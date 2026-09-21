"""
Self-Reward Agent 增强版

继承BaseAgent，集成轨迹记录、Self-Reward评分、Critic反思、经验存储、
偏好数据生成和RLHF训练触发功能。实现多候选生成和行为选择机制。

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.2
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union, Sequence, Callable

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

from .base_agent import BaseAgent
from core.self_reward.scorer import SelfRewardScorer
from core.self_reward.models import SelfRewardResult
from core.critic.critic_agent import CriticAgent
from core.critic.models import CriticReport
from core.trajectory.recorder import TrajectoryRecorder
from core.trajectory.models import ExecutionTrajectory
from core.memory.experience_store import ExperienceStore
from core.memory.models import ExperienceRecord
from rlhf_integration.preference_generator import PreferenceGenerator
from rlhf_integration.training_trigger import TrainingTrigger, TriggerConfig, TrainingStatus
from config import get_logger

logger = get_logger(__name__)


class CandidateResult:
    """候选响应结果"""
    def __init__(
        self,
        response: str,
        score_result: SelfRewardResult,
        trajectory: Optional[ExecutionTrajectory] = None
    ):
        self.response = response
        self.score_result = score_result
        self.trajectory = trajectory


class SelfRewardAgent(BaseAgent):
    """
    Self-Reward增强版Agent
    
    在BaseAgent基础上增加：
    1. 轨迹记录 - 记录完整执行过程
    2. Self-Reward评分 - 对响应进行多维度评分
    3. 多候选生成 - 生成多个候选响应并选择最佳
    4. Patch应用 - 对低分响应应用改进补丁
    5. Critic反思 - 分析执行轨迹生成改进建议
    6. 经验存储 - 持久化存储经验记录
    7. 偏好数据生成 - 自动生成用于RM训练的偏好数据
    8. RLHF训练触发 - 数据量达到阈值时自动触发训练
    
    Attributes:
        scorer: Self-Reward评分器
        recorder: 轨迹记录器
        critic: Critic反思Agent
        memory: 经验存储
        preference_generator: 偏好数据生成器
        training_trigger: 训练触发器
        num_candidates: 候选数量
        score_threshold: 评分阈值
        
    Example:
        >>> agent = SelfRewardAgent(num_candidates=3, score_threshold=6.0)
        >>> response = agent.invoke("请解释什么是机器学习")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: Optional[Union[str, BaseChatModel]] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        num_candidates: int = 1,
        score_threshold: float = 6.0,
        enable_critic: bool = True,
        enable_memory: bool = True,
        enable_rlhf: bool = True,
        rlhf_trigger_threshold: int = 100,
        rm_training_script: Optional[str] = None,
        on_training_triggered: Optional[Callable[[str], None]] = None,
        system_prompt: Optional[str] = None,
        prompt_mode: str = "default",
        debug: bool = False,
        **kwargs: Any,
    ):
        """
        初始化Self-Reward Agent
        
        Args:
            model: LLM模型
            tools: 工具列表
            num_candidates: 候选响应数量（默认1，可配置为多个）
            score_threshold: 评分阈值，低于此值将应用patch
            enable_critic: 是否启用Critic反思
            enable_memory: 是否启用经验存储
            enable_rlhf: 是否启用RLHF训练触发（需要enable_memory=True）
            rlhf_trigger_threshold: 触发RLHF训练的偏好数据量阈值（默认100）
            rm_training_script: RM训练脚本路径（可选）
            on_training_triggered: 训练触发时的回调函数
            system_prompt: 自定义系统提示词
            prompt_mode: 提示词模式
            debug: 是否启用调试日志
            **kwargs: 其他传递给BaseAgent的参数
        """
        # 调用父类初始化
        super().__init__(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            prompt_mode=prompt_mode,
            debug=debug,
            **kwargs,
        )
        
        # 配置参数
        self.num_candidates = max(1, num_candidates)
        self.score_threshold = score_threshold
        self.enable_critic = enable_critic
        self.enable_memory = enable_memory
        self.enable_rlhf = enable_rlhf and enable_memory  # RLHF需要memory
        
        # 初始化评分器（使用与Agent相同的模型）
        self.scorer = SelfRewardScorer(
            model=self.model,
            threshold=score_threshold
        )
        
        # 初始化轨迹记录器
        self.recorder = TrajectoryRecorder()
        
        # 初始化Critic Agent（可选）
        self.critic = CriticAgent(self.model) if enable_critic else None
        
        # 初始化经验存储（可选）
        self._memory: Optional[ExperienceStore] = None
        if enable_memory:
            try:
                self._memory = ExperienceStore()
                logger.info("经验存储初始化成功")
            except Exception as e:
                logger.warning(f"经验存储初始化失败，将禁用: {e}")
                self._memory = None
                self.enable_rlhf = False  # 没有memory就不能启用RLHF
        
        # 初始化偏好数据生成器和训练触发器（可选）
        self._preference_generator: Optional[PreferenceGenerator] = None
        self._training_trigger: Optional[TrainingTrigger] = None
        self._rlhf_trigger_threshold = rlhf_trigger_threshold
        self._last_training_check_count = 0
        
        if self.enable_rlhf and self._memory:
            try:
                # 初始化偏好数据生成器
                self._preference_generator = PreferenceGenerator(
                    self._memory,
                    chosen_threshold=7.0,
                    rejected_threshold=score_threshold
                )
                
                # 初始化训练触发器
                trigger_config = TriggerConfig(
                    min_data_count=rlhf_trigger_threshold,
                    auto_trigger=True,
                    rm_training_script=rm_training_script
                )
                self._training_trigger = TrainingTrigger(
                    experience_store=self._memory,
                    config=trigger_config
                )
                
                # 设置训练回调
                if on_training_triggered:
                    self._training_trigger.set_training_callback(on_training_triggered)
                
                logger.info(f"RLHF训练触发器初始化成功，阈值: {rlhf_trigger_threshold}")
            except Exception as e:
                logger.warning(f"RLHF训练触发器初始化失败: {e}")
                self._preference_generator = None
                self._training_trigger = None
                self.enable_rlhf = False
        
        # 候选记录（用于多候选模式）
        self._candidate_records: List[CandidateResult] = []
        
        logger.info(
            f"SelfRewardAgent初始化完成: "
            f"candidates={self.num_candidates}, "
            f"threshold={self.score_threshold}, "
            f"critic={enable_critic}, "
            f"memory={enable_memory and self._memory is not None}, "
            f"rlhf={self.enable_rlhf}"
        )
    
    @property
    def memory(self) -> Optional[ExperienceStore]:
        """获取经验存储实例"""
        return self._memory

    def invoke(
        self,
        input_text: str,
        chat_history: Optional[List[BaseMessage]] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        增强版invoke，包含完整的Self-Reward流程
        
        流程：
        1. 轨迹记录开始
        2. 生成候选响应（单个或多个）
        3. Self-Reward评分
        4. 行为选择（选择最高分或应用patch）
        5. Critic反思（可选）
        6. 经验存储（可选）
        
        Args:
            input_text: 用户输入
            chat_history: 对话历史
            session_id: 会话ID（可选，自动生成）
            **kwargs: 其他参数
            
        Returns:
            最终响应文本
        """
        # 生成会话ID
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        logger.info(f"SelfRewardAgent执行: session={session_id[:8]}...")
        
        # 清空候选记录
        self._candidate_records = []
        
        try:
            # 生成候选响应并评分
            candidates = self._generate_and_score_candidates(
                input_text, chat_history, session_id
            )
            
            if not candidates:
                logger.error("未能生成任何候选响应")
                return "抱歉，处理您的请求时出现错误。"
            
            # 选择最佳响应
            best_response, best_result, best_trajectory = self._select_best(
                candidates, input_text
            )
            
            # 记录所有候选（用于分析）
            self._candidate_records = candidates
            
            # Critic反思
            critic_report = None
            if self.enable_critic and self.critic and best_trajectory:
                try:
                    critic_report = self.critic.analyze(best_trajectory)
                    logger.info(f"Critic分析完成: {len(critic_report.problems)}个问题")
                except Exception as e:
                    logger.warning(f"Critic分析失败: {e}")
                    critic_report = self._create_empty_critic_report()
            else:
                critic_report = self._create_empty_critic_report()
            
            # 经验存储
            if self._memory and best_trajectory:
                try:
                    record = ExperienceRecord(
                        record_id=str(uuid.uuid4()),
                        session_id=session_id,
                        trajectory=best_trajectory,
                        self_reward=best_result,
                        critic_report=critic_report,
                        timestamp=datetime.now().isoformat(),
                        tags=[]
                    )
                    self._memory.save(record)
                    logger.info(f"经验记录已保存: {record.record_id[:8]}...")
                    
                    # 检查是否需要触发RLHF训练
                    if self.enable_rlhf:
                        self._check_and_trigger_training()
                        
                except Exception as e:
                    logger.warning(f"经验存储失败: {e}")
            
            return best_response
            
        except Exception as e:
            logger.error(f"SelfRewardAgent执行失败: {e}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"

    def _generate_and_score_candidates(
        self,
        input_text: str,
        chat_history: Optional[List[BaseMessage]],
        session_id: str
    ) -> List[CandidateResult]:
        """
        生成候选响应并评分
        
        Args:
            input_text: 用户输入
            chat_history: 对话历史
            session_id: 会话ID
            
        Returns:
            候选结果列表
        """
        candidates = []
        
        for i in range(self.num_candidates):
            logger.debug(f"生成候选 {i+1}/{self.num_candidates}")
            
            # 开始轨迹记录
            self.recorder.start(
                session_id=f"{session_id}_candidate_{i}",
                user_input=input_text,
                system_prompt=self.system_prompt
            )
            
            try:
                # 调用父类生成响应
                response = super().invoke(input_text, chat_history)
                
                # 完成轨迹记录
                trajectory = self.recorder.finish(response)
                
                # 评分
                score_result = self.scorer.score(
                    query=input_text,
                    response=response
                )
                
                candidates.append(CandidateResult(
                    response=response,
                    score_result=score_result,
                    trajectory=trajectory
                ))
                
                logger.debug(
                    f"候选 {i+1} 评分: {score_result.score:.2f}"
                )
                
            except Exception as e:
                logger.warning(f"候选 {i+1} 生成失败: {e}")
                # 取消当前轨迹记录
                if self.recorder.is_recording:
                    self.recorder.cancel()
        
        return candidates
    
    def _select_best(
        self,
        candidates: List[CandidateResult],
        query: str
    ) -> Tuple[str, SelfRewardResult, Optional[ExecutionTrajectory]]:
        """
        选择最佳响应
        
        规则：
        1. 选择评分最高的候选
        2. 如果最高分仍低于阈值，应用patch修正
        
        Args:
            candidates: 候选结果列表
            query: 原始查询
            
        Returns:
            (最终响应, 评分结果, 轨迹) 元组
        """
        if not candidates:
            raise ValueError("候选列表为空")
        
        # 按评分排序，选择最高分
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.score_result.score,
            reverse=True
        )
        
        best = sorted_candidates[0]
        
        logger.info(
            f"选择最佳候选: score={best.score_result.score:.2f}, "
            f"threshold={self.score_threshold}"
        )
        
        # 如果最高分仍低于阈值，尝试应用patch
        if best.score_result.score < self.score_threshold:
            if best.score_result.patch:
                logger.info("应用patch修正响应")
                patched_response = self._apply_patch(
                    best.response,
                    best.score_result.patch
                )
                return patched_response, best.score_result, best.trajectory
            else:
                logger.warning("评分低于阈值但无可用patch")
        
        return best.response, best.score_result, best.trajectory

    def _apply_patch(self, response: str, patch: str) -> str:
        """
        应用改进补丁
        
        patch可能是：
        1. 完整的替换响应
        2. 修改建议（需要合并）
        
        Args:
            response: 原始响应
            patch: 改进补丁
            
        Returns:
            修正后的响应
        """
        # 如果patch看起来是完整响应，直接使用
        if len(patch) > len(response) * 0.5:
            logger.debug("使用patch作为完整替换")
            return patch
        
        # 否则，将patch作为补充附加到响应
        logger.debug("将patch作为补充附加")
        return f"{response}\n\n【补充说明】\n{patch}"
    
    def _create_empty_critic_report(self) -> CriticReport:
        """创建空的Critic报告"""
        return CriticReport(
            problems=[],
            root_causes=[],
            improvements=[],
            overall_assessment="未进行Critic分析",
            timestamp=datetime.now().isoformat()
        )
    
    def get_candidate_records(self) -> List[CandidateResult]:
        """
        获取最近一次执行的所有候选记录
        
        用于分析多候选模式下的所有候选及其评分。
        
        Returns:
            候选结果列表
        """
        return self._candidate_records.copy()
    
    def get_experience_summary(self) -> Dict[str, Any]:
        """
        获取经验统计摘要
        
        Returns:
            包含统计信息的字典
        """
        if not self._memory:
            return {
                "enabled": False,
                "message": "经验存储未启用"
            }
        
        try:
            total_count = self._memory.count()
            high_score_count = self._memory.count({"min_score": 7.0})
            low_score_count = self._memory.count({"max_score": 6.0})
            
            result = {
                "enabled": True,
                "total_records": total_count,
                "high_score_records": high_score_count,
                "low_score_records": low_score_count,
                "score_threshold": self.score_threshold
            }
            
            # 添加RLHF相关统计
            if self.enable_rlhf and self._preference_generator:
                try:
                    pref_stats = self._preference_generator.get_statistics()
                    result["rlhf"] = {
                        "enabled": True,
                        "preference_pairs": pref_stats.get("generated_pairs", 0),
                        "trigger_threshold": self._rlhf_trigger_threshold,
                        "training_status": self._training_trigger.status.value if self._training_trigger else "unknown"
                    }
                except Exception as e:
                    result["rlhf"] = {"enabled": True, "error": str(e)}
            else:
                result["rlhf"] = {"enabled": False}
            
            return result
        except Exception as e:
            logger.warning(f"获取经验统计失败: {e}")
            return {
                "enabled": True,
                "error": str(e)
            }
    
    def _check_and_trigger_training(self) -> None:
        """
        检查偏好数据量并在达到阈值时触发训练
        
        每次经验存储后调用，检查是否有足够的偏好数据来触发RM训练。
        """
        if not self._training_trigger or not self._preference_generator:
            return
        
        try:
            # 获取当前偏好数据统计
            stats = self._preference_generator.get_statistics()
            current_pairs = stats.get("generated_pairs", 0)
            
            # 避免频繁检查：只有当数据量有显著增加时才检查
            if current_pairs <= self._last_training_check_count:
                return
            
            self._last_training_check_count = current_pairs
            
            # 检查是否达到阈值
            check_result = self._training_trigger.check_data_ready()
            
            if check_result.status == TrainingStatus.DATA_READY:
                logger.info(
                    f"偏好数据量达到阈值 ({current_pairs}/{self._rlhf_trigger_threshold})，"
                    f"触发RLHF训练..."
                )
                
                # 触发训练
                trigger_result = self._training_trigger.trigger_training()
                
                if trigger_result.triggered:
                    logger.info(
                        f"RLHF训练已触发！"
                        f"导出路径: {trigger_result.export_path}"
                    )
                else:
                    logger.warning(f"RLHF训练触发失败: {trigger_result.message}")
            else:
                logger.debug(
                    f"偏好数据量: {current_pairs}/{self._rlhf_trigger_threshold}, "
                    f"状态: {check_result.status.value}"
                )
                
        except Exception as e:
            logger.warning(f"检查训练触发失败: {e}")
    
    def force_trigger_training(self) -> Dict[str, Any]:
        """
        强制触发RLHF训练（忽略数据量阈值）
        
        Returns:
            触发结果信息
        """
        if not self._training_trigger:
            return {
                "success": False,
                "message": "RLHF训练触发器未启用"
            }
        
        try:
            result = self._training_trigger.trigger_training(force=True)
            return {
                "success": result.triggered,
                "status": result.status.value,
                "data_count": result.data_count,
                "export_path": result.export_path,
                "message": result.message
            }
        except Exception as e:
            logger.error(f"强制触发训练失败: {e}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def get_preference_pairs(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取当前生成的偏好数据对
        
        Args:
            limit: 返回数量限制
            
        Returns:
            偏好数据对列表
        """
        if not self._preference_generator:
            return []
        
        try:
            pairs = self._preference_generator.generate_pairs(limit=limit)
            return [
                {
                    "prompt": p.prompt,
                    "chosen": p.chosen,
                    "rejected": p.rejected,
                    "chosen_score": p.chosen_score,
                    "rejected_score": p.rejected_score
                }
                for p in pairs
            ]
        except Exception as e:
            logger.warning(f"获取偏好数据对失败: {e}")
            return []
    
    def export_preference_data(self, output_path: str) -> int:
        """
        导出偏好数据到文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            导出的数据对数量
        """
        if not self._preference_generator:
            logger.warning("偏好数据生成器未启用")
            return 0
        
        try:
            count = self._preference_generator.export_for_rm_training(output_path)
            logger.info(f"导出 {count} 条偏好数据到 {output_path}")
            return count
        except Exception as e:
            logger.error(f"导出偏好数据失败: {e}")
            return 0


def create_self_reward_agent(
    model: Optional[Union[str, BaseChatModel]] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    num_candidates: int = 1,
    score_threshold: float = 6.0,
    enable_critic: bool = True,
    enable_memory: bool = True,
    enable_rlhf: bool = True,
    rlhf_trigger_threshold: int = 100,
    rm_training_script: Optional[str] = None,
    on_training_triggered: Optional[Callable[[str], None]] = None,
    prompt_mode: str = "default",
    debug: bool = False,
    **kwargs: Any,
) -> SelfRewardAgent:
    """
    创建Self-Reward Agent的便捷工厂函数
    
    Args:
        model: LLM模型
        tools: 工具列表
        num_candidates: 候选响应数量
        score_threshold: 评分阈值
        enable_critic: 是否启用Critic反思
        enable_memory: 是否启用经验存储
        enable_rlhf: 是否启用RLHF训练触发
        rlhf_trigger_threshold: 触发RLHF训练的偏好数据量阈值
        rm_training_script: RM训练脚本路径
        on_training_triggered: 训练触发时的回调函数
        prompt_mode: 提示词模式
        debug: 是否启用调试日志
        **kwargs: 其他参数
        
    Returns:
        配置好的SelfRewardAgent实例
        
    Example:
        >>> # 创建默认Agent（启用RLHF）
        >>> agent = create_self_reward_agent()
        >>> 
        >>> # 创建多候选Agent
        >>> agent = create_self_reward_agent(num_candidates=3)
        >>> 
        >>> # 创建带自定义训练回调的Agent
        >>> def on_training(path):
        ...     print(f"训练数据已导出到: {path}")
        >>> agent = create_self_reward_agent(
        ...     on_training_triggered=on_training,
        ...     rlhf_trigger_threshold=50
        ... )
        >>> 
        >>> # 创建不带RLHF的轻量Agent
        >>> agent = create_self_reward_agent(
        ...     enable_critic=False,
        ...     enable_memory=False,
        ...     enable_rlhf=False
        ... )
    """
    logger.info(
        f"创建SelfRewardAgent: "
        f"candidates={num_candidates}, "
        f"threshold={score_threshold}, "
        f"rlhf={enable_rlhf}"
    )
    
    return SelfRewardAgent(
        model=model,
        tools=tools,
        num_candidates=num_candidates,
        score_threshold=score_threshold,
        enable_critic=enable_critic,
        enable_memory=enable_memory,
        enable_rlhf=enable_rlhf,
        rlhf_trigger_threshold=rlhf_trigger_threshold,
        rm_training_script=rm_training_script,
        on_training_triggered=on_training_triggered,
        prompt_mode=prompt_mode,
        debug=debug,
        **kwargs,
    )

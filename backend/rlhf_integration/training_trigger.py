"""
训练触发器模块

实现数据量检查和触发逻辑，与现有RM训练流程的集成接口。
支持自动检测偏好数据量并触发奖励模型微调。
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from enum import Enum

from pydantic import BaseModel, Field

from core.memory.experience_store import ExperienceStore
from rlhf_integration.preference_generator import PreferenceGenerator

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    """训练状态枚举"""
    PENDING = "pending"           # 等待中
    DATA_READY = "data_ready"     # 数据就绪
    TRAINING = "training"         # 训练中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败


class TriggerConfig(BaseModel):
    """触发器配置"""
    min_data_count: int = Field(default=100, ge=1, description="触发训练的最小数据量")
    auto_trigger: bool = Field(default=False, description="是否自动触发训练")
    export_path: str = Field(default="data/preference_data", description="偏好数据导出路径")
    rm_training_script: Optional[str] = Field(default=None, description="RM训练脚本路径")
    chosen_threshold: float = Field(default=7.0, ge=0, le=10, description="chosen阈值")
    rejected_threshold: float = Field(default=6.0, ge=0, le=10, description="rejected阈值")


class TriggerResult(BaseModel):
    """触发结果"""
    triggered: bool = Field(description="是否触发了训练")
    status: TrainingStatus = Field(description="当前状态")
    data_count: int = Field(description="当前数据量")
    min_required: int = Field(description="最小要求数据量")
    export_path: Optional[str] = Field(default=None, description="导出文件路径")
    message: str = Field(description="状态消息")
    timestamp: str = Field(description="时间戳")


class TrainingTrigger:
    """
    训练触发器
    
    负责检查偏好数据量，在达到阈值时触发RM训练。
    提供与现有RM训练流程的集成接口。
    """
    
    def __init__(
        self,
        experience_store: Optional[ExperienceStore] = None,
        config: Optional[TriggerConfig] = None
    ):
        """
        初始化训练触发器
        
        Args:
            experience_store: 经验存储实例，如果为None则创建新实例
            config: 触发器配置，如果为None则使用默认配置
        """
        self.store = experience_store or ExperienceStore()
        self.config = config or TriggerConfig()
        self.generator = PreferenceGenerator(
            self.store,
            chosen_threshold=self.config.chosen_threshold,
            rejected_threshold=self.config.rejected_threshold
        )
        self._status = TrainingStatus.PENDING
        self._last_check_time: Optional[str] = None
        self._training_callback: Optional[Callable[[str], None]] = None
    
    @property
    def status(self) -> TrainingStatus:
        """获取当前训练状态"""
        return self._status
    
    def set_training_callback(self, callback: Callable[[str], None]) -> None:
        """
        设置训练回调函数
        
        Args:
            callback: 训练完成后的回调函数，接收导出文件路径作为参数
        """
        self._training_callback = callback
    
    def check_data_ready(self) -> TriggerResult:
        """
        检查数据是否就绪
        
        检查当前偏好数据量是否达到触发阈值。
        
        Returns:
            TriggerResult: 检查结果
        """
        self._last_check_time = datetime.now().isoformat()
        
        try:
            # 获取统计信息
            stats = self.generator.get_statistics()
            data_count = stats.get("generated_pairs", 0)
            
            # 检查是否达到阈值
            is_ready = data_count >= self.config.min_data_count
            
            if is_ready:
                self._status = TrainingStatus.DATA_READY
                message = f"数据就绪: {data_count} 条偏好数据，已达到阈值 {self.config.min_data_count}"
            else:
                self._status = TrainingStatus.PENDING
                message = f"数据不足: {data_count}/{self.config.min_data_count} 条偏好数据"
            
            logger.info(message)
            
            return TriggerResult(
                triggered=False,
                status=self._status,
                data_count=data_count,
                min_required=self.config.min_data_count,
                message=message,
                timestamp=self._last_check_time
            )
            
        except Exception as e:
            logger.error(f"检查数据就绪状态失败: {e}")
            self._status = TrainingStatus.FAILED
            return TriggerResult(
                triggered=False,
                status=self._status,
                data_count=0,
                min_required=self.config.min_data_count,
                message=f"检查失败: {str(e)}",
                timestamp=self._last_check_time or datetime.now().isoformat()
            )
    
    def trigger_training(self, force: bool = False) -> TriggerResult:
        """
        触发训练
        
        如果数据量达到阈值或force=True，则导出数据并触发训练。
        
        Args:
            force: 是否强制触发（忽略数据量检查）
            
        Returns:
            TriggerResult: 触发结果
        """
        timestamp = datetime.now().isoformat()
        
        # 检查数据就绪状态
        check_result = self.check_data_ready()
        
        if not force and check_result.status != TrainingStatus.DATA_READY:
            return TriggerResult(
                triggered=False,
                status=check_result.status,
                data_count=check_result.data_count,
                min_required=self.config.min_data_count,
                message=f"未触发训练: {check_result.message}",
                timestamp=timestamp
            )
        
        try:
            # 导出偏好数据
            export_path = self._export_preference_data()
            
            if export_path is None:
                self._status = TrainingStatus.FAILED
                return TriggerResult(
                    triggered=False,
                    status=self._status,
                    data_count=check_result.data_count,
                    min_required=self.config.min_data_count,
                    message="导出偏好数据失败",
                    timestamp=timestamp
                )
            
            # 更新状态
            self._status = TrainingStatus.TRAINING
            
            # 如果配置了训练脚本，执行训练
            if self.config.rm_training_script:
                self._execute_training(export_path)
            
            # 调用回调
            if self._training_callback:
                self._training_callback(export_path)
            
            self._status = TrainingStatus.COMPLETED
            
            return TriggerResult(
                triggered=True,
                status=self._status,
                data_count=check_result.data_count,
                min_required=self.config.min_data_count,
                export_path=export_path,
                message=f"训练已触发，数据导出到: {export_path}",
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"触发训练失败: {e}")
            self._status = TrainingStatus.FAILED
            return TriggerResult(
                triggered=False,
                status=self._status,
                data_count=check_result.data_count,
                min_required=self.config.min_data_count,
                message=f"触发训练失败: {str(e)}",
                timestamp=timestamp
            )
    
    def _export_preference_data(self) -> Optional[str]:
        """
        导出偏好数据
        
        Returns:
            导出文件路径，失败返回None
        """
        try:
            # 确保导出目录存在
            export_dir = Path(self.config.export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = export_dir / f"preference_data_{timestamp}.json"
            
            # 导出数据
            count = self.generator.export_for_rm_training(str(export_file))
            
            if count > 0:
                logger.info(f"导出 {count} 条偏好数据到 {export_file}")
                return str(export_file)
            else:
                logger.warning("没有可导出的偏好数据")
                return None
                
        except Exception as e:
            logger.error(f"导出偏好数据失败: {e}")
            return None
    
    def _execute_training(self, data_path: str) -> bool:
        """
        执行训练脚本
        
        Args:
            data_path: 偏好数据文件路径
            
        Returns:
            是否执行成功
        """
        if not self.config.rm_training_script:
            logger.warning("未配置训练脚本路径")
            return False
        
        script_path = Path(self.config.rm_training_script)
        if not script_path.exists():
            logger.error(f"训练脚本不存在: {script_path}")
            return False
        
        try:
            # 执行训练脚本
            logger.info(f"执行训练脚本: {script_path} --data {data_path}")
            
            result = subprocess.run(
                [sys.executable, str(script_path), "--data", data_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            if result.returncode == 0:
                logger.info("训练脚本执行成功")
                return True
            else:
                logger.error(f"训练脚本执行失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("训练脚本执行超时")
            return False
        except Exception as e:
            logger.error(f"执行训练脚本失败: {e}")
            return False
    
    def get_integration_interface(self) -> Dict[str, Any]:
        """
        获取与现有RM训练流程的集成接口信息
        
        Returns:
            集成接口信息字典
        """
        return {
            "data_format": {
                "description": "偏好数据格式，与现有RM训练兼容",
                "schema": {
                    "prompt": "str - 用户输入/提示",
                    "chosen": "str - 高质量响应",
                    "rejected": "str - 低质量响应",
                    "metadata": {
                        "chosen_score": "float - chosen响应的评分",
                        "rejected_score": "float - rejected响应的评分",
                        "source": "str - 数据来源标识",
                        "source_records": "List[str] - 来源记录ID"
                    }
                }
            },
            "export_method": "export_for_rm_training(output_path: str) -> int",
            "trigger_method": "trigger_training(force: bool = False) -> TriggerResult",
            "check_method": "check_data_ready() -> TriggerResult",
            "config": self.config.model_dump(),
            "current_status": self._status.value
        }
    
    def reset_status(self) -> None:
        """重置训练状态为PENDING"""
        self._status = TrainingStatus.PENDING
        logger.info("训练状态已重置为PENDING")


class TrainingMetrics(BaseModel):
    """训练指标"""
    total_pairs: int = Field(description="总偏好数据对数量")
    high_score_records: int = Field(description="高分记录数量")
    low_score_records: int = Field(description="低分记录数量")
    records_with_patch: int = Field(description="带patch的记录数量")
    average_score_difference: float = Field(description="平均分数差")
    chosen_threshold: float = Field(description="chosen阈值")
    rejected_threshold: float = Field(description="rejected阈值")


class TrainingReport(BaseModel):
    """训练报告"""
    report_id: str = Field(description="报告ID")
    training_status: TrainingStatus = Field(description="训练状态")
    trigger_result: Optional[TriggerResult] = Field(default=None, description="触发结果")
    metrics: TrainingMetrics = Field(description="训练指标")
    data_export_path: Optional[str] = Field(default=None, description="数据导出路径")
    training_start_time: Optional[str] = Field(default=None, description="训练开始时间")
    training_end_time: Optional[str] = Field(default=None, description="训练结束时间")
    training_duration_seconds: Optional[float] = Field(default=None, description="训练耗时(秒)")
    improvement_indicators: Dict[str, Any] = Field(default_factory=dict, description="改进指标")
    notes: List[str] = Field(default_factory=list, description="备注")
    generated_at: str = Field(description="报告生成时间")


class TrainingReportGenerator:
    """
    训练报告生成器
    
    记录训练数据量、指标，生成训练报告。
    """
    
    def __init__(
        self,
        training_trigger: TrainingTrigger,
        report_output_dir: str = "data/training_reports"
    ):
        """
        初始化训练报告生成器
        
        Args:
            training_trigger: 训练触发器实例
            report_output_dir: 报告输出目录
        """
        self.trigger = training_trigger
        self.report_output_dir = Path(report_output_dir)
        self._current_report: Optional[TrainingReport] = None
        self._training_start_time: Optional[datetime] = None
    
    def start_training_session(self) -> str:
        """
        开始训练会话
        
        Returns:
            报告ID
        """
        import uuid
        
        self._training_start_time = datetime.now()
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 获取当前指标
        stats = self.trigger.generator.get_statistics()
        
        metrics = TrainingMetrics(
            total_pairs=stats.get("generated_pairs", 0),
            high_score_records=stats.get("high_score_records", 0),
            low_score_records=stats.get("low_score_records", 0),
            records_with_patch=stats.get("records_with_patch", 0),
            average_score_difference=stats.get("average_score_difference", 0.0),
            chosen_threshold=stats.get("chosen_threshold", 7.0),
            rejected_threshold=stats.get("rejected_threshold", 6.0)
        )
        
        self._current_report = TrainingReport(
            report_id=report_id,
            training_status=TrainingStatus.PENDING,
            metrics=metrics,
            training_start_time=self._training_start_time.isoformat(),
            generated_at=datetime.now().isoformat()
        )
        
        logger.info(f"训练会话已开始: {report_id}")
        return report_id
    
    def record_trigger_result(self, result: TriggerResult) -> None:
        """
        记录触发结果
        
        Args:
            result: 触发结果
        """
        if self._current_report is None:
            logger.warning("未开始训练会话，无法记录触发结果")
            return
        
        self._current_report.trigger_result = result
        self._current_report.training_status = result.status
        self._current_report.data_export_path = result.export_path
        
        # 更新指标
        self._current_report.metrics.total_pairs = result.data_count
        
        logger.info(f"已记录触发结果: {result.status.value}")
    
    def record_training_completion(
        self,
        success: bool,
        improvement_indicators: Optional[Dict[str, Any]] = None,
        notes: Optional[List[str]] = None
    ) -> None:
        """
        记录训练完成
        
        Args:
            success: 训练是否成功
            improvement_indicators: 改进指标
            notes: 备注
        """
        if self._current_report is None:
            logger.warning("未开始训练会话，无法记录训练完成")
            return
        
        end_time = datetime.now()
        self._current_report.training_end_time = end_time.isoformat()
        
        if self._training_start_time:
            duration = (end_time - self._training_start_time).total_seconds()
            self._current_report.training_duration_seconds = duration
        
        self._current_report.training_status = (
            TrainingStatus.COMPLETED if success else TrainingStatus.FAILED
        )
        
        if improvement_indicators:
            self._current_report.improvement_indicators = improvement_indicators
        
        if notes:
            self._current_report.notes.extend(notes)
        
        logger.info(f"训练完成记录: success={success}")
    
    def add_note(self, note: str) -> None:
        """
        添加备注
        
        Args:
            note: 备注内容
        """
        if self._current_report is None:
            logger.warning("未开始训练会话，无法添加备注")
            return
        
        self._current_report.notes.append(note)
    
    def generate_report(self) -> TrainingReport:
        """
        生成训练报告
        
        Returns:
            训练报告
        """
        if self._current_report is None:
            raise ValueError("未开始训练会话，无法生成报告")
        
        # 更新生成时间
        self._current_report.generated_at = datetime.now().isoformat()
        
        return self._current_report
    
    def save_report(self, report: Optional[TrainingReport] = None) -> str:
        """
        保存训练报告到文件
        
        Args:
            report: 训练报告，如果为None则使用当前报告
            
        Returns:
            报告文件路径
        """
        report = report or self._current_report
        
        if report is None:
            raise ValueError("没有可保存的报告")
        
        # 确保输出目录存在
        self.report_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        filename = f"{report.report_id}.json"
        filepath = self.report_output_dir / filename
        
        # 保存报告
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练报告已保存: {filepath}")
        return str(filepath)
    
    def load_report(self, report_id: str) -> TrainingReport:
        """
        加载训练报告
        
        Args:
            report_id: 报告ID
            
        Returns:
            训练报告
        """
        filepath = self.report_output_dir / f"{report_id}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"报告文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return TrainingReport.model_validate(data)
    
    def list_reports(self) -> List[str]:
        """
        列出所有报告ID
        
        Returns:
            报告ID列表
        """
        if not self.report_output_dir.exists():
            return []
        
        reports = []
        for filepath in self.report_output_dir.glob("report_*.json"):
            report_id = filepath.stem
            reports.append(report_id)
        
        return sorted(reports, reverse=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取报告摘要
        
        Returns:
            摘要信息
        """
        reports = self.list_reports()
        
        if not reports:
            return {
                "total_reports": 0,
                "latest_report": None,
                "summary": "暂无训练报告"
            }
        
        # 加载最新报告
        latest_report = self.load_report(reports[0])
        
        return {
            "total_reports": len(reports),
            "latest_report": {
                "report_id": latest_report.report_id,
                "status": latest_report.training_status.value,
                "total_pairs": latest_report.metrics.total_pairs,
                "generated_at": latest_report.generated_at
            },
            "all_report_ids": reports[:10]  # 最近10个
        }

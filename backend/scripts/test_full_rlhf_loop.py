#!/usr/bin/env python
"""
完整RLHF闭环测试脚本

测试从轨迹生成、Self-Reward评分、Critic反思、经验存储到RM训练触发的完整流程。
当积累10条偏好数据时，自动触发RM训练。

使用方法:
    python scripts/test_full_rlhf_loop.py
    python scripts/test_full_rlhf_loop.py --no-cleanup  # 保留数据到PostgreSQL
    python scripts/test_full_rlhf_loop.py --run-rm-training  # 真正执行RM训练

流程:
    1. 生成多条执行轨迹（模拟Agent执行）
    2. 对每条轨迹进行Self-Reward评分
    3. 使用Critic进行反思分析
    4. 将经验记录存储到PostgreSQL数据库
    5. 当偏好数据达到10条时，触发RM训练
"""

import sys
import os
import uuid
import random
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# 添加backend目录到路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# 添加rlhf目录到路径（用于RM训练）
rlhf_dir = backend_dir.parent.parent.parent / "rlhf"
sys.path.insert(0, str(rlhf_dir))

from config import get_logger

logger = get_logger(__name__)


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_section(title: str):
    """打印小节标题"""
    print(f"\n--- {title} ---\n")


def print_progress(current: int, total: int, message: str):
    """打印进度"""
    print(f"[{current}/{total}] {message}")


# ==================== 测试数据 ====================

# 模拟的用户问题和期望的响应质量
# 关键：相同问题需要有高分和低分版本，才能生成偏好数据对
TEST_QUERIES = [
    # (问题, 期望质量: high/low, 是否需要工具)
    # 问题1: 机器学习 - 高分和低分版本
    ("请解释什么是机器学习？", "high", False),
    ("请解释什么是机器学习？", "low", False),
    
    # 问题2: 深度学习 - 高分和低分版本
    ("什么是深度学习？", "high", False),
    ("什么是深度学习？", "low", False),
    
    # 问题3: 神经网络 - 高分和低分版本
    ("什么是神经网络？", "high", False),
    ("什么是神经网络？", "low", False),
    
    # 问题4: Python - 高分和低分版本
    ("Python和Java哪个更好？", "high", False),
    ("Python和Java哪个更好？", "low", False),
    
    # 问题5: 区块链 - 高分和低分版本
    ("请解释什么是区块链？", "high", False),
    ("请解释什么是区块链？", "low", False),
    
    # 问题6: 人工智能 - 高分和低分版本
    ("如何学习人工智能？", "high", False),
    ("如何学习人工智能？", "low", False),
    
    # 问题7: 自然语言处理 - 高分和低分版本
    ("请解释什么是自然语言处理？", "high", False),
    ("请解释什么是自然语言处理？", "low", False),
    
    # 问题8: 量子计算 - 高分和低分版本
    ("请解释量子计算的基本原理", "high", False),
    ("请解释量子计算的基本原理", "low", False),
    
    # 问题9: 编程书籍 - 高分和低分版本
    ("请推荐几本编程入门书籍", "high", False),
    ("请推荐几本编程入门书籍", "low", False),
    
    # 问题10: 计算 - 高分和低分版本
    ("请帮我计算123*456", "high", False),
    ("请帮我计算123*456", "low", False),
    
    # 问题11: 翻译 - 高分和低分版本
    ("帮我翻译这句话：Hello World", "high", False),
    ("帮我翻译这句话：Hello World", "low", False),
    
    # 问题12: 春天的诗 - 高分和低分版本
    ("请帮我写一首关于春天的诗", "high", False),
    ("请帮我写一首关于春天的诗", "low", False),
]


# ==================== 核心测试类 ====================

class FullRLHFLoopTest:
    """完整RLHF闭环测试"""
    
    def __init__(self, trigger_threshold: int = 10, run_rm_training: bool = False):
        """
        初始化测试
        
        Args:
            trigger_threshold: 触发RM训练的数据量阈值
            run_rm_training: 是否真正执行RM训练
        """
        self.trigger_threshold = trigger_threshold
        self.run_rm_training = run_rm_training
        self.created_record_ids = []
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化所有组件"""
        print_section("初始化组件")
        
        # 导入所需模块
        from core.trajectory.recorder import TrajectoryRecorder
        from core.trajectory.models import ExecutionTrajectory, ToolCall, ReasoningStep
        from core.self_reward.models import ScoreDimension, DimensionScore, SelfRewardResult
        from core.critic.models import ProblemType, Problem, Improvement, CriticReport
        from core.memory.models import ExperienceRecord
        from core.memory.experience_store import ExperienceStore
        from rlhf_integration.preference_generator import PreferenceGenerator
        from rlhf_integration.training_trigger import (
            TrainingTrigger, 
            TriggerConfig,
            TrainingReportGenerator
        )
        
        # 保存模块引用
        self.TrajectoryRecorder = TrajectoryRecorder
        self.ExecutionTrajectory = ExecutionTrajectory
        self.ToolCall = ToolCall
        self.ReasoningStep = ReasoningStep
        self.ScoreDimension = ScoreDimension
        self.DimensionScore = DimensionScore
        self.SelfRewardResult = SelfRewardResult
        self.ProblemType = ProblemType
        self.Problem = Problem
        self.Improvement = Improvement
        self.CriticReport = CriticReport
        self.ExperienceRecord = ExperienceRecord
        
        # 初始化经验存储
        print("  初始化经验存储 (PostgreSQL)...")
        self.store = ExperienceStore()
        print("  ✓ 经验存储初始化成功")
        
        # 初始化偏好数据生成器
        print("  初始化偏好数据生成器...")
        self.generator = PreferenceGenerator(
            self.store,
            chosen_threshold=7.0,
            rejected_threshold=6.0
        )
        print("  ✓ 偏好数据生成器初始化成功")
        
        # 初始化训练触发器
        print(f"  初始化训练触发器 (阈值: {self.trigger_threshold})...")
        self.trigger_config = TriggerConfig(
            min_data_count=self.trigger_threshold,
            auto_trigger=False,
            export_path="data/rlhf_test_data",
            chosen_threshold=7.0,
            rejected_threshold=6.0
        )
        self.trigger = TrainingTrigger(
            experience_store=self.store,
            config=self.trigger_config
        )
        print("  ✓ 训练触发器初始化成功")
        
        # 初始化报告生成器
        print("  初始化训练报告生成器...")
        self.report_generator = TrainingReportGenerator(
            training_trigger=self.trigger,
            report_output_dir="data/rlhf_test_reports"
        )
        print("  ✓ 训练报告生成器初始化成功")
        
        # 初始化轨迹记录器
        self.recorder = TrajectoryRecorder()
        
        print("\n✓ 所有组件初始化完成")
    
    def generate_trajectory(
        self, 
        query: str, 
        quality: str,
        use_tool: bool
    ) -> 'ExecutionTrajectory':
        """
        生成执行轨迹
        
        Args:
            query: 用户问题
            quality: 期望质量 (high/low)
            use_tool: 是否使用工具
            
        Returns:
            ExecutionTrajectory
        """
        session_id = str(uuid.uuid4())
        
        # 开始记录
        self.recorder.start(
            session_id=session_id,
            user_input=query,
            system_prompt="你是一个智能助手，请准确、完整地回答用户问题。"
        )
        
        # 添加推理步骤
        self.recorder.add_reasoning(f"分析用户问题: {query}")
        
        if use_tool:
            self.recorder.add_reasoning("需要调用外部工具获取信息")
            
            # 模拟工具调用（可能失败）
            tool_success = random.random() > 0.5
            tool_result = "工具返回结果" if tool_success else {"error": "API调用失败"}
            
            self.recorder.add_tool_call(
                tool_name="external_api",
                input_params={"query": query},
                output_result=tool_result,
                duration_ms=random.uniform(100, 2000),
                success=tool_success,
                error=None if tool_success else "API rate limit exceeded"
            )
            
            if not tool_success:
                self.recorder.add_reasoning("工具调用失败，尝试基于已有知识回答")
        else:
            self.recorder.add_reasoning("基于已有知识直接回答")
        
        self.recorder.add_reasoning("生成最终响应")
        
        # 生成响应
        if quality == "high":
            final_output = self._generate_high_quality_response(query)
        else:
            final_output = self._generate_low_quality_response(query)
        
        # 完成记录
        trajectory = self.recorder.finish(final_output)
        
        return trajectory
    
    def _generate_high_quality_response(self, query: str) -> str:
        """生成高质量响应"""
        responses = {
            "请解释什么是机器学习？": 
                "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，"
                "而无需明确编程。主要类型包括：监督学习、无监督学习和强化学习。"
                "机器学习广泛应用于图像识别、自然语言处理、推荐系统等领域。",
            "1+1等于多少？": 
                "1+1等于2。这是基本的算术加法运算。",
            "请帮我写一首关于春天的诗": 
                "春风轻拂柳丝长，\n桃花朵朵映斜阳。\n"
                "燕子归来寻旧巢，\n万物复苏满园香。",
            "Python和Java哪个更好？": 
                "Python和Java各有优势：Python语法简洁，适合数据科学和快速开发；"
                "Java性能稳定，适合大型企业应用。选择取决于具体需求和场景。",
        }
        return responses.get(query, f"关于'{query}'的详细解答：这是一个很好的问题，让我为您详细解释...")
    
    def _generate_low_quality_response(self, query: str) -> str:
        """生成低质量响应"""
        # 对所有问题都生成简短、不完整的响应
        return "不太清楚，可能是这样吧。"
    
    def score_trajectory(
        self, 
        trajectory: 'ExecutionTrajectory',
        expected_quality: str
    ) -> 'SelfRewardResult':
        """
        对轨迹进行Self-Reward评分
        
        Args:
            trajectory: 执行轨迹
            expected_quality: 期望质量
            
        Returns:
            SelfRewardResult
        """
        # 根据期望质量生成评分
        if expected_quality == "high":
            base_score = random.uniform(7.0, 9.5)
            patch = None
        else:
            base_score = random.uniform(2.0, 5.5)
            patch = f"改进后的响应：关于'{trajectory.user_input}'，这是一个详细的解答..."
        
        # 生成维度评分
        dimension_scores = []
        for dim in self.ScoreDimension:
            dim_score = base_score + random.uniform(-1.0, 1.0)
            dim_score = max(0.0, min(10.0, dim_score))
            dimension_scores.append(self.DimensionScore(
                dimension=dim,
                score=dim_score,
                reason=f"{dim.value}评分理由"
            ))
        
        # 计算总分
        total_score = sum(ds.score for ds in dimension_scores) / len(dimension_scores)
        total_score = max(0.0, min(10.0, total_score))
        
        return self.SelfRewardResult(
            score=total_score,
            reason=f"总体评分理由：{'响应质量较高' if total_score >= 6.0 else '响应质量需要改进'}",
            patch=patch if total_score < 6.0 else None,
            dimension_scores=dimension_scores,
            timestamp=datetime.now().isoformat()
        )
    
    def analyze_with_critic(
        self, 
        trajectory: 'ExecutionTrajectory',
        score_result: 'SelfRewardResult'
    ) -> 'CriticReport':
        """
        使用Critic分析轨迹
        
        Args:
            trajectory: 执行轨迹
            score_result: 评分结果
            
        Returns:
            CriticReport
        """
        problems = []
        improvements = []
        root_causes = []
        
        # 检查工具调用
        for tool_call in trajectory.tool_calls:
            if not tool_call.success:
                problems.append(self.Problem(
                    type=self.ProblemType.TOOL_ERROR,
                    description=f"工具 {tool_call.tool_name} 调用失败: {tool_call.error}",
                    severity=4,
                    location="工具调用"
                ))
                improvements.append(self.Improvement(
                    problem_ref=f"工具 {tool_call.tool_name} 调用失败",
                    suggestion="实现重试机制或使用备选方案",
                    priority=4,
                    correction="添加错误处理和重试逻辑"
                ))
                root_causes.append("外部API不稳定")
        
        # 检查评分
        if score_result.score < 6.0:
            problems.append(self.Problem(
                type=self.ProblemType.INFO_MISSING,
                description="响应质量不足，信息不完整",
                severity=3,
                location="最终输出"
            ))
            improvements.append(self.Improvement(
                problem_ref="响应质量不足",
                suggestion="提供更详细、准确的信息",
                priority=5,
                correction=score_result.patch
            ))
            root_causes.append("响应生成策略需要优化")
        
        # 如果没有问题，添加正面评价
        if not problems:
            overall = "执行过程顺利，响应质量良好。"
        else:
            overall = f"发现{len(problems)}个问题，需要改进。"
        
        return self.CriticReport(
            problems=problems,
            root_causes=root_causes,
            improvements=improvements,
            overall_assessment=overall,
            timestamp=datetime.now().isoformat()
        )
    
    def store_experience(
        self,
        trajectory: 'ExecutionTrajectory',
        score_result: 'SelfRewardResult',
        critic_report: 'CriticReport'
    ) -> str:
        """
        存储经验记录
        
        Args:
            trajectory: 执行轨迹
            score_result: 评分结果
            critic_report: Critic报告
            
        Returns:
            record_id
        """
        record = self.ExperienceRecord(
            record_id=str(uuid.uuid4()),
            session_id=trajectory.session_id,
            trajectory=trajectory,
            self_reward=score_result,
            critic_report=critic_report,
            timestamp=datetime.now().isoformat(),
            tags=["rlhf_test"]
        )
        
        record_id = self.store.save(record)
        self.created_record_ids.append(record_id)
        
        return record_id
    
    def check_and_trigger_training(self, run_rm_training: bool = False) -> Tuple[bool, dict]:
        """
        检查并触发训练
        
        Args:
            run_rm_training: 是否真正执行RM训练
        
        Returns:
            (是否触发, 结果信息)
        """
        # 获取统计信息
        stats = self.generator.get_statistics()
        pair_count = stats.get("generated_pairs", 0)
        
        print(f"\n当前偏好数据对数量: {pair_count}/{self.trigger_threshold}")
        
        if pair_count >= self.trigger_threshold:
            print("\n🎯 达到训练阈值，触发RM训练!")
            
            # 开始训练会话
            report_id = self.report_generator.start_training_session()
            
            # 触发训练（导出数据）
            result = self.trigger.trigger_training(force=False)
            
            # 记录结果
            self.report_generator.record_trigger_result(result)
            
            if result.triggered:
                # 如果需要真正执行RM训练
                rm_training_success = False
                rm_training_output = ""
                
                if run_rm_training and result.export_path:
                    print("\n🚀 开始执行RM训练...")
                    rm_training_success, rm_training_output = self._run_rm_training(
                        result.export_path
                    )
                
                self.report_generator.record_training_completion(
                    success=True,
                    improvement_indicators={
                        "data_count": pair_count,
                        "rm_training_executed": run_rm_training,
                        "rm_training_success": rm_training_success
                    },
                    notes=[
                        "自动触发的RM训练",
                        f"RM训练{'已执行' if run_rm_training else '未执行（仅导出数据）'}",
                        rm_training_output[:200] if rm_training_output else ""
                    ]
                )
                
                # 保存报告
                report = self.report_generator.generate_report()
                report_path = self.report_generator.save_report(report)
                
                return True, {
                    "triggered": True,
                    "pair_count": pair_count,
                    "export_path": result.export_path,
                    "report_path": report_path,
                    "status": result.status.value,
                    "rm_training_executed": run_rm_training,
                    "rm_training_success": rm_training_success
                }
            else:
                return False, {
                    "triggered": False,
                    "message": result.message
                }
        
        return False, {
            "triggered": False,
            "pair_count": pair_count,
            "remaining": self.trigger_threshold - pair_count
        }
    
    def _run_rm_training(self, data_path: str) -> Tuple[bool, str]:
        """
        真正执行RM训练
        
        Args:
            data_path: 偏好数据文件路径
            
        Returns:
            (是否成功, 输出信息)
        """
        try:
            import json
            
            # 找到RM训练目录
            rm_dir = backend_dir.parent.parent.parent / "rlhf" / "rm"
            
            if not rm_dir.exists():
                return False, f"RM训练目录不存在: {rm_dir}"
            
            print(f"  RM训练目录: {rm_dir}")
            print(f"  偏好数据路径: {data_path}")
            
            # 首先转换数据格式为RM训练所需格式
            converted_data_path = self._convert_data_for_rm(data_path)
            
            if not converted_data_path:
                return False, "数据格式转换失败"
            
            # 获取绝对路径
            abs_data_path = str(Path(converted_data_path).resolve())
            print(f"  转换后数据路径: {abs_data_path}")
            
            # 验证数据格式
            with open(abs_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  数据条数: {len(data)}")
            if data:
                print(f"  样本字段: {list(data[0].keys())}")
            
            # 真正执行RM训练
            print("\n  🚀 开始执行RM训练...")
            print(f"  工作目录: {rm_dir}")
            
            # 设置输出目录（在backend/data下）
            output_dir = str((backend_dir / "data" / "rm_model_output").resolve())
            
            # 使用subprocess调用训练脚本，传入自定义数据路径
            import subprocess
            
            # 创建训练命令
            train_cmd = [
                "python", "-c", f"""
import sys
sys.path.insert(0, '{rm_dir}')

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
import json
import os

# 加载本地数据
print("=== 步骤1: 加载本地数据 ===")
with open('{abs_data_path}', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
print(f"加载了 {{len(raw_data)}} 条数据")

# 转换为Dataset格式
dataset = Dataset.from_list(raw_data)

# 分割训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']
print(f"训练集: {{len(train_dataset)}}, 验证集: {{len(eval_dataset)}}")

# 加载tokenizer
print("=== 步骤2: 加载模型和tokenizer ===")
from config import TrainingConfig
config = TrainingConfig()

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# tokenize函数
def tokenize_function(examples):
    # 构建chosen和rejected文本
    chosen_texts = []
    rejected_texts = []
    
    for i in range(len(examples['context'])):
        context = examples['context'][i]
        chosen = examples['chosen'][i]
        rejected = examples['rejected'][i]
        
        # 构建消息
        messages = []
        for turn in context:
            role = "user" if turn.get("role") == "human" else "assistant"
            messages.append({{"role": role, "content": turn.get("text", "")}})
        
        chosen_msgs = messages + [{{"role": "assistant", "content": chosen.get("text", "")}}]
        rejected_msgs = messages + [{{"role": "assistant", "content": rejected.get("text", "")}}]
        
        chosen_texts.append(tokenizer.apply_chat_template(chosen_msgs, tokenize=False, add_generation_prompt=False))
        rejected_texts.append(tokenizer.apply_chat_template(rejected_msgs, tokenize=False, add_generation_prompt=False))
    
    chosen_enc = tokenizer(chosen_texts, truncation=True, max_length=config.max_length, padding="max_length")
    rejected_enc = tokenizer(rejected_texts, truncation=True, max_length=config.max_length, padding="max_length")
    
    return {{
        "input_ids_chosen": chosen_enc["input_ids"],
        "attention_mask_chosen": chosen_enc["attention_mask"],
        "input_ids_rejected": rejected_enc["input_ids"],
        "attention_mask_rejected": rejected_enc["attention_mask"],
    }}

print("=== 步骤3: Tokenize数据 ===")
train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
eval_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# 加载模型
print("=== 步骤4: 加载RewardModel ===")
from model import RewardModel
model = RewardModel(config.model_name_or_path)
model = model.to(config.device)

# DataCollator
from dataclasses import dataclass
@dataclass
class RewardDataCollator:
    def __call__(self, features):
        return {{
            "input_ids_chosen": torch.stack([torch.tensor(f["input_ids_chosen"]) for f in features]),
            "attention_mask_chosen": torch.stack([torch.tensor(f["attention_mask_chosen"]) for f in features]),
            "input_ids_rejected": torch.stack([torch.tensor(f["input_ids_rejected"]) for f in features]),
            "attention_mask_rejected": torch.stack([torch.tensor(f["attention_mask_rejected"]) for f in features]),
        }}

# 自定义Trainer
class CustomRewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        chosen_rewards = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])
        rejected_rewards = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])
        loss = -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        center_loss = 0.01 * (chosen_rewards.pow(2).mean() + rejected_rewards.pow(2).mean())
        loss = loss + center_loss
        return (loss, {{"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}}) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss, None, None)

# 训练参数
print("=== 步骤5: 配置训练参数 ===")
output_dir = '{output_dir}'
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    eval_strategy="epoch",
    save_strategy="no",  # 不自动保存checkpoint
    logging_steps=1,
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to="none",
)

# 开始训练
print("=== 步骤6: 开始训练 ===")
trainer = CustomRewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=RewardDataCollator(),
    processing_class=tokenizer,
)

train_result = trainer.train()

# 保存模型
print("=== 步骤7: 保存模型 ===")
# 使用safe_serialization=False避免共享张量问题
model.model.save_pretrained(output_dir, safe_serialization=False)
torch.save(model.value_head.state_dict(), f"{{output_dir}}/value_head.pt")
tokenizer.save_pretrained(output_dir)

print(f"\\n✅ RM训练完成！模型已保存到: {{output_dir}}")
print(f"训练损失: {{train_result.metrics.get('train_loss', 'N/A')}}")
"""
            ]
            
            # 执行训练
            result = subprocess.run(
                train_cmd,
                cwd=str(rm_dir),
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            print(result.stdout)
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")
            
            if result.returncode == 0:
                return True, f"RM训练完成，模型保存到: {output_dir}"
            else:
                return False, f"RM训练失败: {result.stderr[-500:]}"
            
        except subprocess.TimeoutExpired:
            return False, "RM训练超时（超过10分钟）"
        except Exception as e:
            import traceback
            return False, f"RM训练失败: {e}\n{traceback.format_exc()}"
    
    def _convert_data_for_rm(self, data_path: str) -> Optional[str]:
        """
        将偏好数据转换为RM训练格式
        
        Args:
            data_path: 原始偏好数据路径
            
        Returns:
            转换后的数据路径
        """
        try:
            import json
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为RM训练格式
            # 原格式: {"prompt": ..., "chosen": ..., "rejected": ..., "metadata": ...}
            # 目标格式: {"context": [...], "chosen": {"text": ...}, "rejected": {"text": ...}}
            converted_data = []
            
            for item in data:
                converted_item = {
                    "context": [
                        {"role": "human", "text": item["prompt"]}
                    ],
                    "chosen": {"text": item["chosen"]},
                    "rejected": {"text": item["rejected"]}
                }
                converted_data.append(converted_item)
            
            # 保存转换后的数据
            output_path = data_path.replace(".json", "_rm_format.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"  数据转换失败: {e}")
            return None
    
    def run(self):
        """运行完整测试"""
        print_header("完整RLHF闭环测试")
        
        print(f"""
测试配置:
  - 测试样本数: {len(TEST_QUERIES)}
  - 训练触发阈值: {self.trigger_threshold} 条偏好数据
  - 高分阈值 (chosen): >= 7.0
  - 低分阈值 (rejected): < 6.0
        """)
        
        # 处理每个测试样本
        print_section("执行测试样本")
        
        for i, (query, quality, use_tool) in enumerate(TEST_QUERIES, 1):
            print_progress(i, len(TEST_QUERIES), f"处理: {query[:30]}...")
            
            try:
                # 1. 生成轨迹
                trajectory = self.generate_trajectory(query, quality, use_tool)
                print(f"    ✓ 轨迹生成完成 (ID: {trajectory.trajectory_id[:8]}...)")
                
                # 2. Self-Reward评分
                score_result = self.score_trajectory(trajectory, quality)
                score_emoji = "🟢" if score_result.score >= 7.0 else "🟡" if score_result.score >= 6.0 else "🔴"
                print(f"    ✓ 评分完成: {score_emoji} {score_result.score:.2f}/10")
                
                # 3. Critic反思
                critic_report = self.analyze_with_critic(trajectory, score_result)
                print(f"    ✓ Critic分析完成: {len(critic_report.problems)} 个问题")
                
                # 4. 存储经验
                record_id = self.store_experience(trajectory, score_result, critic_report)
                print(f"    ✓ 经验存储完成 (ID: {record_id[:8]}...)")
                
                # 5. 检查是否触发训练
                triggered, info = self.check_and_trigger_training(
                    run_rm_training=self.run_rm_training
                )
                
                if triggered:
                    print_section("RM训练已触发!")
                    print(f"  导出路径: {info.get('export_path')}")
                    print(f"  报告路径: {info.get('report_path')}")
                    print(f"  偏好数据对: {info.get('pair_count')}")
                    if info.get('rm_training_executed'):
                        print(f"  RM训练执行: {'成功' if info.get('rm_training_success') else '失败'}")
                    break
                
                # 短暂延迟，模拟真实场景
                time.sleep(0.1)
                
            except Exception as e:
                print(f"    ✗ 处理失败: {e}")
                continue
        
        # 最终统计
        print_section("测试结果统计")
        
        stats = self.generator.get_statistics()
        print(f"""
经验记录统计:
  - 总记录数: {stats.get('total_records', 0)}
  - 高分记录 (>=7.0): {stats.get('high_score_records', 0)}
  - 低分记录 (<6.0): {stats.get('low_score_records', 0)}
  - 带Patch记录: {stats.get('records_with_patch', 0)}

偏好数据统计:
  - 生成的偏好对: {stats.get('generated_pairs', 0)}
  - 平均分数差: {stats.get('average_score_difference', 0):.2f}
  - Chosen阈值: {stats.get('chosen_threshold', 7.0)}
  - Rejected阈值: {stats.get('rejected_threshold', 6.0)}
        """)
        
        # 检查最终训练状态
        final_triggered, final_info = self.check_and_trigger_training()
        
        if not final_triggered and final_info.get('pair_count', 0) < self.trigger_threshold:
            print(f"\n⚠️ 偏好数据不足，还需要 {final_info.get('remaining', 0)} 条才能触发训练")
            print("   提示: 需要更多高分和低分记录的配对")
        
        print("\n✅ 测试完成")
    
    def cleanup(self):
        """清理测试数据"""
        print_section("清理测试数据")
        
        cleaned = 0
        for record_id in self.created_record_ids:
            try:
                self.store.delete(record_id)
                cleaned += 1
            except Exception as e:
                print(f"  清理记录 {record_id[:8]}... 失败: {e}")
        
        print(f"  已清理 {cleaned}/{len(self.created_record_ids)} 条记录")


# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="完整RLHF闭环测试")
    parser.add_argument(
        "--threshold", 
        type=int, 
        default=10,
        help="触发RM训练的数据量阈值 (默认: 10)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="测试后不清理数据"
    )
    parser.add_argument(
        "--run-rm-training",
        action="store_true",
        help="真正执行RM训练（需要GPU和较长时间）"
    )
    
    args = parser.parse_args()
    
    print("\n" + "🔄 完整RLHF闭环测试 🔄".center(70))
    print("=" * 70)
    
    if args.run_rm_training:
        print("⚠️  已启用RM训练模式，将在达到阈值后执行真正的RM训练")
    
    test = None
    try:
        test = FullRLHFLoopTest(
            trigger_threshold=args.threshold,
            run_rm_training=args.run_rm_training
        )
        test.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if test and not args.no_cleanup:
            test.cleanup()
    
    print("\n" + "=" * 70)
    print("测试结束".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

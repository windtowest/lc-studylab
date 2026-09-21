#!/usr/bin/env python
"""
完整RLHF闭环训练流水线

实现从经验收集到模型优化的完整闭环：
1. 经验收集 → 2. 偏好数据生成 → 3. RM训练 → 4. PPO训练 → 5. 模型更新

使用方法:
    # 运行完整流水线
    python scripts/run_full_rlhf_pipeline.py --mode full
    
    # 只运行RM训练
    python scripts/run_full_rlhf_pipeline.py --mode rm
    
    # 只运行PPO训练（使用已有RM）
    python scripts/run_full_rlhf_pipeline.py --mode ppo
    
    # 检查当前状态
    python scripts/run_full_rlhf_pipeline.py --mode status
"""

import sys
import os
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# 添加路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

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


class RLHFPipeline:
    """
    完整RLHF训练流水线
    
    管理从数据收集到模型优化的完整流程
    """
    
    def __init__(
        self,
        min_preference_pairs: int = 100,
        rm_output_dir: str = "data/rm_model",
        ppo_output_dir: str = "data/ppo_model",
        preference_data_dir: str = "data/rlhf_preference_data"
    ):
        """
        初始化流水线
        
        Args:
            min_preference_pairs: 触发训练的最小偏好数据对数量
            rm_output_dir: RM模型输出目录
            ppo_output_dir: PPO模型输出目录
            preference_data_dir: 偏好数据目录
        """
        self.min_preference_pairs = min_preference_pairs
        self.rm_base_dir = backend_dir / rm_output_dir
        self.ppo_base_dir = backend_dir / ppo_output_dir
        self.preference_data_dir = backend_dir / preference_data_dir
        
        # 当前训练的时间戳（用于创建子目录）
        self.training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 带时间戳的输出目录
        self.rm_output_dir = self.rm_base_dir / self.training_timestamp
        self.ppo_output_dir = self.ppo_base_dir / self.training_timestamp
        
        # RLHF代码目录
        self.rm_dir = rlhf_dir / "rm"
        self.ppo_dir = rlhf_dir / "ppo"
        
        # 确保目录存在
        self.rm_base_dir.mkdir(parents=True, exist_ok=True)
        self.ppo_base_dir.mkdir(parents=True, exist_ok=True)
        self.preference_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._init_components()
    
    def _get_latest_model_dir(self, base_dir: Path) -> Optional[Path]:
        """获取最新的模型目录"""
        if not base_dir.exists():
            return None
        
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return None
        
        # 按名称排序（时间戳格式，所以字典序就是时间序）
        return sorted(subdirs)[-1]
    
    def _init_components(self):
        """初始化组件"""
        from core.memory.experience_store import ExperienceStore
        from rlhf_integration.preference_generator import PreferenceGenerator
        
        self.store = ExperienceStore()
        self.generator = PreferenceGenerator(
            self.store,
            chosen_threshold=7.0,
            rejected_threshold=6.0
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取当前流水线状态
        
        Returns:
            状态信息字典
        """
        stats = self.generator.get_statistics()
        
        # 检查最新的RM模型
        latest_rm_dir = self._get_latest_model_dir(self.rm_base_dir)
        rm_exists = False
        rm_path = None
        if latest_rm_dir:
            rm_exists = (latest_rm_dir / "adapter_model.bin").exists() or \
                        (latest_rm_dir / "value_head.pt").exists()
            rm_path = str(latest_rm_dir)
        
        # 检查最新的PPO模型
        latest_ppo_dir = self._get_latest_model_dir(self.ppo_base_dir)
        ppo_exists = False
        ppo_path = None
        if latest_ppo_dir:
            ppo_exists = (latest_ppo_dir / "final_model").exists() or \
                         any(latest_ppo_dir.glob("checkpoint-*"))
            ppo_path = str(latest_ppo_dir)
        
        # 列出所有历史版本
        rm_versions = sorted([d.name for d in self.rm_base_dir.iterdir() if d.is_dir()]) \
                      if self.rm_base_dir.exists() else []
        ppo_versions = sorted([d.name for d in self.ppo_base_dir.iterdir() if d.is_dir()]) \
                       if self.ppo_base_dir.exists() else []
        
        return {
            "experience_records": {
                "total": stats.get("total_records", 0),
                "high_score": stats.get("high_score_records", 0),
                "low_score": stats.get("low_score_records", 0)
            },
            "preference_pairs": {
                "count": stats.get("generated_pairs", 0),
                "threshold": self.min_preference_pairs,
                "ready_for_training": stats.get("generated_pairs", 0) >= self.min_preference_pairs
            },
            "rm_model": {
                "exists": rm_exists,
                "latest_path": rm_path,
                "versions": rm_versions,
                "base_dir": str(self.rm_base_dir)
            },
            "ppo_model": {
                "exists": ppo_exists,
                "latest_path": ppo_path,
                "versions": ppo_versions,
                "base_dir": str(self.ppo_base_dir)
            },
            "next_training_dir": {
                "rm": str(self.rm_output_dir),
                "ppo": str(self.ppo_output_dir),
                "timestamp": self.training_timestamp
            }
        }
    
    def export_preference_data(self) -> Optional[str]:
        """
        导出偏好数据
        
        Returns:
            导出文件路径，如果数据不足返回None
        """
        stats = self.generator.get_statistics()
        pair_count = stats.get("generated_pairs", 0)
        
        if pair_count < self.min_preference_pairs:
            print(f"偏好数据不足: {pair_count}/{self.min_preference_pairs}")
            return None
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.preference_data_dir / f"preference_data_{timestamp}.json"
        
        # 导出数据
        count = self.generator.export_for_rm_training(str(output_path))
        print(f"导出 {count} 条偏好数据到: {output_path}")
        
        # 转换为RM训练格式
        rm_format_path = self._convert_to_rm_format(str(output_path))
        
        return rm_format_path
    
    def _convert_to_rm_format(self, data_path: str) -> str:
        """
        将偏好数据转换为RM训练格式
        
        Args:
            data_path: 原始数据路径
            
        Returns:
            转换后的数据路径
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换格式
        # 原格式: {"prompt": ..., "chosen": ..., "rejected": ...}
        # 目标格式: {"context": [...], "chosen": {"text": ...}, "rejected": {"text": ...}}
        converted = []
        for item in data:
            converted.append({
                "context": [{"role": "human", "text": item["prompt"]}],
                "chosen": {"text": item["chosen"]},
                "rejected": {"text": item["rejected"]}
            })
        
        output_path = data_path.replace(".json", "_rm_format.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        
        print(f"转换为RM格式: {output_path}")
        return output_path
    
    def train_rm(self, data_path: str) -> Tuple[bool, str]:
        """
        训练奖励模型
        
        Args:
            data_path: 偏好数据路径
            
        Returns:
            (是否成功, 输出信息)
        """
        print_section("开始RM训练")
        
        output_dir = str(self.rm_output_dir)
        
        # 构建训练脚本
        train_script = f'''
import sys
sys.path.insert(0, '{self.rm_dir}')

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
import json
import os

# 加载数据
print("加载偏好数据...")
with open('{data_path}', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
print(f"数据量: {{len(raw_data)}}")

dataset = Dataset.from_list(raw_data)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = split['train'], split['test']
print(f"训练集: {{len(train_dataset)}}, 验证集: {{len(eval_dataset)}}")

# 加载模型
from config import TrainingConfig
from model import RewardModel

config = TrainingConfig()
print(f"基础模型: {{config.base_model_name}}")
print(f"SFT模型: {{config.model_name_or_path}}")

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize
def tokenize_fn(examples):
    chosen_texts, rejected_texts = [], []
    for i in range(len(examples['context'])):
        ctx = examples['context'][i]
        chosen = examples['chosen'][i]
        rejected = examples['rejected'][i]
        
        msgs = [{{"role": "user" if t.get("role") == "human" else "assistant", 
                 "content": t.get("text", "")}} for t in ctx]
        
        chosen_msgs = msgs + [{{"role": "assistant", "content": chosen.get("text", "")}}]
        rejected_msgs = msgs + [{{"role": "assistant", "content": rejected.get("text", "")}}]
        
        chosen_texts.append(tokenizer.apply_chat_template(chosen_msgs, tokenize=False))
        rejected_texts.append(tokenizer.apply_chat_template(rejected_msgs, tokenize=False))
    
    chosen_enc = tokenizer(chosen_texts, truncation=True, max_length=512, padding="max_length")
    rejected_enc = tokenizer(rejected_texts, truncation=True, max_length=512, padding="max_length")
    
    return {{
        "input_ids_chosen": chosen_enc["input_ids"],
        "attention_mask_chosen": chosen_enc["attention_mask"],
        "input_ids_rejected": rejected_enc["input_ids"],
        "attention_mask_rejected": rejected_enc["attention_mask"],
    }}

print("Tokenizing数据...")
train_tok = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
eval_tok = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)

# 模型
print("加载RewardModel...")
model = RewardModel(config.model_name_or_path).to(config.device)
print(f"设备: {{config.device}}")

# DataCollator
from dataclasses import dataclass
@dataclass
class RMCollator:
    def __call__(self, features):
        return {{
            "input_ids_chosen": torch.stack([torch.tensor(f["input_ids_chosen"]) for f in features]),
            "attention_mask_chosen": torch.stack([torch.tensor(f["attention_mask_chosen"]) for f in features]),
            "input_ids_rejected": torch.stack([torch.tensor(f["input_ids_rejected"]) for f in features]),
            "attention_mask_rejected": torch.stack([torch.tensor(f["attention_mask_rejected"]) for f in features]),
        }}

# Trainer
class RMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        chosen = model(inputs["input_ids_chosen"], inputs["attention_mask_chosen"])
        rejected = model(inputs["input_ids_rejected"], inputs["attention_mask_rejected"])
        loss = -nn.functional.logsigmoid(chosen - rejected).mean()
        loss += 0.01 * (chosen.pow(2).mean() + rejected.pow(2).mean())
        return (loss, {{}}) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

# 训练
os.makedirs('{output_dir}', exist_ok=True)
args = TrainingArguments(
    output_dir='{output_dir}',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=5,
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to="none",
)

print("\\n开始训练...")
trainer = RMTrainer(
    model=model, args=args,
    train_dataset=train_tok, eval_dataset=eval_tok,
    data_collator=RMCollator(), processing_class=tokenizer,
)
trainer.train()

# 保存
print("\\n保存模型...")
model.model.save_pretrained('{output_dir}', safe_serialization=False)
torch.save(model.value_head.state_dict(), '{output_dir}/value_head.pt')
tokenizer.save_pretrained('{output_dir}')
print(f"✅ RM模型已保存到: {output_dir}")
'''
        
        # 使用Popen实时输出
        print("\n" + "=" * 60)
        print("RM训练输出")
        print("=" * 60 + "\n")
        
        process = subprocess.Popen(
            ["python", "-u", "-c", train_script],  # -u 禁用缓冲
            cwd=str(self.rm_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # 行缓冲
        )
        
        output_lines = []
        try:
            for line in process.stdout:
                print(line, end='')  # 实时打印
                output_lines.append(line)
            
            process.wait(timeout=1800)
            success = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            process.kill()
            return False, "RM训练超时"
        except Exception as e:
            process.kill()
            return False, str(e)
        
        return success, "".join(output_lines)
    
    def train_ppo(self, rm_path: str = None, data_path: str = None) -> Tuple[bool, str]:
        """
        训练PPO模型（使用自己收集的经验数据）
        
        Args:
            rm_path: RM模型路径，如果为None则使用最新的RM模型
            data_path: 偏好数据路径，用于PPO训练
            
        Returns:
            (是否成功, 输出信息)
        """
        print_section("开始PPO训练")
        
        # 如果没有指定RM路径，使用最新的
        if rm_path is None:
            latest_rm = self._get_latest_model_dir(self.rm_base_dir)
            if latest_rm:
                rm_path = str(latest_rm)
            else:
                return False, "没有找到RM模型，请先训练RM"
        
        output_dir = str(self.ppo_output_dir)
        
        # 检查RM模型是否存在
        if not Path(rm_path).exists():
            return False, f"RM模型不存在: {rm_path}"
        
        # 如果没有指定数据路径，使用最新的偏好数据
        if data_path is None:
            # 查找最新的偏好数据文件
            data_files = list(self.preference_data_dir.glob("*_rm_format.json"))
            if not data_files:
                return False, "没有找到偏好数据文件"
            data_path = str(sorted(data_files)[-1])
        
        print(f"使用偏好数据: {data_path}")
        print(f"使用RM模型: {rm_path}")
        print(f"输出目录: {output_dir}")
        
        # 构建PPO训练脚本（使用自己的数据）
        train_script = f'''
import sys
import os
sys.path.insert(0, '{self.ppo_dir}')
os.chdir('{self.ppo_dir}')

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import PPOConfig
from trainer import PPOTrainer

# 自定义数据集：使用我们收集的经验数据
class CustomPPODataset(Dataset):
    """使用自己收集的经验数据"""
    
    def __init__(self, data_path, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        
        print(f"加载自定义数据: {{data_path}}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"数据量: {{len(self.data)}}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        
        # 构建prompt
        prompt_parts = []
        for turn in context:
            role = turn["role"]
            text = turn["text"]
            if role == "human":
                prompt_parts.append(f"Human: {{text}}")
            else:
                prompt_parts.append(f"Assistant: {{text}}")
        
        prompt = "\\n\\n".join(prompt_parts) + "\\n\\nAssistant:"
        
        return {{
            "prompt": prompt,
            "chosen_response": item["chosen"]["text"],
        }}
    
    def collate_fn(self, batch):
        prompts = [item["prompt"] for item in batch]
        chosen_responses = [item["chosen_response"] for item in batch]
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length // 2,
            return_tensors="pt"
        )
        
        return {{
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "prompts": prompts,
            "chosen_responses": chosen_responses,
        }}

# 配置
config = PPOConfig()
config.reward_model_path = '{rm_path}'
config.output_dir = '{output_dir}'
config.total_steps = 100  # 小数据集用较少步数
config.batch_size = 4
config.mini_batch_size = 2
config.log_interval = 10
config.save_interval = 50
config.eval_interval = 25

print("=" * 60)
print("PPO训练配置")
print("=" * 60)
print(f"RM模型: {{config.reward_model_path}}")
print(f"输出目录: {{config.output_dir}}")
print(f"总步数: {{config.total_steps}}")
print(f"批次大小: {{config.batch_size}}")

# 创建数据加载器
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = CustomPPODataset('{data_path}', tokenizer, config)
dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=dataset.collate_fn,
    drop_last=True
)

print(f"数据批次数: {{len(dataloader)}}")

# 创建训练器并训练
trainer = PPOTrainer(config)

# 替换eval_dataset为我们的数据
trainer.eval_dataset = dataset

print("\\n开始训练...")
trainer.train(dataloader)

print("\\n✅ PPO训练完成!")
print(f"模型保存在: {{config.output_dir}}")
'''
        
        # 使用Popen实时输出
        print("\n" + "=" * 60)
        print("PPO训练输出")
        print("=" * 60 + "\n")
        
        process = subprocess.Popen(
            ["python", "-u", "-c", train_script],  # -u 禁用缓冲
            cwd=str(self.ppo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # 行缓冲
        )
        
        output_lines = []
        try:
            for line in process.stdout:
                print(line, end='')  # 实时打印
                output_lines.append(line)
            
            process.wait(timeout=3600)
            success = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            process.kill()
            return False, "PPO训练超时"
        except Exception as e:
            process.kill()
            return False, str(e)
        
        return success, "".join(output_lines)
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        运行完整的RLHF流水线
        
        Returns:
            执行结果
        """
        print_header("完整RLHF训练流水线")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # 阶段1: 检查状态
        print_section("阶段1: 检查当前状态")
        status = self.get_status()
        print(f"经验记录: {status['experience_records']['total']}")
        print(f"偏好数据对: {status['preference_pairs']['count']}/{status['preference_pairs']['threshold']}")
        print(f"RM模型: {'存在' if status['rm_model']['exists'] else '不存在'}")
        print(f"PPO模型: {'存在' if status['ppo_model']['exists'] else '不存在'}")
        
        results["stages"]["status"] = status
        
        # 阶段2: 导出偏好数据
        print_section("阶段2: 导出偏好数据")
        if not status['preference_pairs']['ready_for_training']:
            print(f"⚠️ 偏好数据不足，需要至少 {self.min_preference_pairs} 对")
            results["stages"]["export"] = {"success": False, "reason": "数据不足"}
            return results
        
        data_path = self.export_preference_data()
        if not data_path:
            results["stages"]["export"] = {"success": False, "reason": "导出失败"}
            return results
        
        results["stages"]["export"] = {"success": True, "path": data_path}
        
        # 阶段3: RM训练
        print_section("阶段3: RM训练")
        rm_success, rm_output = self.train_rm(data_path)
        results["stages"]["rm_training"] = {
            "success": rm_success,
            "output_dir": str(self.rm_output_dir)
        }
        
        if not rm_success:
            print(f"❌ RM训练失败")
            return results
        
        print(f"✅ RM训练完成")
        
        # 阶段4: PPO训练
        print_section("阶段4: PPO训练")
        ppo_success, ppo_output = self.train_ppo(data_path=data_path)
        results["stages"]["ppo_training"] = {
            "success": ppo_success,
            "output_dir": str(self.ppo_output_dir)
        }
        
        if not ppo_success:
            print(f"❌ PPO训练失败")
            return results
        
        print(f"✅ PPO训练完成")
        
        # 阶段5: 总结
        print_section("阶段5: 训练完成")
        print(f"""
训练结果 (时间戳: {self.training_timestamp}):
  - RM模型: {self.rm_output_dir}
  - PPO模型: {self.ppo_output_dir}
  
下一步:
  1. 使用PPO模型替换Agent的基础模型
  2. 继续收集经验数据
  3. 定期重新训练以持续优化
  
历史版本:
  - RM模型目录: {self.rm_base_dir}
  - PPO模型目录: {self.ppo_base_dir}
        """)
        
        results["success"] = True
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="完整RLHF训练流水线")
    parser.add_argument(
        "--mode",
        choices=["full", "rm", "ppo", "status", "export"],
        default="status",
        help="运行模式"
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=100,
        help="触发训练的最小偏好数据对数量"
    )
    parser.add_argument(
        "--rm-path",
        type=str,
        default=None,
        help="RM模型路径（用于PPO训练）"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="偏好数据路径（用于RM训练）"
    )
    
    args = parser.parse_args()
    
    pipeline = RLHFPipeline(min_preference_pairs=args.min_pairs)
    
    if args.mode == "status":
        print_header("RLHF流水线状态")
        status = pipeline.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
    elif args.mode == "export":
        print_header("导出偏好数据")
        path = pipeline.export_preference_data()
        if path:
            print(f"✅ 导出成功: {path}")
        else:
            print("❌ 导出失败（数据不足）")
            
    elif args.mode == "rm":
        print_header("RM训练")
        if args.data_path:
            success, output = pipeline.train_rm(args.data_path)
        else:
            # 先导出数据
            data_path = pipeline.export_preference_data()
            if not data_path:
                print("❌ 无法导出偏好数据")
                return
            success, output = pipeline.train_rm(data_path)
        
        if success:
            print("✅ RM训练完成")
        else:
            print("❌ RM训练失败")
            
    elif args.mode == "ppo":
        print_header("PPO训练")
        success, output = pipeline.train_ppo(args.rm_path)
        if success:
            print("✅ PPO训练完成")
        else:
            print("❌ PPO训练失败")
            
    elif args.mode == "full":
        results = pipeline.run_full_pipeline()
        
        # 保存结果
        result_path = backend_dir / "data" / "rlhf_pipeline_results" / \
                      f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()

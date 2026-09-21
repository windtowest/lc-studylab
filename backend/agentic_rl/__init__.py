"""
Agentic RL 模块

基于 Agent-Lightning + VeRL + GRPO 架构实现的可训练 Agent 系统。

主要组件:
- models: 数据模型 (Transition, Trajectory, TrajectoryCollector)
- rewards: 奖励函数系统
- trainers: GRPO 训练器和 VeRL 集成
- adapters: Lightning Client 适配器
- pipeline: 训练流水线
- agents: 可训练 Agent 实现

使用示例:
    >>> from agentic_rl import TrainableAgent, GRPOTrainer
    >>> agent = TrainableAgent(training_mode=True)
    >>> response, trajectory = agent.invoke("你好")
"""

from .models import Transition, Trajectory, TransitionMeta, ToolCallRecord
from .models import TrajectoryCollector

__version__ = "0.1.0"

__all__ = [
    # 数据模型
    "Transition",
    "Trajectory", 
    "TransitionMeta",
    "ToolCallRecord",
    "TrajectoryCollector",
]

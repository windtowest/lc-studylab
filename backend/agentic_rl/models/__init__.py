"""
数据模型模块

包含 RL 训练所需的核心数据结构:
- Transition: RL 训练的基本数据单元
- Trajectory: 一次任务执行的完整轨迹
- TrajectoryCollector: 轨迹收集器
"""

from .transition import (
    Transition,
    TransitionMeta,
    ToolCallRecord,
    Trajectory,
)
from .collector import TrajectoryCollector

__all__ = [
    "Transition",
    "TransitionMeta", 
    "ToolCallRecord",
    "Trajectory",
    "TrajectoryCollector",
]

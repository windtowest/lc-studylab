"""
智能体模块
提供各种类型的 Agent 实现

第 1 阶段：基础 Agent + Streaming + 工具
- BaseAgent: 通用智能体，支持工具调用和流式输出

第 2 阶段：Self-Reward Agent
- SelfRewardAgent: 增强版智能体，支持自评分、反思和经验存储
"""

from .base_agent import BaseAgent, create_base_agent
from .self_reward_agent import SelfRewardAgent, create_self_reward_agent

__all__ = [
    "BaseAgent",
    "create_base_agent",
    "SelfRewardAgent",
    "create_self_reward_agent",
]


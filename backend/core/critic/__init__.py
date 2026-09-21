"""
Critic 反思模块

该模块提供Critic Agent能力，从执行轨迹生成改进建议，与Memory模块联动形成经验沉淀。
"""

from .models import ProblemType, Problem, Improvement, CriticReport
from .critic_agent import CriticAgent
from .prompts import (
    TRAJECTORY_ANALYSIS_SYSTEM_PROMPT,
    CORRECTION_SYSTEM_PROMPT,
    format_trajectory_for_analysis,
    format_correction_prompt,
)

__all__ = [
    "ProblemType",
    "Problem",
    "Improvement",
    "CriticReport",
    "CriticAgent",
    "TRAJECTORY_ANALYSIS_SYSTEM_PROMPT",
    "CORRECTION_SYSTEM_PROMPT",
    "format_trajectory_for_analysis",
    "format_correction_prompt",
]

"""
Self-Reward 自评分模块

该模块提供Agent自我评分能力，通过标准化的reason/score/patch输出结构进行行为选择。
"""

from .models import ScoreDimension, DimensionScore, SelfRewardResult
from .scorer import SelfRewardScorer
from .prompts import (
    SCORING_SYSTEM_PROMPT,
    PATCH_SYSTEM_PROMPT,
    format_scoring_prompt,
    format_patch_prompt,
)

__all__ = [
    "ScoreDimension",
    "DimensionScore", 
    "SelfRewardResult",
    "SelfRewardScorer",
    "SCORING_SYSTEM_PROMPT",
    "PATCH_SYSTEM_PROMPT",
    "format_scoring_prompt",
    "format_patch_prompt",
]

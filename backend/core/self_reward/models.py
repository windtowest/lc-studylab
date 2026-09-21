"""
Self-Reward 数据模型

定义自评分相关的数据结构，包括评分维度、维度评分和评分结果。
"""

from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class ScoreDimension(str, Enum):
    """评分维度枚举"""
    ACCURACY = "accuracy"           # 准确性
    RELEVANCE = "relevance"         # 相关性
    COMPLETENESS = "completeness"   # 完整性
    CLARITY = "clarity"             # 清晰度


class DimensionScore(BaseModel):
    """维度评分"""
    dimension: ScoreDimension
    score: float  # 0-10
    reason: str


class SelfRewardResult(BaseModel):
    """Self-Reward评分结果"""
    score: float                              # 总分 0-10
    reason: str                               # 评分理由
    patch: Optional[str] = None               # 改进补丁（低分时生成）
    dimension_scores: List[DimensionScore]    # 各维度评分
    timestamp: str

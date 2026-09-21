"""
Critic 数据模型

定义Critic反思相关的数据结构，包括问题类型、问题、改进建议和反思报告。
"""

from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class ProblemType(str, Enum):
    """问题类型枚举"""
    TOOL_ERROR = "tool_error"               # 工具调用错误
    REASONING_BREAK = "reasoning_break"     # 推理链断裂
    INFO_MISSING = "info_missing"           # 信息遗漏
    LOGIC_ERROR = "logic_error"             # 逻辑错误
    FORMAT_ERROR = "format_error"           # 格式错误


class Problem(BaseModel):
    """问题记录"""
    type: ProblemType
    description: str
    severity: int  # 1-5, 5最严重
    location: str  # 问题位置（步骤编号）


class Improvement(BaseModel):
    """改进建议"""
    problem_ref: str  # 关联的问题
    suggestion: str
    priority: int  # 1-5, 5最高优先级
    correction: Optional[str] = None  # 具体修正方案


class CriticReport(BaseModel):
    """Critic反思报告"""
    problems: List[Problem]
    root_causes: List[str]
    improvements: List[Improvement]
    overall_assessment: str
    timestamp: str

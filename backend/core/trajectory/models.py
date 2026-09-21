"""
轨迹数据模型

定义执行轨迹相关的数据结构，包括工具调用、推理步骤和完整轨迹记录。
"""

from pydantic import BaseModel
from typing import List, Optional, Any


class ToolCall(BaseModel):
    """工具调用记录"""
    tool_name: str
    input_params: dict
    output_result: Any
    duration_ms: float
    timestamp: str
    success: bool
    error: Optional[str] = None


class ReasoningStep(BaseModel):
    """推理步骤记录"""
    step_number: int
    content: str
    timestamp: str


class ExecutionTrajectory(BaseModel):
    """执行轨迹"""
    trajectory_id: str
    session_id: str
    user_input: str
    system_prompt: Optional[str] = None
    reasoning_steps: List[ReasoningStep] = []
    tool_calls: List[ToolCall] = []
    intermediate_results: List[str] = []
    final_output: str
    total_duration_ms: float
    start_time: str
    end_time: str
    
    def to_json(self) -> str:
        """序列化为JSON"""
        return self.model_dump_json()
    
    @classmethod
    def from_json(cls, json_str: str) -> "ExecutionTrajectory":
        """从JSON反序列化"""
        return cls.model_validate_json(json_str)

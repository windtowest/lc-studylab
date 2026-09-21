"""
轨迹记录器

实现Agent执行轨迹的记录功能，包括推理步骤、工具调用等。
"""

import uuid
from datetime import datetime
from typing import Optional, Any

from .models import (
    ToolCall,
    ReasoningStep,
    ExecutionTrajectory,
)


class TrajectoryRecorder:
    """
    轨迹记录器
    
    用于记录Agent执行过程中的完整轨迹，包括：
    - 用户输入和系统提示
    - 推理步骤
    - 工具调用（名称、参数、结果、耗时）
    - 中间结果
    - 最终输出
    - 执行时间统计
    """
    
    def __init__(self):
        self._current_trajectory: Optional[ExecutionTrajectory] = None
        self._start_timestamp: Optional[datetime] = None
        self._reasoning_step_counter: int = 0
        self._intermediate_results: list[str] = []
    
    @property
    def is_recording(self) -> bool:
        """检查是否正在记录"""
        return self._current_trajectory is not None
    
    def start(
        self, 
        session_id: str, 
        user_input: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        开始记录新轨迹
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            system_prompt: 系统提示（可选）
            
        Returns:
            trajectory_id: 新创建的轨迹ID
            
        Raises:
            RuntimeError: 如果已经在记录中
        """
        if self._current_trajectory is not None:
            raise RuntimeError("已经在记录轨迹中，请先调用 finish() 完成当前记录")
        
        trajectory_id = str(uuid.uuid4())
        self._start_timestamp = datetime.now()
        self._reasoning_step_counter = 0
        self._intermediate_results = []
        
        # 创建初始轨迹对象（final_output 和 end_time 将在 finish 时填充）
        self._current_trajectory = ExecutionTrajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            user_input=user_input,
            system_prompt=system_prompt,
            reasoning_steps=[],
            tool_calls=[],
            intermediate_results=[],
            final_output="",  # 占位，finish时更新
            total_duration_ms=0.0,  # 占位，finish时更新
            start_time=self._start_timestamp.isoformat(),
            end_time="",  # 占位，finish时更新
        )
        
        return trajectory_id
    
    def add_reasoning(self, content: str) -> ReasoningStep:
        """
        添加推理步骤
        
        Args:
            content: 推理内容
            
        Returns:
            创建的 ReasoningStep 对象
            
        Raises:
            RuntimeError: 如果未开始记录
        """
        if self._current_trajectory is None:
            raise RuntimeError("未开始记录，请先调用 start()")
        
        self._reasoning_step_counter += 1
        step = ReasoningStep(
            step_number=self._reasoning_step_counter,
            content=content,
            timestamp=datetime.now().isoformat(),
        )
        self._current_trajectory.reasoning_steps.append(step)
        
        return step
    
    def add_tool_call(
        self,
        tool_name: str,
        input_params: dict,
        output_result: Any,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> ToolCall:
        """
        添加工具调用记录
        
        Args:
            tool_name: 工具名称
            input_params: 输入参数
            output_result: 输出结果
            duration_ms: 执行耗时（毫秒）
            success: 是否成功
            error: 错误信息（可选）
            
        Returns:
            创建的 ToolCall 对象
            
        Raises:
            RuntimeError: 如果未开始记录
        """
        if self._current_trajectory is None:
            raise RuntimeError("未开始记录，请先调用 start()")
        
        tool_call = ToolCall(
            tool_name=tool_name,
            input_params=input_params,
            output_result=output_result,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat(),
            success=success,
            error=error,
        )
        self._current_trajectory.tool_calls.append(tool_call)
        
        return tool_call
    
    def add_intermediate_result(self, result: str) -> None:
        """
        添加中间结果
        
        Args:
            result: 中间结果内容
            
        Raises:
            RuntimeError: 如果未开始记录
        """
        if self._current_trajectory is None:
            raise RuntimeError("未开始记录，请先调用 start()")
        
        self._intermediate_results.append(result)
    
    def finish(self, final_output: str) -> ExecutionTrajectory:
        """
        完成记录并返回轨迹
        
        Args:
            final_output: 最终输出
            
        Returns:
            完整的 ExecutionTrajectory 对象
            
        Raises:
            RuntimeError: 如果未开始记录
        """
        if self._current_trajectory is None:
            raise RuntimeError("未开始记录，请先调用 start()")
        
        end_timestamp = datetime.now()
        
        # 计算总执行时间
        total_duration_ms = (end_timestamp - self._start_timestamp).total_seconds() * 1000
        
        # 更新轨迹对象
        self._current_trajectory.final_output = final_output
        self._current_trajectory.end_time = end_timestamp.isoformat()
        self._current_trajectory.total_duration_ms = total_duration_ms
        self._current_trajectory.intermediate_results = self._intermediate_results.copy()
        
        # 获取完成的轨迹
        completed_trajectory = self._current_trajectory
        
        # 重置状态
        self._current_trajectory = None
        self._start_timestamp = None
        self._reasoning_step_counter = 0
        self._intermediate_results = []
        
        return completed_trajectory
    
    def cancel(self) -> None:
        """
        取消当前记录
        
        重置记录器状态，丢弃当前轨迹。
        """
        self._current_trajectory = None
        self._start_timestamp = None
        self._reasoning_step_counter = 0
        self._intermediate_results = []
    
    def get_current_trajectory(self) -> Optional[ExecutionTrajectory]:
        """
        获取当前正在记录的轨迹（快照）
        
        Returns:
            当前轨迹的副本，如果未在记录则返回 None
        """
        if self._current_trajectory is None:
            return None
        
        # 返回当前状态的副本
        return ExecutionTrajectory(
            trajectory_id=self._current_trajectory.trajectory_id,
            session_id=self._current_trajectory.session_id,
            user_input=self._current_trajectory.user_input,
            system_prompt=self._current_trajectory.system_prompt,
            reasoning_steps=self._current_trajectory.reasoning_steps.copy(),
            tool_calls=self._current_trajectory.tool_calls.copy(),
            intermediate_results=self._intermediate_results.copy(),
            final_output=self._current_trajectory.final_output,
            total_duration_ms=self._current_trajectory.total_duration_ms,
            start_time=self._current_trajectory.start_time,
            end_time=self._current_trajectory.end_time,
        )

"""
轨迹数据模型属性测试

Feature: agent-self-reward-critic, Property 6: 轨迹序列化Round-Trip
Validates: Requirements 3.6
"""

import sys
from pathlib import Path

# 确保可以导入 core.trajectory 模块
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from hypothesis import given, strategies as st, settings

from core.trajectory.models import (
    ToolCall,
    ReasoningStep,
    ExecutionTrajectory,
)


# 自定义策略：生成有效的时间戳字符串
timestamp_strategy = st.datetimes().map(lambda dt: dt.isoformat())

# JSON 兼容的 float 策略（排除 inf 和 nan）
json_safe_float = st.floats(allow_nan=False, allow_infinity=False)

# 自定义策略：生成 ToolCall 对象
tool_call_strategy = st.builds(
    ToolCall,
    tool_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    input_params=st.dictionaries(
        keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
        values=st.one_of(st.text(max_size=100), st.integers(), json_safe_float, st.booleans()),
        max_size=5
    ),
    output_result=st.one_of(st.text(max_size=200), st.integers(), json_safe_float, st.booleans(), st.none()),
    duration_ms=st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
    timestamp=timestamp_strategy,
    success=st.booleans(),
    error=st.one_of(st.none(), st.text(max_size=200)),
)

# 自定义策略：生成 ReasoningStep 对象
reasoning_step_strategy = st.builds(
    ReasoningStep,
    step_number=st.integers(min_value=1, max_value=1000),
    content=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    timestamp=timestamp_strategy,
)

# 自定义策略：生成 ExecutionTrajectory 对象
execution_trajectory_strategy = st.builds(
    ExecutionTrajectory,
    trajectory_id=st.uuids().map(str),
    session_id=st.uuids().map(str),
    user_input=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    system_prompt=st.one_of(st.none(), st.text(max_size=500)),
    reasoning_steps=st.lists(reasoning_step_strategy, max_size=10),
    tool_calls=st.lists(tool_call_strategy, max_size=10),
    intermediate_results=st.lists(st.text(max_size=200), max_size=10),
    final_output=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
    total_duration_ms=st.floats(min_value=0, max_value=1e9, allow_nan=False),
    start_time=timestamp_strategy,
    end_time=timestamp_strategy,
)


class TestTrajectorySerializationRoundTrip:
    """
    Property 6: 轨迹序列化Round-Trip
    
    For any ExecutionTrajectory对象，序列化为JSON后再反序列化应得到等价的对象。
    Validates: Requirements 3.6
    """

    @given(trajectory=execution_trajectory_strategy)
    @settings(max_examples=100)
    def test_execution_trajectory_round_trip(self, trajectory: ExecutionTrajectory):
        """
        Feature: agent-self-reward-critic, Property 6: 轨迹序列化Round-Trip
        
        验证 ExecutionTrajectory 对象序列化为 JSON 后再反序列化得到等价对象。
        """
        # 序列化为 JSON
        json_str = trajectory.to_json()
        
        # 从 JSON 反序列化
        restored = ExecutionTrajectory.from_json(json_str)
        
        # 验证等价性
        assert restored == trajectory
        assert restored.trajectory_id == trajectory.trajectory_id
        assert restored.session_id == trajectory.session_id
        assert restored.user_input == trajectory.user_input
        assert restored.system_prompt == trajectory.system_prompt
        assert restored.final_output == trajectory.final_output
        assert len(restored.reasoning_steps) == len(trajectory.reasoning_steps)
        assert len(restored.tool_calls) == len(trajectory.tool_calls)
        assert len(restored.intermediate_results) == len(trajectory.intermediate_results)

    @given(tool_call=tool_call_strategy)
    @settings(max_examples=100)
    def test_tool_call_round_trip(self, tool_call: ToolCall):
        """
        验证 ToolCall 对象序列化为 JSON 后再反序列化得到等价对象。
        """
        json_str = tool_call.model_dump_json()
        restored = ToolCall.model_validate_json(json_str)
        assert restored == tool_call

    @given(reasoning_step=reasoning_step_strategy)
    @settings(max_examples=100)
    def test_reasoning_step_round_trip(self, reasoning_step: ReasoningStep):
        """
        验证 ReasoningStep 对象序列化为 JSON 后再反序列化得到等价对象。
        """
        json_str = reasoning_step.model_dump_json()
        restored = ReasoningStep.model_validate_json(json_str)
        assert restored == reasoning_step

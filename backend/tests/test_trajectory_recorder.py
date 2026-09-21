"""
轨迹记录器属性测试

Feature: agent-self-reward-critic, Property 5: 轨迹记录完整性
Validates: Requirements 3.2, 3.3, 3.4, 3.5
"""

import sys
from pathlib import Path

# 确保可以导入 core.trajectory 模块
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from hypothesis import given, strategies as st, settings, assume

from core.trajectory import TrajectoryRecorder
from core.trajectory.models import ExecutionTrajectory


# 自定义策略：生成有效的会话ID
session_id_strategy = st.uuids().map(str)

# 自定义策略：生成非空文本
non_empty_text = st.text(min_size=1, max_size=200).filter(lambda x: x.strip())

# 自定义策略：生成工具调用参数
tool_params_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    values=st.one_of(
        st.text(max_size=100),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans()
    ),
    max_size=5
)

# 自定义策略：生成工具调用结果
tool_result_strategy = st.one_of(
    st.text(max_size=200),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none()
)

# 自定义策略：生成正数的耗时
duration_strategy = st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False)


class TestTrajectoryRecorderCompleteness:
    """
    Property 5: 轨迹记录完整性
    
    For any Agent执行过程，生成的ExecutionTrajectory应包含user_input、final_output、
    total_duration_ms，且所有tool_calls记录应包含tool_name、input_params、output_result和duration_ms。
    
    Validates: Requirements 3.2, 3.3, 3.4, 3.5
    """

    @given(
        session_id=session_id_strategy,
        user_input=non_empty_text,
        system_prompt=st.one_of(st.none(), non_empty_text),
        final_output=non_empty_text,
    )
    @settings(max_examples=100)
    def test_basic_trajectory_completeness(
        self,
        session_id: str,
        user_input: str,
        system_prompt: str,
        final_output: str,
    ):
        """
        Feature: agent-self-reward-critic, Property 5: 轨迹记录完整性
        
        验证基本轨迹记录包含必需字段：user_input、final_output、total_duration_ms
        Validates: Requirements 3.2, 3.5
        """
        recorder = TrajectoryRecorder()
        
        # 开始记录
        trajectory_id = recorder.start(session_id, user_input, system_prompt)
        
        # 完成记录
        trajectory = recorder.finish(final_output)
        
        # 验证必需字段存在且正确
        assert trajectory.trajectory_id == trajectory_id
        assert trajectory.session_id == session_id
        assert trajectory.user_input == user_input
        assert trajectory.system_prompt == system_prompt
        assert trajectory.final_output == final_output
        assert trajectory.total_duration_ms >= 0
        assert trajectory.start_time != ""
        assert trajectory.end_time != ""

    @given(
        session_id=session_id_strategy,
        user_input=non_empty_text,
        reasoning_contents=st.lists(non_empty_text, min_size=1, max_size=5),
        final_output=non_empty_text,
    )
    @settings(max_examples=100)
    def test_reasoning_steps_completeness(
        self,
        session_id: str,
        user_input: str,
        reasoning_contents: list,
        final_output: str,
    ):
        """
        Feature: agent-self-reward-critic, Property 5: 轨迹记录完整性
        
        验证推理步骤记录完整性
        Validates: Requirements 3.2, 3.4
        """
        recorder = TrajectoryRecorder()
        recorder.start(session_id, user_input)
        
        # 添加推理步骤
        for content in reasoning_contents:
            recorder.add_reasoning(content)
        
        trajectory = recorder.finish(final_output)
        
        # 验证推理步骤数量正确
        assert len(trajectory.reasoning_steps) == len(reasoning_contents)
        
        # 验证每个推理步骤包含必需字段
        for i, step in enumerate(trajectory.reasoning_steps):
            assert step.step_number == i + 1
            assert step.content == reasoning_contents[i]
            assert step.timestamp != ""

    @given(
        session_id=session_id_strategy,
        user_input=non_empty_text,
        tool_name=non_empty_text,
        input_params=tool_params_strategy,
        output_result=tool_result_strategy,
        duration_ms=duration_strategy,
        success=st.booleans(),
        error=st.one_of(st.none(), non_empty_text),
        final_output=non_empty_text,
    )
    @settings(max_examples=100)
    def test_tool_call_completeness(
        self,
        session_id: str,
        user_input: str,
        tool_name: str,
        input_params: dict,
        output_result,
        duration_ms: float,
        success: bool,
        error: str,
        final_output: str,
    ):
        """
        Feature: agent-self-reward-critic, Property 5: 轨迹记录完整性
        
        验证工具调用记录包含所有必需字段：tool_name、input_params、output_result、duration_ms
        Validates: Requirements 3.3
        """
        recorder = TrajectoryRecorder()
        recorder.start(session_id, user_input)
        
        # 添加工具调用
        recorder.add_tool_call(
            tool_name=tool_name,
            input_params=input_params,
            output_result=output_result,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        
        trajectory = recorder.finish(final_output)
        
        # 验证工具调用数量
        assert len(trajectory.tool_calls) == 1
        
        # 验证工具调用包含所有必需字段
        tool_call = trajectory.tool_calls[0]
        assert tool_call.tool_name == tool_name
        assert tool_call.input_params == input_params
        assert tool_call.output_result == output_result
        assert tool_call.duration_ms == duration_ms
        assert tool_call.success == success
        assert tool_call.error == error
        assert tool_call.timestamp != ""

    @given(
        session_id=session_id_strategy,
        user_input=non_empty_text,
        num_tool_calls=st.integers(min_value=1, max_value=5),
        final_output=non_empty_text,
    )
    @settings(max_examples=100)
    def test_multiple_tool_calls_completeness(
        self,
        session_id: str,
        user_input: str,
        num_tool_calls: int,
        final_output: str,
    ):
        """
        Feature: agent-self-reward-critic, Property 5: 轨迹记录完整性
        
        验证多个工具调用都被完整记录
        Validates: Requirements 3.3
        """
        recorder = TrajectoryRecorder()
        recorder.start(session_id, user_input)
        
        # 添加多个工具调用
        for i in range(num_tool_calls):
            recorder.add_tool_call(
                tool_name=f"tool_{i}",
                input_params={"index": i},
                output_result=f"result_{i}",
                duration_ms=float(i + 1) * 10,
                success=True,
            )
        
        trajectory = recorder.finish(final_output)
        
        # 验证所有工具调用都被记录
        assert len(trajectory.tool_calls) == num_tool_calls
        
        # 验证每个工具调用的完整性
        for i, tool_call in enumerate(trajectory.tool_calls):
            assert tool_call.tool_name == f"tool_{i}"
            assert tool_call.input_params == {"index": i}
            assert tool_call.output_result == f"result_{i}"
            assert tool_call.duration_ms == float(i + 1) * 10
            assert tool_call.success is True
            assert tool_call.timestamp != ""

    @given(
        session_id=session_id_strategy,
        user_input=non_empty_text,
        intermediate_results=st.lists(st.text(max_size=100), min_size=0, max_size=5),
        final_output=non_empty_text,
    )
    @settings(max_examples=100)
    def test_intermediate_results_completeness(
        self,
        session_id: str,
        user_input: str,
        intermediate_results: list,
        final_output: str,
    ):
        """
        Feature: agent-self-reward-critic, Property 5: 轨迹记录完整性
        
        验证中间结果被完整记录
        Validates: Requirements 3.2
        """
        recorder = TrajectoryRecorder()
        recorder.start(session_id, user_input)
        
        # 添加中间结果
        for result in intermediate_results:
            recorder.add_intermediate_result(result)
        
        trajectory = recorder.finish(final_output)
        
        # 验证中间结果数量和内容
        assert len(trajectory.intermediate_results) == len(intermediate_results)
        assert trajectory.intermediate_results == intermediate_results

"""
Critic Agent 属性测试

测试Critic反思报告结构完整性。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime

# 确保可以导入模块
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from hypothesis import given, strategies as st, settings

from core.critic.models import ProblemType, Problem, Improvement, CriticReport
from core.critic.critic_agent import CriticAgent
from core.trajectory.models import ExecutionTrajectory, ToolCall, ReasoningStep


# ============== 测试策略定义 ==============

# 非空文本策略
non_empty_text = st.text(min_size=1, max_size=200).filter(lambda x: x.strip())

# 时间戳策略
timestamp_strategy = st.datetimes().map(lambda dt: dt.isoformat())

# 问题类型策略
problem_type_strategy = st.sampled_from(list(ProblemType))

# 严重程度策略 (1-5)
severity_strategy = st.integers(min_value=1, max_value=5)

# 优先级策略 (1-5)
priority_strategy = st.integers(min_value=1, max_value=5)

# Problem策略
problem_strategy = st.builds(
    Problem,
    type=problem_type_strategy,
    description=non_empty_text,
    severity=severity_strategy,
    location=non_empty_text
)

# Improvement策略
improvement_strategy = st.builds(
    Improvement,
    problem_ref=non_empty_text,
    suggestion=non_empty_text,
    priority=priority_strategy,
    correction=st.one_of(st.none(), non_empty_text)
)

# ToolCall策略
tool_call_strategy = st.builds(
    ToolCall,
    tool_name=non_empty_text,
    input_params=st.fixed_dictionaries({"param": non_empty_text}),
    output_result=non_empty_text,
    duration_ms=st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False),
    timestamp=timestamp_strategy,
    success=st.booleans(),
    error=st.one_of(st.none(), non_empty_text)
)

# ReasoningStep策略
reasoning_step_strategy = st.builds(
    ReasoningStep,
    step_number=st.integers(min_value=1, max_value=100),
    content=non_empty_text,
    timestamp=timestamp_strategy
)

# ExecutionTrajectory策略
execution_trajectory_strategy = st.builds(
    ExecutionTrajectory,
    trajectory_id=st.uuids().map(str),
    session_id=st.uuids().map(str),
    user_input=non_empty_text,
    system_prompt=st.one_of(st.none(), non_empty_text),
    reasoning_steps=st.lists(reasoning_step_strategy, min_size=0, max_size=5),
    tool_calls=st.lists(tool_call_strategy, min_size=0, max_size=3),
    intermediate_results=st.lists(non_empty_text, min_size=0, max_size=3),
    final_output=non_empty_text,
    total_duration_ms=st.floats(min_value=0.1, max_value=100000.0, allow_nan=False, allow_infinity=False),
    start_time=timestamp_strategy,
    end_time=timestamp_strategy
)


# ============== Mock LLM 响应生成 ==============

def create_mock_analysis_response(
    problems: list = None,
    root_causes: list = None,
    improvements: list = None,
    overall_assessment: str = "测试评估"
) -> str:
    """创建模拟的分析LLM响应"""
    import json
    
    if problems is None:
        problems = []
    if root_causes is None:
        root_causes = []
    if improvements is None:
        improvements = []
    
    data = {
        "problems": problems,
        "root_causes": root_causes,
        "improvements": improvements,
        "overall_assessment": overall_assessment
    }
    
    return f'```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```'


def create_mock_model(response_content: str):
    """创建模拟的LLM模型"""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_content
    mock_model.invoke.return_value = mock_response
    return mock_model


# ============== Property 4: Critic反思报告结构完整性 ==============

class TestCriticReportStructureCompleteness:
    """
    Property 4: Critic反思报告结构完整性
    
    For any Critic分析操作，输出的CriticReport应包含problems、root_causes、
    improvements和overall_assessment字段。
    
    Validates: Requirements 2.3, 2.4
    """

    @given(trajectory=execution_trajectory_strategy)
    @settings(max_examples=100)
    def test_critic_report_has_all_required_fields(
        self,
        trajectory: ExecutionTrajectory
    ):
        """
        Feature: agent-self-reward-critic, Property 4: Critic反思报告结构完整性
        
        验证CriticReport包含所有必需字段：problems、root_causes、improvements、overall_assessment。
        Validates: Requirements 2.3, 2.4
        """
        # 创建模拟响应
        mock_response = create_mock_analysis_response(
            problems=[
                {
                    "type": "logic_error",
                    "description": "测试问题描述",
                    "severity": 3,
                    "location": "步骤1"
                }
            ],
            root_causes=["测试根因"],
            improvements=[
                {
                    "problem_ref": "测试问题描述",
                    "suggestion": "测试改进建议",
                    "priority": 3,
                    "correction": None
                }
            ],
            overall_assessment="测试总体评估"
        )
        mock_model = create_mock_model(mock_response)
        
        # 创建CriticAgent并分析
        critic = CriticAgent(model=mock_model)
        report = critic.analyze(trajectory)
        
        # 验证结果是 CriticReport 类型
        assert isinstance(report, CriticReport), "结果应为 CriticReport 类型"
        
        # 验证 problems 字段存在且为列表
        assert hasattr(report, 'problems'), "报告应包含 problems 字段"
        assert isinstance(report.problems, list), "problems 应为列表"
        
        # 验证 root_causes 字段存在且为列表
        assert hasattr(report, 'root_causes'), "报告应包含 root_causes 字段"
        assert isinstance(report.root_causes, list), "root_causes 应为列表"
        
        # 验证 improvements 字段存在且为列表
        assert hasattr(report, 'improvements'), "报告应包含 improvements 字段"
        assert isinstance(report.improvements, list), "improvements 应为列表"
        
        # 验证 overall_assessment 字段存在且非空
        assert hasattr(report, 'overall_assessment'), "报告应包含 overall_assessment 字段"
        assert report.overall_assessment is not None, "overall_assessment 不应为 None"
        assert len(report.overall_assessment.strip()) > 0, "overall_assessment 不应为空字符串"
        
        # 验证 timestamp 字段存在
        assert hasattr(report, 'timestamp'), "报告应包含 timestamp 字段"
        assert report.timestamp is not None, "timestamp 不应为 None"

    @given(
        num_problems=st.integers(min_value=0, max_value=5),
        trajectory=execution_trajectory_strategy
    )
    @settings(max_examples=100)
    def test_problems_structure_completeness(
        self,
        num_problems: int,
        trajectory: ExecutionTrajectory
    ):
        """
        Feature: agent-self-reward-critic, Property 4: Critic反思报告结构完整性
        
        验证problems列表中每个Problem都包含type、description、severity、location字段。
        Validates: Requirements 2.3, 2.4
        """
        # 生成指定数量的问题
        problems_data = [
            {
                "type": list(ProblemType)[i % len(ProblemType)].value,
                "description": f"问题描述{i+1}",
                "severity": (i % 5) + 1,
                "location": f"步骤{i+1}"
            }
            for i in range(num_problems)
        ]
        
        # 创建模拟响应
        mock_response = create_mock_analysis_response(
            problems=problems_data,
            root_causes=["根因分析"],
            improvements=[
                {
                    "problem_ref": f"问题描述{i+1}",
                    "suggestion": f"改进建议{i+1}",
                    "priority": 3,
                    "correction": None
                }
                for i in range(num_problems)
            ],
            overall_assessment="总体评估"
        )
        mock_model = create_mock_model(mock_response)
        
        # 创建CriticAgent并分析
        critic = CriticAgent(model=mock_model)
        report = critic.analyze(trajectory)
        
        # 验证每个问题的结构
        for problem in report.problems:
            assert isinstance(problem, Problem), "问题应为 Problem 类型"
            
            # 验证 type 字段
            assert hasattr(problem, 'type'), "问题应包含 type 字段"
            assert isinstance(problem.type, ProblemType), "type 应为 ProblemType 枚举类型"
            
            # 验证 description 字段
            assert hasattr(problem, 'description'), "问题应包含 description 字段"
            assert problem.description is not None, "description 不应为 None"
            
            # 验证 severity 字段
            assert hasattr(problem, 'severity'), "问题应包含 severity 字段"
            assert isinstance(problem.severity, int), "severity 应为整数"
            assert 1 <= problem.severity <= 5, f"severity {problem.severity} 应在 1-5 范围内"
            
            # 验证 location 字段
            assert hasattr(problem, 'location'), "问题应包含 location 字段"
            assert problem.location is not None, "location 不应为 None"

    @given(
        num_improvements=st.integers(min_value=0, max_value=5),
        trajectory=execution_trajectory_strategy
    )
    @settings(max_examples=100)
    def test_improvements_structure_completeness(
        self,
        num_improvements: int,
        trajectory: ExecutionTrajectory
    ):
        """
        Feature: agent-self-reward-critic, Property 4: Critic反思报告结构完整性
        
        验证improvements列表中每个Improvement都包含problem_ref、suggestion、priority字段。
        Validates: Requirements 2.3, 2.4
        """
        # 生成指定数量的改进建议
        improvements_data = [
            {
                "problem_ref": f"问题{i+1}",
                "suggestion": f"改进建议{i+1}",
                "priority": (i % 5) + 1,
                "correction": f"修正方案{i+1}" if i % 2 == 0 else None
            }
            for i in range(num_improvements)
        ]
        
        # 创建模拟响应
        mock_response = create_mock_analysis_response(
            problems=[],
            root_causes=["根因分析"],
            improvements=improvements_data,
            overall_assessment="总体评估"
        )
        mock_model = create_mock_model(mock_response)
        
        # 创建CriticAgent并分析
        critic = CriticAgent(model=mock_model)
        report = critic.analyze(trajectory)
        
        # 验证每个改进建议的结构
        for improvement in report.improvements:
            assert isinstance(improvement, Improvement), "改进建议应为 Improvement 类型"
            
            # 验证 problem_ref 字段
            assert hasattr(improvement, 'problem_ref'), "改进建议应包含 problem_ref 字段"
            assert improvement.problem_ref is not None, "problem_ref 不应为 None"
            
            # 验证 suggestion 字段
            assert hasattr(improvement, 'suggestion'), "改进建议应包含 suggestion 字段"
            assert improvement.suggestion is not None, "suggestion 不应为 None"
            
            # 验证 priority 字段
            assert hasattr(improvement, 'priority'), "改进建议应包含 priority 字段"
            assert isinstance(improvement.priority, int), "priority 应为整数"
            assert 1 <= improvement.priority <= 5, f"priority {improvement.priority} 应在 1-5 范围内"
            
            # 验证 correction 字段存在（可以为 None）
            assert hasattr(improvement, 'correction'), "改进建议应包含 correction 字段"

    @given(trajectory=execution_trajectory_strategy)
    @settings(max_examples=100)
    def test_improvements_sorted_by_priority(
        self,
        trajectory: ExecutionTrajectory
    ):
        """
        Feature: agent-self-reward-critic, Property 4: Critic反思报告结构完整性
        
        验证improvements列表按优先级降序排序（高优先级在前）。
        Validates: Requirements 2.4
        """
        # 创建不同优先级的改进建议
        improvements_data = [
            {"problem_ref": "问题1", "suggestion": "建议1", "priority": 2, "correction": None},
            {"problem_ref": "问题2", "suggestion": "建议2", "priority": 5, "correction": None},
            {"problem_ref": "问题3", "suggestion": "建议3", "priority": 1, "correction": None},
            {"problem_ref": "问题4", "suggestion": "建议4", "priority": 4, "correction": None},
        ]
        
        # 创建模拟响应
        mock_response = create_mock_analysis_response(
            problems=[],
            root_causes=[],
            improvements=improvements_data,
            overall_assessment="总体评估"
        )
        mock_model = create_mock_model(mock_response)
        
        # 创建CriticAgent并分析
        critic = CriticAgent(model=mock_model)
        report = critic.analyze(trajectory)
        
        # 验证改进建议按优先级降序排序
        if len(report.improvements) > 1:
            for i in range(len(report.improvements) - 1):
                assert report.improvements[i].priority >= report.improvements[i + 1].priority, \
                    f"改进建议未按优先级降序排序: {report.improvements[i].priority} < {report.improvements[i + 1].priority}"

    @given(trajectory=execution_trajectory_strategy)
    @settings(max_examples=100)
    def test_empty_report_on_analysis_failure(
        self,
        trajectory: ExecutionTrajectory
    ):
        """
        Feature: agent-self-reward-critic, Property 4: Critic反思报告结构完整性
        
        验证当分析失败时，返回的报告仍然包含所有必需字段（空列表和失败说明）。
        Validates: Requirements 2.3, 2.4
        """
        # 创建会导致解析失败的模拟响应
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("模拟LLM调用失败")
        
        # 创建CriticAgent并分析
        critic = CriticAgent(model=mock_model, max_retries=0)
        report = critic.analyze(trajectory)
        
        # 验证即使失败，报告仍然包含所有必需字段
        assert isinstance(report, CriticReport), "结果应为 CriticReport 类型"
        assert hasattr(report, 'problems'), "报告应包含 problems 字段"
        assert hasattr(report, 'root_causes'), "报告应包含 root_causes 字段"
        assert hasattr(report, 'improvements'), "报告应包含 improvements 字段"
        assert hasattr(report, 'overall_assessment'), "报告应包含 overall_assessment 字段"
        assert hasattr(report, 'timestamp'), "报告应包含 timestamp 字段"
        
        # 验证失败时返回空列表
        assert isinstance(report.problems, list), "problems 应为列表"
        assert isinstance(report.root_causes, list), "root_causes 应为列表"
        assert isinstance(report.improvements, list), "improvements 应为列表"

    @given(trajectory=execution_trajectory_strategy)
    @settings(max_examples=100)
    def test_all_problem_types_recognized(
        self,
        trajectory: ExecutionTrajectory
    ):
        """
        Feature: agent-self-reward-critic, Property 4: Critic反思报告结构完整性
        
        验证所有问题类型都能被正确识别和解析。
        Validates: Requirements 2.3
        """
        # 创建包含所有问题类型的模拟响应
        all_problem_types = list(ProblemType)
        problems_data = [
            {
                "type": pt.value,
                "description": f"测试{pt.value}问题",
                "severity": 3,
                "location": f"步骤{i+1}"
            }
            for i, pt in enumerate(all_problem_types)
        ]
        
        # 创建模拟响应
        mock_response = create_mock_analysis_response(
            problems=problems_data,
            root_causes=["根因分析"],
            improvements=[],
            overall_assessment="总体评估"
        )
        mock_model = create_mock_model(mock_response)
        
        # 创建CriticAgent并分析
        critic = CriticAgent(model=mock_model)
        report = critic.analyze(trajectory)
        
        # 验证所有问题类型都被正确解析
        parsed_types = {p.type for p in report.problems}
        expected_types = set(all_problem_types)
        
        assert parsed_types == expected_types, \
            f"未能识别所有问题类型。缺少: {expected_types - parsed_types}"

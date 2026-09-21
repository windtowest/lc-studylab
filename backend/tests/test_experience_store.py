"""
经验记录存储完整性属性测试

Feature: agent-self-reward-critic, Property 7: 经验记录存储完整性
Validates: Requirements 4.1, 4.2
"""

import sys
from pathlib import Path

# 确保可以导入模块
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime
import uuid

from core.memory.models import ExperienceRecord
from core.memory.experience_store import ExperienceStore
from core.trajectory.models import (
    ToolCall,
    ReasoningStep,
    ExecutionTrajectory,
)
from core.self_reward.models import (
    ScoreDimension,
    DimensionScore,
    SelfRewardResult,
)
from core.critic.models import (
    ProblemType,
    Problem,
    Improvement,
    CriticReport,
)


# ==================== 自定义策略 ====================

# 时间戳策略
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31)
).map(lambda dt: dt.isoformat())

# JSON 兼容的 float 策略
json_safe_float = st.floats(
    min_value=-1e9, 
    max_value=1e9, 
    allow_nan=False, 
    allow_infinity=False
)


def filter_null_chars(s: str) -> str:
    """过滤掉 PostgreSQL 不支持的 null 字符"""
    return s.replace('\x00', '').replace('\u0000', '')


def is_valid_pg_text(s: str) -> bool:
    """检查字符串是否可以存储到 PostgreSQL"""
    return '\x00' not in s and '\u0000' not in s and s.strip()


# 非空文本策略 - 过滤掉 PostgreSQL 不支持的 null 字符
# 使用 ASCII 可打印字符 + 常见 Unicode 字符，避免生成 null 字符
pg_safe_alphabet = st.characters(
    blacklist_characters='\x00\u0000',
    blacklist_categories=('Cs',)  # 排除代理对字符
)

non_empty_text = st.text(
    alphabet=pg_safe_alphabet,
    min_size=1, 
    max_size=100
).filter(lambda x: x.strip() and '\x00' not in x)

# PostgreSQL 安全的文本策略（用于可选字段）
pg_safe_text = st.text(
    alphabet=pg_safe_alphabet,
    max_size=200
).filter(lambda x: '\x00' not in x)

# ToolCall 策略
tool_call_strategy = st.builds(
    ToolCall,
    tool_name=non_empty_text,
    input_params=st.dictionaries(
        keys=non_empty_text,
        values=st.one_of(pg_safe_text, st.integers(-1000, 1000), st.booleans()),
        max_size=3
    ),
    output_result=st.one_of(pg_safe_text, st.integers(-1000, 1000), st.none()),
    duration_ms=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    timestamp=timestamp_strategy,
    success=st.booleans(),
    error=st.one_of(st.none(), pg_safe_text),
)

# ReasoningStep 策略
reasoning_step_strategy = st.builds(
    ReasoningStep,
    step_number=st.integers(min_value=1, max_value=100),
    content=non_empty_text,
    timestamp=timestamp_strategy,
)

# ExecutionTrajectory 策略
execution_trajectory_strategy = st.builds(
    ExecutionTrajectory,
    trajectory_id=st.uuids().map(str),
    session_id=st.uuids().map(str),
    user_input=non_empty_text,
    system_prompt=st.one_of(st.none(), pg_safe_text),
    reasoning_steps=st.lists(reasoning_step_strategy, max_size=3),
    tool_calls=st.lists(tool_call_strategy, max_size=3),
    intermediate_results=st.lists(pg_safe_text, max_size=3),
    final_output=non_empty_text,
    total_duration_ms=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    start_time=timestamp_strategy,
    end_time=timestamp_strategy,
)

# DimensionScore 策略
dimension_score_strategy = st.builds(
    DimensionScore,
    dimension=st.sampled_from(list(ScoreDimension)),
    score=st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
    reason=non_empty_text,
)

# SelfRewardResult 策略
self_reward_result_strategy = st.builds(
    SelfRewardResult,
    score=st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
    reason=non_empty_text,
    patch=st.one_of(st.none(), pg_safe_text),
    dimension_scores=st.lists(dimension_score_strategy, min_size=1, max_size=4),
    timestamp=timestamp_strategy,
)

# Problem 策略
problem_strategy = st.builds(
    Problem,
    type=st.sampled_from(list(ProblemType)),
    description=non_empty_text,
    severity=st.integers(min_value=1, max_value=5),
    location=non_empty_text,
)

# Improvement 策略
improvement_strategy = st.builds(
    Improvement,
    problem_ref=non_empty_text,
    suggestion=non_empty_text,
    priority=st.integers(min_value=1, max_value=5),
    correction=st.one_of(st.none(), pg_safe_text),
)

# CriticReport 策略
critic_report_strategy = st.builds(
    CriticReport,
    problems=st.lists(problem_strategy, max_size=3),
    root_causes=st.lists(non_empty_text, max_size=3),
    improvements=st.lists(improvement_strategy, max_size=3),
    overall_assessment=non_empty_text,
    timestamp=timestamp_strategy,
)

# ExperienceRecord 策略
experience_record_strategy = st.builds(
    ExperienceRecord,
    record_id=st.uuids().map(str),
    session_id=st.uuids().map(str),
    trajectory=execution_trajectory_strategy,
    self_reward=self_reward_result_strategy,
    critic_report=critic_report_strategy,
    timestamp=timestamp_strategy,
    tags=st.lists(non_empty_text, max_size=5),
)


# ==================== 测试类 ====================

class TestExperienceRecordStorageCompleteness:
    """
    Property 7: 经验记录存储完整性
    
    For any 完整的执行-评分-反思流程，存储的ExperienceRecord应包含
    trajectory、self_reward、critic_report、timestamp和session_id字段。
    
    Validates: Requirements 4.1, 4.2
    """
    
    @pytest.fixture(autouse=True)
    def setup_store(self):
        """设置测试用的ExperienceStore"""
        try:
            self.store = ExperienceStore()
            yield
        except Exception as e:
            pytest.skip(f"无法连接数据库: {e}")
        finally:
            # 清理测试数据
            pass
    
    @given(record=experience_record_strategy)
    @settings(max_examples=100)
    def test_experience_record_storage_completeness(self, record: ExperienceRecord):
        """
        Feature: agent-self-reward-critic, Property 7: 经验记录存储完整性
        
        验证存储的ExperienceRecord包含所有必需字段：
        - trajectory
        - self_reward
        - critic_report
        - timestamp
        - session_id
        """
        try:
            # 保存记录
            saved_id = self.store.save(record)
            assert saved_id == record.record_id
            
            # 读取记录
            retrieved = self.store.get(record.record_id)
            assert retrieved is not None
            
            # 验证必需字段存在且完整
            # 1. trajectory 字段
            assert retrieved.trajectory is not None
            assert retrieved.trajectory.trajectory_id == record.trajectory.trajectory_id
            assert retrieved.trajectory.session_id == record.trajectory.session_id
            assert retrieved.trajectory.user_input == record.trajectory.user_input
            assert retrieved.trajectory.final_output == record.trajectory.final_output
            assert retrieved.trajectory.total_duration_ms == record.trajectory.total_duration_ms
            
            # 2. self_reward 字段
            assert retrieved.self_reward is not None
            assert retrieved.self_reward.score == record.self_reward.score
            assert retrieved.self_reward.reason == record.self_reward.reason
            assert retrieved.self_reward.patch == record.self_reward.patch
            assert len(retrieved.self_reward.dimension_scores) == len(record.self_reward.dimension_scores)
            
            # 3. critic_report 字段
            assert retrieved.critic_report is not None
            assert retrieved.critic_report.overall_assessment == record.critic_report.overall_assessment
            assert len(retrieved.critic_report.problems) == len(record.critic_report.problems)
            assert len(retrieved.critic_report.root_causes) == len(record.critic_report.root_causes)
            assert len(retrieved.critic_report.improvements) == len(record.critic_report.improvements)
            
            # 4. timestamp 字段
            assert retrieved.timestamp is not None
            assert len(retrieved.timestamp) > 0
            
            # 5. session_id 字段
            assert retrieved.session_id is not None
            assert retrieved.session_id == record.session_id
            
            # 6. record_id 字段
            assert retrieved.record_id == record.record_id
            
            # 7. tags 字段
            assert retrieved.tags == record.tags
            
        finally:
            # 清理测试数据
            try:
                self.store.delete(record.record_id)
            except Exception:
                pass

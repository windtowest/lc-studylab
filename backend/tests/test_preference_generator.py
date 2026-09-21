"""
偏好数据生成器属性测试

Feature: agent-self-reward-critic
Property 8: 偏好数据来源正确性 - Validates: Requirements 5.2, 5.3
Property 9: 偏好数据输入一致性 - Validates: Requirements 5.4
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
from rlhf_integration.preference_generator import PreferenceGenerator, PreferencePair


# ==================== 自定义策略 ====================

# 时间戳策略
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31)
).map(lambda dt: dt.isoformat())

# PostgreSQL 安全字符策略
pg_safe_alphabet = st.characters(
    blacklist_characters='\x00\u0000',
    blacklist_categories=('Cs',)
)

non_empty_text = st.text(
    alphabet=pg_safe_alphabet,
    min_size=1, 
    max_size=100
).filter(lambda x: x.strip() and '\x00' not in x)

# ToolCall 策略
tool_call_strategy = st.builds(
    ToolCall,
    tool_name=non_empty_text,
    input_params=st.dictionaries(
        keys=non_empty_text,
        values=st.one_of(st.text(max_size=50), st.integers(-1000, 1000), st.booleans()),
        max_size=3
    ),
    output_result=st.one_of(st.text(max_size=100), st.integers(-1000, 1000), st.none()),
    duration_ms=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    timestamp=timestamp_strategy,
    success=st.booleans(),
    error=st.one_of(st.none(), st.text(max_size=100)),
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
    system_prompt=st.one_of(st.none(), st.text(max_size=200)),
    reasoning_steps=st.lists(reasoning_step_strategy, max_size=3),
    tool_calls=st.lists(tool_call_strategy, max_size=3),
    intermediate_results=st.lists(st.text(max_size=100), max_size=3),
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
    correction=st.one_of(st.none(), st.text(max_size=200)),
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


def create_self_reward_result(score: float, patch: str = None) -> SelfRewardResult:
    """创建指定评分的SelfRewardResult"""
    return SelfRewardResult(
        score=score,
        reason="Test reason",
        patch=patch,
        dimension_scores=[
            DimensionScore(
                dimension=ScoreDimension.ACCURACY,
                score=score,
                reason="Test dimension reason"
            )
        ],
        timestamp=datetime.now().isoformat()
    )


def create_experience_record(
    user_input: str,
    final_output: str,
    score: float,
    patch: str = None,
    session_id: str = None
) -> ExperienceRecord:
    """创建测试用的ExperienceRecord"""
    session_id = session_id or str(uuid.uuid4())
    
    trajectory = ExecutionTrajectory(
        trajectory_id=str(uuid.uuid4()),
        session_id=session_id,
        user_input=user_input,
        system_prompt=None,
        reasoning_steps=[],
        tool_calls=[],
        intermediate_results=[],
        final_output=final_output,
        total_duration_ms=100.0,
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat()
    )
    
    self_reward = create_self_reward_result(score, patch)
    
    critic_report = CriticReport(
        problems=[],
        root_causes=[],
        improvements=[],
        overall_assessment="Test assessment",
        timestamp=datetime.now().isoformat()
    )
    
    return ExperienceRecord(
        record_id=str(uuid.uuid4()),
        session_id=session_id,
        trajectory=trajectory,
        self_reward=self_reward,
        critic_report=critic_report,
        timestamp=datetime.now().isoformat(),
        tags=[]
    )


# ==================== 高分/低分记录策略 ====================

# 高分记录策略 (score >= 7.0)
@st.composite
def high_score_record_strategy(draw):
    """生成高分记录 (score >= 7.0)"""
    score = draw(st.floats(min_value=7.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    user_input = draw(non_empty_text)
    final_output = draw(non_empty_text)
    return create_experience_record(user_input, final_output, score)


# 低分记录策略 (score < 6.0)
@st.composite
def low_score_record_strategy(draw):
    """生成低分记录 (score < 6.0)"""
    score = draw(st.floats(min_value=0.0, max_value=5.99, allow_nan=False, allow_infinity=False))
    user_input = draw(non_empty_text)
    final_output = draw(non_empty_text)
    return create_experience_record(user_input, final_output, score)


# 带patch的低分记录策略
@st.composite
def low_score_with_patch_record_strategy(draw):
    """生成带patch的低分记录"""
    score = draw(st.floats(min_value=0.0, max_value=5.99, allow_nan=False, allow_infinity=False))
    user_input = draw(non_empty_text)
    final_output = draw(non_empty_text)
    patch = draw(non_empty_text)
    return create_experience_record(user_input, final_output, score, patch=patch)


# 相同输入的高低分记录对策略
@st.composite
def matching_record_pair_strategy(draw):
    """生成相同输入的高低分记录对"""
    user_input = draw(non_empty_text)
    session_id = str(uuid.uuid4())
    
    high_score = draw(st.floats(min_value=7.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    high_output = draw(non_empty_text)
    high_record = create_experience_record(user_input, high_output, high_score, session_id=session_id)
    
    low_score = draw(st.floats(min_value=0.0, max_value=5.99, allow_nan=False, allow_infinity=False))
    low_output = draw(non_empty_text)
    low_record = create_experience_record(user_input, low_output, low_score, session_id=session_id)
    
    return high_record, low_record


# ==================== 测试类 ====================

class TestPreferenceDataSourceCorrectness:
    """
    Property 8: 偏好数据来源正确性
    
    For any 生成的PreferencePair，chosen响应的原始评分应>=chosen_threshold
    或来自patch修正，rejected响应的原始评分应<rejected_threshold。
    
    Validates: Requirements 5.2, 5.3
    """
    
    @pytest.fixture(autouse=True)
    def setup_store(self):
        """设置测试用的ExperienceStore"""
        try:
            self.store = ExperienceStore()
            self.generator = PreferenceGenerator(
                self.store,
                chosen_threshold=7.0,
                rejected_threshold=6.0
            )
            self.created_record_ids = []
            yield
        except Exception as e:
            pytest.skip(f"无法连接数据库: {e}")
        finally:
            # 清理测试数据
            for record_id in self.created_record_ids:
                try:
                    self.store.delete(record_id)
                except Exception:
                    pass
    
    @given(record_pair=matching_record_pair_strategy())
    @settings(max_examples=100)
    def test_preference_data_source_correctness(self, record_pair):
        """
        Feature: agent-self-reward-critic, Property 8: 偏好数据来源正确性
        
        验证生成的偏好数据对满足：
        1. chosen响应来自评分>=7.0的记录或patch修正
        2. rejected响应来自评分<6.0的记录
        """
        high_record, low_record = record_pair
        
        try:
            # 保存记录
            self.store.save(high_record)
            self.created_record_ids.append(high_record.record_id)
            
            self.store.save(low_record)
            self.created_record_ids.append(low_record.record_id)
            
            # 生成偏好对
            pairs = self.generator.generate_pairs()
            
            # 验证每个偏好对的来源正确性
            for pair in pairs:
                # 验证chosen_score >= chosen_threshold 或来自patch
                # 如果chosen_score是patch的假定分数(8.0)，也是有效的
                assert pair.chosen_score >= self.generator.chosen_threshold, \
                    f"chosen_score {pair.chosen_score} < threshold {self.generator.chosen_threshold}"
                
                # 验证rejected_score < rejected_threshold
                assert pair.rejected_score < self.generator.rejected_threshold, \
                    f"rejected_score {pair.rejected_score} >= threshold {self.generator.rejected_threshold}"
                
        finally:
            pass  # cleanup in fixture
    
    @given(patch_record=low_score_with_patch_record_strategy(), 
           low_record=low_score_record_strategy())
    @settings(max_examples=100)
    def test_patch_as_chosen_source(self, patch_record, low_record):
        """
        Feature: agent-self-reward-critic, Property 8: 偏好数据来源正确性
        
        验证带patch的低分记录可以作为chosen来源：
        - patch内容作为chosen响应
        - 原始低分响应作为rejected
        """
        # 确保两个记录有相同的user_input以便匹配
        low_record.trajectory.user_input = patch_record.trajectory.user_input
        low_record.session_id = patch_record.session_id
        
        try:
            # 保存记录
            self.store.save(patch_record)
            self.created_record_ids.append(patch_record.record_id)
            
            self.store.save(low_record)
            self.created_record_ids.append(low_record.record_id)
            
            # 生成偏好对
            pairs = self.generator.generate_pairs()
            
            # 验证偏好对
            for pair in pairs:
                # chosen_score应该是patch的假定分数或高分
                assert pair.chosen_score >= self.generator.chosen_threshold
                
                # rejected_score应该低于阈值
                assert pair.rejected_score < self.generator.rejected_threshold
                
        finally:
            pass


class TestPreferenceDataInputConsistency:
    """
    Property 9: 偏好数据输入一致性
    
    For any PreferencePair，chosen和rejected应对应相同或语义相似的prompt。
    
    Validates: Requirements 5.4
    """
    
    @pytest.fixture(autouse=True)
    def setup_store(self):
        """设置测试用的ExperienceStore"""
        try:
            self.store = ExperienceStore()
            self.generator = PreferenceGenerator(
                self.store,
                chosen_threshold=7.0,
                rejected_threshold=6.0
            )
            self.created_record_ids = []
            yield
        except Exception as e:
            pytest.skip(f"无法连接数据库: {e}")
        finally:
            # 清理测试数据
            for record_id in self.created_record_ids:
                try:
                    self.store.delete(record_id)
                except Exception:
                    pass
    
    @given(record_pair=matching_record_pair_strategy())
    @settings(max_examples=100)
    def test_preference_data_input_consistency(self, record_pair):
        """
        Feature: agent-self-reward-critic, Property 9: 偏好数据输入一致性
        
        验证生成的偏好数据对中，chosen和rejected对应相同的prompt。
        """
        high_record, low_record = record_pair
        
        try:
            # 保存记录
            self.store.save(high_record)
            self.created_record_ids.append(high_record.record_id)
            
            self.store.save(low_record)
            self.created_record_ids.append(low_record.record_id)
            
            # 生成偏好对
            pairs = self.generator.generate_pairs()
            
            # 验证每个偏好对的输入一致性
            for pair in pairs:
                # prompt应该是chosen记录的user_input
                # 由于我们使用相同的user_input创建记录对，prompt应该匹配
                assert pair.prompt is not None and len(pair.prompt) > 0
                
                # 验证source_records存在
                assert len(pair.source_records) == 2
                chosen_id, rejected_id = pair.source_records
                
                # 获取原始记录验证
                chosen_record = self.store.get(chosen_id)
                rejected_record = self.store.get(rejected_id)
                
                if chosen_record and rejected_record:
                    # 验证prompt与chosen记录的user_input一致
                    assert pair.prompt == chosen_record.trajectory.user_input
                    
                    # 验证chosen和rejected来自相同或相似的输入
                    # 由于我们的匹配策略，它们应该有相同的session_id或相似的user_input
                    same_session = chosen_record.session_id == rejected_record.session_id
                    same_input = chosen_record.trajectory.user_input == rejected_record.trajectory.user_input
                    
                    # 计算相似度
                    similarity = self._compute_similarity(
                        chosen_record.trajectory.user_input,
                        rejected_record.trajectory.user_input
                    )
                    
                    # 至少满足以下条件之一：相同session、相同输入、或相似输入
                    assert same_session or same_input or similarity >= 0.3, \
                        f"chosen和rejected的输入不一致: session={same_session}, input={same_input}, similarity={similarity}"
                
        finally:
            pass
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（基于词重叠）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0

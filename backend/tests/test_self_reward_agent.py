"""
Self-Reward Agent 属性测试

测试行为选择正确性和多候选记录完整性。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

# 确保可以导入模块
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from hypothesis import given, strategies as st, settings, assume

from core.self_reward.models import ScoreDimension, DimensionScore, SelfRewardResult
from core.trajectory.models import ExecutionTrajectory, ToolCall, ReasoningStep
from agents.self_reward_agent import SelfRewardAgent, CandidateResult


# ============== 测试策略定义 ==============

# 有效的评分范围策略
valid_score_strategy = st.floats(
    min_value=0.0, max_value=10.0, 
    allow_nan=False, allow_infinity=False
)

# 非空文本策略
non_empty_text = st.text(min_size=1, max_size=200).filter(lambda x: x.strip())

# 时间戳策略
timestamp_strategy = st.datetimes().map(lambda dt: dt.isoformat())

# 维度评分策略
dimension_score_strategy = st.builds(
    DimensionScore,
    dimension=st.sampled_from(list(ScoreDimension)),
    score=valid_score_strategy,
    reason=non_empty_text
)

# SelfRewardResult策略
def self_reward_result_strategy(score: float = None, with_patch: bool = False):
    """创建SelfRewardResult策略"""
    score_st = st.just(score) if score is not None else valid_score_strategy
    patch_st = non_empty_text if with_patch else st.none()
    
    return st.builds(
        SelfRewardResult,
        score=score_st,
        reason=non_empty_text,
        patch=patch_st,
        dimension_scores=st.lists(dimension_score_strategy, min_size=4, max_size=4),
        timestamp=timestamp_strategy
    )


# ExecutionTrajectory策略
def execution_trajectory_strategy():
    """创建ExecutionTrajectory策略"""
    return st.builds(
        ExecutionTrajectory,
        trajectory_id=st.uuids().map(str),
        session_id=st.uuids().map(str),
        user_input=non_empty_text,
        system_prompt=st.one_of(st.none(), non_empty_text),
        reasoning_steps=st.lists(
            st.builds(
                ReasoningStep,
                step_number=st.integers(min_value=1, max_value=10),
                content=non_empty_text,
                timestamp=timestamp_strategy
            ),
            min_size=0, max_size=3
        ),
        tool_calls=st.lists(
            st.builds(
                ToolCall,
                tool_name=non_empty_text,
                input_params=st.just({}),
                output_result=non_empty_text,
                duration_ms=st.floats(min_value=0.0, max_value=10000.0),
                timestamp=timestamp_strategy,
                success=st.booleans(),
                error=st.one_of(st.none(), non_empty_text)
            ),
            min_size=0, max_size=3
        ),
        intermediate_results=st.lists(non_empty_text, min_size=0, max_size=3),
        final_output=non_empty_text,
        total_duration_ms=st.floats(min_value=0.0, max_value=100000.0),
        start_time=timestamp_strategy,
        end_time=timestamp_strategy
    )


# CandidateResult策略
@st.composite
def candidate_result_strategy(draw, score: float = None, with_patch: bool = False):
    """创建CandidateResult策略"""
    response = draw(non_empty_text)
    
    # 创建评分结果
    if score is not None:
        score_val = score
    else:
        score_val = draw(valid_score_strategy)
    
    patch_val = draw(non_empty_text) if with_patch else None
    
    score_result = SelfRewardResult(
        score=score_val,
        reason=draw(non_empty_text),
        patch=patch_val,
        dimension_scores=[
            DimensionScore(
                dimension=dim,
                score=draw(valid_score_strategy),
                reason=draw(non_empty_text)
            )
            for dim in ScoreDimension
        ],
        timestamp=datetime.now().isoformat()
    )
    
    trajectory = draw(execution_trajectory_strategy())
    
    return CandidateResult(
        response=response,
        score_result=score_result,
        trajectory=trajectory
    )


# 多候选列表策略
@st.composite
def candidates_list_strategy(draw, min_size: int = 2, max_size: int = 5):
    """创建候选列表策略，确保评分各不相同"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # 生成不同的评分
    scores = draw(st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size, unique=True
    ))
    
    candidates = []
    for score in scores:
        candidate = draw(candidate_result_strategy(score=score))
        candidates.append(candidate)
    
    return candidates


# ============== Property 10: 行为选择正确性 ==============

class TestBehaviorSelectionCorrectness:
    """
    Property 10: 行为选择正确性
    
    For any 多候选评分场景，最终选择的响应应是评分最高的候选，
    或是应用patch后的修正版本。
    
    Validates: Requirements 6.2, 6.3
    """

    @given(candidates=candidates_list_strategy(min_size=2, max_size=5))
    @settings(max_examples=100)
    def test_selects_highest_score_candidate(self, candidates: list):
        """
        Feature: agent-self-reward-critic, Property 10: 行为选择正确性
        
        验证在多候选场景中，选择评分最高的候选。
        Validates: Requirements 6.2
        """
        # 确保有多个候选
        assume(len(candidates) >= 2)
        
        # 找出最高分候选
        expected_best = max(candidates, key=lambda c: c.score_result.score)
        
        # 创建mock agent来测试_select_best方法
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent.score_threshold = 6.0
        
        # 调用选择方法
        selected_response, selected_result, _ = agent._select_best(
            candidates, "test query"
        )
        
        # 验证选择了最高分候选
        assert selected_result.score == expected_best.score_result.score, \
            f"应选择评分最高的候选 ({expected_best.score_result.score})，" \
            f"实际选择了 ({selected_result.score})"

    @given(
        high_score=st.floats(min_value=6.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        threshold=st.floats(min_value=1.0, max_value=9.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_no_patch_when_score_above_threshold(
        self, 
        high_score: float,
        threshold: float
    ):
        """
        Feature: agent-self-reward-critic, Property 10: 行为选择正确性
        
        验证当最高分高于阈值时，直接返回原始响应，不应用patch。
        Validates: Requirements 6.2
        """
        # 确保评分高于阈值
        assume(high_score >= threshold)
        
        # 创建高分候选
        candidate = CandidateResult(
            response="原始高分响应",
            score_result=SelfRewardResult(
                score=high_score,
                reason="高分理由",
                patch="不应该使用的patch",  # 即使有patch也不应使用
                dimension_scores=[
                    DimensionScore(dimension=dim, score=high_score, reason="理由")
                    for dim in ScoreDimension
                ],
                timestamp=datetime.now().isoformat()
            ),
            trajectory=None
        )
        
        # 创建mock agent
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent.score_threshold = threshold
        
        # 调用选择方法
        selected_response, _, _ = agent._select_best([candidate], "test query")
        
        # 验证返回原始响应
        assert selected_response == "原始高分响应", \
            f"评分 {high_score} >= 阈值 {threshold} 时，应返回原始响应"


    @given(
        low_score=st.floats(min_value=0.0, max_value=5.99, allow_nan=False, allow_infinity=False),
        threshold=st.floats(min_value=6.0, max_value=9.0, allow_nan=False, allow_infinity=False),
        patch_text=non_empty_text
    )
    @settings(max_examples=100)
    def test_applies_patch_when_score_below_threshold(
        self,
        low_score: float,
        threshold: float,
        patch_text: str
    ):
        """
        Feature: agent-self-reward-critic, Property 10: 行为选择正确性
        
        验证当最高分低于阈值且有patch时，应用patch修正。
        Validates: Requirements 6.3
        """
        # 确保评分低于阈值
        assume(low_score < threshold)
        
        # 创建低分候选（带patch）
        original_response = "原始低分响应"
        candidate = CandidateResult(
            response=original_response,
            score_result=SelfRewardResult(
                score=low_score,
                reason="低分理由",
                patch=patch_text,
                dimension_scores=[
                    DimensionScore(dimension=dim, score=low_score, reason="理由")
                    for dim in ScoreDimension
                ],
                timestamp=datetime.now().isoformat()
            ),
            trajectory=None
        )
        
        # 创建mock agent
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent.score_threshold = threshold
        
        # 调用选择方法
        selected_response, _, _ = agent._select_best([candidate], "test query")
        
        # 验证应用了patch（响应应该包含patch内容或被patch替换）
        assert selected_response != original_response or patch_text in selected_response, \
            f"评分 {low_score} < 阈值 {threshold} 且有patch时，应应用patch修正"

    @given(
        low_score=st.floats(min_value=0.0, max_value=5.99, allow_nan=False, allow_infinity=False),
        threshold=st.floats(min_value=6.0, max_value=9.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_returns_original_when_no_patch_available(
        self,
        low_score: float,
        threshold: float
    ):
        """
        Feature: agent-self-reward-critic, Property 10: 行为选择正确性
        
        验证当最高分低于阈值但无patch时，返回原始响应。
        Validates: Requirements 6.3
        """
        # 确保评分低于阈值
        assume(low_score < threshold)
        
        # 创建低分候选（无patch）
        original_response = "原始低分响应无patch"
        candidate = CandidateResult(
            response=original_response,
            score_result=SelfRewardResult(
                score=low_score,
                reason="低分理由",
                patch=None,  # 无patch
                dimension_scores=[
                    DimensionScore(dimension=dim, score=low_score, reason="理由")
                    for dim in ScoreDimension
                ],
                timestamp=datetime.now().isoformat()
            ),
            trajectory=None
        )
        
        # 创建mock agent
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent.score_threshold = threshold
        
        # 调用选择方法
        selected_response, _, _ = agent._select_best([candidate], "test query")
        
        # 验证返回原始响应
        assert selected_response == original_response, \
            f"评分 {low_score} < 阈值 {threshold} 但无patch时，应返回原始响应"

    @given(candidates=candidates_list_strategy(min_size=3, max_size=5))
    @settings(max_examples=100)
    def test_selection_is_deterministic(self, candidates: list):
        """
        Feature: agent-self-reward-critic, Property 10: 行为选择正确性
        
        验证对于相同的候选列表，选择结果是确定性的。
        Validates: Requirements 6.2
        """
        assume(len(candidates) >= 3)
        
        # 创建mock agent
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent.score_threshold = 6.0
        
        # 多次调用选择方法
        results = []
        for _ in range(3):
            response, result, _ = agent._select_best(candidates, "test query")
            results.append((response, result.score))
        
        # 验证所有结果相同
        assert all(r == results[0] for r in results), \
            "对于相同的候选列表，选择结果应该是确定性的"



# ============== Property 11: 多候选记录完整性 ==============

class TestMultiCandidateRecordCompleteness:
    """
    Property 11: 多候选记录完整性
    
    For any 启用多候选模式的执行，所有候选响应及其评分应被完整记录。
    
    Validates: Requirements 6.5
    """

    @given(
        num_candidates=st.integers(min_value=1, max_value=5),
        scores=st.lists(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=100)
    def test_all_candidates_recorded(
        self,
        num_candidates: int,
        scores: list
    ):
        """
        Feature: agent-self-reward-critic, Property 11: 多候选记录完整性
        
        验证所有生成的候选都被记录。
        Validates: Requirements 6.5
        """
        # 确保scores列表长度与num_candidates匹配
        actual_num = min(num_candidates, len(scores))
        assume(actual_num >= 1)
        
        # 创建候选列表
        candidates = []
        for i in range(actual_num):
            score = scores[i] if i < len(scores) else 5.0
            candidate = CandidateResult(
                response=f"候选响应 {i+1}",
                score_result=SelfRewardResult(
                    score=score,
                    reason=f"评分理由 {i+1}",
                    patch=None,
                    dimension_scores=[
                        DimensionScore(dimension=dim, score=score, reason="理由")
                        for dim in ScoreDimension
                    ],
                    timestamp=datetime.now().isoformat()
                ),
                trajectory=None
            )
            candidates.append(candidate)
        
        # 创建mock agent并设置候选记录
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent._candidate_records = candidates
        
        # 获取候选记录
        recorded = agent.get_candidate_records()
        
        # 验证记录数量
        assert len(recorded) == len(candidates), \
            f"应记录 {len(candidates)} 个候选，实际记录了 {len(recorded)} 个"

    @given(candidates=candidates_list_strategy(min_size=2, max_size=5))
    @settings(max_examples=100)
    def test_candidate_records_contain_response(self, candidates: list):
        """
        Feature: agent-self-reward-critic, Property 11: 多候选记录完整性
        
        验证每个候选记录都包含响应内容。
        Validates: Requirements 6.5
        """
        assume(len(candidates) >= 2)
        
        # 创建mock agent并设置候选记录
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent._candidate_records = candidates
        
        # 获取候选记录
        recorded = agent.get_candidate_records()
        
        # 验证每个记录都有响应
        for i, record in enumerate(recorded):
            assert hasattr(record, 'response'), \
                f"候选 {i+1} 应包含 response 属性"
            assert record.response is not None, \
                f"候选 {i+1} 的 response 不应为 None"
            assert len(record.response) > 0, \
                f"候选 {i+1} 的 response 不应为空"

    @given(candidates=candidates_list_strategy(min_size=2, max_size=5))
    @settings(max_examples=100)
    def test_candidate_records_contain_score_result(self, candidates: list):
        """
        Feature: agent-self-reward-critic, Property 11: 多候选记录完整性
        
        验证每个候选记录都包含评分结果。
        Validates: Requirements 6.5
        """
        assume(len(candidates) >= 2)
        
        # 创建mock agent并设置候选记录
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent._candidate_records = candidates
        
        # 获取候选记录
        recorded = agent.get_candidate_records()
        
        # 验证每个记录都有评分结果
        for i, record in enumerate(recorded):
            assert hasattr(record, 'score_result'), \
                f"候选 {i+1} 应包含 score_result 属性"
            assert record.score_result is not None, \
                f"候选 {i+1} 的 score_result 不应为 None"
            assert isinstance(record.score_result, SelfRewardResult), \
                f"候选 {i+1} 的 score_result 应为 SelfRewardResult 类型"
            
            # 验证评分结果的完整性
            assert hasattr(record.score_result, 'score'), \
                f"候选 {i+1} 的评分结果应包含 score"
            assert 0.0 <= record.score_result.score <= 10.0, \
                f"候选 {i+1} 的评分 {record.score_result.score} 应在 [0, 10] 范围内"


    @given(candidates=candidates_list_strategy(min_size=2, max_size=5))
    @settings(max_examples=100)
    def test_candidate_records_are_independent_copies(self, candidates: list):
        """
        Feature: agent-self-reward-critic, Property 11: 多候选记录完整性
        
        验证获取的候选记录是独立副本，修改不影响原始记录。
        Validates: Requirements 6.5
        """
        assume(len(candidates) >= 2)
        
        # 创建mock agent并设置候选记录
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent._candidate_records = candidates
        
        # 获取候选记录
        recorded1 = agent.get_candidate_records()
        recorded2 = agent.get_candidate_records()
        
        # 验证是独立副本
        assert recorded1 is not recorded2, \
            "每次获取应返回独立的列表副本"
        assert len(recorded1) == len(recorded2), \
            "两次获取的记录数量应相同"

    @given(
        num_candidates=st.integers(min_value=2, max_value=5),
        scores=st.lists(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=5, unique=True
        )
    )
    @settings(max_examples=100)
    def test_all_scores_preserved_in_records(
        self,
        num_candidates: int,
        scores: list
    ):
        """
        Feature: agent-self-reward-critic, Property 11: 多候选记录完整性
        
        验证所有候选的评分都被正确保存。
        Validates: Requirements 6.5
        """
        # 确保有足够的分数
        actual_num = min(num_candidates, len(scores))
        assume(actual_num >= 2)
        
        # 创建候选列表
        candidates = []
        expected_scores = set()
        for i in range(actual_num):
            score = scores[i]
            expected_scores.add(score)
            candidate = CandidateResult(
                response=f"候选响应 {i+1}",
                score_result=SelfRewardResult(
                    score=score,
                    reason=f"评分理由 {i+1}",
                    patch=None,
                    dimension_scores=[
                        DimensionScore(dimension=dim, score=score, reason="理由")
                        for dim in ScoreDimension
                    ],
                    timestamp=datetime.now().isoformat()
                ),
                trajectory=None
            )
            candidates.append(candidate)
        
        # 创建mock agent并设置候选记录
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent._candidate_records = candidates
        
        # 获取候选记录
        recorded = agent.get_candidate_records()
        
        # 提取记录的评分
        recorded_scores = {r.score_result.score for r in recorded}
        
        # 验证所有评分都被保存
        assert recorded_scores == expected_scores, \
            f"期望评分 {expected_scores}，实际记录 {recorded_scores}"

    @given(candidates=candidates_list_strategy(min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_candidate_records_preserve_order(self, candidates: list):
        """
        Feature: agent-self-reward-critic, Property 11: 多候选记录完整性
        
        验证候选记录保持原始顺序。
        Validates: Requirements 6.5
        """
        assume(len(candidates) >= 1)
        
        # 记录原始响应顺序
        original_responses = [c.response for c in candidates]
        
        # 创建mock agent并设置候选记录
        with patch.object(SelfRewardAgent, '__init__', lambda self, **kwargs: None):
            agent = SelfRewardAgent.__new__(SelfRewardAgent)
            agent._candidate_records = candidates
        
        # 获取候选记录
        recorded = agent.get_candidate_records()
        
        # 提取记录的响应顺序
        recorded_responses = [r.response for r in recorded]
        
        # 验证顺序一致
        assert recorded_responses == original_responses, \
            "候选记录应保持原始顺序"

"""
Self-Reward 评分器属性测试

测试评分范围有效性、输出结构完整性和评分与Patch一致性。
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
from core.self_reward.scorer import SelfRewardScorer


# ============== 测试策略定义 ==============

# 有效的评分范围策略
valid_score_strategy = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# 超出范围的评分策略（用于测试边界处理）
out_of_range_score_strategy = st.one_of(
    st.floats(min_value=-100.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=10.01, max_value=100.0, allow_nan=False, allow_infinity=False)
)

# 非空文本策略
non_empty_text = st.text(min_size=1, max_size=200).filter(lambda x: x.strip())

# 维度评分策略
dimension_score_strategy = st.builds(
    DimensionScore,
    dimension=st.sampled_from(list(ScoreDimension)),
    score=valid_score_strategy,
    reason=non_empty_text
)

# 时间戳策略
timestamp_strategy = st.datetimes().map(lambda dt: dt.isoformat())

# SelfRewardResult策略
self_reward_result_strategy = st.builds(
    SelfRewardResult,
    score=valid_score_strategy,
    reason=non_empty_text,
    patch=st.one_of(st.none(), non_empty_text),
    dimension_scores=st.lists(dimension_score_strategy, min_size=4, max_size=4),
    timestamp=timestamp_strategy
)


# ============== Mock LLM 响应生成 ==============

def create_mock_scoring_response(
    accuracy_score: float,
    relevance_score: float,
    completeness_score: float,
    clarity_score: float,
    overall_score: float,
    overall_reason: str = "测试评分理由"
) -> str:
    """创建模拟的评分LLM响应"""
    return f'''```json
{{
    "dimension_scores": [
        {{"dimension": "accuracy", "score": {accuracy_score}, "reason": "准确性评分理由"}},
        {{"dimension": "relevance", "score": {relevance_score}, "reason": "相关性评分理由"}},
        {{"dimension": "completeness", "score": {completeness_score}, "reason": "完整性评分理由"}},
        {{"dimension": "clarity", "score": {clarity_score}, "reason": "清晰度评分理由"}}
    ],
    "overall_score": {overall_score},
    "overall_reason": "{overall_reason}"
}}
```'''


def create_mock_model(response_content: str):
    """创建模拟的LLM模型"""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_content
    mock_model.invoke.return_value = mock_response
    return mock_model


# ============== Property 1: Self-Reward评分范围有效性 ==============

class TestScoreRangeValidity:
    """
    Property 1: Self-Reward评分范围有效性
    
    For any Agent响应，Self-Reward评分结果的score字段应在0-10范围内，
    且各维度评分也应在0-10范围内。
    
    Validates: Requirements 1.1
    """

    @given(
        accuracy=valid_score_strategy,
        relevance=valid_score_strategy,
        completeness=valid_score_strategy,
        clarity=valid_score_strategy,
        overall=valid_score_strategy,
        query=non_empty_text,
        response=non_empty_text
    )
    @settings(max_examples=100)
    def test_score_range_validity_with_valid_scores(
        self,
        accuracy: float,
        relevance: float,
        completeness: float,
        clarity: float,
        overall: float,
        query: str,
        response: str
    ):
        """
        Feature: agent-self-reward-critic, Property 1: Self-Reward评分范围有效性
        
        验证当LLM返回有效评分时，结果评分在0-10范围内。
        Validates: Requirements 1.1
        """
        # 创建模拟响应
        mock_response = create_mock_scoring_response(
            accuracy, relevance, completeness, clarity, overall
        )
        mock_model = create_mock_model(mock_response)
        
        # 创建评分器并评分
        scorer = SelfRewardScorer(model=mock_model, threshold=6.0)
        result = scorer.score(query=query, response=response)
        
        # 验证总分在0-10范围内
        assert 0.0 <= result.score <= 10.0, f"总分 {result.score} 超出范围 [0, 10]"
        
        # 验证各维度评分在0-10范围内
        for dim_score in result.dimension_scores:
            assert 0.0 <= dim_score.score <= 10.0, \
                f"维度 {dim_score.dimension} 评分 {dim_score.score} 超出范围 [0, 10]"

    @given(
        out_of_range=out_of_range_score_strategy,
        query=non_empty_text,
        response=non_empty_text
    )
    @settings(max_examples=100)
    def test_score_range_clamping_with_out_of_range_scores(
        self,
        out_of_range: float,
        query: str,
        response: str
    ):
        """
        Feature: agent-self-reward-critic, Property 1: Self-Reward评分范围有效性
        
        验证当LLM返回超出范围的评分时，结果被正确限制在0-10范围内。
        Validates: Requirements 1.1
        """
        # 创建包含超出范围评分的模拟响应
        mock_response = create_mock_scoring_response(
            out_of_range, out_of_range, out_of_range, out_of_range, out_of_range
        )
        mock_model = create_mock_model(mock_response)
        
        # 创建评分器并评分
        scorer = SelfRewardScorer(model=mock_model, threshold=6.0)
        result = scorer.score(query=query, response=response)
        
        # 验证总分被限制在0-10范围内
        assert 0.0 <= result.score <= 10.0, f"总分 {result.score} 未被正确限制在 [0, 10]"
        
        # 验证各维度评分被限制在0-10范围内
        for dim_score in result.dimension_scores:
            assert 0.0 <= dim_score.score <= 10.0, \
                f"维度 {dim_score.dimension} 评分 {dim_score.score} 未被正确限制在 [0, 10]"



# ============== Property 2: Self-Reward输出结构完整性 ==============

class TestOutputStructureCompleteness:
    """
    Property 2: Self-Reward输出结构完整性
    
    For any Self-Reward评分操作，输出结果应包含reason、score、patch和dimension_scores字段，
    且reason非空。
    
    Validates: Requirements 1.2, 1.6
    """

    @given(
        accuracy=valid_score_strategy,
        relevance=valid_score_strategy,
        completeness=valid_score_strategy,
        clarity=valid_score_strategy,
        overall=valid_score_strategy,
        query=non_empty_text,
        response=non_empty_text
    )
    @settings(max_examples=100)
    def test_output_structure_completeness(
        self,
        accuracy: float,
        relevance: float,
        completeness: float,
        clarity: float,
        overall: float,
        query: str,
        response: str
    ):
        """
        Feature: agent-self-reward-critic, Property 2: Self-Reward输出结构完整性
        
        验证评分结果包含所有必需字段：reason、score、patch、dimension_scores，且reason非空。
        Validates: Requirements 1.2, 1.6
        """
        # 创建模拟响应
        mock_response = create_mock_scoring_response(
            accuracy, relevance, completeness, clarity, overall
        )
        mock_model = create_mock_model(mock_response)
        
        # 创建评分器并评分
        scorer = SelfRewardScorer(model=mock_model, threshold=6.0)
        result = scorer.score(query=query, response=response)
        
        # 验证结果是 SelfRewardResult 类型
        assert isinstance(result, SelfRewardResult), "结果应为 SelfRewardResult 类型"
        
        # 验证 score 字段存在且为数值
        assert hasattr(result, 'score'), "结果应包含 score 字段"
        assert isinstance(result.score, (int, float)), "score 应为数值类型"
        
        # 验证 reason 字段存在且非空
        assert hasattr(result, 'reason'), "结果应包含 reason 字段"
        assert result.reason is not None, "reason 不应为 None"
        assert len(result.reason.strip()) > 0, "reason 不应为空字符串"
        
        # 验证 patch 字段存在（可以为 None）
        assert hasattr(result, 'patch'), "结果应包含 patch 字段"
        
        # 验证 dimension_scores 字段存在且为列表
        assert hasattr(result, 'dimension_scores'), "结果应包含 dimension_scores 字段"
        assert isinstance(result.dimension_scores, list), "dimension_scores 应为列表"
        
        # 验证 timestamp 字段存在
        assert hasattr(result, 'timestamp'), "结果应包含 timestamp 字段"
        assert result.timestamp is not None, "timestamp 不应为 None"

    @given(
        query=non_empty_text,
        response=non_empty_text
    )
    @settings(max_examples=100)
    def test_dimension_scores_structure(
        self,
        query: str,
        response: str
    ):
        """
        Feature: agent-self-reward-critic, Property 2: Self-Reward输出结构完整性
        
        验证dimension_scores中每个维度评分都包含dimension、score和reason字段。
        Validates: Requirements 1.2, 1.6
        """
        # 创建模拟响应
        mock_response = create_mock_scoring_response(7.0, 8.0, 6.5, 7.5, 7.25)
        mock_model = create_mock_model(mock_response)
        
        # 创建评分器并评分
        scorer = SelfRewardScorer(model=mock_model, threshold=6.0)
        result = scorer.score(query=query, response=response)
        
        # 验证每个维度评分的结构
        for dim_score in result.dimension_scores:
            assert isinstance(dim_score, DimensionScore), "维度评分应为 DimensionScore 类型"
            
            # 验证 dimension 字段
            assert hasattr(dim_score, 'dimension'), "维度评分应包含 dimension 字段"
            assert isinstance(dim_score.dimension, ScoreDimension), \
                "dimension 应为 ScoreDimension 枚举类型"
            
            # 验证 score 字段
            assert hasattr(dim_score, 'score'), "维度评分应包含 score 字段"
            assert isinstance(dim_score.score, (int, float)), "score 应为数值类型"
            
            # 验证 reason 字段
            assert hasattr(dim_score, 'reason'), "维度评分应包含 reason 字段"
            assert dim_score.reason is not None, "reason 不应为 None"

    @given(query=non_empty_text, response=non_empty_text)
    @settings(max_examples=100)
    def test_all_dimensions_present(
        self,
        query: str,
        response: str
    ):
        """
        Feature: agent-self-reward-critic, Property 2: Self-Reward输出结构完整性
        
        验证评分结果包含所有四个评分维度。
        Validates: Requirements 1.2, 1.6
        """
        # 创建模拟响应
        mock_response = create_mock_scoring_response(7.0, 8.0, 6.5, 7.5, 7.25)
        mock_model = create_mock_model(mock_response)
        
        # 创建评分器并评分
        scorer = SelfRewardScorer(model=mock_model, threshold=6.0)
        result = scorer.score(query=query, response=response)
        
        # 获取结果中的所有维度
        result_dimensions = {ds.dimension for ds in result.dimension_scores}
        
        # 验证包含所有四个维度
        expected_dimensions = {
            ScoreDimension.ACCURACY,
            ScoreDimension.RELEVANCE,
            ScoreDimension.COMPLETENESS,
            ScoreDimension.CLARITY
        }
        
        assert result_dimensions == expected_dimensions, \
            f"缺少维度: {expected_dimensions - result_dimensions}"



# ============== Property 3: 评分与Patch的一致性 ==============

class TestScorePatchConsistency:
    """
    Property 3: 评分与Patch的一致性
    
    For any Self-Reward评分结果，当score < threshold时patch应非空，
    当score >= threshold时patch应为null。
    
    Validates: Requirements 1.3, 1.4
    """

    @given(
        low_score=st.floats(min_value=0.0, max_value=5.99, allow_nan=False, allow_infinity=False),
        query=non_empty_text,
        response=non_empty_text
    )
    @settings(max_examples=100)
    def test_low_score_generates_patch(
        self,
        low_score: float,
        query: str,
        response: str
    ):
        """
        Feature: agent-self-reward-critic, Property 3: 评分与Patch的一致性
        
        验证当评分低于阈值（6.0）时，patch应非空。
        Validates: Requirements 1.3
        """
        # 创建低分模拟响应
        mock_response = create_mock_scoring_response(
            low_score, low_score, low_score, low_score, low_score
        )
        mock_model = create_mock_model(mock_response)
        
        # 配置patch生成响应
        patch_response = MagicMock()
        patch_response.content = "这是改进后的响应内容"
        mock_model.invoke.side_effect = [
            MagicMock(content=mock_response),  # 评分响应
            patch_response  # patch生成响应
        ]
        
        # 创建评分器并评分（阈值为6.0）
        scorer = SelfRewardScorer(model=mock_model, threshold=6.0)
        result = scorer.score(query=query, response=response)
        
        # 验证低分时patch非空
        assert result.score < 6.0, f"测试前提：评分 {result.score} 应低于阈值 6.0"
        assert result.patch is not None, \
            f"当评分 {result.score} < 阈值 6.0 时，patch 应非空"

    @given(
        high_score=st.floats(min_value=6.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        query=non_empty_text,
        response=non_empty_text
    )
    @settings(max_examples=100)
    def test_high_score_no_patch(
        self,
        high_score: float,
        query: str,
        response: str
    ):
        """
        Feature: agent-self-reward-critic, Property 3: 评分与Patch的一致性
        
        验证当评分高于或等于阈值（6.0）时，patch应为null。
        Validates: Requirements 1.4
        """
        # 创建高分模拟响应
        mock_response = create_mock_scoring_response(
            high_score, high_score, high_score, high_score, high_score
        )
        mock_model = create_mock_model(mock_response)
        
        # 创建评分器并评分（阈值为6.0）
        scorer = SelfRewardScorer(model=mock_model, threshold=6.0)
        result = scorer.score(query=query, response=response)
        
        # 验证高分时patch为null
        assert result.score >= 6.0, f"测试前提：评分 {result.score} 应高于或等于阈值 6.0"
        assert result.patch is None, \
            f"当评分 {result.score} >= 阈值 6.0 时，patch 应为 null"

    @given(
        threshold=st.floats(min_value=1.0, max_value=9.0, allow_nan=False, allow_infinity=False),
        score=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        query=non_empty_text,
        response=non_empty_text
    )
    @settings(max_examples=100)
    def test_patch_consistency_with_custom_threshold(
        self,
        threshold: float,
        score: float,
        query: str,
        response: str
    ):
        """
        Feature: agent-self-reward-critic, Property 3: 评分与Patch的一致性
        
        验证对于任意阈值，评分与patch的一致性关系都成立。
        Validates: Requirements 1.3, 1.4
        """
        # 创建模拟响应
        mock_response = create_mock_scoring_response(
            score, score, score, score, score
        )
        mock_model = create_mock_model(mock_response)
        
        # 如果分数低于阈值，配置patch生成响应
        if score < threshold:
            patch_response = MagicMock()
            patch_response.content = "改进后的响应"
            mock_model.invoke.side_effect = [
                MagicMock(content=mock_response),
                patch_response
            ]
        
        # 创建评分器并评分
        scorer = SelfRewardScorer(model=mock_model, threshold=threshold)
        result = scorer.score(query=query, response=response)
        
        # 验证一致性
        if result.score < threshold:
            assert result.patch is not None, \
                f"当评分 {result.score} < 阈值 {threshold} 时，patch 应非空"
        else:
            assert result.patch is None, \
                f"当评分 {result.score} >= 阈值 {threshold} 时，patch 应为 null"

"""
Self-Reward 评分器

实现Agent自评分核心逻辑，包括多维度评分和patch生成。
"""

import json
import re
from datetime import datetime
from typing import Optional, List, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .models import ScoreDimension, DimensionScore, SelfRewardResult
from .prompts import (
    SCORING_SYSTEM_PROMPT,
    PATCH_SYSTEM_PROMPT,
    format_scoring_prompt,
    format_patch_prompt,
)


class SelfRewardScorer:
    """
    Self-Reward评分器
    
    对Agent输出进行多维度评分，并为低分输出生成改进建议。
    """
    
    # 维度权重
    DIMENSION_WEIGHTS = {
        ScoreDimension.ACCURACY: 0.3,
        ScoreDimension.RELEVANCE: 0.25,
        ScoreDimension.COMPLETENESS: 0.25,
        ScoreDimension.CLARITY: 0.2,
    }
    
    def __init__(
        self, 
        model: BaseChatModel,
        threshold: float = 6.0,
        max_retries: int = 2
    ):
        """
        初始化评分器
        
        Args:
            model: LangChain聊天模型
            threshold: 评分阈值，低于此值将生成patch
            max_retries: LLM调用失败时的最大重试次数
        """
        self.model = model
        self.threshold = threshold
        self.max_retries = max_retries
    
    def score(
        self, 
        query: str, 
        response: str, 
        context: Optional[str] = None
    ) -> SelfRewardResult:
        """
        对响应进行评分
        
        Args:
            query: 用户问题
            response: AI响应
            context: 可选的参考上下文
            
        Returns:
            SelfRewardResult: 评分结果
        """
        timestamp = datetime.now().isoformat()
        
        # 调用LLM进行评分
        scoring_result = self._call_scoring_llm(query, response, context)
        
        if scoring_result is None:
            # 评分失败，返回默认中等评分
            return self._create_default_result(timestamp)
        
        # 解析评分结果
        dimension_scores, overall_score, overall_reason = self._parse_scoring_result(
            scoring_result
        )
        
        # 如果解析失败，返回默认结果
        if dimension_scores is None:
            return self._create_default_result(timestamp)
        
        # 确保评分在有效范围内
        overall_score = self._clamp_score(overall_score)
        dimension_scores = [
            DimensionScore(
                dimension=ds.dimension,
                score=self._clamp_score(ds.score),
                reason=ds.reason
            )
            for ds in dimension_scores
        ]
        
        # 如果评分低于阈值，生成patch
        patch = None
        if overall_score < self.threshold:
            patch = self._generate_patch(
                query, response, overall_score, overall_reason, dimension_scores
            )
        
        return SelfRewardResult(
            score=overall_score,
            reason=overall_reason,
            patch=patch,
            dimension_scores=dimension_scores,
            timestamp=timestamp
        )
    
    def score_batch(
        self, 
        items: List[dict]
    ) -> List[SelfRewardResult]:
        """
        批量评分
        
        Args:
            items: 包含query、response和可选context的字典列表
            
        Returns:
            评分结果列表
        """
        results = []
        for item in items:
            result = self.score(
                query=item.get("query", ""),
                response=item.get("response", ""),
                context=item.get("context")
            )
            results.append(result)
        return results
    
    def _call_scoring_llm(
        self, 
        query: str, 
        response: str, 
        context: Optional[str]
    ) -> Optional[str]:
        """
        调用LLM进行评分
        
        Args:
            query: 用户问题
            response: AI响应
            context: 可选的参考上下文
            
        Returns:
            LLM输出的评分结果字符串，失败返回None
        """
        messages = [
            SystemMessage(content=SCORING_SYSTEM_PROMPT),
            HumanMessage(content=format_scoring_prompt(query, response, context))
        ]
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self.model.invoke(messages)
                return result.content
            except Exception as e:
                if attempt == self.max_retries:
                    # 记录错误但不抛出
                    print(f"评分LLM调用失败: {e}")
                    return None
        
        return None
    
    def _parse_scoring_result(
        self, 
        result: str
    ) -> tuple:
        """
        解析评分结果
        
        Args:
            result: LLM输出的评分结果字符串
            
        Returns:
            (dimension_scores, overall_score, overall_reason) 元组
        """
        try:
            # 尝试提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析整个结果
                json_str = result
            
            data = json.loads(json_str)
            
            # 解析维度评分
            dimension_scores = []
            for ds in data.get("dimension_scores", []):
                dimension = ds.get("dimension", "")
                # 将字符串转换为枚举
                try:
                    dim_enum = ScoreDimension(dimension)
                except ValueError:
                    continue
                
                dimension_scores.append(DimensionScore(
                    dimension=dim_enum,
                    score=float(ds.get("score", 5.0)),
                    reason=ds.get("reason", "")
                ))
            
            # 如果没有解析到维度评分，创建默认的
            if not dimension_scores:
                dimension_scores = self._create_default_dimension_scores()
            
            overall_score = float(data.get("overall_score", 5.0))
            overall_reason = data.get("overall_reason", "评分解析不完整")
            
            return dimension_scores, overall_score, overall_reason
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # 尝试从原始输出中提取数字作为分数
            score = self._extract_score_from_text(result)
            if score is not None:
                return (
                    self._create_default_dimension_scores(),
                    score,
                    "评分解析失败，使用提取的分数"
                )
            return None, None, None
    
    def _extract_score_from_text(self, text: str) -> Optional[float]:
        """
        从文本中提取分数
        
        Args:
            text: 包含分数的文本
            
        Returns:
            提取的分数，失败返回None
        """
        # 尝试匹配常见的分数模式
        patterns = [
            r'overall_score["\s:]+(\d+(?:\.\d+)?)',
            r'总分[：:]\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 10:
                        return score
                except ValueError:
                    continue
        
        return None
    
    def _generate_patch(
        self,
        query: str,
        response: str,
        score: float,
        reason: str,
        dimension_scores: List[DimensionScore]
    ) -> Optional[str]:
        """
        生成改进补丁
        
        Args:
            query: 用户问题
            response: 原始响应
            score: 总评分
            reason: 评分理由
            dimension_scores: 各维度评分
            
        Returns:
            改进后的响应，失败返回None
        """
        messages = [
            SystemMessage(content=PATCH_SYSTEM_PROMPT),
            HumanMessage(content=format_patch_prompt(
                query, response, score, reason, dimension_scores
            ))
        ]
        
        try:
            result = self.model.invoke(messages)
            return result.content.strip()
        except Exception as e:
            print(f"Patch生成失败: {e}")
            return None
    
    def _create_default_result(self, timestamp: str) -> SelfRewardResult:
        """
        创建默认评分结果（评分失败时使用）
        
        Args:
            timestamp: 时间戳
            
        Returns:
            默认的SelfRewardResult
        """
        return SelfRewardResult(
            score=5.0,
            reason="评分失败，使用默认中等评分",
            patch=None,
            dimension_scores=self._create_default_dimension_scores(),
            timestamp=timestamp
        )
    
    def _create_default_dimension_scores(self) -> List[DimensionScore]:
        """
        创建默认维度评分
        
        Returns:
            默认的维度评分列表
        """
        return [
            DimensionScore(
                dimension=ScoreDimension.ACCURACY,
                score=5.0,
                reason="默认评分"
            ),
            DimensionScore(
                dimension=ScoreDimension.RELEVANCE,
                score=5.0,
                reason="默认评分"
            ),
            DimensionScore(
                dimension=ScoreDimension.COMPLETENESS,
                score=5.0,
                reason="默认评分"
            ),
            DimensionScore(
                dimension=ScoreDimension.CLARITY,
                score=5.0,
                reason="默认评分"
            ),
        ]
    
    @staticmethod
    def _clamp_score(score: float) -> float:
        """
        将评分限制在0-10范围内
        
        Args:
            score: 原始评分
            
        Returns:
            限制后的评分
        """
        return max(0.0, min(10.0, score))

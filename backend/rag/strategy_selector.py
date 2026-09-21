"""
检索策略自动选择器

根据用户 query 的特征自动选择最合适的检索策略。
支持两种模式：
1. 基于规则的快速选择（rule-based）
2. 基于 LLM 的智能选择（llm-based）
"""

from typing import Literal, Optional
from enum import Enum
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_logger

logger = get_logger(__name__)


class RetrievalStrategy(str, Enum):
    """检索策略枚举"""
    SIMILARITY = "similarity"
    MMR = "mmr"
    THRESHOLD = "similarity_score_threshold"


class QueryIntent(str, Enum):
    """查询意图分类"""
    BROAD_EXPLORATION = "broad_exploration"      # 宽泛探索："介绍一下XX"
    SPECIFIC_QUESTION = "specific_question"      # 具体问题："XX的定义是什么"
    COMPARISON = "comparison"                    # 对比分析："XX和YY的区别"
    RECOMMENDATION = "recommendation"            # 推荐请求："最好的XX是什么"
    HOW_TO = "how_to"                           # 操作指南："如何做XX"
    FACTUAL = "factual"                         # 事实查询："XX发生在什么时候"


# ==================== 方式一：基于规则的快速选择 ====================

class RuleBasedSelector:
    """
    基于规则的策略选择器
    
    优点：速度快，无需调用 LLM，适合高并发场景
    缺点：规则固定，可能不够灵活
    """
    
    # 意图识别规则（关键词匹配）
    INTENT_PATTERNS = {
        QueryIntent.BROAD_EXPLORATION: [
            r"介绍.*?一下", r"概述", r"总结", r"有哪些", r"都有什么",
            r"列举", r"罗列", r"全面.*?了解", r"整体.*?认识"
        ],
        QueryIntent.SPECIFIC_QUESTION: [
            r".*?是什么", r".*?的定义", r"什么是", r"解释.*?含义",
            r".*?指的是", r".*?意思"
        ],
        QueryIntent.COMPARISON: [
            r".*?和.*?的区别", r".*?与.*?的差异", r"对比", r"比较",
            r".*?还是.*?", r"哪个更好"
        ],
        QueryIntent.RECOMMENDATION: [
            r"最好的", r"推荐", r"优秀的", r"高质量", r"哪个好",
            r"建议.*?使用", r"应该.*?选择"
        ],
        QueryIntent.HOW_TO: [
            r"如何", r"怎么", r"怎样", r"如何.*?实现", r"怎么.*?做",
            r"步骤", r"方法", r"教程"
        ],
        QueryIntent.FACTUAL: [
            r"什么时候", r"在哪里", r"谁", r"多少", r"几个",
            r"发生在", r"位于"
        ]
    }
    
    # 意图到策略的映射
    INTENT_TO_STRATEGY = {
        QueryIntent.BROAD_EXPLORATION: RetrievalStrategy.MMR,
        QueryIntent.SPECIFIC_QUESTION: RetrievalStrategy.SIMILARITY,
        QueryIntent.COMPARISON: RetrievalStrategy.MMR,
        QueryIntent.RECOMMENDATION: RetrievalStrategy.THRESHOLD,
        QueryIntent.HOW_TO: RetrievalStrategy.SIMILARITY,
        QueryIntent.FACTUAL: RetrievalStrategy.SIMILARITY,
    }
    
    def __init__(self):
        # 预编译正则表达式
        self.compiled_patterns = {
            intent: [re.compile(pattern) for pattern in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
    
    def detect_intent(self, query: str) -> Optional[QueryIntent]:
        """
        检测查询意图
        
        Args:
            query: 用户查询
            
        Returns:
            检测到的意图，如果无法判断则返回 None
        """
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    logger.info(f"[Rule Selector] 检测到意图: {intent.value}")
                    return intent
        
        logger.info("[Rule Selector] 未检测到明确意图，使用默认策略")
        return None
    
    def select_strategy(self, query: str) -> RetrievalStrategy:
        """
        选择检索策略
        
        Args:
            query: 用户查询
            
        Returns:
            推荐的检索策略
        """
        logger.info(f"[Rule Selector] 分析查询: {query[:50]}...")
        
        # 检测意图
        intent = self.detect_intent(query)
        
        # 根据意图选择策略
        if intent:
            strategy = self.INTENT_TO_STRATEGY[intent]
            logger.info(f"[Rule Selector] 选择策略: {strategy.value} (基于意图: {intent.value})")
            return strategy
        
        # 默认策略：相似度检索
        logger.info("[Rule Selector] 使用默认策略: similarity")
        return RetrievalStrategy.SIMILARITY


# ==================== 方式二：基于 LLM 的智能选择 ====================

class LLMBasedSelector:
    """
    基于 LLM 的策略选择器
    
    优点：更智能，能理解复杂的语义
    缺点：需要调用 LLM，有延迟和成本
    """
    
    # 策略选择提示词
    STRATEGY_SELECTION_PROMPT = """你是一个检索策略专家。根据用户的查询，选择最合适的检索策略。

可选策略：
1. similarity - 相似度检索
   适用场景：具体问题、事实查询、定义解释
   特点：返回最相关的文档，速度快

2. mmr - 最大边际相关性检索
   适用场景：宽泛探索、对比分析、需要多样性的查询
   特点：在相关性和多样性之间平衡，避免结果重复

3. similarity_score_threshold - 相似度阈值过滤
   适用场景：推荐请求、需要高质量结果、宁缺毋滥
   特点：只返回高相关度的文档，可能返回较少结果

用户查询：{query}

请分析这个查询的特点，并选择最合适的策略。只需要输出策略名称（similarity/mmr/similarity_score_threshold），不要有其他内容。

策略："""
    
    def __init__(self, model=None):
        """
        初始化 LLM 选择器
        
        Args:
            model: LangChain 聊天模型实例，如果为 None 则使用默认模型
        """
        if model is None:
            # 使用轻量级模型（比如 gpt-3.5-turbo 或本地小模型）
            from core.models import get_chat_model
            self.model = get_chat_model("openai:gpt-3.5-turbo")
        else:
            self.model = model
        
        # 创建提示词模板
        self.prompt = ChatPromptTemplate.from_template(self.STRATEGY_SELECTION_PROMPT)
        
        # 创建链
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def select_strategy(self, query: str) -> RetrievalStrategy:
        """
        使用 LLM 选择检索策略
        
        Args:
            query: 用户查询
            
        Returns:
            推荐的检索策略
        """
        logger.info(f"[LLM Selector] 分析查询: {query[:50]}...")
        
        try:
            # 调用 LLM
            result = self.chain.invoke({"query": query})
            
            # 解析结果
            result = result.strip().lower()
            
            # 映射到策略枚举
            if "mmr" in result:
                strategy = RetrievalStrategy.MMR
            elif "threshold" in result:
                strategy = RetrievalStrategy.THRESHOLD
            else:
                strategy = RetrievalStrategy.SIMILARITY
            
            logger.info(f"[LLM Selector] LLM 推荐策略: {strategy.value}")
            return strategy
            
        except Exception as e:
            logger.error(f"[LLM Selector] LLM 调用失败: {e}，使用默认策略")
            return RetrievalStrategy.SIMILARITY


# ==================== 方式三：混合选择器（推荐） ====================

class HybridSelector:
    """
    混合选择器：结合规则和 LLM
    
    策略：
    1. 先用规则快速判断（覆盖常见场景）
    2. 如果规则无法判断，再调用 LLM（处理复杂场景）
    
    优点：兼顾速度和准确性
    """
    
    def __init__(self, model=None, use_llm_fallback: bool = True):
        """
        初始化混合选择器
        
        Args:
            model: LLM 模型实例
            use_llm_fallback: 当规则无法判断时是否使用 LLM 兜底
        """
        self.rule_selector = RuleBasedSelector()
        self.use_llm_fallback = use_llm_fallback
        
        if use_llm_fallback:
            self.llm_selector = LLMBasedSelector(model)
        else:
            self.llm_selector = None
    
    def select_strategy(self, query: str) -> RetrievalStrategy:
        """
        选择检索策略
        
        Args:
            query: 用户查询
            
        Returns:
            推荐的检索策略
        """
        logger.info(f"[Hybrid Selector] 分析查询: {query[:50]}...")
        
        # 1. 先尝试规则匹配
        intent = self.rule_selector.detect_intent(query)
        
        if intent:
            # 规则能判断，直接返回
            strategy = self.rule_selector.INTENT_TO_STRATEGY[intent]
            logger.info(f"[Hybrid Selector] 规则匹配成功，策略: {strategy.value}")
            return strategy
        
        # 2. 规则无法判断，使用 LLM 兜底
        if self.use_llm_fallback and self.llm_selector:
            logger.info("[Hybrid Selector] 规则无法判断，调用 LLM...")
            return self.llm_selector.select_strategy(query)
        
        # 3. 默认策略
        logger.info("[Hybrid Selector] 使用默认策略: similarity")
        return RetrievalStrategy.SIMILARITY


# ==================== 工厂函数 ====================

def create_strategy_selector(
    mode: Literal["rule", "llm", "hybrid"] = "hybrid",
    model=None
):
    """
    创建策略选择器
    
    Args:
        mode: 选择器模式
            - "rule": 基于规则（快速）
            - "llm": 基于 LLM（智能）
            - "hybrid": 混合模式（推荐）
        model: LLM 模型实例（仅 llm 和 hybrid 模式需要）
        
    Returns:
        策略选择器实例
        
    Example:
        >>> # 方式一：快速规则选择
        >>> selector = create_strategy_selector(mode="rule")
        >>> strategy = selector.select_strategy("介绍一下机器学习")
        >>> print(strategy)  # RetrievalStrategy.MMR
        >>> 
        >>> # 方式二：LLM 智能选择
        >>> selector = create_strategy_selector(mode="llm")
        >>> strategy = selector.select_strategy("比较深度学习和传统机器学习的优缺点")
        >>> print(strategy)  # RetrievalStrategy.MMR
        >>> 
        >>> # 方式三：混合模式（推荐）
        >>> selector = create_strategy_selector(mode="hybrid")
        >>> strategy = selector.select_strategy("什么是神经网络？")
        >>> print(strategy)  # RetrievalStrategy.SIMILARITY
    """
    if mode == "rule":
        return RuleBasedSelector()
    elif mode == "llm":
        return LLMBasedSelector(model)
    elif mode == "hybrid":
        return HybridSelector(model)
    else:
        raise ValueError(f"不支持的模式: {mode}，可选: rule, llm, hybrid")


# ==================== 测试函数 ====================

def test_strategy_selector():
    """测试策略选择器"""
    print("=" * 60)
    print("测试检索策略自动选择")
    print("=" * 60)
    
    # 测试用例
    test_queries = [
        ("介绍一下机器学习的基本概念", "broad_exploration → MMR"),
        ("什么是神经网络？", "specific_question → similarity"),
        ("深度学习和传统机器学习的区别是什么？", "comparison → MMR"),
        ("推荐一些高质量的机器学习教程", "recommendation → threshold"),
        ("如何训练一个神经网络模型？", "how_to → similarity"),
        ("AlphaGo 是什么时候战胜李世石的？", "factual → similarity"),
    ]
    
    # 测试规则选择器
    print("\n1. 测试规则选择器")
    print("-" * 60)
    rule_selector = RuleBasedSelector()
    
    for query, expected in test_queries:
        strategy = rule_selector.select_strategy(query)
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        print(f"Selected: {strategy.value}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")


if __name__ == "__main__":
    test_strategy_selector()

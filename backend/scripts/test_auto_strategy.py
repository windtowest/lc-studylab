"""
测试自动策略选择功能

演示如何使用自动策略选择来优化 RAG 检索效果
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag import (
    get_embeddings,
    load_vector_store,
    create_retriever,
)
from rag.strategy_selector import create_strategy_selector, RetrievalStrategy


def test_strategy_selector():
    """测试策略选择器"""
    print("=" * 70)
    print("测试 1: 策略选择器")
    print("=" * 70)
    
    # 创建选择器
    selector = create_strategy_selector(mode="rule")
    
    # 测试用例
    test_cases = [
        "介绍一下深度学习的基本概念",
        "什么是卷积神经网络？",
        "深度学习和传统机器学习有什么区别？",
        "推荐一些高质量的机器学习资源",
        "如何训练一个图像分类模型？",
        "Transformer 是什么时候提出的？",
    ]
    
    print("\n测试查询 → 自动选择策略：\n")
    for query in test_cases:
        strategy = selector.select_strategy(query)
        print(f"Query: {query}")
        print(f"  → Strategy: {strategy.value}")
        print()


def test_auto_retrieval():
    """测试自动检索"""
    print("=" * 70)
    print("测试 2: 自动策略检索")
    print("=" * 70)
    
    # 检查索引是否存在
    index_path = "../data/indexes/test_index"
    if not os.path.exists(index_path):
        print(f"\n⚠️  索引不存在: {index_path}")
        print("请先运行 scripts/update_index.py 创建索引")
        return
    
    # 加载向量库
    print("\n1. 加载向量库...")
    embeddings = get_embeddings()
    vector_store = load_vector_store(index_path, embeddings)
    print("   ✓ 向量库加载完成")
    
    # 测试不同查询的自动策略选择
    test_queries = [
        {
            "query": "介绍一下机器学习",
            "expected_strategy": "mmr",
            "reason": "宽泛探索，需要多样性"
        },
        {
            "query": "什么是神经网络？",
            "expected_strategy": "similarity",
            "reason": "具体问题，需要精确答案"
        },
        {
            "query": "推荐最好的深度学习框架",
            "expected_strategy": "similarity_score_threshold",
            "reason": "推荐请求，需要高质量结果"
        }
    ]
    
    print("\n2. 测试自动策略检索：\n")
    
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected = test["expected_strategy"]
        reason = test["reason"]
        
        print(f"测试 {i}: {query}")
        print(f"  预期策略: {expected} ({reason})")
        
        # 创建自动选择的检索器
        retriever = create_retriever(
            vector_store,
            auto_select=True,
            query=query,
            k=3
        )
        
        # 执行检索
        docs = retriever.invoke(query)
        
        print(f"  检索结果: {len(docs)} 个文档")
        if docs:
            print(f"  第一个文档: {docs[0].page_content[:100]}...")
        print()


def compare_strategies():
    """对比不同策略的检索效果"""
    print("=" * 70)
    print("测试 3: 对比不同策略")
    print("=" * 70)
    
    # 检查索引
    index_path = "../data/indexes/test_index"
    if not os.path.exists(index_path):
        print(f"\n⚠️  索引不存在: {index_path}")
        return
    
    # 加载向量库
    print("\n加载向量库...")
    embeddings = get_embeddings()
    vector_store = load_vector_store(index_path, embeddings)
    
    # 测试查询
    query = "介绍一下机器学习的应用"
    print(f"\n查询: {query}\n")
    
    # 对比三种策略
    strategies = ["similarity", "mmr", "similarity_score_threshold"]
    
    for strategy in strategies:
        print(f"策略: {strategy}")
        print("-" * 50)
        
        retriever = create_retriever(
            vector_store,
            search_type=strategy,
            k=3,
            score_threshold=0.7 if strategy == "similarity_score_threshold" else None
        )
        
        docs = retriever.invoke(query)
        
        print(f"检索到 {len(docs)} 个文档：")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc.page_content[:80]}...")
        print()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("自动策略选择测试")
    print("=" * 70 + "\n")
    
    try:
        # 测试 1: 策略选择器
        test_strategy_selector()
        
        # 测试 2: 自动检索（需要索引）
        # test_auto_retrieval()
        
        # 测试 3: 对比策略（需要索引）
        # compare_strategies()
        
        print("\n" + "=" * 70)
        print("✅ 测试完成")
        print("=" * 70)
        
        print("\n使用建议：")
        print("1. 对于大多数场景，使用 auto_select=True 即可")
        print("2. 如果需要精确控制，手动指定 search_type")
        print("3. 混合模式（hybrid）兼顾速度和准确性，推荐使用")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

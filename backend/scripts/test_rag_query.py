#!/usr/bin/env python3
"""
简单的 RAG 查询测试脚本
用于测试 RAG Agent 是否正常工作
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from rag import (
    IndexManager,
    get_embeddings,
    create_retriever,
    create_rag_agent,
    query_rag_agent,
)

def main():
    print("\n" + "="*60)
    print("RAG 查询测试")
    print("="*60 + "\n")
    
    # 索引名称
    index_name = "test_index"
    query = "什么是机器学习？"
    
    print(f"📝 索引: {index_name}")
    print(f"🔍 查询: {query}\n")
    
    try:
        # 1. 加载索引
        print("1️⃣  加载索引...")
        manager = IndexManager()
        
        if not manager.index_exists(index_name):
            print(f"❌ 索引不存在: {index_name}")
            print("   请先创建索引:")
            print(f"   python scripts/rag_cli.py index create {index_name} data/documents/test")
            return 1
        
        embeddings = get_embeddings()
        vector_store = manager.load_index(index_name, embeddings)
        print("✅ 索引加载成功\n")
        
        # 2. 创建检索器
        print("2️⃣  创建检索器...")
        retriever = create_retriever(vector_store, k=4)
        print("✅ 检索器创建成功\n")
        
        # 3. 创建 RAG Agent
        print("3️⃣  创建 RAG Agent...")
        agent = create_rag_agent(retriever)
        print("✅ RAG Agent 创建成功\n")
        
        # 4. 执行查询
        print("4️⃣  执行查询...")
        result = query_rag_agent(agent, query)
        print("✅ 查询完成\n")
        
        # 5. 显示结果
        print("="*60)
        print("回答:")
        print("="*60)
        print(result["answer"])
        print("="*60 + "\n")
        
        print("✅ 测试成功！")
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())


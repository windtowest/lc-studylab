"""
文档检索节点 (Retrieval Node)

本节点负责根据学习计划检索相关文档。
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

from ..state import StudyFlowState, RetrievedDocument
from rag.index_manager import IndexManager
from rag.embeddings import get_embeddings
from rag.retrievers import create_retriever
from config.logging import get_logger

logger = get_logger(__name__)


def retrieval_node(state: StudyFlowState) -> Dict[str, Any]:
    """
    文档检索节点
    
    功能：
    1. 根据学习计划的主题和关键点检索相关文档
    2. 对检索结果进行排序和过滤
    3. 返回最相关的文档列表
    
    Args:
        state: 当前工作流状态
        
    Returns:
        更新后的状态字典，包含 retrieved_docs
    """
    logger.info("[Retrieval Node] 开始检索相关文档")
    
    try:
        learning_plan = state.get("learning_plan")
        if not learning_plan:
            raise ValueError("学习计划不存在，无法进行文档检索")
        
        # 构建检索查询
        # 使用主题 + 关键知识点作为检索查询
        topic = learning_plan["topic"]
        key_points = learning_plan["key_points"]
        
        # 主查询：使用主题
        main_query = f"{topic}"
        logger.info(f"[Retrieval Node] 主查询: {main_query}")
        
        # 加载向量存储和创建检索器
        index_manager = IndexManager()
        embeddings = get_embeddings()
        
        # 尝试加载索引
        try:
            vector_store = index_manager.load_index("test_index", embeddings)
            retriever = create_retriever(vector_store, k=5)
            
            # 执行检索
            logger.info("[Retrieval Node] 执行文档检索...")
            docs = retriever.invoke(main_query)
        except FileNotFoundError:
            logger.warning("[Retrieval Node] 索引不存在，返回空结果")
            docs = []
        
        # 转换为标准格式
        retrieved_docs: List[RetrievedDocument] = []
        for i, doc in enumerate(docs):
            retrieved_doc: RetrievedDocument = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": 1.0 - (i * 0.1)  # 简单的相关性评分
            }
            retrieved_docs.append(retrieved_doc)
        
        logger.info(f"[Retrieval Node] 检索到 {len(retrieved_docs)} 个相关文档")
        
        # 如果检索结果较少，尝试用关键点补充检索
        if len(retrieved_docs) < 3 and key_points and docs:  # 确保 retriever 可用
            logger.info("[Retrieval Node] 文档较少，使用关键点补充检索...")
            try:
                for point in key_points[:2]:  # 最多用前2个关键点
                    additional_docs = retriever.invoke(point)
                    for doc in additional_docs[:2]:  # 每个关键点最多取2个文档
                        if doc.page_content not in [d["content"] for d in retrieved_docs]:
                            retrieved_doc: RetrievedDocument = {
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "relevance_score": 0.7
                            }
                            retrieved_docs.append(retrieved_doc)
            except Exception as e:
                logger.warning(f"[Retrieval Node] 补充检索失败: {e}")
        
        logger.info(f"[Retrieval Node] 最终检索到 {len(retrieved_docs)} 个文档")
        
        # 构建检索摘要
        retrieval_summary = f"\n\n 已检索到 {len(retrieved_docs)} 个相关文档，将用于生成学习内容和练习题。"
        
        # 更新状态
        return {
            "retrieved_docs": retrieved_docs,
            "messages": [{"role": "assistant", "content": retrieval_summary}],
            "current_step": "retrieval",
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[Retrieval Node] 文档检索失败: {str(e)}", exc_info=True)
        
        # 如果检索失败，返回空列表而不是报错，让流程继续
        # 这样即使没有文档，也可以基于 LLM 的知识生成内容
        logger.warning("[Retrieval Node] 检索失败，将继续使用 LLM 内置知识")
        return {
            "retrieved_docs": [],
            "messages": [{"role": "assistant", "content": "\n\n文档检索遇到问题，将使用 AI 内置知识继续生成内容。"}],
            "current_step": "retrieval",
            "updated_at": datetime.now().isoformat()
        }


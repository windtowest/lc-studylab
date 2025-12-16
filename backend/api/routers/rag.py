"""
RAG API 路由

提供 RAG 相关的 HTTP 接口：
- 索引管理（创建、列表、删除、统计）
- 文档管理（上传、添加目录）
- 查询接口（RAG 问答、纯检索）
- 流式查询接口

使用 FastAPI 实现 RESTful API。
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import asyncio

from config import settings, get_logger
from rag import (
    IndexManager,
    load_document,
    load_directory,
    split_documents,
    get_embeddings,
    create_retriever,
    create_rag_agent,
    query_rag_agent,
)

logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/rag", tags=["RAG"])

# 全局索引管理器
index_manager = IndexManager()


# ==================== Pydantic 模型 ====================

class CreateIndexRequest(BaseModel):
    """创建索引请求"""
    name: str = Field(..., description="索引名称")
    directory_path: str = Field(..., description="文档目录路径")
    description: str = Field(default="", description="索引描述")
    chunk_size: Optional[int] = Field(default=None, description="分块大小")
    chunk_overlap: Optional[int] = Field(default=None, description="分块重叠")
    overwrite: bool = Field(default=False, description="是否覆盖已存在的索引")


class IndexInfo(BaseModel):
    """索引信息"""
    name: str
    description: str
    created_at: str
    updated_at: str
    num_documents: int
    store_type: str = "faiss"
    embedding_model: str


class QueryRequest(BaseModel):
    """查询请求"""
    index_name: str = Field(..., description="索引名称")
    query: str = Field(..., description="查询问题")
    k: Optional[int] = Field(default=4, description="返回文档数量")
    return_sources: bool = Field(default=True, description="是否返回来源")


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str
    sources: List[str] = []
    # retrieved_documents: List[dict] = []


class SearchRequest(BaseModel):
    """检索请求（纯检索，不生成回答）"""
    index_name: str = Field(..., description="索引名称")
    query: str = Field(..., description="检索查询")
    k: Optional[int] = Field(default=4, description="返回文档数量")
    score_threshold: Optional[float] = Field(default=None, description="相似度阈值")


class SearchResult(BaseModel):
    """检索结果"""
    content: str
    metadata: dict
    score: Optional[float] = None


# ==================== 索引管理接口 ====================

@router.post("/index", response_model=IndexInfo)
async def create_index(request: CreateIndexRequest):
    """
    创建新索引
    
    从指定目录加载文档，创建向量索引。
    
    Example:
        ```bash
        curl -X POST "http://localhost:8000/rag/index" \\
          -H "Content-Type: application/json" \\
          -d '{
            "name": "my_docs",
            "directory_path": "data/documents/test",
            "description": "测试文档索引",
            "chunk_size": 1000
          }'
        ```
    """
    try:
        logger.info(f"创建索引请求: {request.name}")
        
        # 检查目录是否存在
        directory_path = Path(request.directory_path)
        if not directory_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"目录不存在: {request.directory_path}"
            )
        
        # 检查索引是否已存在
        if index_manager.index_exists(request.name) and not request.overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"索引已存在: {request.name}。使用 overwrite=true 来覆盖。"
            )
        
        # 加载文档
        logger.info(f"加载文档: {directory_path}")
        documents = load_directory(str(directory_path))
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="目录中没有找到支持的文档"
            )
        
        # 分块文档
        logger.info("分块文档...")
        chunks = split_documents(
            documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        
        # 创建 embeddings
        logger.info("创建 embeddings...")
        embeddings = get_embeddings()
        
        # 创建索引
        logger.info("创建向量索引...")
        index_manager.create_index(
            name=request.name,
            documents=chunks,
            embeddings=embeddings,
            description=request.description,
            overwrite=request.overwrite,
        )
        
        # 获取索引信息
        index_info = index_manager.get_index_info(request.name)
        
        logger.info(f"索引创建成功: {request.name}")
        return IndexInfo(**index_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建索引失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/list", response_model=List[IndexInfo])
async def list_indexes():
    """
    列出所有索引
    
    Example:
        ```bash
        curl "http://localhost:8000/rag/index/list"
        ```
    """
    try:
        indexes = index_manager.list_indexes()
        return [IndexInfo(**idx) for idx in indexes]
    except Exception as e:
        logger.error(f"列出索引失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/{name}", response_model=IndexInfo)
async def get_index_info(name: str):
    """
    获取索引详细信息
    
    Example:
        ```bash
        curl "http://localhost:8000/rag/index/my_docs"
        ```
    """
    try:
        index_info = index_manager.get_index_info(name)
        
        if not index_info:
            raise HTTPException(
                status_code=404,
                detail=f"索引不存在: {name}"
            )
        
        return IndexInfo(**index_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取索引信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index/{name}")
async def delete_index(name: str):
    """
    删除索引
    
    Example:
        ```bash
        curl -X DELETE "http://localhost:8000/rag/index/my_docs"
        ```
    """
    try:
        if not index_manager.index_exists(name):
            raise HTTPException(
                status_code=404,
                detail=f"索引不存在: {name}"
            )
        
        index_manager.delete_index(name)
        
        return {"message": f"索引已删除: {name}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除索引失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 查询接口 ====================

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG 查询（非流式）
    
    基于索引内容回答问题。
    
    Example:
        ```bash
        curl -X POST "http://localhost:8000/rag/query" \\
          -H "Content-Type: application/json" \\
          -d '{
            "index_name": "my_docs",
            "query": "什么是机器学习？",
            "k": 4
          }'
        ```
    """
    try:
        logger.info(f"RAG 查询: {request.query[:50]}...")
        
        # 检查索引是否存在
        if not index_manager.index_exists(request.index_name):
            raise HTTPException(
                status_code=404,
                detail=f"索引不存在: {request.index_name}"
            )
        
        # 加载索引
        embeddings = get_embeddings()
        vector_store = index_manager.load_index(request.index_name, embeddings)
        
        # 创建检索器
        retriever = create_retriever(vector_store, k=request.k)
        
        # 创建 RAG Agent
        agent = create_rag_agent(retriever)
        
        # 查询
        result = query_rag_agent(
            agent,
            request.query,
            return_sources=request.return_sources,
        )
        
        logger.info("查询完成")
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            # retrieved_documents=[
            #     {
            #         "content": doc.page_content,
            #         "metadata": doc.metadata,
            #     }
            #     for doc in result.get("retrieved_documents", [])
            # ],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    RAG 查询（流式）
    
    使用 Server-Sent Events (SSE) 返回流式响应。
    
    Example:
        ```bash
        curl -X POST "http://localhost:8000/rag/query/stream" \\
          -H "Content-Type: application/json" \\
          -d '{
            "index_name": "my_docs",
            "query": "什么是机器学习？"
          }'
        ```
    """
    try:
        logger.info(f"RAG 流式查询: {request.query[:50]}...")
        
        # 检查索引是否存在
        if not index_manager.index_exists(request.index_name):
            raise HTTPException(
                status_code=404,
                detail=f"索引不存在: {request.index_name}"
            )
        
        # 加载索引
        embeddings = get_embeddings()
        vector_store = index_manager.load_index(request.index_name, embeddings)
        
        # 创建检索器
        retriever = create_retriever(vector_store, k=request.k)
        
        # 创建 RAG Agent
        agent = create_rag_agent(retriever, streaming=True)
        
        # 流式生成器
        async def event_generator():
            try:
                # 流式执行 - 使用字典输入
                async for chunk in agent.astream({"messages": [{"role": "user", "content": request.query}]}):
                    # 提取内容
                    if isinstance(chunk, dict) and "messages" in chunk:
                        messages = chunk["messages"]
                        if messages:
                            content = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
                        else:
                            content = str(chunk)
                    else:
                        content = str(chunk)
                    
                    # 输出内容
                    data = {
                        "type": "content",
                        "content": content,
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                # 发送完成信号
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"流式查询错误: {e}")
                error_data = {
                    "type": "error",
                    "error": str(e),
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"流式查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """
    纯检索（不生成回答）
    
    只返回相关文档，不使用 LLM 生成回答。
    
    Example:
        ```bash
        curl -X POST "http://localhost:8000/rag/search" \\
          -H "Content-Type: application/json" \\
          -d '{
            "index_name": "my_docs",
            "query": "机器学习",
            "k": 3
          }'
        ```
    """
    try:
        logger.info(f"检索: {request.query[:50]}...")
        
        # 检查索引是否存在
        if not index_manager.index_exists(request.index_name):
            raise HTTPException(
                status_code=404,
                detail=f"索引不存在: {request.index_name}"
            )
        
        # 加载索引
        embeddings = get_embeddings()
        vector_store = index_manager.load_index(request.index_name, embeddings)
        
        # 执行检索
        from rag.vector_stores import search_vector_store
        results = search_vector_store(
            vector_store,
            request.query,
            k=request.k,
            score_threshold=request.score_threshold,
        )
        
        logger.info(f"找到 {len(results)} 个文档")
        
        return [
            SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
            )
            for doc, score in results
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 健康检查 ====================

@router.get("/health")
async def health_check():
    """
    健康检查
    
    Example:
        ```bash
        curl "http://localhost:8000/rag/health"
        ```
    """
    return {
        "status": "healthy",
        "indexes_count": len(index_manager.list_indexes()),
        "base_path": str(index_manager.base_path),
    }


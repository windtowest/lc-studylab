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
            from langchain_core.messages import AIMessage, ToolMessage
            
            try:
                # 发送开始事件
                yield f"data: {json.dumps({'type': 'start', 'message': '开始查询...'}, ensure_ascii=False)}\n\n"
                
                # 准备输入
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=request.query)]
                graph_input = {"messages": messages}
                
                # 追踪状态
                current_content = ""
                tool_calls_map = {}
                all_messages = []
                
                # 使用 astream 获取详细输出
                async for chunk in agent.astream(graph_input, stream_mode="messages"):
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        message, metadata = chunk
                    else:
                        message = chunk
                        metadata = {}
                    
                    # 保存所有消息
                    all_messages.append(message)
                    
                    # 处理 AI 消息
                    if isinstance(message, AIMessage):
                        # 提取并发送工具调用
                        tool_calls = getattr(message, "tool_calls", [])
                        if tool_calls:
                            for tool_call in tool_calls:
                                tool_id = tool_call.get("id", "")
                                tool_name = tool_call.get("name", "")
                                
                                tool_info = {
                                    "id": tool_id,
                                    "name": tool_name,
                                    "type": f"tool-call-{tool_name}",
                                    "state": "input-available",
                                    "parameters": tool_call.get("args", {}),
                                }
                                tool_calls_map[tool_id] = tool_info
                                
                                # 发送工具调用事件
                                yield f"data: {json.dumps({'type': 'tool', 'data': tool_info}, ensure_ascii=False)}\n\n"
                        
                        # 发送内容增量
                        if message.content and not tool_calls:
                            # 计算新增内容
                            def _lcp_len(a: str, b: str) -> int:
                                i = 0
                                for ca, cb in zip(a, b):
                                    if ca != cb:
                                        break
                                    i += 1
                                return i
                            
                            lcp = _lcp_len(current_content, message.content)
                            if lcp < len(message.content):
                                new_content = message.content[lcp:]
                                current_content = message.content
                                
                                chunk_data = {
                                    "type": "chunk",
                                    "content": new_content,
                                }
                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    
                    # 处理工具结果
                    elif isinstance(message, ToolMessage):
                        tool_call_id = getattr(message, "tool_call_id", "")
                        is_error = getattr(message, "status", None) == "error"
                        
                        if tool_call_id in tool_calls_map:
                            tool_info = tool_calls_map[tool_call_id]
                            tool_info["state"] = "output-error" if is_error else "output-available"
                            tool_info["result"] = None if is_error else message.content
                            tool_info["error"] = message.content if is_error else None
                            
                            # 发送工具结果更新
                            yield f"data: {json.dumps({'type': 'tool_result', 'data': tool_info}, ensure_ascii=False)}\n\n"
                    
                    # 小延迟
                    await asyncio.sleep(0.01)
                
                # 查找最终 AI 消息
                final_ai_message = None
                for msg in reversed(all_messages):
                    if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                        final_ai_message = msg
                        break
                
                # 发送剩余内容
                if final_ai_message and final_ai_message.content:
                    final_content = final_ai_message.content
                    if len(final_content) > len(current_content):
                        remaining_content = final_content[len(current_content):]
                        if remaining_content:
                            chunk_data = {
                                "type": "chunk",
                                "content": remaining_content,
                            }
                            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                # 发送结束事件
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                
                logger.info("流式查询完成")
                
            except Exception as e:
                logger.error(f"流式查询错误: {e}")
                logger.exception(e)
                error_data = {
                    "type": "error",
                    "message": "查询过程中出现错误",
                    "error": str(e),
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
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


"""
文本分块器模块

提供多种文本分块策略，用于将长文档分割成适合向量化的小块。

支持的分块器：
- RecursiveCharacterTextSplitter: 递归字符分块（推荐）
- CharacterTextSplitter: 简单字符分块
- MarkdownTextSplitter: Markdown 专用分块
- TokenTextSplitter: 基于 Token 的分块

参考：
- https://reference.langchain.com/python/langchain_text_splitters/
"""

from typing import List, Optional, Literal
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter,
)

from config import settings, get_logger

logger = get_logger(__name__)


# 分块器类型
SplitterType = Literal["recursive", "character", "markdown", "token"]


def get_text_splitter(
    splitter_type: SplitterType = "recursive",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    **kwargs,
):
    """
    获取文本分块器
    
    Args:
        splitter_type: 分块器类型
            - "recursive": 递归字符分块（推荐，适合大多数情况）
            - "character": 简单字符分块
            - "markdown": Markdown 专用分块
            - "token": 基于 Token 的分块
        chunk_size: 分块大小（字符数或 token 数），默认使用配置值
        chunk_overlap: 分块重叠大小，默认使用配置值
        **kwargs: 其他传递给分块器的参数
        
    Returns:
        文本分块器实例
        
    Example:
        >>> # 使用默认配置
        >>> splitter = get_text_splitter()
        >>> 
        >>> # 自定义参数
        >>> splitter = get_text_splitter(
        ...     splitter_type="recursive",
        ...     chunk_size=500,
        ...     chunk_overlap=50
        ... )
        >>> 
        >>> # Markdown 专用
        >>> splitter = get_text_splitter(splitter_type="markdown")
    """
    # 使用配置中的默认值
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    logger.debug(
        f"创建文本分块器: type={splitter_type}, "
        f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )
    
    if splitter_type == "recursive":
        # 递归字符分块器（推荐）
        # 会尝试按照 \n\n, \n, 空格等分隔符递归分割
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            **kwargs,
        )
    
    elif splitter_type == "character":
        # 简单字符分块器
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n",  # 默认按段落分割
            length_function=len,
            is_separator_regex=False,
            **kwargs,
        )
    
    elif splitter_type == "markdown":
        # Markdown 专用分块器
        # 会按照 Markdown 的标题层级进行分割
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    
    elif splitter_type == "token":
        # 基于 Token 的分块器
        # 使用 tiktoken 计算 token 数量
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    
    else:
        raise ValueError(
            f"不支持的分块器类型: {splitter_type}。"
            f"支持的类型: recursive, character, markdown, token"
        )


def split_documents(
    documents: List[Document],
    splitter_type: SplitterType = "recursive",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    **kwargs,
) -> List[Document]:
    """
    分块文档列表
    
    Args:
        documents: 文档列表
        splitter_type: 分块器类型
        chunk_size: 分块大小
        chunk_overlap: 分块重叠大小
        **kwargs: 其他参数
        
    Returns:
        分块后的文档列表
        
    Example:
        >>> from rag.loaders import load_document
        >>> 
        >>> # 加载文档
        >>> documents = load_document("document.pdf")
        >>> print(f"原始文档: {len(documents)} 个")
        >>> 
        >>> # 分块
        >>> chunks = split_documents(documents)
        >>> print(f"分块后: {len(chunks)} 个")
        >>> 
        >>> # 查看第一个块
        >>> print(chunks[0].page_content[:200])
    """
    if not documents:
        logger.warning("文档列表为空，无需分块")
        return []
    
    logger.info(f"开始分块: {len(documents)} 个文档")
    
    # 获取分块器
    splitter = get_text_splitter(
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    
    # 执行分块
    try:
        chunks = splitter.split_documents(documents)
        
        logger.info(f"分块完成: {len(chunks)} 个文本块")
        
        # 统计信息
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0
        
        logger.info(f"   平均块大小: {avg_chars:.0f} 字符")
        logger.info(f"   总字符数: {total_chars}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"分块失败: {e}")
        raise


def split_text(
    text: str,
    splitter_type: SplitterType = "recursive",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    metadata: Optional[dict] = None,
    **kwargs,
) -> List[Document]:
    """
    分块纯文本
    
    Args:
        text: 要分块的文本
        splitter_type: 分块器类型
        chunk_size: 分块大小
        chunk_overlap: 分块重叠大小
        metadata: 要添加到所有块的元数据
        **kwargs: 其他参数
        
    Returns:
        分块后的文档列表
        
    Example:
        >>> text = "这是一段很长的文本..." * 1000
        >>> chunks = split_text(
        ...     text,
        ...     chunk_size=500,
        ...     metadata={"source": "manual_input"}
        ... )
    """
    if not text:
        logger.warning("文本为空，无需分块")
        return []
    
    logger.info(f"开始分块文本: {len(text)} 字符")
    
    # 获取分块器
    splitter = get_text_splitter(
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    
    # 执行分块
    try:
        # 使用 create_documents 方法，可以添加元数据
        metadatas = [metadata] if metadata else None
        chunks = splitter.create_documents(
            texts=[text],
            metadatas=metadatas,
        )
        
        logger.info(f"分块完成: {len(chunks)} 个文本块")
        
        return chunks
        
    except Exception as e:
        logger.error(f"分块失败: {e}")
        raise


def get_optimal_chunk_size(
    document_type: str = "general",
) -> tuple[int, int]:
    """
    根据文档类型获取推荐的分块参数
    
    Args:
        document_type: 文档类型
            - "general": 通用文档（默认）
            - "code": 代码文档
            - "markdown": Markdown 文档
            - "academic": 学术论文
            - "chat": 对话记录
            
    Returns:
        (chunk_size, chunk_overlap) 元组
        
    Example:
        >>> chunk_size, overlap = get_optimal_chunk_size("code")
        >>> splitter = get_text_splitter(
        ...     chunk_size=chunk_size,
        ...     chunk_overlap=overlap
        ... )
    """
    # 不同类型文档的推荐参数
    recommendations = {
        "general": (1000, 200),      # 通用文档
        "code": (1500, 300),          # 代码需要更大的上下文
        "markdown": (800, 150),       # Markdown 通常结构清晰
        "academic": (1200, 250),      # 学术论文需要保持上下文
        "chat": (500, 50),            # 对话记录可以更小
    }
    
    if document_type not in recommendations:
        logger.warning(
            f"未知的文档类型: {document_type}，使用默认参数"
        )
        return recommendations["general"]
    
    chunk_size, overlap = recommendations[document_type]
    logger.info(
        f"推荐的分块参数 ({document_type}): "
        f"chunk_size={chunk_size}, overlap={overlap}"
    )
    
    return chunk_size, overlap


def analyze_chunks(chunks: List[Document]) -> dict:
    """
    分析分块结果的统计信息
    
    Args:
        chunks: 分块后的文档列表
        
    Returns:
        包含统计信息的字典
        
    Example:
        >>> chunks = split_documents(documents)
        >>> stats = analyze_chunks(chunks)
        >>> print(f"平均块大小: {stats['avg_chunk_size']}")
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_chars": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
        }
    
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    total_chars = sum(chunk_sizes)
    
    stats = {
        "total_chunks": len(chunks),
        "total_chars": total_chars,
        "avg_chunk_size": total_chars / len(chunks),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes),
    }
    
    logger.info("分块统计:")
    logger.info(f"   总块数: {stats['total_chunks']}")
    logger.info(f"   总字符数: {stats['total_chars']}")
    logger.info(f"   平均大小: {stats['avg_chunk_size']:.0f} 字符")
    logger.info(f"   最小块: {stats['min_chunk_size']} 字符")
    logger.info(f"   最大块: {stats['max_chunk_size']} 字符")
    
    return stats


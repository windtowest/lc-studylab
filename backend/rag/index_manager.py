"""
索引管理器模块

提供向量索引的统一管理接口，包括：
- 索引的创建、加载、保存、删除
- 索引元数据管理
- 索引列表和统计

索引结构：
data/indexes/
  ├── index_name/
  │   ├── index.faiss        # FAISS 索引文件
  │   ├── index.pkl          # FAISS 文档存储
  │   └── metadata.json      # 索引元数据
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config import settings, get_logger
from rag.vector_stores import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
    add_documents_to_vector_store,
    delete_vector_store,
    get_vector_store_stats,
)

logger = get_logger(__name__)


class IndexManager:
    """
    索引管理器
    
    管理所有向量索引的生命周期，包括创建、加载、更新、删除等操作。
    
    Example:
        >>> from rag import IndexManager, get_embeddings
        >>> 
        >>> # 创建管理器
        >>> manager = IndexManager()
        >>> embeddings = get_embeddings()
        >>> 
        >>> # 创建索引
        >>> manager.create_index(
        ...     name="my_docs",
        ...     documents=chunks,
        ...     embeddings=embeddings,
        ...     description="我的文档索引"
        ... )
        >>> 
        >>> # 列出所有索引
        >>> indexes = manager.list_indexes()
        >>> 
        >>> # 加载索引
        >>> vector_store = manager.load_index("my_docs", embeddings)
        >>> 
        >>> # 删除索引
        >>> manager.delete_index("my_docs")
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        初始化索引管理器
        
        Args:
            base_path: 索引存储的基础路径，默认使用配置中的路径
        """
        self.base_path = Path(base_path or settings.vector_store_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"索引管理器初始化: {self.base_path}")
    
    def _get_index_path(self, name: str) -> Path:
        """获取索引的完整路径"""
        return self.base_path / name
    
    def _get_metadata_path(self, name: str) -> Path:
        """获取索引元数据文件路径"""
        return self._get_index_path(name) / "metadata.json"
    
    def _save_metadata(
        self,
        name: str,
        metadata: Dict[str, Any],
    ) -> None:
        """保存索引元数据"""
        metadata_path = self._get_metadata_path(name)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"保存元数据: {metadata_path}")
    
    def _load_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """加载索引元数据"""
        metadata_path = self._get_metadata_path(name)
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            return None
    
    def create_index(
        self,
        name: str,
        documents: List[Document],
        embeddings: Embeddings,
        description: str = "",
        store_type: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> VectorStore:
        """
        创建新索引
        
        Args:
            name: 索引名称
            documents: 文档列表
            embeddings: Embedding 模型
            description: 索引描述
            store_type: 向量库类型
            overwrite: 是否覆盖已存在的索引
            **kwargs: 其他参数
            
        Returns:
            创建的 VectorStore 实例
            
        Raises:
            ValueError: 如果索引已存在且 overwrite=False
            
        Example:
            >>> manager.create_index(
            ...     name="my_docs",
            ...     documents=chunks,
            ...     embeddings=embeddings,
            ...     description="我的文档集合"
            ... )
        """
        index_path = self._get_index_path(name)
        
        # 检查索引是否已存在
        if index_path.exists() and not overwrite:
            raise ValueError(
                f"索引已存在: {name}。使用 overwrite=True 来覆盖。"
            )
        
        logger.info(f"   创建索引: {name}")
        logger.info(f"   文档数量: {len(documents)}")
        logger.info(f"   描述: {description}")
        
        try:
            # 创建向量库
            vector_store = create_vector_store(
                documents=documents,
                embeddings=embeddings,
                store_type=store_type,
                **kwargs,
            )
            
            # 保存向量库
            save_vector_store(vector_store, str(index_path), embeddings)
            
            # 保存元数据
            metadata = {
                "name": name,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "num_documents": len(documents),
                "store_type": store_type or settings.vector_store_type,
                "embedding_model": settings.embedding_model,
            }
            self._save_metadata(name, metadata)
            
            logger.info(f"索引创建成功: {name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            # 清理失败的索引
            if index_path.exists():
                delete_vector_store(str(index_path))
            raise
    
    def load_index(
        self,
        name: str,
        embeddings: Embeddings,
        **kwargs,
    ) -> VectorStore:
        """
        加载索引
        
        Args:
            name: 索引名称
            embeddings: Embedding 模型
            **kwargs: 其他参数
            
        Returns:
            VectorStore 实例
            
        Raises:
            FileNotFoundError: 如果索引不存在
            
        Example:
            >>> vector_store = manager.load_index("my_docs", embeddings)
        """
        index_path = self._get_index_path(name)
        if not index_path.exists():
            raise FileNotFoundError(f"索引不存在: {name}")
        
        logger.info(f"加载索引: {name}")
        
        try:
            # 加载元数据
            metadata = self._load_metadata(name)
            logger.info(f"元数据为 metadata: {metadata}")
            if metadata:
                logger.info(f"   描述: {metadata.get('description', 'N/A')}")
                logger.info(f"   文档数: {metadata.get('num_documents', 'N/A')}")
            
            # 加载向量库
            vector_store = load_vector_store(
                load_path=str(index_path),
                embeddings=embeddings,
                **kwargs,
            )
            
            logger.info(f"索引加载成功: {name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            raise
    
    def update_index(
        self,
        name: str,
        documents: List[Document],
        embeddings: Embeddings,
        **kwargs,
    ) -> VectorStore:
        """
        更新索引（添加新文档）
        
        Args:
            name: 索引名称
            documents: 要添加的文档列表
            embeddings: Embedding 模型
            **kwargs: 其他参数
            
        Returns:
            更新后的 VectorStore 实例
            
        Example:
            >>> # 加载新文档
            >>> new_docs = load_document("new_doc.pdf")
            >>> chunks = split_documents(new_docs)
            >>> 
            >>> # 更新索引
            >>> manager.update_index("my_docs", chunks, embeddings)
        """
        logger.info(f"   更新索引: {name}")
        logger.info(f"   新增文档: {len(documents)}")
        
        try:
            # 加载现有索引
            vector_store = self.load_index(name, embeddings, **kwargs)
            
            # 添加新文档
            add_documents_to_vector_store(vector_store, documents)
            
            # 保存更新后的向量库
            index_path = self._get_index_path(name)
            save_vector_store(vector_store, str(index_path), embeddings)
            
            # 更新元数据
            metadata = self._load_metadata(name) or {}
            metadata["updated_at"] = datetime.now().isoformat()
            metadata["num_documents"] = metadata.get("num_documents", 0) + len(documents)
            self._save_metadata(name, metadata)
            
            logger.info(f"索引更新成功: {name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"更新索引失败: {e}")
            raise
    
    def delete_index(self, name: str) -> None:
        """
        删除索引
        
        Args:
            name: 索引名称
            
        Example:
            >>> manager.delete_index("old_index")
        """
        index_path = self._get_index_path(name)
        
        if not index_path.exists():
            logger.warning(f"索引不存在: {name}")
            return
        
        logger.info(f"删除索引: {name}")
        
        try:
            delete_vector_store(str(index_path))
            logger.info(f"索引删除成功: {name}")
            
        except Exception as e:
            logger.error(f"删除索引失败: {e}")
            raise
    
    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        列出所有索引
        
        Returns:
            索引信息列表
            
        Example:
            >>> indexes = manager.list_indexes()
            >>> for idx in indexes:
            ...     print(f"{idx['name']}: {idx['description']}")
        """
        logger.info("列出所有索引")
        
        indexes = []
        
        # 遍历索引目录
        if not self.base_path.exists():
            return indexes
        
        for index_dir in self.base_path.iterdir():
            if not index_dir.is_dir():
                continue
            
            name = index_dir.name
            metadata = self._load_metadata(name)
            
            if metadata:
                indexes.append(metadata)
            else:
                # 如果没有元数据，创建基本信息
                indexes.append({
                    "name": name,
                    "description": "N/A",
                    "created_at": "N/A",
                    "updated_at": "N/A",
                    "num_documents": "N/A",
                })
        
        logger.info(f"   找到 {len(indexes)} 个索引")
        return indexes
    
    def get_index_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取索引详细信息
        
        Args:
            name: 索引名称
            
        Returns:
            索引信息字典，如果不存在返回 None
            
        Example:
            >>> info = manager.get_index_info("my_docs")
            >>> print(info)
        """
        index_path = self._get_index_path(name)
        
        if not index_path.exists():
            logger.warning(f"索引不存在: {name}")
            return None
        
        # 加载元数据
        metadata = self._load_metadata(name)
        
        if not metadata:
            metadata = {
                "name": name,
                "description": "N/A",
            }
        
        # 添加路径信息
        metadata["path"] = str(index_path)
        
        # 计算索引大小
        try:
            total_size = sum(
                f.stat().st_size
                for f in index_path.rglob("*")
                if f.is_file()
            )
            metadata["size_bytes"] = total_size
            metadata["size_mb"] = total_size / (1024 * 1024)
        except Exception as e:
            logger.warning(f"计算索引大小失败: {e}")
        
        return metadata
    
    def index_exists(self, name: str) -> bool:
        """
        检查索引是否存在
        
        Args:
            name: 索引名称
            
        Returns:
            是否存在
            
        Example:
            >>> if manager.index_exists("my_docs"):
            ...     print("索引存在")
        """
        return self._get_index_path(name).exists()


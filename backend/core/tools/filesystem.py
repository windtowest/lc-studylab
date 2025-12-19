"""
文件系统工具模块

为 DeepAgent 提供虚拟文件系统功能，用于：
1. 存储研究计划和中间结果
2. 管理研究档案
3. 控制上下文窗口大小

这是 Stage 4 的核心工具之一，支持 DeepAgent 进行长时间运行的研究任务。

技术要点：
- 使用 LangChain 的 @tool 装饰器创建工具
- 提供完整的文件 CRUD 操作
- 支持目录管理和文件搜索
- 自动创建必要的目录结构

参考：
- https://docs.langchain.com/oss/python/langchain/tools
- https://docs.langchain.com/oss/python/deepagents/harness
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from langchain_core.tools import tool
from sqlalchemy import True_

from config import settings, get_logger

logger = get_logger(__name__)


class ResearchFileSystem:
    """
    研究文件系统类
    
    为 DeepAgent 提供虚拟文件系统，用于管理研究过程中的所有文件。
    
    特性：
    - 独立的工作空间（基于 thread_id）
    - 自动创建目录结构
    - 文件版本管理
    - 元数据跟踪
    
    Attributes:
        base_path: 文件系统根目录
        thread_id: 当前研究任务的线程 ID
        workspace_path: 当前工作空间路径
    
    Example:
        >>> fs = ResearchFileSystem(thread_id="research_123")
        >>> fs.write_file("plan.md", "# 研究计划\\n...")
        >>> content = fs.read_file("plan.md")
        >>> files = fs.list_files()
    """
    
    def __init__(
        self,
        thread_id: str,
        base_path: Optional[str] = None,
    ):
        """
        初始化研究文件系统
        
        Args:
            thread_id: 研究任务的唯一标识符
            base_path: 文件系统根目录，默认使用配置中的路径
        """
        self.thread_id = thread_id
        
        # 使用配置的基础路径或默认路径
        if base_path is None:
            base_path = os.path.join(settings.DATA_DIR, "research")
        
        self.base_path = Path(base_path)
        
        # 每个研究任务有独立的工作空间
        self.workspace_path = self.base_path / thread_id
        
        # 创建必要的目录结构
        self._init_workspace()
        
        logger.info(f"初始化研究文件系统: {self.workspace_path}")
    
    def _init_workspace(self) -> None:
        """
        初始化工作空间目录结构
        
        创建以下目录：
        - plans/: 存储研究计划
        - notes/: 存储研究笔记
        - reports/: 存储最终报告
        - temp/: 存储临时文件
        """
        directories = [
            self.workspace_path,
            self.workspace_path / "plans",
            self.workspace_path / "notes",
            self.workspace_path / "reports",
            self.workspace_path / "temp",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"   工作空间目录已创建: {self.workspace_path}")
    
    def write_file(
        self,
        filename: str,
        content: str,
        subdirectory: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        写入文件（同步操作，确保写入完成）
        
        Args:
            filename: 文件名
            content: 文件内容
            subdirectory: 子目录（plans/notes/reports/temp）
            metadata: 文件元数据（可选）
            
        Returns:
            文件的完整路径
            
        Example:
            >>> fs.write_file("plan.md", "# 研究计划", subdirectory="plans")
            '/path/to/research/thread_123/plans/plan.md'
        """
        # 确定文件路径
        if subdirectory:
            file_path = self.workspace_path / subdirectory / filename
        else:
            file_path = self.workspace_path / filename
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                # 强制刷新到磁盘
                f.flush()
                import os
                os.fsync(f.fileno())
            
            # 写入元数据（如果提供）
            if metadata:
                metadata_path = file_path.with_suffix(file_path.suffix + '.meta.json')
                metadata['created_at'] = datetime.now().isoformat()
                metadata['filename'] = filename
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
            
            logger.info(f"文件已写入: {file_path.relative_to(self.base_path)}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"写入文件失败: {e}")
            raise
    
    def read_file(
        self,
        filename: str,
        subdirectory: Optional[str] = None,
    ) -> str:
        """
        读取文件内容
        
        Args:
            filename: 文件名
            subdirectory: 子目录
            
        Returns:
            文件内容
            
        Raises:
            FileNotFoundError: 如果文件不存在
            
        Example:
            >>> content = fs.read_file("plan.md", subdirectory="plans")
        """
        # 确定文件路径
        if subdirectory:
            file_path = self.workspace_path / subdirectory / filename
        else:
            file_path = self.workspace_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.debug(f"读取文件: {file_path.relative_to(self.base_path)}")
            return content
            
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            raise
    
    def list_files(
        self,
        subdirectory: Optional[str] = None,
        pattern: str = "*",
    ) -> List[str]:
        """
        列出文件
        
        Args:
            subdirectory: 子目录，None 表示列出所有文件
            pattern: 文件名模式（支持通配符）
            
        Returns:
            文件名列表
            
        Example:
            >>> files = fs.list_files(subdirectory="notes")
            >>> md_files = fs.list_files(pattern="*.md")
        """
        if subdirectory:
            search_path = self.workspace_path / subdirectory
        else:
            search_path = self.workspace_path
        
        if not search_path.exists():
            return []
        
        # 查找文件（排除元数据文件）
        files = []
        for file_path in search_path.glob(pattern):
            if file_path.is_file() and not file_path.name.endswith('.meta.json'):
                # 返回相对于工作空间的路径
                relative_path = file_path.relative_to(self.workspace_path)
                files.append(str(relative_path))
        
        logger.debug(f"列出文件: {len(files)} 个文件")
        return sorted(files)
    
    def delete_file(
        self,
        filename: str,
        subdirectory: Optional[str] = None,
    ) -> bool:
        """
        删除文件
        
        Args:
            filename: 文件名
            subdirectory: 子目录
            
        Returns:
            是否删除成功
            
        Example:
            >>> fs.delete_file("temp.txt", subdirectory="temp")
        """
        if subdirectory:
            file_path = self.workspace_path / subdirectory / filename
        else:
            file_path = self.workspace_path / filename
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {filename}")
            return False
        
        try:
            # 删除文件
            file_path.unlink()
            
            # 删除元数据文件（如果存在）
            metadata_path = file_path.with_suffix(file_path.suffix + '.meta.json')
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"文件已删除: {file_path.relative_to(self.base_path)}")
            return True
            
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False
    
    def file_exists(
        self,
        filename: str,
        subdirectory: Optional[str] = None,
    ) -> bool:
        """
        检查文件是否存在
        
        Args:
            filename: 文件名
            subdirectory: 子目录
            
        Returns:
            文件是否存在
        """
        if subdirectory:
            file_path = self.workspace_path / subdirectory / filename
        else:
            file_path = self.workspace_path / filename
        
        return file_path.exists()
    
    def get_file_info(
        self,
        filename: str,
        subdirectory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            filename: 文件名
            subdirectory: 子目录
            
        Returns:
            文件信息字典
            
        Example:
            >>> info = fs.get_file_info("plan.md", subdirectory="plans")
            >>> print(info['size'], info['modified_at'])
        """
        if subdirectory:
            file_path = self.workspace_path / subdirectory / filename
        else:
            file_path = self.workspace_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {filename}")
        
        stat = file_path.stat()
        
        info = {
            "filename": filename,
            "path": str(file_path),
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
        
        # 读取元数据（如果存在）
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta.json')
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    info['metadata'] = metadata
            except Exception as e:
                logger.warning(f"读取元数据失败: {e}")
        
        return info
    
    def search_files(
        self,
        keyword: str,
        subdirectory: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索包含关键词的文件
        
        Args:
            keyword: 搜索关键词
            subdirectory: 子目录
            
        Returns:
            匹配的文件列表，每个元素包含文件名和匹配的行
            
        Example:
            >>> results = fs.search_files("机器学习")
            >>> for result in results:
            ...     print(result['filename'], result['matches'])
        """
        if subdirectory:
            search_path = self.workspace_path / subdirectory
        else:
            search_path = self.workspace_path
        
        if not search_path.exists():
            return []
        
        results = []
        
        # 递归搜索所有文本文件
        for file_path in search_path.rglob("*"):
            if not file_path.is_file() or file_path.name.endswith('.meta.json'):
                continue
            
            # 只搜索文本文件
            if file_path.suffix not in ['.md', '.txt', '.json', '.py', '.yaml', '.yml']:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找匹配的行
                matches = []
                for i, line in enumerate(content.split('\n'), 1):
                    if keyword.lower() in line.lower():
                        matches.append({
                            "line_number": i,
                            "line": line.strip()
                        })
                
                if matches:
                    relative_path = file_path.relative_to(self.workspace_path)
                    results.append({
                        "filename": str(relative_path),
                        "matches": matches,
                        "match_count": len(matches)
                    })
                    
            except Exception as e:
                logger.warning(f"搜索文件失败 {file_path}: {e}")
                continue
        
        logger.debug(f"搜索完成: 找到 {len(results)} 个匹配文件")
        return results
    
    def cleanup(self) -> None:
        """
        清理工作空间
        
        删除所有临时文件，保留重要文件（plans, reports）
        """
        temp_path = self.workspace_path / "temp"
        if temp_path.exists():
            for file_path in temp_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
        
        logger.info(f"临时文件已清理: {self.workspace_path}")


# ==================== LangChain 工具封装 ====================

# 全局文件系统实例缓存
_filesystem_cache: Dict[str, ResearchFileSystem] = {}


def get_filesystem(thread_id: str) -> ResearchFileSystem:
    """
    获取或创建文件系统实例
    
    使用缓存避免重复创建
    
    Args:
        thread_id: 研究任务 ID
        
    Returns:
        文件系统实例
    """
    if thread_id not in _filesystem_cache:
        _filesystem_cache[thread_id] = ResearchFileSystem(thread_id)
    
    return _filesystem_cache[thread_id]


@tool(description="""写入研究文件
    
    将研究过程中的内容保存到文件系统。
    适用于保存研究计划、笔记、中间结果等。""")
def write_research_file(
    filename: str,
    content: str,
    thread_id: str,
    subdirectory: str = "notes",
) -> str:
    """
    写入研究文件
    
    将研究过程中的内容保存到文件系统。
    适用于保存研究计划、笔记、中间结果等。
    
    Args:
        filename: 文件名（包含扩展名，如 "plan.md"）
        content: 文件内容
        thread_id: 研究任务 ID
        subdirectory: 子目录，可选值：plans, notes, reports, temp
        
    Returns:
        成功消息和文件路径

    Example:
        >>> write_research_file(
        ...     filename="research_plan.md",
        ...     content="# 研究计划\\n\\n1. 搜索相关资料\\n2. 分析数据",
        ...     thread_id="research_123",
        ...     subdirectory="plans"
        ... )
        '文件已保存: research_plan.md'
    """
    try:
        fs = get_filesystem(thread_id)
        file_path = fs.write_file(filename, content, subdirectory)
        return f"文件已保存: {filename} (路径: {file_path})"
    except Exception as e:
        return f"保存文件失败: {str(e)}"


@tool(description="""读取研究文件
    
    从文件系统读取之前保存的文件内容。""")
def read_research_file(
    filename: str,
    thread_id: str,
    subdirectory: str = "notes",
) -> str:
    """
    读取研究文件
    
    从文件系统读取之前保存的文件内容。
    
    Args:
        filename: 文件名
        thread_id: 研究任务 ID
        subdirectory: 子目录
        
    Returns:
        文件内容
        
    Example:
        >>> read_research_file(
        ...     filename="research_plan.md",
        ...     thread_id="research_123",
        ...     subdirectory="plans"
        ... )
        '# 研究计划\\n\\n1. 搜索相关资料\\n2. 分析数据'
    """
    try:
        fs = get_filesystem(thread_id)
        content = fs.read_file(filename, subdirectory)
        return content
    except FileNotFoundError:
        return f"文件不存在: {filename}"
    except Exception as e:
        return f"读取文件失败: {str(e)}"


@tool(description="""列出研究文件
    
    列出工作空间中的所有文件。""")
def list_research_files(
    thread_id: str,
    subdirectory: Optional[str] = None,
) -> str:
    """
    列出研究文件
    
    列出工作空间中的所有文件。
    
    Args:
        thread_id: 研究任务 ID
        subdirectory: 子目录（可选），不指定则列出所有文件
        
    Returns:
        文件列表（格式化的字符串）
        
    Example:
        >>> list_research_files(thread_id="research_123", subdirectory="notes")
        '找到 3 个文件:\\n- notes/note1.md\\n- notes/note2.md\\n- notes/summary.md'
    """
    try:
        fs = get_filesystem(thread_id)
        files = fs.list_files(subdirectory)
        
        if not files:
            return f"没有找到文件（目录: {subdirectory or '全部'}）"
        
        file_list = "\n".join([f"- {f}" for f in files])
        return f"找到 {len(files)} 个文件:\n{file_list}"
        
    except Exception as e:
        return f"列出文件失败: {str(e)}"


@tool(description="""搜索研究文件
    
    在文件内容中搜索关键词。""")
def search_research_files(
    keyword: str,
    thread_id: str,
    subdirectory: Optional[str] = None,
) -> str:
    """
    搜索研究文件
    
    在文件内容中搜索关键词。
    
    Args:
        keyword: 搜索关键词
        thread_id: 研究任务 ID
        subdirectory: 子目录（可选）
        
    Returns:
        搜索结果（格式化的字符串）
        
    Example:
        >>> search_research_files(
        ...     keyword="机器学习",
        ...     thread_id="research_123"
        ... )
        '找到 2 个匹配文件:\\n\\n文件: notes/ml_basics.md\\n- 第 5 行: 机器学习是...'
    """
    try:
        fs = get_filesystem(thread_id)
        results = fs.search_files(keyword, subdirectory)
        
        if not results:
            return f"没有找到包含 '{keyword}' 的文件"
        
        # 格式化结果
        output = [f"找到 {len(results)} 个匹配文件:\n"]
        
        for result in results:
            output.append(f"\n文件: {result['filename']}")
            output.append(f"匹配次数: {result['match_count']}")
            
            # 显示前 3 个匹配
            for match in result['matches'][:3]:
                output.append(f"- 第 {match['line_number']} 行: {match['line']}")
            
            if result['match_count'] > 3:
                output.append(f"  ... 还有 {result['match_count'] - 3} 个匹配")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"搜索失败: {str(e)}"


# 导出所有文件系统工具
FILESYSTEM_TOOLS = [
    write_research_file,
    read_research_file,
    list_research_files,
    search_research_files,
]


# 导出工具名称列表（用于文档）
FILESYSTEM_TOOL_NAMES = [tool.name for tool in FILESYSTEM_TOOLS]


logger.info(f"文件系统工具已加载: {', '.join(FILESYSTEM_TOOL_NAMES)}")


"""
SubAgents 子智能体模块

定义专门的子智能体，每个负责特定的研究任务：
1. WebResearcher: 网络搜索和信息整理
2. DocAnalyst: 文档分析和知识提取
3. ReportWriter: 报告撰写和内容组织

这些子智能体由 DeepAgent 协调，共同完成复杂的研究任务。

技术要点：
- 基于 LangChain 1.0.3 的 create_agent API
- 每个子智能体有专门的系统提示词
- 配备特定的工具集
- 支持流式输出

参考：
- https://docs.langchain.com/oss/python/deepagents/subagents
- https://docs.langchain.com/oss/python/langchain/agents
"""

from typing import Optional, List, Sequence
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

from config import settings, get_logger
from core.models import get_model_string
from core.my_llm import my_deepseek
from core.prompts import WRITER_GUIDELINES
from core.tools import zhipu_web_search
from core.tools.web_search import create_tavily_search_tool
from core.tools.filesystem import FILESYSTEM_TOOLS

logger = get_logger(__name__)


# ==================== 系统提示词 ====================

WEB_RESEARCHER_PROMPT = (
    "你是一个专业的网络研究员，负责从互联网搜索与整理信息。"
    "使用搜索工具查找并评估来源，提取关键数据，"
    "按来源类型自适配呈现（官方文档、论文、标准、新闻、博客），"
    "采用要点与段落混合的方式记录，使用内联引用并在结尾列出参考来源，"
    "将研究笔记保存到文件系统。"
)


DOC_ANALYST_PROMPT = (
    "你是一个专业的文档分析师，负责在知识库中检索并提炼信息。"
    "根据研究问题执行多次检索与评估，直接引用关键段落，"
    "整理为要点与段落混合的分析笔记，列出文档来源与位置，"
    "并保存到文件系统。"
)


REPORT_WRITER_PROMPT = (
    "你是一个专业的研究报告撰写者，负责整合研究材料并产出高质量报告。"
    "列出并阅读研究笔记与分析，识别关键发现与证据，"
    "根据主题与信息密度选择合适结构，提供真实示例或代码片段（技术主题），"
    "使用内联引用与参考列表，最终保存报告到文件系统。"
)


# ==================== SubAgent 创建函数 ====================

def create_web_researcher(
    model: Optional[str] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    **kwargs,
):
    """
    创建网络研究员子智能体

    专门负责网络搜索和信息整理。

    Args:
        model: 模型字符串（如 "openai:gpt-4o"）
        tools: 工具列表，默认包含搜索和文件系统工具
        **kwargs: 其他传递给 create_agent 的参数

    Returns:
        WebResearcher Agent

    Example:
        >>> researcher = create_web_researcher()
        >>> result = researcher.invoke({
        ...     "messages": [{"role": "user", "content": "搜索 LangChain 1.0 的新特性"}]
        ... })
    """
    logger.info("创建 WebResearcher 子智能体")

    # 使用默认模型
    if model is None:
        # model = get_model_string()
        model = my_deepseek
    # 配置工具：搜索 + 文件系统
    if tools is None:
        agent_tools = []

        # 添加搜索工具
        try:
            if settings.zhipu_api_key:
                # search_tool = create_tavily_search_tool()
                # agent_tools.append(search_tool)
                agent_tools.append(zhipu_web_search)
                logger.debug("   添加 智谱ai 搜索工具")
        except Exception as e:
            logger.warning(f"无法添加搜索工具: {e}")

        # 添加文件系统工具
        agent_tools.extend(FILESYSTEM_TOOLS)
        logger.debug(f"   添加文件系统工具: {len(FILESYSTEM_TOOLS)} 个")

        tools = agent_tools

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=f"{WEB_RESEARCHER_PROMPT}\n\n{WRITER_GUIDELINES}",
        **kwargs,
    )

    logger.info("WebResearcher 创建成功")
    return agent


def create_doc_analyst(
    model: Optional[str] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    retriever_tool: Optional[BaseTool] = None,
    **kwargs,
):
    """
    创建文档分析师子智能体

    专门负责文档分析和知识提取。

    Args:
        model: 模型字符串
        tools: 工具列表，默认包含 RAG 检索和文件系统工具
        retriever_tool: RAG 检索工具（可选）
        **kwargs: 其他参数

    Returns:
        DocAnalyst Agent

    Example:
        >>> from rag import create_retriever_tool, get_embeddings, load_vector_store
        >>> 
        >>> # 创建检索工具
        >>> embeddings = get_embeddings()
        >>> vector_store = load_vector_store("data/indexes/test_index", embeddings)
        >>> retriever = vector_store.as_retriever()
        >>> retriever_tool = create_retriever_tool(retriever)
        >>> 
        >>> # 创建文档分析师
        >>> analyst = create_doc_analyst(retriever_tool=retriever_tool)
    """
    logger.info("创建 DocAnalyst 子智能体")
    
    # 使用默认模型
    if model is None:
        # model = get_model_string()
        model = my_deepseek
    # 配置工具：RAG 检索 + 文件系统
    if tools is None:
        agent_tools = []
        
        # 添加 RAG 检索工具
        if retriever_tool is not None:
            agent_tools.append(retriever_tool)
            logger.debug("   添加 RAG 检索工具")
        else:
            logger.warning("未提供 retriever_tool，DocAnalyst 将无法检索文档")
        
        # 添加文件系统工具
        agent_tools.extend(FILESYSTEM_TOOLS)
        logger.debug(f"   添加文件系统工具: {len(FILESYSTEM_TOOLS)} 个")
        
        tools = agent_tools
    
    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=f"{DOC_ANALYST_PROMPT}\n\n{WRITER_GUIDELINES}",
        **kwargs,
    )
    
    logger.info("DocAnalyst 创建成功")
    return agent


def create_report_writer(
    model: Optional[str] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    **kwargs,
):
    """
    创建报告撰写者子智能体
    
    专门负责报告撰写和内容组织。
    
    Args:
        model: 模型字符串
        tools: 工具列表，默认只包含文件系统工具
        **kwargs: 其他参数
        
    Returns:
        ReportWriter Agent
        
    Example:
        >>> writer = create_report_writer()
        >>> result = writer.invoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": "根据研究笔记撰写最终报告，thread_id: research_123"
        ...     }]
        ... })
    """
    logger.info("创建 ReportWriter 子智能体")
    
    # 使用默认模型（可以使用更强大的模型）
    if model is None:
        # ReportWriter 使用主模型，确保报告质量
        # model = f"openai:{settings.openai_model}"
        model = my_deepseek
    # 配置工具：只需要文件系统工具
    if tools is None:
        tools = FILESYSTEM_TOOLS
        logger.debug(f"   添加文件系统工具: {len(FILESYSTEM_TOOLS)} 个")
    
    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=f"{REPORT_WRITER_PROMPT}\n\n{WRITER_GUIDELINES}",
        **kwargs,
    )
    
    logger.info("ReportWriter 创建成功")
    return agent


# ==================== SubAgent 辅助函数 ====================

def get_subagent_info() -> dict:
    """
    获取所有子智能体的信息
    
    Returns:
        包含子智能体信息的字典
    """
    return {
        "web_researcher": {
            "name": "WebResearcher",
            "description": "网络搜索和信息整理专家",
            "capabilities": [
                "网络搜索",
                "信息筛选",
                "来源评估",
                "笔记整理"
            ],
            "tools": ["zhipu_web_search", "write_research_file", "read_research_file"]
        },
        "doc_analyst": {
            "name": "DocAnalyst",
            "description": "文档分析和知识提取专家",
            "capabilities": [
                "文档检索",
                "内容分析",
                "信息提炼",
                "关联识别"
            ],
            "tools": ["knowledge_base", "write_research_file", "read_research_file"]
        },
        "report_writer": {
            "name": "ReportWriter",
            "description": "研究报告撰写专家",
            "capabilities": [
                "内容组织",
                "报告撰写",
                "引用管理",
                "质量把控"
            ],
            "tools": ["write_research_file", "read_research_file", "list_research_files"]
        }
    }


logger.info("SubAgents 模块已加载")

"""
DeepAgent 深度研究智能体

这是 Stage 4 的核心模块，实现基于 LangChain 1.0.3 的深度研究智能体。

核心功能：
1. 自动规划研究任务
2. 协调多个子智能体
3. 管理研究流程
4. 生成最终报告

技术架构：
- 基于 LangGraph 构建工作流
- 使用 StateGraph 管理状态
- 集成 SubAgents 处理子任务
- 使用文件系统存储中间结果

参考：
- https://docs.langchain.com/oss/python/deepagents/quickstart
- https://docs.langchain.com/oss/python/langgraph/quickstart
"""

from typing import Optional, List, Dict, Any, Sequence, TypedDict, Annotated
import json
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from core.my_llm import my_deepseek
from config import settings, get_logger
from core.models import get_chat_model
from core.tools.filesystem import get_filesystem, ResearchFileSystem
from core.prompts import WRITER_GUIDELINES
from core.guardrails.output_validators import OutputValidator
from deep_research.subagents import (
    create_web_researcher,
    create_doc_analyst,
    create_report_writer,
)

logger = get_logger(__name__)


# ==================== State 定义 ====================

class ResearchState(TypedDict):
    """
    研究状态定义
    
    包含研究过程中的所有状态信息。
    
    Attributes:
        messages: 消息历史
        query: 研究问题
        thread_id: 研究任务 ID
        plan: 研究计划
        web_research_done: 网络研究是否完成
        doc_analysis_done: 文档分析是否完成
        report_done: 报告是否完成
        current_step: 当前步骤
        error: 错误信息
        final_report: 最终报告
    """
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    thread_id: str
    plan: Optional[Dict[str, Any]]
    web_research_done: bool
    doc_analysis_done: bool
    report_done: bool
    current_step: str
    error: Optional[str]
    final_report: Optional[str]


# ==================== DeepAgent 类 ====================

class DeepResearchAgent:
    """
    深度研究智能体
    
    协调多个子智能体完成复杂的研究任务。
    
    工作流程：
    1. Planner: 生成研究计划
    2. WebResearcher: 搜索网络信息
    3. DocAnalyst: 分析文档（可选）
    4. ReportWriter: 撰写最终报告
    
    Attributes:
        thread_id: 研究任务 ID
        filesystem: 文件系统实例
        web_researcher: 网络研究员
        doc_analyst: 文档分析师（可选）
        report_writer: 报告撰写者
        graph: LangGraph 工作流
        
    Example:
        >>> agent = DeepResearchAgent(
        ...     thread_id="research_123",
        ...     enable_doc_analysis=True
        ... )
        >>> result = agent.research("分析 LangChain 1.0 的新特性")
        >>> print(result["final_report"])
    """
    
    def __init__(
        self,
        thread_id: str,
        enable_web_search: bool = True,
        enable_doc_analysis: bool = False,
        retriever_tool: Optional[BaseTool] = None,
        checkpointer: Optional[Any] = None,
        **kwargs,
    ):
        """
        初始化深度研究智能体
        
        Args:
            thread_id: 研究任务的唯一标识符
            enable_web_search: 是否启用网络搜索
            enable_doc_analysis: 是否启用文档分析
            retriever_tool: RAG 检索工具（如果启用文档分析）
            checkpointer: 状态检查点（可选）
            **kwargs: 其他参数
        """
        self.thread_id = thread_id
        self.enable_web_search = enable_web_search
        self.enable_doc_analysis = enable_doc_analysis
        
        logger.info(f"初始化 DeepResearchAgent: {thread_id}")
        logger.info(f"   网络搜索: {enable_web_search}")
        logger.info(f"   文档分析: {enable_doc_analysis}")
        
        # 初始化文件系统
        self.filesystem = get_filesystem(thread_id)
        
        # 创建子智能体
        self._init_subagents(retriever_tool)
        
        # 创建工作流
        self.graph = self._build_graph(checkpointer)
        
        logger.info("DeepResearchAgent 初始化完成")
    
    def _init_subagents(self, retriever_tool: Optional[BaseTool] = None) -> None:
        """
        初始化子智能体
        
        Args:
            retriever_tool: RAG 检索工具
        """
        logger.info("初始化子智能体...")
        
        # WebResearcher
        if self.enable_web_search:
            self.web_researcher = create_web_researcher()
            logger.debug("   ✓ WebResearcher")
        else:
            self.web_researcher = None
            logger.debug("   ✗ WebResearcher (禁用)")
        
        # DocAnalyst
        if self.enable_doc_analysis:
            if retriever_tool is None:
                logger.warning("启用了文档分析但未提供 retriever_tool")
            self.doc_analyst = create_doc_analyst(retriever_tool=retriever_tool)
            logger.debug("   ✓ DocAnalyst")
        else:
            self.doc_analyst = None
            logger.debug("   ✗ DocAnalyst (禁用)")
        
        # ReportWriter（总是需要）
        self.report_writer = create_report_writer()
        logger.debug("   ✓ ReportWriter")
    
    def _build_graph(self, checkpointer: Optional[Any] = None) -> Any:
        """
        构建 LangGraph 工作流
        
        Args:
            checkpointer: 状态检查点
            
        Returns:
            编译后的 StateGraph
        """
        logger.info("构建研究工作流...")
        
        # 创建 StateGraph
        workflow = StateGraph(ResearchState)
        
        # 添加节点
        workflow.add_node("planner", self._planner_node)
        
        if self.enable_web_search:
            workflow.add_node("web_research", self._web_research_node)
        
        if self.enable_doc_analysis:
            workflow.add_node("doc_analysis", self._doc_analysis_node)
        
        workflow.add_node("report_writing", self._report_writing_node)
        
        # 设置入口点
        workflow.set_entry_point("planner")
        
        # 添加边
        if self.enable_web_search:
            workflow.add_edge("planner", "web_research")
            
            if self.enable_doc_analysis:
                workflow.add_edge("web_research", "doc_analysis")
                workflow.add_edge("doc_analysis", "report_writing")
            else:
                workflow.add_edge("web_research", "report_writing")
        else:
            if self.enable_doc_analysis:
                workflow.add_edge("planner", "doc_analysis")
                workflow.add_edge("doc_analysis", "report_writing")
            else:
                workflow.add_edge("planner", "report_writing")
        
        workflow.add_edge("report_writing", END)
        
        # 编译
        if checkpointer is None:
            # 使用内存检查点
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        
        graph = workflow.compile(checkpointer=checkpointer)
        
        logger.info("工作流构建完成")
        return graph
    
    # ==================== 节点函数 ====================
    
    def _planner_node(self, state: ResearchState) -> ResearchState:
        """
        规划节点：生成研究计划
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行规划节点...")
        
        query = state["query"]
        thread_id = state["thread_id"]
        
        # 生成研究计划
        plan_prompt = f"""请为以下研究问题制定详细的研究计划：

研究问题：{query}

可用资源：
- 网络搜索：{"是" if self.enable_web_search else "否"}
- 文档分析：{"是" if self.enable_doc_analysis else "否"}

请输出 JSON 格式的研究计划：
{{
    "research_goal": "研究目标",
    "key_questions": ["问题1", "问题2", ...],
    "search_keywords": ["关键词1", "关键词2", ...],
    "expected_outcomes": ["预期成果1", "预期成果2", ...]
}}
"""
        
        # 使用 LLM 生成计划
        try:
            # model = get_chat_model()
            model = my_deepseek
            response = model.invoke([HumanMessage(content=plan_prompt)])
            
            # 解析 JSON
            plan_text = response.content
            
            # 尝试提取 JSON
            import re
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                # 如果没有 JSON，创建默认计划
                plan = {
                    "research_goal": query,
                    "key_questions": [query],
                    "search_keywords": query.split(),
                    "expected_outcomes": ["完整的研究报告"]
                }
            
            # 保存计划
            plan_content = f"""# 研究计划

## 研究目标
{plan.get('research_goal', query)}

## 关键问题
{chr(10).join([f"- {q}" for q in plan.get('key_questions', [query])])}

## 搜索关键词
{', '.join(plan.get('search_keywords', []))}

## 预期成果
{chr(10).join([f"- {o}" for o in plan.get('expected_outcomes', [])])}

---
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            self.filesystem.write_file(
                "research_plan.md",
                plan_content,
                subdirectory="plans"
            )
            
            logger.info("研究计划已生成")
            
            # 更新状态
            state["plan"] = plan
            state["current_step"] = "planner"
            state["messages"] = [AIMessage(content=f"研究计划已生成：{plan.get('research_goal')}")]
            
        except Exception as e:
            logger.error(f"生成计划失败: {e}")
            state["error"] = str(e)
            state["plan"] = {"research_goal": query}
        
        return state
    
    def _web_research_node(self, state: ResearchState) -> ResearchState:
        """
        网络研究节点：搜索和整理网络信息
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行网络研究节点...")
        
        query = state["query"]
        thread_id = state["thread_id"]
        plan = state.get("plan", {})
        
        # 构建研究指令
        research_instruction = f"""请对以下问题进行深入的网络研究：

研究问题：{query}

研究计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

任务要求：
1. 使用搜索工具查找相关信息
2. 评估信息的可信度和相关性
3. 提取关键信息和数据
4. 整理为要点与段落混合的研究笔记，按来源类型自适配呈现（官方文档、论文、标准、新闻、博客）
5. 使用内联引用并在结尾列出参考来源
6. 使用 write_research_file 保存到 notes/web_research.md

写作准则：
{WRITER_GUIDELINES}

thread_id: {thread_id}
"""
        
        try:
            # 调用 WebResearcher
            result = self.web_researcher.invoke({
                "messages": [HumanMessage(content=research_instruction)]
            })
            
            # 验证笔记是否已保存（文件系统已保证同步写入）
            notes_saved = False
            try:
                notes = self.filesystem.read_file("web_research.md", subdirectory="notes")
                notes_saved = True
                logger.info("网络研究笔记已保存")
            except Exception:
                logger.info("   未在文件系统中找到笔记，尝试从 Agent 输出提取...")
            
            # 如果笔记没有保存，尝试从 Agent 输出中提取
            if not notes_saved:
                if isinstance(result, dict) and "messages" in result:
                    messages = result["messages"]
                    
                    # 收集所有 AI 消息内容
                    research_content = []
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.content:
                            # 跳过工具调用的消息
                            if not msg.content.startswith("找到") and not msg.content.startswith("搜索"):
                                research_content.append(msg.content)
                    
                    if research_content:
                        # 合并内容并保存
                        combined_content = "\n\n".join(research_content)
                        
                        # 如果内容不是 Markdown 格式，添加标题
                        if not combined_content.strip().startswith("#"):
                            combined_content = f"""# 研究笔记：{query}

## 研究内容

{combined_content}

## 说明
本笔记由系统自动从研究过程中提取。

---
*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
                        
                        try:
                            self.filesystem.write_file(
                                "web_research.md",
                                combined_content,
                                subdirectory="notes",
                                metadata={"source": "agent_output_extraction"}
                            )
                            logger.info("已从 Agent 输出提取并保存研究笔记")
                        except Exception as save_error:
                            logger.error(f"保存提取的笔记失败: {save_error}")
            
            logger.info("网络研究完成")
            
            # 更新状态
            state["web_research_done"] = True
            state["current_step"] = "web_research"
            state["messages"] = [AIMessage(content="网络研究已完成")]
            
        except Exception as e:
            logger.error(f"网络研究失败: {e}")
            state["error"] = str(e)
        
        return state
    
    def _doc_analysis_node(self, state: ResearchState) -> ResearchState:
        """
        文档分析节点：分析本地文档
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行文档分析节点...")
        
        query = state["query"]
        thread_id = state["thread_id"]
        plan = state.get("plan", {})
        
        # 构建分析指令
        analysis_instruction = f"""请对以下问题进行深入的文档分析：

研究问题：{query}

研究计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

任务要求：
1. 使用 knowledge_base 工具检索相关文档
2. 分析文档内容的相关性
3. 提炼关键段落和数据
4. 整理成结构化的分析报告
5. 使用 write_research_file 保存报告到 notes 目录，文件名：doc_analysis.md

请确保：
- 检索多个相关文档
- 直接引用原文
- 记录文档来源
- 提炼核心观点

thread_id: {thread_id}
"""
        
        try:
            # 调用 DocAnalyst
            result = self.doc_analyst.invoke({
                "messages": [HumanMessage(content=analysis_instruction)]
            })
            
            logger.info("文档分析完成")
            
            # 更新状态
            state["doc_analysis_done"] = True
            state["current_step"] = "doc_analysis"
            state["messages"] = [AIMessage(content="文档分析已完成")]
            
        except Exception as e:
            logger.error(f"文档分析失败: {e}")
            state["error"] = str(e)
        
        return state
    
    def _report_writing_node(self, state: ResearchState) -> ResearchState:
        """
        报告撰写节点：生成最终报告
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        logger.info("执行报告撰写节点...")
        
        query = state["query"]
        thread_id = state["thread_id"]
        
        # 构建撰写指令
        writing_instruction = f"""请根据所有研究材料撰写最终研究报告：

研究问题：{query}

任务要求：
1. 使用 list_research_files 列出研究笔记（thread_id: {thread_id}）并逐一读取
2. 整合研究发现与证据，避免模板化结构，按主题与信息密度选择合适章节
3. 提供真实示例或代码片段（技术主题）与实践建议
4. 使用内联引用并在结尾列出参考来源
5. 使用 write_research_file 保存到 reports/final_report.md

写作准则：
{WRITER_GUIDELINES}

thread_id: {thread_id}
"""
        
        try:
            # 调用 ReportWriter
            result = self.report_writer.invoke({
                "messages": [HumanMessage(content=writing_instruction)]
            })
            
            # 尝试从多个来源获取报告（文件系统已保证同步写入）
            final_report = None
            
            # 1. 先尝试从文件系统读取
            try:
                final_report = self.filesystem.read_file(
                    "final_report.md",
                    subdirectory="reports"
                )
                logger.info("从文件系统读取最终报告")
            except Exception as e:
                logger.debug(f"   无法从文件系统读取: {e}")
            
            # 2. 如果文件系统没有，尝试从 Agent 的输出中提取
            if not final_report:
                logger.info("从 Agent 输出中提取报告内容...")
                
                # 从 result 中提取 AI 消息
                if isinstance(result, dict) and "messages" in result:
                    messages = result["messages"]
                    
                    # 收集所有 AI 消息内容（可能包含报告）
                    ai_contents = []
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.content:
                            # 跳过工具调用结果消息
                            content = msg.content.strip()
                            if content and not content.startswith("找到") and not content.startswith("文件已保存"):
                                ai_contents.append(content)
                    
                    # 尝试找到最长的、看起来像报告的内容
                    for content in sorted(ai_contents, key=len, reverse=True):
                        # 检查是否包含报告特征
                        is_report = (
                            len(content) > 200 and  # 足够长
                            (content.startswith("#") or  # Markdown 标题
                             "##" in content or  # 包含二级标题
                             "执行摘要" in content or  # 包含报告关键词
                             "研究背景" in content or
                             "主要发现" in content)
                        )
                        
                        if is_report:
                            final_report = content
                            logger.info(f"从 Agent 输出中提取到报告（长度: {len(content)} 字符）")
                            
                            # 保存到文件系统
                            try:
                                self.filesystem.write_file(
                                    "final_report.md",
                                    final_report,
                                    subdirectory="reports",
                                    metadata={"source": "agent_output"}
                                )
                                logger.info("报告已保存到文件系统")
                            except Exception as save_error:
                                logger.warning(f"保存报告失败: {save_error}")
                            
                            break
            
            # 3. 如果还是没有，生成综合报告
            if not final_report:
                logger.info("Agent 未直接生成报告，使用研究材料生成综合报告")
                
                # 读取所有可用的研究材料
                research_materials = []
                
                # 尝试读取研究计划
                try:
                    plan_content = self.filesystem.read_file(
                        "research_plan.md",
                        subdirectory="plans"
                    )
                    research_materials.append(("研究计划", plan_content))
                    logger.debug("   ✓ 读取研究计划")
                except Exception:
                    logger.debug("   ✗ 未找到研究计划")
                
                # 尝试读取网络研究笔记
                try:
                    web_notes = self.filesystem.read_file(
                        "web_research.md",
                        subdirectory="notes"
                    )
                    research_materials.append(("网络研究笔记", web_notes))
                    logger.debug("   ✓ 读取网络研究笔记")
                except Exception:
                    logger.debug("   ✗ 未找到网络研究笔记")
                
                # 尝试读取文档分析报告
                try:
                    doc_notes = self.filesystem.read_file(
                        "doc_analysis.md",
                        subdirectory="notes"
                    )
                    research_materials.append(("文档分析报告", doc_notes))
                    logger.debug("   ✓ 读取文档分析报告")
                except Exception:
                    logger.debug("   ✗ 未找到文档分析报告")
                
                # 生成基础报告
                if research_materials:
                    logger.info(f"   找到 {len(research_materials)} 个研究材料，生成综合报告")
                    
                    # 构建报告内容
                    materials_section = ""
                    for title, content in research_materials:
                        materials_section += f"\n### {title}\n\n{content}\n\n"
                    
                    final_report = f"""# {query}

{materials_section}

结论与建议：基于上述材料给出清晰结论与可执行建议，并在文中保留关键证据的内联引用。

参考来源：请在文末列出来源。

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
研究任务ID：{thread_id}
"""
                else:
                    logger.warning("   未找到任何研究材料")
                    final_report = f"""# {query}

## 执行摘要

本研究针对"{query}"进行了调研。

## 说明

研究过程已完成，但未能找到保存的研究材料。这可能是由于：
1. 研究任务刚刚启动，材料尚未生成
2. 文件保存过程中出现问题
3. 研究工具未能正确调用

建议：
- 查看日志文件了解详细情况
- 重新运行研究任务
- 检查 API 配置和网络连接

---
*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*研究任务 ID：{thread_id}*
"""
                
                # 保存基础报告
                try:
                    self.filesystem.write_file(
                        "final_report.md",
                        final_report,
                        subdirectory="reports",
                        metadata={
                            "source": "fallback",
                            "materials_count": len(research_materials)
                        }
                    )
                    logger.info("基础报告已生成并保存")
                except Exception as save_error:
                    logger.error(f"保存基础报告失败: {save_error}")
            
            is_technical = any(w in query.lower() for w in ["react", "hook", "api", "编程", "代码", "javascript", "python"])            
            validator = OutputValidator(require_examples=is_technical)
            result = validator.validate(final_report)
            if not result.is_valid:
                try:
                    revision_prompt = f"请在保持现有结构与引用的前提下，补充示例或代码片段，并提升信息密度与可操作性。\n\n写作准则：\n{WRITER_GUIDELINES}\n\n原文：\n{final_report}"
                    model = my_deepseek
                    revised = model.invoke([HumanMessage(content=revision_prompt)])
                    revised_text = revised.content or final_report
                    final_report = revised_text
                    self.filesystem.write_file(
                        "final_report.md",
                        final_report,
                        subdirectory="reports",
                        metadata={"source": "revision"}
                    )
                except Exception:
                    pass

            state["final_report"] = final_report
            state["report_done"] = True
            state["current_step"] = "report_writing"
            state["messages"] = [AIMessage(content="最终报告已完成")]
            
            logger.info("报告撰写节点完成")
            
        except Exception as e:
            logger.error(f"报告撰写失败: {e}")
            state["error"] = str(e)
        
        return state
    
    # ==================== 公共方法 ====================
    
    def research(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行研究任务
        
        Args:
            query: 研究问题
            config: 配置参数（可选）
            
        Returns:
            研究结果字典
            
        Example:
            >>> agent = DeepResearchAgent(thread_id="research_123")
            >>> result = agent.research("分析 LangChain 1.0 的新特性")
            >>> print(result["final_report"])
        """
        logger.info(f"开始研究任务: {query}")
        
        # 初始化状态
        initial_state: ResearchState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "thread_id": self.thread_id,
            "plan": None,
            "web_research_done": False,
            "doc_analysis_done": False,
            "report_done": False,
            "current_step": "init",
            "error": None,
            "final_report": None,
        }
        
        # 执行工作流
        try:
            if config is None:
                config = {"configurable": {"thread_id": self.thread_id}}
            
            final_state = self.graph.invoke(initial_state, config)
            
            logger.info("研究任务完成")
            
            # 返回结果
            return {
                "status": "completed",
                "query": query,
                "thread_id": self.thread_id,
                "final_report": final_state.get("final_report"),
                "plan": final_state.get("plan"),
                "error": final_state.get("error"),
                "steps_completed": {
                    "web_research": final_state.get("web_research_done", False),
                    "doc_analysis": final_state.get("doc_analysis_done", False),
                    "report": final_state.get("report_done", False),
                }
            }
            
        except Exception as e:
            logger.error(f"研究任务失败: {e}")
            return {
                "status": "failed",
                "query": query,
                "thread_id": self.thread_id,
                "error": str(e),
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取研究状态
        
        Returns:
            状态信息字典
        """
        # 列出文件系统中的文件
        files = self.filesystem.list_files()
        
        return {
            "thread_id": self.thread_id,
            "filesystem_files": files,
            "web_search_enabled": self.enable_web_search,
            "doc_analysis_enabled": self.enable_doc_analysis,
        }


# ==================== 工厂函数 ====================

def create_deep_research_agent(
    thread_id: str,
    enable_web_search: bool = True,
    enable_doc_analysis: bool = False,
    retriever_tool: Optional[BaseTool] = None,
    **kwargs,
) -> DeepResearchAgent:
    """
    创建深度研究智能体的便捷工厂函数
    
    Args:
        thread_id: 研究任务 ID
        enable_web_search: 是否启用网络搜索
        enable_doc_analysis: 是否启用文档分析
        retriever_tool: RAG 检索工具
        **kwargs: 其他参数
        
    Returns:
        DeepResearchAgent 实例
        
    Example:
        >>> agent = create_deep_research_agent(
        ...     thread_id="research_123",
        ...     enable_web_search=True,
        ...     enable_doc_analysis=True,
        ...     retriever_tool=my_retriever_tool
        ... )
        >>> result = agent.research("分析 AI 领域的最新趋势")
    """
    return DeepResearchAgent(
        thread_id=thread_id,
        enable_web_search=enable_web_search,
        enable_doc_analysis=enable_doc_analysis,
        retriever_tool=retriever_tool,
        **kwargs,
    )


logger.info("DeepAgent 模块已加载")

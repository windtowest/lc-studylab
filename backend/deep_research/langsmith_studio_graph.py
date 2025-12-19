import json
from datetime import datetime

from fsspec import filesystem
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from config import get_logger
from core.my_llm import my_deepseek
from core.prompts import WRITER_GUIDELINES
from core.tools.filesystem import get_filesystem
from deep_research import create_deep_research_agent
from deep_research.deep_agent import ResearchState

logger = get_logger(__name__)
filesystem = get_filesystem("langsmith_test")

def _planner_node(self, state: ResearchState) -> ResearchState:
    """
    规划节点：生成研究计划

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """

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

        filesystem.write_file(
            "research_plan.md",
            plan_content,
            subdirectory="plans"
        )


        # 更新状态
        state["plan"] = plan
        state["current_step"] = "planner"
        state["messages"] = [AIMessage(content=f"研究计划已生成：{plan.get('research_goal')}")]

    except Exception as e:
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


workflow = StateGraph(ResearchState)
workflow.add_node("planner", _planner_node)
workflow.add_node("web_research", _web_research_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "web_research")
workflow.add_edge("web_research", END)
graph = workflow.compile()
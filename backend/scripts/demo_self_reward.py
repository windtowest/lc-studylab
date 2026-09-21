#!/usr/bin/env python
"""
Self-Reward Agent 演示脚本

演示Self-Reward评分、Critic反思和偏好数据生成功能。

使用方法:
    python scripts/demo_self_reward.py [--mode MODE]

模式:
    - scoring: 演示Self-Reward评分功能
    - critic: 演示Critic反思功能
    - preference: 演示偏好数据生成
    - full: 演示完整流程（默认）
    - interactive: 交互式演示

示例:
    python scripts/demo_self_reward.py --mode scoring
    python scripts/demo_self_reward.py --mode full
    python scripts/demo_self_reward.py --mode interactive
"""

import sys
import argparse
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
# 添加backend目录到路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from core.tools import BASIC_TOOLS, WEB_SEARCH_TOOLS, WEATHER_TOOLS


from config import get_logger, settings

logger = get_logger(__name__)


# ==================== 辅助函数 ====================

def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_section(title: str):
    """打印小节标题"""
    print(f"\n--- {title} ---\n")


def print_json(data: dict, indent: int = 2):
    """格式化打印JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=indent, default=str))


# ==================== 演示函数 ====================

def demo_self_reward_scoring():
    """
    演示Self-Reward评分功能
    
    展示如何对Agent响应进行多维度评分。
    """
    print_header("Self-Reward 评分演示")
    
    from core.self_reward.models import ScoreDimension, DimensionScore, SelfRewardResult
    
    # 模拟评分结果（实际使用时会调用LLM）
    print_section("1. 创建评分结果示例")
    
    # 高分示例
    high_score_result = SelfRewardResult(
        score=8.5,
        reason="响应准确、完整，清晰地回答了用户问题，提供了有价值的信息。",
        patch=None,
        dimension_scores=[
            DimensionScore(
                dimension=ScoreDimension.ACCURACY,
                score=9.0,
                reason="信息准确，没有事实错误"
            ),
            DimensionScore(
                dimension=ScoreDimension.RELEVANCE,
                score=8.5,
                reason="高度相关，直接回答了用户问题"
            ),
            DimensionScore(
                dimension=ScoreDimension.COMPLETENESS,
                score=8.0,
                reason="覆盖了主要方面，但可以补充更多细节"
            ),
            DimensionScore(
                dimension=ScoreDimension.CLARITY,
                score=8.5,
                reason="表达清晰，结构合理"
            ),
        ],
        timestamp=datetime.now().isoformat()
    )
    
    print("高分响应评分结果:")
    print(f"  总分: {high_score_result.score}/10")
    print(f"  评分理由: {high_score_result.reason}")
    print(f"  是否需要Patch: {'否' if high_score_result.patch is None else '是'}")
    print("\n  各维度评分:")
    for ds in high_score_result.dimension_scores:
        print(f"    - {ds.dimension.value}: {ds.score}/10 ({ds.reason})")
    
    # 低分示例（带patch）
    print_section("2. 低分响应示例（带改进建议）")
    
    low_score_result = SelfRewardResult(
        score=4.5,
        reason="响应不够准确，遗漏了关键信息，表达也不够清晰。",
        patch="改进后的响应：机器学习是人工智能的一个分支，它使计算机能够从数据中学习，"
              "而无需明确编程。主要类型包括：1) 监督学习 - 使用标记数据训练；"
              "2) 无监督学习 - 发现数据中的模式；3) 强化学习 - 通过奖励信号学习。",
        dimension_scores=[
            DimensionScore(
                dimension=ScoreDimension.ACCURACY,
                score=5.0,
                reason="部分信息不准确"
            ),
            DimensionScore(
                dimension=ScoreDimension.RELEVANCE,
                score=6.0,
                reason="基本相关但偏离主题"
            ),
            DimensionScore(
                dimension=ScoreDimension.COMPLETENESS,
                score=3.5,
                reason="遗漏了重要内容"
            ),
            DimensionScore(
                dimension=ScoreDimension.CLARITY,
                score=4.0,
                reason="表达混乱，难以理解"
            ),
        ],
        timestamp=datetime.now().isoformat()
    )
    
    print("低分响应评分结果:")
    print(f"  总分: {low_score_result.score}/10")
    print(f"  评分理由: {low_score_result.reason}")
    print(f"  是否需要Patch: {'是' if low_score_result.patch else '否'}")
    if low_score_result.patch:
        print(f"\n  改进建议 (Patch):")
        print(f"    {low_score_result.patch[:200]}...")
    
    print_section("3. 评分维度说明")
    print("""
    Self-Reward评分基于四个维度：
    
    1. 准确性 (Accuracy) - 权重30%
       评估响应中信息的正确性和可靠性
    
    2. 相关性 (Relevance) - 权重25%
       评估响应与用户问题的相关程度
    
    3. 完整性 (Completeness) - 权重25%
       评估响应是否覆盖了问题的所有方面
    
    4. 清晰度 (Clarity) - 权重20%
       评估响应的表达是否清晰易懂
    """)
    
    print("\n✅ Self-Reward评分演示完成")


def demo_critic_reflection():
    """
    演示Critic反思功能
    
    展示如何分析执行轨迹并生成改进建议。
    """
    print_header("Critic 反思演示")
    
    from core.critic.models import ProblemType, Problem, Improvement, CriticReport
    from core.trajectory.models import ExecutionTrajectory, ToolCall, ReasoningStep
    
    print_section("1. 创建执行轨迹示例")
    
    # 创建一个有问题的执行轨迹
    trajectory = ExecutionTrajectory(
        trajectory_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        user_input="请帮我查询北京今天的天气，并推荐适合的穿着",
        system_prompt="你是一个智能助手",
        reasoning_steps=[
            ReasoningStep(
                step_number=1,
                content="用户想知道北京的天气和穿着建议",
                timestamp=datetime.now().isoformat()
            ),
            ReasoningStep(
                step_number=2,
                content="需要调用天气查询工具",
                timestamp=datetime.now().isoformat()
            ),
            ReasoningStep(
                step_number=3,
                content="根据天气给出穿着建议",
                timestamp=datetime.now().isoformat()
            ),
        ],
        tool_calls=[
            ToolCall(
                tool_name="weather_query",
                input_params={"city": "北京"},
                output_result={"error": "API调用失败"},
                duration_ms=1500.0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error="API rate limit exceeded"
            ),
        ],
        intermediate_results=["天气查询失败"],
        final_output="抱歉，我无法获取天气信息。",
        total_duration_ms=2500.0,
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat()
    )
    
    print("执行轨迹概要:")
    print(f"  用户输入: {trajectory.user_input}")
    print(f"  推理步骤数: {len(trajectory.reasoning_steps)}")
    print(f"  工具调用数: {len(trajectory.tool_calls)}")
    print(f"  最终输出: {trajectory.final_output}")
    print(f"  总耗时: {trajectory.total_duration_ms}ms")
    
    print_section("2. Critic分析报告")
    
    # 创建Critic报告
    critic_report = CriticReport(
        problems=[
            Problem(
                type=ProblemType.TOOL_ERROR,
                description="天气查询工具调用失败，API返回错误",
                severity=4,
                location="步骤2-工具调用"
            ),
            Problem(
                type=ProblemType.INFO_MISSING,
                description="未能提供用户请求的天气信息和穿着建议",
                severity=5,
                location="最终输出"
            ),
            Problem(
                type=ProblemType.REASONING_BREAK,
                description="工具失败后未尝试其他方案",
                severity=3,
                location="步骤3"
            ),
        ],
        root_causes=[
            "API调用频率限制导致工具失败",
            "缺乏错误恢复机制",
            "未考虑备选方案"
        ],
        improvements=[
            Improvement(
                problem_ref="天气查询工具调用失败",
                suggestion="实现API调用重试机制，添加指数退避策略",
                priority=5,
                correction="添加重试逻辑：最多重试3次，每次间隔翻倍"
            ),
            Improvement(
                problem_ref="未能提供用户请求的信息",
                suggestion="工具失败时提供基于历史数据的估计或建议用户稍后重试",
                priority=4,
                correction="当天气API失败时，可以说：'目前无法获取实时天气，"
                          "根据北京这个季节的一般情况，建议...'"
            ),
            Improvement(
                problem_ref="工具失败后未尝试其他方案",
                suggestion="添加备选工具或信息源",
                priority=3,
                correction=None
            ),
        ],
        overall_assessment="执行过程中遇到工具调用失败，且缺乏有效的错误处理机制。"
                          "建议增强系统的容错能力和用户体验。",
        timestamp=datetime.now().isoformat()
    )
    
    print("发现的问题:")
    for i, problem in enumerate(critic_report.problems, 1):
        severity_emoji = "🔴" if problem.severity >= 4 else "🟡" if problem.severity >= 3 else "🟢"
        print(f"  {i}. {severity_emoji} [{problem.type.value}] {problem.description}")
        print(f"     位置: {problem.location}, 严重程度: {problem.severity}/5")
    
    print("\n根因分析:")
    for i, cause in enumerate(critic_report.root_causes, 1):
        print(f"  {i}. {cause}")
    
    print("\n改进建议 (按优先级排序):")
    for i, imp in enumerate(critic_report.improvements, 1):
        print(f"  {i}. [优先级{imp.priority}] {imp.suggestion}")
        if imp.correction:
            print(f"     具体修正: {imp.correction[:100]}...")
    
    print(f"\n总体评估: {critic_report.overall_assessment}")
    
    print_section("3. 问题类型说明")
    print("""
    Critic可以识别以下类型的问题：
    
    1. TOOL_ERROR - 工具调用错误
       工具执行失败、返回错误或超时
    
    2. REASONING_BREAK - 推理链断裂
       推理过程中出现逻辑跳跃或不连贯
    
    3. INFO_MISSING - 信息遗漏
       响应中缺少用户请求的关键信息
    
    4. LOGIC_ERROR - 逻辑错误
       推理或结论中存在逻辑谬误
    
    5. FORMAT_ERROR - 格式错误
       输出格式不符合要求或难以解析
    """)
    
    print("\n✅ Critic反思演示完成")


def demo_preference_generation():
    """
    演示偏好数据生成功能
    
    展示如何从经验记录生成用于RM训练的偏好数据。
    """
    print_header("偏好数据生成演示")
    
    from rlhf_integration.preference_generator import PreferencePair
    
    print_section("1. 偏好数据对示例")
    
    # 创建示例偏好对
    preference_pair = PreferencePair(
        prompt="请解释什么是机器学习？",
        chosen="机器学习是人工智能的一个分支，它使计算机能够从数据中学习，"
               "而无需明确编程。主要类型包括：\n"
               "1. 监督学习 - 使用标记数据训练模型\n"
               "2. 无监督学习 - 发现数据中的隐藏模式\n"
               "3. 强化学习 - 通过奖励信号学习最优策略\n\n"
               "机器学习广泛应用于图像识别、自然语言处理、推荐系统等领域。",
        rejected="机器学习就是让机器学习东西。",
        chosen_score=8.5,
        rejected_score=3.0,
        source_records=("record_001", "record_002")
    )
    
    print("偏好数据对:")
    print(f"\n  Prompt (用户输入):")
    print(f"    {preference_pair.prompt}")
    
    print(f"\n  Chosen (高质量响应) - 评分: {preference_pair.chosen_score}/10:")
    print(f"    {preference_pair.chosen[:200]}...")
    
    print(f"\n  Rejected (低质量响应) - 评分: {preference_pair.rejected_score}/10:")
    print(f"    {preference_pair.rejected}")
    
    print(f"\n  来源记录: {preference_pair.source_records}")
    
    print_section("2. 偏好数据生成规则")
    print("""
    偏好数据生成遵循以下规则：
    
    Chosen响应来源：
    - 评分 >= 7.0 的原始响应
    - 或低分响应的Patch修正版本
    
    Rejected响应来源：
    - 评分 < 6.0 的原始响应
    
    匹配策略：
    1. 优先匹配相同session_id的记录
    2. 其次匹配相同user_input的记录
    3. 最后匹配语义相似的输入
    """)
    
    print_section("3. 导出格式示例")
    
    export_example = {
        "prompt": preference_pair.prompt,
        "chosen": preference_pair.chosen,
        "rejected": preference_pair.rejected,
        "metadata": {
            "chosen_score": preference_pair.chosen_score,
            "rejected_score": preference_pair.rejected_score,
            "source": "self_reward_agent",
            "source_records": list(preference_pair.source_records)
        }
    }
    
    print("RM训练数据格式 (JSON):")
    print_json(export_example)
    
    print_section("4. 与现有RM训练的集成")
    print("""
    导出的偏好数据与现有RM训练流程兼容：
    
    1. 数据格式：标准的 prompt/chosen/rejected 结构
    2. 元数据：包含评分和来源信息，便于追溯
    3. 导出方法：export_for_rm_training(output_path)
    4. 触发训练：当数据量达到阈值时自动触发
    """)
    
    print("\n✅ 偏好数据生成演示完成")


def demo_full_workflow():
    """
    演示完整工作流程
    
    展示从执行到训练的完整闭环。
    """
    print_header("完整工作流程演示")
    
    print("""
    Self-Reward Agent 完整工作流程：
    
    ┌─────────────────────────────────────────────────────────────┐
    │                     1. Agent 执行                           │
    │  用户输入 → Agent处理 → 工具调用 → 生成响应                  │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                   2. 轨迹记录                               │
    │  记录推理步骤、工具调用、中间结果、最终输出                   │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                 3. Self-Reward 评分                         │
    │  多维度评分（准确性、相关性、完整性、清晰度）                 │
    │  低分时生成Patch改进建议                                    │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                   4. Critic 反思                            │
    │  分析轨迹 → 识别问题 → 生成改进建议                         │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                   5. 经验存储                               │
    │  保存完整经验记录到PostgreSQL                               │
    │  支持按评分、时间、问题类型索引                              │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                 6. 偏好数据生成                              │
    │  从高分/低分记录构造(chosen, rejected)数据对                 │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                 7. RLHF 训练触发                             │
    │  数据量达到阈值 → 导出数据 → 触发RM微调                      │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    print_section("关键组件")
    print("""
    1. SelfRewardAgent
       - 继承BaseAgent，集成所有Self-Reward功能
       - 支持多候选生成和行为选择
       - 自动记录轨迹和触发反思
    
    2. SelfRewardScorer
       - 多维度评分（准确性、相关性、完整性、清晰度）
       - 低分时自动生成Patch
       - 支持批量评分
    
    3. CriticAgent
       - 分析执行轨迹
       - 识别问题类型和严重程度
       - 生成改进建议和修正方案
    
    4. ExperienceStore
       - PostgreSQL持久化存储
       - 多条件查询和相似检索
       - 数据导出功能
    
    5. PreferenceGenerator
       - 自动构造偏好数据对
       - 支持Patch作为chosen来源
       - 导出为RM训练格式
    
    6. TrainingTrigger
       - 数据量检查
       - 自动触发训练
       - 生成训练报告
    """)
    
    print_section("使用示例")
    print("""
    # 创建Self-Reward Agent
    from agents.self_reward_agent import create_self_reward_agent
    
    agent = create_self_reward_agent(
        num_candidates=3,      # 生成3个候选响应
        score_threshold=6.0,   # 评分阈值
        enable_critic=True,    # 启用Critic反思
        enable_memory=True     # 启用经验存储
    )
    
    # 执行对话
    response = agent.invoke("请解释什么是机器学习")
    
    # 获取候选记录
    candidates = agent.get_candidate_records()
    for c in candidates:
        print(f"评分: {c.score_result.score}, 响应: {c.response[:50]}...")
    
    # 获取经验统计
    summary = agent.get_experience_summary()
    print(f"总记录数: {summary['total_records']}")
    """)
    
    print("\n✅ 完整工作流程演示完成")


def demo_interactive():
    """
    交互式演示
    
    允许用户输入问题并查看评分结果。
    """
    print_header("交互式演示")
    
    print("""
    注意：交互式演示需要配置好LLM API。
    
    如果API未配置，将使用模拟数据演示。
    """)
    
    try:
        # 尝试导入Agent
        from agents.self_reward_agent import create_self_reward_agent
        
        print("\n正在初始化Self-Reward Agent...")
        tools: List = list(BASIC_TOOLS)
        if settings.zhipu_api_key:
            for tool in WEB_SEARCH_TOOLS:
                if tool not in tools:
                    tools.append(tool)
        else:
            logger.debug("未配置 zhipu API Key，网络搜索工具不可用")
        agent = create_self_reward_agent(
            num_candidates=1,
            score_threshold=6.0,
            tools=tools,
            enable_critic=False,  # 简化演示
            enable_memory=False   # 简化演示
        )
        
        print("Agent初始化成功！\n")
        
        while True:
            user_input = input("\n请输入问题 (输入 'quit' 退出): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n感谢使用！再见！")
                break
            
            if not user_input:
                print("请输入有效的问题。")
                continue
            
            print("\n正在处理...")
            
            try:
                response = agent.invoke(user_input)
                
                print(f"\n响应: {response}")
                
                # 获取候选记录
                candidates = agent.get_candidate_records()
                if candidates:
                    print(f"\n评分: {candidates[0].score_result.score}/10")
                    print(f"评分理由: {candidates[0].score_result.reason}")
                    print(f"改进补丁: {candidates[0].score_result.patch}")
                    
            except Exception as e:
                print(f"\n处理出错: {e}")
                
    except Exception as e:
        print(f"\n无法初始化Agent: {e}")
        print("\n将使用模拟数据进行演示...")
        
        # 使用模拟数据
        demo_self_reward_scoring()


# ==================== 主函数 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Self-Reward Agent 演示脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/demo_self_reward.py --mode scoring
    python scripts/demo_self_reward.py --mode critic
    python scripts/demo_self_reward.py --mode preference
    python scripts/demo_self_reward.py --mode full
    python scripts/demo_self_reward.py --mode interactive
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["scoring", "critic", "preference", "full", "interactive"],
        help="演示模式 (默认: full)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "🤖 Self-Reward Agent 演示程序 🤖".center(60))
    print("=" * 60)
    
    if args.mode == "scoring":
        demo_self_reward_scoring()
    elif args.mode == "critic":
        demo_critic_reflection()
    elif args.mode == "preference":
        demo_preference_generation()
    elif args.mode == "full":
        demo_self_reward_scoring()
        demo_critic_reflection()
        demo_preference_generation()
        demo_full_workflow()
    elif args.mode == "interactive":
        demo_interactive()
    
    print("\n" + "=" * 60)
    print("演示结束".center(60))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

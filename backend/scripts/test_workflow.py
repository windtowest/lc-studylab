#!/usr/bin/env python3
"""
学习工作流测试脚本

本脚本用于测试 LangGraph 学习工作流的完整功能，包括：
- 启动工作流
- 生成学习计划和练习题
- 模拟用户答题
- 自动评分和反馈
- 检查点恢复
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import json
import uuid
from datetime import datetime

from workflows.study_flow_graph import (
    start_study_flow,
    submit_answers,
    get_workflow_state,
    get_workflow_history
)
from config.logging import get_logger

# 初始化日志
logger = get_logger(__name__)


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_state(state: dict, title: str = "当前状态"):
    """打印工作流状态"""
    print(f"\n{'─' * 60}")
    print(f"📊 {title}")
    print(f"{'─' * 60}")
    print(f"步骤: {state.get('current_step')}")
    print(f"更新时间: {state.get('updated_at')}")
    
    if state.get('error'):
        print(f"❌ 错误: {state['error']}")
    
    print(f"{'─' * 60}\n")


def test_complete_workflow():
    """测试完整的工作流"""
    print_section("测试 1: 完整工作流")
    
    # 生成唯一的 thread_id
    thread_id = f"test_{uuid.uuid4().hex[:8]}"
    print(f"🆔 Thread ID: {thread_id}\n")
    
    # 步骤 1: 启动工作流
    print("📝 步骤 1: 启动工作流，提出学习问题...")
    user_question = "我想学习 Python 的基础知识，包括变量、数据类型和控制流"
    
    try:
        result = start_study_flow(
            user_question=user_question,
            thread_id=thread_id
        )
        
        print("✅ 工作流启动成功！")
        print_state(result, "启动后状态")
        
        # 显示学习计划
        if result.get('learning_plan'):
            plan = result['learning_plan']
            print("📚 学习计划:")
            print(f"   主题: {plan.get('topic')}")
            print(f"   难度: {plan.get('difficulty')}")
            print(f"   预计时间: {plan.get('estimated_time')} 分钟")
            print(f"   学习目标: {len(plan.get('objectives', []))} 个")
            print(f"   关键知识点: {len(plan.get('key_points', []))} 个\n")
        
        # 显示练习题
        if result.get('quiz'):
            quiz = result['quiz']
            print(f"📝 练习题: {len(quiz.get('questions', []))} 题")
            print(f"   总分: {quiz.get('total_points')} 分")
            print(f"   建议用时: {quiz.get('time_limit')} 分钟\n")
            
            # 显示题目
            for i, q in enumerate(quiz['questions'], 1):
                print(f"   第 {i} 题 ({q['points']} 分) - {q['type']}")
                print(f"   {q['question']}")
                if q.get('options'):
                    for opt in q['options']:
                        print(f"      {opt}")
                print()
        
        # 步骤 2: 模拟用户答题
        print("✍步骤 2: 模拟用户提交答案...")
        
        # 构造答案（故意答错一些）
        quiz = result.get('quiz', {})
        questions = quiz.get('questions', [])
        
        user_answers = {}
        for i, q in enumerate(questions):
            q_id = q['id']
            q_type = q['type']
            
            if q_type == 'multiple_choice':
                # 第一题答对，其他答错
                if i == 0:
                    user_answers[q_id] = q['answer']
                else:
                    # 随便选一个错误答案
                    wrong_options = ['A', 'B', 'C', 'D']
                    if q['answer'] in wrong_options:
                        wrong_options.remove(q['answer'])
                    user_answers[q_id] = wrong_options[0]
            
            elif q_type == 'fill_blank':
                # 填空题答对
                user_answers[q_id] = q['answer']
            
            elif q_type == 'short_answer':
                # 简答题给一个部分正确的答案
                user_answers[q_id] = "这是一个简短的回答，包含了部分关键信息。"
        
        print(f"   提交 {len(user_answers)} 个答案\n")
        
        # 提交答案
        result = submit_answers(
            thread_id=thread_id,
            user_answers=user_answers
        )
        
        print("答案提交成功！")
        print_state(result, "评分后状态")
        
        # 显示评分结果
        if result.get('score') is not None:
            score = result['score']
            score_details = result.get('score_details', {})
            
            print(f"评分结果:")
            print(f"   总分: {score} 分")
            print(f"   答对: {score_details.get('correct_count', 0)}/{score_details.get('total_count', 0)} 题")
            print(f"   是否需要重试: {'是' if result.get('should_retry') else '否'}\n")
        
        # 显示反馈
        if result.get('feedback'):
            print(f"反馈:")
            print(f"   {result['feedback']}\n")
        
        # 步骤 3: 查看工作流历史
        print("步骤 3: 查看工作流历史...")
        history = get_workflow_history(thread_id)
        print(f"   共 {len(history)} 个检查点\n")
        
        for i, h in enumerate(history[:5], 1):  # 只显示前5个
            print(f"   {i}. {h.get('step')} - {h.get('timestamp')}")
        
        print("\n完整工作流测试完成！")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        print(f"\n测试失败: {str(e)}")
        return False


def test_checkpoint_recovery():
    """测试检查点恢复功能"""
    print_section("测试 2: 检查点恢复")
    
    thread_id = f"test_recovery_{uuid.uuid4().hex[:8]}"
    print(f"Thread ID: {thread_id}\n")
    
    try:
        # 启动工作流
        print("启动工作流...")
        result = start_study_flow(
            user_question="学习机器学习的基础概念",
            thread_id=thread_id
        )
        
        print("工作流已暂停在答题环节")
        
        # 模拟程序重启，从检查点恢复
        print("\n模拟程序重启，从检查点恢复状态...")
        
        recovered_state = get_workflow_state(thread_id)
        
        if recovered_state:
            print("成功从检查点恢复状态！")
            print_state(recovered_state, "恢复的状态")
            
            # 验证数据完整性
            if recovered_state.get('quiz'):
                print("练习题数据完整")
            if recovered_state.get('learning_plan'):
                print("学习计划数据完整")
            
            print("\n检查点恢复测试完成！")
            return True
        else:
            print("无法恢复状态")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        print(f"\n测试失败: {str(e)}")
        return False


def test_retry_mechanism():
    """测试重试机制"""
    print_section("测试 3: 重试机制（得分低于60分）")
    
    thread_id = f"test_retry_{uuid.uuid4().hex[:8]}"
    print(f"Thread ID: {thread_id}\n")
    
    try:
        # 启动工作流
        print("启动工作流...")
        result = start_study_flow(
            user_question="学习深度学习的基本概念",
            thread_id=thread_id
        )
        
        # 故意全部答错
        print("\n提交全错答案（测试重试机制）...")
        
        quiz = result.get('quiz', {})
        questions = quiz.get('questions', [])
        
        # 全部答错
        user_answers = {}
        for q in questions:
            q_id = q['id']
            q_type = q['type']
            
            if q_type == 'multiple_choice':
                # 选择一个错误答案
                wrong_options = ['A', 'B', 'C', 'D']
                if q['answer'] in wrong_options:
                    wrong_options.remove(q['answer'])
                user_answers[q_id] = wrong_options[0]
            else:
                user_answers[q_id] = "错误答案"
        
        result = submit_answers(thread_id, user_answers)
        
        score = result.get('score', 0)
        should_retry = result.get('should_retry', False)
        
        print(f"\n评分结果: {score} 分")
        print(f"🔄 是否触发重试: {'是' if should_retry else '否'}")
        
        if should_retry:
            print("重试机制正常工作！")
            print("   系统已自动生成新的练习题")
            
            # 查看新题目
            state = get_workflow_state(thread_id)
            if state.get('quiz'):
                new_quiz = state['quiz']
                print(f"   新练习题: {len(new_quiz.get('questions', []))} 题\n")
            
            return True
        else:
            print("未触发重试（可能得分高于60分）")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        print(f"\n测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("\n" + "🧪" * 30)
    print("  LangGraph 学习工作流测试套件")
    print("🧪" * 30 + "\n")
    
    results = []
    
    # 测试 1: 完整工作流
    results.append(("完整工作流", test_complete_workflow()))
    
    # # 测试 2: 检查点恢复
    # results.append(("检查点恢复", test_checkpoint_recovery()))
    #
    # # 测试 3: 重试机制
    # results.append(("重试机制", test_retry_mechanism()))
    
    # 汇总结果
    print_section("测试结果汇总")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {status}  {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    try:
        from workflows.study_flow_graph import cleanup_study_flow
        cleanup_study_flow()
    except Exception as e:
        logger.error(f"清理工作流资源时出错: {e}")

    logger.info("资源清理完成")
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())



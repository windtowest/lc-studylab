"""
反馈生成节点 (Feedback Node)

本节点负责根据评分结果生成个性化反馈，并决定是否需要重新出题。
"""

import logging
from datetime import datetime
from typing import Dict, Any

from ..state import StudyFlowState
from core.models import get_chat_model
from config.logging import get_logger

logger = get_logger(__name__)


def feedback_node(state: StudyFlowState) -> Dict[str, Any]:
    """
    反馈生成节点
    
    功能：
    1. 根据得分生成个性化反馈
    2. 提供学习建议和改进方向
    3. 决定是否需要重新出题（得分低于60分且重试次数<3）
    
    Args:
        state: 当前工作流状态
        
    Returns:
        更新后的状态字典，包含 feedback 和 should_retry
    """
    logger.info("[Feedback Node] 开始生成反馈")
    
    try:
        score = state.get("score", 0)
        score_details = state.get("score_details", {})
        learning_plan = state.get("learning_plan", {})
        retry_count = state.get("retry_count", 0)
        
        logger.info(f"[Feedback Node] 当前得分: {score}, 重试次数: {retry_count}")
        
        # 获取聊天模型
        model = get_chat_model()
        
        # 分析错题
        question_scores = score_details.get("question_scores", [])
        wrong_questions = [q for q in question_scores if not q["is_correct"]]
        
        # 构建错题分析
        wrong_analysis = ""
        if wrong_questions:
            wrong_analysis = "\n\n错题分析:\n"
            for q in wrong_questions:
                wrong_analysis += f"- 题目ID {q['question_id']}: {q['feedback']}\n"
        
        # 生成个性化反馈
        feedback_prompt = f"""作为一位耐心的学习导师，请根据学生的测验结果提供个性化反馈。

学习主题: {learning_plan.get('topic', '未知')}
难度级别: {learning_plan.get('difficulty', '未知')}

测验结果:
- 得分: {score} 分
- 答对题数: {score_details.get('correct_count', 0)}/{score_details.get('total_count', 0)}
{wrong_analysis}

请提供:
1. 对整体表现的评价（鼓励性的）
2. 针对错题的学习建议
3. 下一步学习方向
4. 鼓励的话语

请用温暖、鼓励的语气，帮助学生建立信心。字数控制在200字以内。"""

        logger.info("[Feedback Node] 调用 LLM 生成个性化反馈...")
        response = model.invoke([{"role": "user", "content": feedback_prompt}])
        feedback = response.content
        
        logger.info(f"[Feedback Node] 反馈生成完成")
        
        # 决定是否需要重新出题
        # 规则：得分低于60分且重试次数小于3次
        should_retry = score < 60 and retry_count < 3
        
        # 构建反馈消息
        feedback_message = f"\n\n💬 **学习反馈**\n\n{feedback}\n\n"
        
        if should_retry:
            feedback_message += f"由于得分未达到60分，系统将为您重新生成练习题。（第 {retry_count + 1} 次重试）\n"
            feedback_message += "请继续努力，相信您一定能掌握这些知识点！"
        elif retry_count >= 3:
            feedback_message += "您已经尝试了3次，建议先回顾学习资料，巩固基础知识后再来挑战。"
        else:
            feedback_message += "恭喜您通过测验！继续保持这样的学习状态！"
        
        # 更新重试计数
        new_retry_count = retry_count + 1 if should_retry else retry_count
        
        logger.info(f"[Feedback Node] should_retry={should_retry}, new_retry_count={new_retry_count}")
        
        # 更新状态
        return {
            "feedback": feedback,
            "should_retry": should_retry,
            "retry_count": new_retry_count,
            "messages": [{"role": "assistant", "content": feedback_message}],
            "current_step": "feedback",
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[Feedback Node] 生成反馈失败: {str(e)}", exc_info=True)
        return {
            "error": f"反馈生成失败: {str(e)}",
            "error_node": "feedback",
            "current_step": "feedback_error",
            "updated_at": datetime.now().isoformat()
        }


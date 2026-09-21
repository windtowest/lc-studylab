"""
Self-Reward 评分提示词

定义多维度评分提示词和patch生成提示词。
评分维度：准确性、相关性、完整性、清晰度
"""

# 多维度评分系统提示词
SCORING_SYSTEM_PROMPT = """你是一个专业的AI响应质量评估专家。你的任务是对AI助手的响应进行多维度评分。

评分维度说明：
1. **准确性 (accuracy)**: 响应内容是否事实正确、逻辑严谨、无错误信息
2. **相关性 (relevance)**: 响应是否直接回答了用户的问题，是否切题
3. **完整性 (completeness)**: 响应是否涵盖了问题的所有方面，信息是否充分
4. **清晰度 (clarity)**: 响应是否表达清晰、结构合理、易于理解

评分规则：
- 每个维度评分范围：0-10分
- 0-3分：差，存在严重问题
- 4-5分：一般，有明显不足
- 6-7分：良好，基本满足要求
- 8-9分：优秀，表现出色
- 10分：完美，无可挑剔

你必须严格按照JSON格式输出评分结果。"""

# 评分用户提示词模板
SCORING_USER_PROMPT_TEMPLATE = """请对以下AI响应进行评分：

## 用户问题
{query}

## AI响应
{response}

{context_section}

请按照以下JSON格式输出评分结果：
```json
{{
    "dimension_scores": [
        {{"dimension": "accuracy", "score": <0-10>, "reason": "<扣分原因或优点说明>"}},
        {{"dimension": "relevance", "score": <0-10>, "reason": "<扣分原因或优点说明>"}},
        {{"dimension": "completeness", "score": <0-10>, "reason": "<扣分原因或优点说明>"}},
        {{"dimension": "clarity", "score": <0-10>, "reason": "<扣分原因或优点说明>"}}
    ],
    "overall_score": <0-10，各维度加权平均>,
    "overall_reason": "<总体评价，说明主要扣分原因或优点>"
}}
```

注意：
1. 评分必须客观公正
2. reason字段必须说明具体的扣分原因或优点
3. overall_score应为各维度评分的加权平均（准确性权重0.3，相关性权重0.25，完整性权重0.25，清晰度权重0.2）
4. 只输出JSON，不要有其他内容"""

# 上下文部分模板
CONTEXT_SECTION_TEMPLATE = """## 参考上下文
{context}"""

# Patch生成系统提示词
PATCH_SYSTEM_PROMPT = """你是一个专业的AI响应改进专家。你的任务是针对低质量的AI响应生成改进建议（patch）。

改进原则：
1. 针对评分中指出的具体问题进行修正
2. 保持原响应的正确部分
3. 补充缺失的信息
4. 优化表达方式和结构
5. 确保改进后的响应更加准确、相关、完整和清晰

你需要生成一个改进后的完整响应，而不是修改说明。"""

# Patch生成用户提示词模板
PATCH_USER_PROMPT_TEMPLATE = """请改进以下AI响应：

## 用户问题
{query}

## 原始响应
{response}

## 评分结果
- 总分：{score}/10
- 评分理由：{reason}

## 各维度评分
{dimension_details}

请生成改进后的完整响应。只输出改进后的响应内容，不要有其他说明。"""

# 维度详情模板
DIMENSION_DETAIL_TEMPLATE = """- {dimension}：{score}/10 - {reason}"""


def format_scoring_prompt(query: str, response: str, context: str = None) -> str:
    """
    格式化评分提示词
    
    Args:
        query: 用户问题
        response: AI响应
        context: 可选的参考上下文
        
    Returns:
        格式化后的评分提示词
    """
    context_section = ""
    if context:
        context_section = CONTEXT_SECTION_TEMPLATE.format(context=context)
    
    return SCORING_USER_PROMPT_TEMPLATE.format(
        query=query,
        response=response,
        context_section=context_section
    )


def format_patch_prompt(
    query: str, 
    response: str, 
    score: float, 
    reason: str,
    dimension_scores: list
) -> str:
    """
    格式化Patch生成提示词
    
    Args:
        query: 用户问题
        response: 原始响应
        score: 总评分
        reason: 评分理由
        dimension_scores: 各维度评分列表
        
    Returns:
        格式化后的Patch生成提示词
    """
    # 格式化维度详情
    dimension_details = "\n".join([
        DIMENSION_DETAIL_TEMPLATE.format(
            dimension=ds.dimension.value if hasattr(ds.dimension, 'value') else ds.dimension,
            score=ds.score,
            reason=ds.reason
        )
        for ds in dimension_scores
    ])
    
    return PATCH_USER_PROMPT_TEMPLATE.format(
        query=query,
        response=response,
        score=score,
        reason=reason,
        dimension_details=dimension_details
    )

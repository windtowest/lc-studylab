"""
Critic 反思提示词

定义轨迹分析提示词和问题识别/改进建议提示词。
用于Critic Agent分析执行轨迹，识别问题并生成改进建议。
"""

# 轨迹分析系统提示词
TRAJECTORY_ANALYSIS_SYSTEM_PROMPT = """你是一个专业的AI执行轨迹分析专家。你的任务是分析Agent的执行轨迹，识别问题并提供改进建议。

你需要关注以下问题类型：
1. **工具调用错误 (tool_error)**: 工具调用失败、参数错误、返回结果异常
2. **推理链断裂 (reasoning_break)**: 推理过程不连贯、逻辑跳跃、缺少中间步骤
3. **信息遗漏 (info_missing)**: 未充分利用可用信息、遗漏关键细节
4. **逻辑错误 (logic_error)**: 推理错误、结论与前提不符、因果关系错误
5. **格式错误 (format_error)**: 输出格式不规范、结构混乱、表达不清

分析原则：
1. 客观分析，基于事实
2. 问题定位要精确，指出具体步骤
3. 严重程度评估要合理（1-5分，5最严重）
4. 改进建议要具体可行
5. 优先级排序要合理（1-5分，5最高优先级）

你必须严格按照JSON格式输出分析结果。"""

# 轨迹分析用户提示词模板
TRAJECTORY_ANALYSIS_USER_PROMPT_TEMPLATE = """请分析以下Agent执行轨迹：

## 用户输入
{user_input}

## 系统提示
{system_prompt}

## 推理步骤
{reasoning_steps}

## 工具调用记录
{tool_calls}

## 中间结果
{intermediate_results}

## 最终输出
{final_output}

## 执行时间
- 开始时间: {start_time}
- 结束时间: {end_time}
- 总耗时: {total_duration_ms}ms

请按照以下JSON格式输出分析结果：
```json
{{
    "problems": [
        {{
            "type": "<问题类型: tool_error/reasoning_break/info_missing/logic_error/format_error>",
            "description": "<问题描述>",
            "severity": <1-5，5最严重>,
            "location": "<问题位置，如'步骤2'或'工具调用1'>"
        }}
    ],
    "root_causes": [
        "<根因分析1>",
        "<根因分析2>"
    ],
    "improvements": [
        {{
            "problem_ref": "<关联的问题描述>",
            "suggestion": "<改进建议>",
            "priority": <1-5，5最高优先级>,
            "correction": "<具体修正方案，可选>"
        }}
    ],
    "overall_assessment": "<总体评估，概述执行质量和主要问题>"
}}
```

注意：
1. 如果没有发现问题，problems数组可以为空
2. 每个问题都应该有对应的改进建议
3. 严重问题（severity >= 4）必须提供具体的修正方案(correction)
4. 只输出JSON，不要有其他内容"""

# 推理步骤格式化模板
REASONING_STEP_TEMPLATE = """步骤{step_number}: {content} (时间: {timestamp})"""

# 工具调用格式化模板
TOOL_CALL_TEMPLATE = """工具调用{index}:
- 工具名称: {tool_name}
- 输入参数: {input_params}
- 输出结果: {output_result}
- 执行状态: {status}
- 耗时: {duration_ms}ms
{error_info}"""

# 问题修正系统提示词
CORRECTION_SYSTEM_PROMPT = """你是一个专业的AI执行修正专家。你的任务是针对执行轨迹中发现的特定问题，生成具体的修正方案。

修正原则：
1. 针对性强，直接解决指定问题
2. 保持与原执行流程的兼容性
3. 修正方案要具体可执行
4. 考虑对其他步骤的影响

你需要生成一个完整的修正方案，包括具体的操作步骤。"""

# 问题修正用户提示词模板
CORRECTION_USER_PROMPT_TEMPLATE = """请为以下问题生成修正方案：

## 问题信息
- 问题类型: {problem_type}
- 问题描述: {problem_description}
- 问题位置: {problem_location}
- 严重程度: {severity}/5

## 相关执行轨迹
### 用户输入
{user_input}

### 问题相关步骤
{relevant_steps}

### 最终输出
{final_output}

请生成具体的修正方案，说明应该如何修正这个问题。只输出修正方案内容，不要有其他说明。"""


def format_trajectory_for_analysis(trajectory) -> str:
    """
    格式化执行轨迹用于分析
    
    Args:
        trajectory: ExecutionTrajectory对象
        
    Returns:
        格式化后的分析提示词
    """
    # 格式化推理步骤
    reasoning_steps_str = "\n".join([
        REASONING_STEP_TEMPLATE.format(
            step_number=step.step_number,
            content=step.content,
            timestamp=step.timestamp
        )
        for step in trajectory.reasoning_steps
    ]) if trajectory.reasoning_steps else "无推理步骤记录"
    
    # 格式化工具调用
    tool_calls_str = "\n\n".join([
        TOOL_CALL_TEMPLATE.format(
            index=i + 1,
            tool_name=tc.tool_name,
            input_params=tc.input_params,
            output_result=tc.output_result,
            status="成功" if tc.success else "失败",
            duration_ms=tc.duration_ms,
            error_info=f"- 错误信息: {tc.error}" if tc.error else ""
        )
        for i, tc in enumerate(trajectory.tool_calls)
    ]) if trajectory.tool_calls else "无工具调用记录"
    
    # 格式化中间结果
    intermediate_results_str = "\n".join([
        f"- {result}" for result in trajectory.intermediate_results
    ]) if trajectory.intermediate_results else "无中间结果记录"
    
    return TRAJECTORY_ANALYSIS_USER_PROMPT_TEMPLATE.format(
        user_input=trajectory.user_input,
        system_prompt=trajectory.system_prompt or "无系统提示",
        reasoning_steps=reasoning_steps_str,
        tool_calls=tool_calls_str,
        intermediate_results=intermediate_results_str,
        final_output=trajectory.final_output,
        start_time=trajectory.start_time,
        end_time=trajectory.end_time,
        total_duration_ms=trajectory.total_duration_ms
    )


def format_correction_prompt(trajectory, problem) -> str:
    """
    格式化问题修正提示词
    
    Args:
        trajectory: ExecutionTrajectory对象
        problem: Problem对象
        
    Returns:
        格式化后的修正提示词
    """
    # 提取问题相关的步骤
    relevant_steps = _extract_relevant_steps(trajectory, problem.location)
    
    return CORRECTION_USER_PROMPT_TEMPLATE.format(
        problem_type=problem.type.value if hasattr(problem.type, 'value') else problem.type,
        problem_description=problem.description,
        problem_location=problem.location,
        severity=problem.severity,
        user_input=trajectory.user_input,
        relevant_steps=relevant_steps,
        final_output=trajectory.final_output
    )


def _extract_relevant_steps(trajectory, location: str) -> str:
    """
    提取与问题位置相关的步骤
    
    Args:
        trajectory: ExecutionTrajectory对象
        location: 问题位置字符串
        
    Returns:
        相关步骤的格式化字符串
    """
    relevant_parts = []
    
    # 尝试解析位置信息
    location_lower = location.lower()
    
    # 检查是否涉及推理步骤
    if "步骤" in location_lower or "step" in location_lower:
        for step in trajectory.reasoning_steps:
            relevant_parts.append(
                f"推理步骤{step.step_number}: {step.content}"
            )
    
    # 检查是否涉及工具调用
    if "工具" in location_lower or "tool" in location_lower:
        for i, tc in enumerate(trajectory.tool_calls):
            status = "成功" if tc.success else f"失败({tc.error})"
            relevant_parts.append(
                f"工具调用{i+1} [{tc.tool_name}]: {status}"
            )
    
    # 如果没有匹配到特定位置，返回所有步骤摘要
    if not relevant_parts:
        if trajectory.reasoning_steps:
            relevant_parts.append("推理步骤:")
            for step in trajectory.reasoning_steps:
                relevant_parts.append(f"  {step.step_number}. {step.content[:100]}...")
        if trajectory.tool_calls:
            relevant_parts.append("工具调用:")
            for i, tc in enumerate(trajectory.tool_calls):
                relevant_parts.append(f"  {i+1}. {tc.tool_name}")
    
    return "\n".join(relevant_parts) if relevant_parts else "无相关步骤信息"

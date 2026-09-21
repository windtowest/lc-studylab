"""
Critic Agent 核心逻辑

实现执行轨迹分析、问题识别和改进建议生成。
"""

import json
import re
from datetime import datetime
from typing import Optional, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .models import ProblemType, Problem, Improvement, CriticReport
from .prompts import (
    TRAJECTORY_ANALYSIS_SYSTEM_PROMPT,
    CORRECTION_SYSTEM_PROMPT,
    format_trajectory_for_analysis,
    format_correction_prompt,
)
from ..trajectory.models import ExecutionTrajectory


class CriticAgent:
    """
    Critic反思Agent
    
    分析执行轨迹，识别问题，生成结构化的反思报告和改进建议。
    """
    
    def __init__(
        self, 
        model: BaseChatModel,
        max_retries: int = 2,
        analysis_timeout: float = 30.0
    ):
        """
        初始化Critic Agent
        
        Args:
            model: LangChain聊天模型
            max_retries: LLM调用失败时的最大重试次数
            analysis_timeout: 分析超时时间（秒）
        """
        self.model = model
        self.max_retries = max_retries
        self.analysis_timeout = analysis_timeout
    
    def analyze(self, trajectory: ExecutionTrajectory) -> CriticReport:
        """
        分析执行轨迹
        
        Args:
            trajectory: 执行轨迹对象
            
        Returns:
            CriticReport: 反思报告
        """
        timestamp = datetime.now().isoformat()
        
        # 调用LLM进行轨迹分析
        analysis_result = self._call_analysis_llm(trajectory)
        
        if analysis_result is None:
            # 分析失败，返回空报告
            return self._create_empty_report(timestamp, "分析失败")
        
        # 解析分析结果
        report = self._parse_analysis_result(analysis_result, timestamp)
        
        if report is None:
            return self._create_empty_report(timestamp, "分析结果解析失败")
        
        # 为严重问题生成修正方案（如果尚未提供）
        report = self._ensure_corrections_for_severe_problems(trajectory, report)
        
        return report
    
    def generate_correction(
        self, 
        trajectory: ExecutionTrajectory,
        problem: Problem
    ) -> str:
        """
        为特定问题生成修正方案
        
        Args:
            trajectory: 执行轨迹对象
            problem: 问题对象
            
        Returns:
            修正方案字符串
        """
        messages = [
            SystemMessage(content=CORRECTION_SYSTEM_PROMPT),
            HumanMessage(content=format_correction_prompt(trajectory, problem))
        ]
        
        try:
            result = self.model.invoke(messages)
            return result.content.strip()
        except Exception as e:
            print(f"修正方案生成失败: {e}")
            return f"无法生成修正方案: {str(e)}"
    
    def _call_analysis_llm(
        self, 
        trajectory: ExecutionTrajectory
    ) -> Optional[str]:
        """
        调用LLM进行轨迹分析
        
        Args:
            trajectory: 执行轨迹对象
            
        Returns:
            LLM输出的分析结果字符串，失败返回None
        """
        messages = [
            SystemMessage(content=TRAJECTORY_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=format_trajectory_for_analysis(trajectory))
        ]
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self.model.invoke(messages)
                return result.content
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"轨迹分析LLM调用失败: {e}")
                    return None
        
        return None
    
    def _parse_analysis_result(
        self, 
        result: str,
        timestamp: str
    ) -> Optional[CriticReport]:
        """
        解析分析结果
        
        Args:
            result: LLM输出的分析结果字符串
            timestamp: 时间戳
            
        Returns:
            CriticReport对象，解析失败返回None
        """
        try:
            # 尝试提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析整个结果
                json_str = result
            
            data = json.loads(json_str)
            
            # 解析问题列表
            problems = self._parse_problems(data.get("problems", []))
            
            # 解析根因分析
            root_causes = data.get("root_causes", [])
            if not isinstance(root_causes, list):
                root_causes = [str(root_causes)] if root_causes else []
            
            # 解析改进建议
            improvements = self._parse_improvements(data.get("improvements", []))
            
            # 获取总体评估
            overall_assessment = data.get("overall_assessment", "分析完成")
            
            return CriticReport(
                problems=problems,
                root_causes=root_causes,
                improvements=improvements,
                overall_assessment=overall_assessment,
                timestamp=timestamp
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"分析结果解析失败: {e}")
            return None
    
    def _parse_problems(self, problems_data: list) -> List[Problem]:
        """
        解析问题列表
        
        Args:
            problems_data: 问题数据列表
            
        Returns:
            Problem对象列表
        """
        problems = []
        for p in problems_data:
            try:
                # 解析问题类型
                problem_type_str = p.get("type", "logic_error")
                try:
                    problem_type = ProblemType(problem_type_str)
                except ValueError:
                    # 如果类型无效，默认为逻辑错误
                    problem_type = ProblemType.LOGIC_ERROR
                
                # 确保severity在有效范围内
                severity = int(p.get("severity", 3))
                severity = max(1, min(5, severity))
                
                problems.append(Problem(
                    type=problem_type,
                    description=p.get("description", "未知问题"),
                    severity=severity,
                    location=p.get("location", "未知位置")
                ))
            except Exception as e:
                print(f"解析问题失败: {e}")
                continue
        
        return problems
    
    def _parse_improvements(self, improvements_data: list) -> List[Improvement]:
        """
        解析改进建议列表
        
        Args:
            improvements_data: 改进建议数据列表
            
        Returns:
            Improvement对象列表
        """
        improvements = []
        for imp in improvements_data:
            try:
                # 确保priority在有效范围内
                priority = int(imp.get("priority", 3))
                priority = max(1, min(5, priority))
                
                improvements.append(Improvement(
                    problem_ref=imp.get("problem_ref", ""),
                    suggestion=imp.get("suggestion", ""),
                    priority=priority,
                    correction=imp.get("correction")
                ))
            except Exception as e:
                print(f"解析改进建议失败: {e}")
                continue
        
        # 按优先级排序（高优先级在前）
        improvements.sort(key=lambda x: x.priority, reverse=True)
        
        return improvements
    
    def _ensure_corrections_for_severe_problems(
        self,
        trajectory: ExecutionTrajectory,
        report: CriticReport
    ) -> CriticReport:
        """
        确保严重问题有修正方案
        
        Args:
            trajectory: 执行轨迹对象
            report: 原始反思报告
            
        Returns:
            更新后的反思报告
        """
        # 找出严重问题（severity >= 4）
        severe_problems = [p for p in report.problems if p.severity >= 4]
        
        if not severe_problems:
            return report
        
        # 检查是否已有对应的修正方案
        updated_improvements = list(report.improvements)
        
        for problem in severe_problems:
            # 检查是否已有修正方案
            has_correction = any(
                imp.problem_ref == problem.description and imp.correction
                for imp in updated_improvements
            )
            
            if not has_correction:
                # 生成修正方案
                correction = self.generate_correction(trajectory, problem)
                
                # 查找或创建对应的改进建议
                found = False
                for imp in updated_improvements:
                    if imp.problem_ref == problem.description:
                        # 更新现有改进建议
                        imp_index = updated_improvements.index(imp)
                        updated_improvements[imp_index] = Improvement(
                            problem_ref=imp.problem_ref,
                            suggestion=imp.suggestion,
                            priority=imp.priority,
                            correction=correction
                        )
                        found = True
                        break
                
                if not found:
                    # 创建新的改进建议
                    updated_improvements.append(Improvement(
                        problem_ref=problem.description,
                        suggestion=f"修正{problem.type.value}问题",
                        priority=5,
                        correction=correction
                    ))
        
        # 重新排序
        updated_improvements.sort(key=lambda x: x.priority, reverse=True)
        
        return CriticReport(
            problems=report.problems,
            root_causes=report.root_causes,
            improvements=updated_improvements,
            overall_assessment=report.overall_assessment,
            timestamp=report.timestamp
        )
    
    def _create_empty_report(
        self, 
        timestamp: str,
        assessment: str = "分析失败"
    ) -> CriticReport:
        """
        创建空的反思报告（分析失败时使用）
        
        Args:
            timestamp: 时间戳
            assessment: 总体评估
            
        Returns:
            空的CriticReport
        """
        return CriticReport(
            problems=[],
            root_causes=[],
            improvements=[],
            overall_assessment=assessment,
            timestamp=timestamp
        )

"""
端到端集成测试

测试完整的执行-评分-反思-存储流程和偏好数据生成导出。

Requirements: 所有需求
"""

import sys
import uuid
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

# 确保可以导入模块
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest

# 直接导入需要的模块，避免触发core/__init__.py的全部导入
from core.self_reward.models import ScoreDimension, DimensionScore, SelfRewardResult
from core.critic.models import ProblemType, Problem, Improvement, CriticReport
from core.trajectory.models import ExecutionTrajectory, ToolCall, ReasoningStep
from core.trajectory.recorder import TrajectoryRecorder
from core.memory.models import ExperienceRecord
from rlhf_integration.preference_generator import PreferenceGenerator, PreferencePair
from rlhf_integration.training_trigger import (
    TrainingTrigger, 
    TriggerConfig, 
    TrainingStatus,
    TrainingReportGenerator
)


# ==================== 辅助函数 ====================

def create_mock_self_reward_result(score: float, patch: str = None) -> SelfRewardResult:
    """创建模拟的SelfRewardResult"""
    return SelfRewardResult(
        score=score,
        reason=f"测试评分理由 (score={score})",
        patch=patch,
        dimension_scores=[
            DimensionScore(
                dimension=ScoreDimension.ACCURACY,
                score=score,
                reason="准确性评分"
            ),
            DimensionScore(
                dimension=ScoreDimension.RELEVANCE,
                score=score,
                reason="相关性评分"
            ),
            DimensionScore(
                dimension=ScoreDimension.COMPLETENESS,
                score=score,
                reason="完整性评分"
            ),
            DimensionScore(
                dimension=ScoreDimension.CLARITY,
                score=score,
                reason="清晰度评分"
            ),
        ],
        timestamp=datetime.now().isoformat()
    )


def create_mock_critic_report(num_problems: int = 0) -> CriticReport:
    """创建模拟的CriticReport"""
    problems = []
    improvements = []
    
    for i in range(num_problems):
        problems.append(Problem(
            type=ProblemType.LOGIC_ERROR,
            description=f"测试问题 {i+1}",
            severity=3,
            location=f"步骤 {i+1}"
        ))
        improvements.append(Improvement(
            problem_ref=f"测试问题 {i+1}",
            suggestion=f"改进建议 {i+1}",
            priority=3,
            correction=None
        ))
    
    return CriticReport(
        problems=problems,
        root_causes=["测试根因"] if num_problems > 0 else [],
        improvements=improvements,
        overall_assessment="测试总体评估",
        timestamp=datetime.now().isoformat()
    )


def create_mock_trajectory(
    user_input: str = "测试输入",
    final_output: str = "测试输出",
    session_id: str = None
) -> ExecutionTrajectory:
    """创建模拟的ExecutionTrajectory"""
    return ExecutionTrajectory(
        trajectory_id=str(uuid.uuid4()),
        session_id=session_id or str(uuid.uuid4()),
        user_input=user_input,
        system_prompt="测试系统提示",
        reasoning_steps=[
            ReasoningStep(
                step_number=1,
                content="测试推理步骤",
                timestamp=datetime.now().isoformat()
            )
        ],
        tool_calls=[
            ToolCall(
                tool_name="test_tool",
                input_params={"param": "value"},
                output_result="工具输出",
                duration_ms=100.0,
                timestamp=datetime.now().isoformat(),
                success=True,
                error=None
            )
        ],
        intermediate_results=["中间结果"],
        final_output=final_output,
        total_duration_ms=500.0,
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat()
    )


def create_mock_experience_record(
    score: float,
    user_input: str = "测试输入",
    final_output: str = "测试输出",
    patch: str = None,
    session_id: str = None
) -> ExperienceRecord:
    """创建模拟的ExperienceRecord"""
    session_id = session_id or str(uuid.uuid4())
    
    return ExperienceRecord(
        record_id=str(uuid.uuid4()),
        session_id=session_id,
        trajectory=create_mock_trajectory(user_input, final_output, session_id),
        self_reward=create_mock_self_reward_result(score, patch),
        critic_report=create_mock_critic_report(1 if score < 6.0 else 0),
        timestamp=datetime.now().isoformat(),
        tags=[]
    )


# ==================== 集成测试类 ====================

class TestTrajectoryRecordingFlow:
    """测试轨迹记录流程"""
    
    def test_complete_trajectory_recording_flow(self):
        """
        测试完整的轨迹记录流程
        
        验证：
        1. 开始记录
        2. 添加推理步骤
        3. 添加工具调用
        4. 完成记录
        5. 轨迹数据完整性
        """
        recorder = TrajectoryRecorder()
        session_id = str(uuid.uuid4())
        user_input = "请帮我计算1+1"
        
        # 1. 开始记录
        trajectory_id = recorder.start(
            session_id=session_id,
            user_input=user_input,
            system_prompt="你是一个计算助手"
        )
        
        assert trajectory_id is not None
        assert recorder.is_recording
        
        # 2. 添加推理步骤
        step1 = recorder.add_reasoning("分析用户请求：需要进行加法计算")
        assert step1.step_number == 1
        
        step2 = recorder.add_reasoning("准备调用计算工具")
        assert step2.step_number == 2
        
        # 3. 添加工具调用
        tool_call = recorder.add_tool_call(
            tool_name="calculator",
            input_params={"expression": "1+1"},
            output_result="2",
            duration_ms=50.0,
            success=True
        )
        assert tool_call.tool_name == "calculator"
        assert tool_call.success
        
        # 4. 添加中间结果
        recorder.add_intermediate_result("计算结果: 2")
        
        # 5. 完成记录
        final_output = "1+1的结果是2"
        trajectory = recorder.finish(final_output)
        
        # 验证轨迹完整性
        assert trajectory.trajectory_id == trajectory_id
        assert trajectory.session_id == session_id
        assert trajectory.user_input == user_input
        assert trajectory.final_output == final_output
        assert len(trajectory.reasoning_steps) == 2
        assert len(trajectory.tool_calls) == 1
        assert len(trajectory.intermediate_results) == 1
        assert trajectory.total_duration_ms > 0
        
        # 验证记录器状态已重置
        assert not recorder.is_recording
    
    def test_trajectory_serialization_roundtrip(self):
        """
        测试轨迹序列化往返
        
        验证轨迹可以正确序列化为JSON并反序列化回来
        """
        original = create_mock_trajectory(
            user_input="测试序列化",
            final_output="序列化测试输出"
        )
        
        # 序列化
        json_str = original.to_json()
        assert isinstance(json_str, str)
        
        # 反序列化
        restored = ExecutionTrajectory.from_json(json_str)
        
        # 验证关键字段
        assert restored.trajectory_id == original.trajectory_id
        assert restored.session_id == original.session_id
        assert restored.user_input == original.user_input
        assert restored.final_output == original.final_output
        assert len(restored.reasoning_steps) == len(original.reasoning_steps)
        assert len(restored.tool_calls) == len(original.tool_calls)


class TestSelfRewardScoringFlow:
    """测试Self-Reward评分流程"""
    
    def test_score_result_structure_completeness(self):
        """
        测试评分结果结构完整性
        
        验证SelfRewardResult包含所有必需字段
        """
        result = create_mock_self_reward_result(7.5)
        
        # 验证必需字段
        assert hasattr(result, 'score')
        assert hasattr(result, 'reason')
        assert hasattr(result, 'patch')
        assert hasattr(result, 'dimension_scores')
        assert hasattr(result, 'timestamp')
        
        # 验证评分范围
        assert 0.0 <= result.score <= 10.0
        
        # 验证维度评分
        assert len(result.dimension_scores) == 4
        for ds in result.dimension_scores:
            assert 0.0 <= ds.score <= 10.0
            assert ds.reason is not None
    
    def test_score_patch_consistency_high_score(self):
        """
        测试高分时patch应为None
        """
        result = create_mock_self_reward_result(8.0, patch=None)
        assert result.score >= 6.0
        assert result.patch is None
    
    def test_score_patch_consistency_low_score(self):
        """
        测试低分时应有patch
        """
        result = create_mock_self_reward_result(4.0, patch="改进后的响应内容")
        assert result.score < 6.0
        assert result.patch is not None
        assert len(result.patch) > 0


class TestCriticReflectionFlow:
    """测试Critic反思流程"""
    
    def test_critic_report_structure_completeness(self):
        """
        测试Critic报告结构完整性
        
        验证CriticReport包含所有必需字段
        """
        report = create_mock_critic_report(num_problems=2)
        
        # 验证必需字段
        assert hasattr(report, 'problems')
        assert hasattr(report, 'root_causes')
        assert hasattr(report, 'improvements')
        assert hasattr(report, 'overall_assessment')
        assert hasattr(report, 'timestamp')
        
        # 验证问题列表
        assert len(report.problems) == 2
        for problem in report.problems:
            assert hasattr(problem, 'type')
            assert hasattr(problem, 'description')
            assert hasattr(problem, 'severity')
            assert 1 <= problem.severity <= 5
        
        # 验证改进建议
        assert len(report.improvements) == 2
        for imp in report.improvements:
            assert hasattr(imp, 'suggestion')
            assert hasattr(imp, 'priority')
            assert 1 <= imp.priority <= 5


class TestExperienceStorageFlow:
    """测试经验存储流程"""
    
    @pytest.fixture(autouse=True)
    def setup_store(self):
        """设置测试用的ExperienceStore"""
        try:
            from core.memory.experience_store import ExperienceStore
            self.store = ExperienceStore()
            self.created_record_ids = []
            yield
        except Exception as e:
            pytest.skip(f"无法连接数据库: {e}")
        finally:
            # 清理测试数据
            for record_id in self.created_record_ids:
                try:
                    self.store.delete(record_id)
                except Exception:
                    pass
    
    def test_experience_record_save_and_retrieve(self):
        """
        测试经验记录的保存和检索
        """
        record = create_mock_experience_record(
            score=7.5,
            user_input="测试保存检索",
            final_output="保存检索测试输出"
        )
        
        # 保存
        saved_id = self.store.save(record)
        self.created_record_ids.append(saved_id)
        
        assert saved_id == record.record_id
        
        # 检索
        retrieved = self.store.get(saved_id)
        
        assert retrieved is not None
        assert retrieved.record_id == record.record_id
        assert retrieved.session_id == record.session_id
        assert retrieved.trajectory.user_input == record.trajectory.user_input
        assert retrieved.self_reward.score == record.self_reward.score
    
    def test_experience_record_query_by_score(self):
        """
        测试按评分查询经验记录
        """
        # 创建高分和低分记录
        high_score_record = create_mock_experience_record(score=8.5)
        low_score_record = create_mock_experience_record(score=4.5)
        
        self.store.save(high_score_record)
        self.created_record_ids.append(high_score_record.record_id)
        
        self.store.save(low_score_record)
        self.created_record_ids.append(low_score_record.record_id)
        
        # 查询高分记录
        high_results = self.store.query(min_score=7.0)
        high_ids = [r.record_id for r in high_results]
        assert high_score_record.record_id in high_ids
        
        # 查询低分记录
        low_results = self.store.query(max_score=6.0)
        low_ids = [r.record_id for r in low_results]
        assert low_score_record.record_id in low_ids


class TestPreferenceDataGenerationFlow:
    """测试偏好数据生成流程"""
    
    @pytest.fixture(autouse=True)
    def setup_store(self):
        """设置测试用的ExperienceStore和PreferenceGenerator"""
        try:
            from core.memory.experience_store import ExperienceStore
            self.store = ExperienceStore()
            self.generator = PreferenceGenerator(
                self.store,
                chosen_threshold=7.0,
                rejected_threshold=6.0
            )
            self.created_record_ids = []
            yield
        except Exception as e:
            pytest.skip(f"无法连接数据库: {e}")
        finally:
            # 清理测试数据
            for record_id in self.created_record_ids:
                try:
                    self.store.delete(record_id)
                except Exception:
                    pass
    
    def test_preference_pair_generation(self):
        """
        测试偏好数据对生成
        
        验证从高分和低分记录生成正确的偏好对
        """
        session_id = str(uuid.uuid4())
        user_input = "测试偏好生成"
        
        # 创建高分记录
        high_record = create_mock_experience_record(
            score=8.0,
            user_input=user_input,
            final_output="高质量响应",
            session_id=session_id
        )
        self.store.save(high_record)
        self.created_record_ids.append(high_record.record_id)
        
        # 创建低分记录（相同输入）
        low_record = create_mock_experience_record(
            score=4.0,
            user_input=user_input,
            final_output="低质量响应",
            session_id=session_id
        )
        self.store.save(low_record)
        self.created_record_ids.append(low_record.record_id)
        
        # 生成偏好对
        pairs = self.generator.generate_pairs()
        
        # 验证生成的偏好对
        if pairs:
            for pair in pairs:
                # 验证chosen_score >= chosen_threshold
                assert pair.chosen_score >= self.generator.chosen_threshold
                # 验证rejected_score < rejected_threshold
                assert pair.rejected_score < self.generator.rejected_threshold
                # 验证source_records存在
                assert len(pair.source_records) == 2
    
    def test_preference_data_export(self):
        """
        测试偏好数据导出
        """
        session_id = str(uuid.uuid4())
        user_input = "测试导出"
        
        # 创建测试记录
        high_record = create_mock_experience_record(
            score=8.5,
            user_input=user_input,
            final_output="高质量导出测试",
            session_id=session_id
        )
        self.store.save(high_record)
        self.created_record_ids.append(high_record.record_id)
        
        low_record = create_mock_experience_record(
            score=3.5,
            user_input=user_input,
            final_output="低质量导出测试",
            session_id=session_id
        )
        self.store.save(low_record)
        self.created_record_ids.append(low_record.record_id)
        
        # 导出到临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            export_path = f.name
        
        try:
            count = self.generator.export_for_rm_training(export_path)
            
            # 验证导出文件
            if count > 0:
                with open(export_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                assert isinstance(data, list)
                for item in data:
                    assert 'prompt' in item
                    assert 'chosen' in item
                    assert 'rejected' in item
                    assert 'metadata' in item
                    assert item['metadata']['source'] == 'self_reward_agent'
        finally:
            Path(export_path).unlink(missing_ok=True)
    
    def test_preference_statistics(self):
        """
        测试偏好数据统计
        """
        stats = self.generator.get_statistics()
        
        # 验证统计信息结构
        assert 'total_records' in stats
        assert 'high_score_records' in stats
        assert 'low_score_records' in stats
        assert 'chosen_threshold' in stats
        assert 'rejected_threshold' in stats
        assert 'generated_pairs' in stats


class TestTrainingTriggerFlow:
    """测试训练触发流程"""
    
    @pytest.fixture(autouse=True)
    def setup_trigger(self):
        """设置测试用的TrainingTrigger"""
        try:
            from core.memory.experience_store import ExperienceStore
            self.store = ExperienceStore()
            self.config = TriggerConfig(
                min_data_count=5,  # 低阈值便于测试
                auto_trigger=False,
                export_path="data/test_preference_data"
            )
            self.trigger = TrainingTrigger(
                experience_store=self.store,
                config=self.config
            )
            yield
        except Exception as e:
            pytest.skip(f"无法初始化训练触发器: {e}")
    
    def test_check_data_ready(self):
        """
        测试数据就绪检查
        """
        result = self.trigger.check_data_ready()
        
        # 验证结果结构
        assert hasattr(result, 'triggered')
        assert hasattr(result, 'status')
        assert hasattr(result, 'data_count')
        assert hasattr(result, 'min_required')
        assert hasattr(result, 'message')
        assert hasattr(result, 'timestamp')
        
        # 验证状态
        assert result.status in [TrainingStatus.PENDING, TrainingStatus.DATA_READY]
        assert result.min_required == self.config.min_data_count
    
    def test_integration_interface(self):
        """
        测试集成接口信息
        """
        interface = self.trigger.get_integration_interface()
        
        # 验证接口信息结构
        assert 'data_format' in interface
        assert 'export_method' in interface
        assert 'trigger_method' in interface
        assert 'check_method' in interface
        assert 'config' in interface
        assert 'current_status' in interface
        
        # 验证数据格式说明
        data_format = interface['data_format']
        assert 'schema' in data_format
        schema = data_format['schema']
        assert 'prompt' in schema
        assert 'chosen' in schema
        assert 'rejected' in schema


class TestTrainingReportFlow:
    """测试训练报告流程"""
    
    @pytest.fixture(autouse=True)
    def setup_report_generator(self):
        """设置测试用的TrainingReportGenerator"""
        try:
            from core.memory.experience_store import ExperienceStore
            self.store = ExperienceStore()
            self.trigger = TrainingTrigger(experience_store=self.store)
            self.report_generator = TrainingReportGenerator(
                training_trigger=self.trigger,
                report_output_dir="data/test_training_reports"
            )
            self.created_reports = []
            yield
        except Exception as e:
            pytest.skip(f"无法初始化报告生成器: {e}")
        finally:
            # 清理测试报告
            for report_id in self.created_reports:
                try:
                    report_path = Path(f"data/test_training_reports/{report_id}.json")
                    report_path.unlink(missing_ok=True)
                except Exception:
                    pass
    
    def test_training_session_lifecycle(self):
        """
        测试训练会话生命周期
        """
        # 开始会话
        report_id = self.report_generator.start_training_session()
        self.created_reports.append(report_id)
        
        assert report_id is not None
        assert report_id.startswith("report_")
        
        # 添加备注
        self.report_generator.add_note("测试备注1")
        self.report_generator.add_note("测试备注2")
        
        # 记录完成
        self.report_generator.record_training_completion(
            success=True,
            improvement_indicators={"accuracy_improvement": 0.05},
            notes=["完成备注"]
        )
        
        # 生成报告
        report = self.report_generator.generate_report()
        
        # 验证报告
        assert report.report_id == report_id
        assert report.training_status == TrainingStatus.COMPLETED
        assert len(report.notes) >= 2
        assert "accuracy_improvement" in report.improvement_indicators
    
    def test_report_save_and_load(self):
        """
        测试报告保存和加载
        """
        # 创建并保存报告
        report_id = self.report_generator.start_training_session()
        self.created_reports.append(report_id)
        
        self.report_generator.record_training_completion(success=True)
        report = self.report_generator.generate_report()
        
        saved_path = self.report_generator.save_report(report)
        assert Path(saved_path).exists()
        
        # 加载报告
        loaded_report = self.report_generator.load_report(report_id)
        
        assert loaded_report.report_id == report.report_id
        assert loaded_report.training_status == report.training_status


class TestEndToEndIntegration:
    """端到端集成测试"""
    
    @pytest.fixture(autouse=True)
    def setup_all(self):
        """设置所有测试组件"""
        try:
            from core.memory.experience_store import ExperienceStore
            self.store = ExperienceStore()
            self.generator = PreferenceGenerator(
                self.store,
                chosen_threshold=7.0,
                rejected_threshold=6.0
            )
            self.trigger = TrainingTrigger(experience_store=self.store)
            self.report_generator = TrainingReportGenerator(
                training_trigger=self.trigger
            )
            self.created_record_ids = []
            self.created_reports = []
            yield
        except Exception as e:
            pytest.skip(f"无法初始化测试组件: {e}")
        finally:
            # 清理
            for record_id in self.created_record_ids:
                try:
                    self.store.delete(record_id)
                except Exception:
                    pass
    
    def test_complete_workflow(self):
        """
        测试完整工作流程
        
        1. 创建轨迹记录
        2. 评分
        3. Critic反思
        4. 存储经验
        5. 生成偏好数据
        6. 检查训练就绪状态
        """
        session_id = str(uuid.uuid4())
        
        # 1. 创建轨迹
        recorder = TrajectoryRecorder()
        recorder.start(
            session_id=session_id,
            user_input="完整流程测试输入",
            system_prompt="测试系统提示"
        )
        recorder.add_reasoning("测试推理")
        trajectory = recorder.finish("完整流程测试输出")
        
        assert trajectory is not None
        assert trajectory.session_id == session_id
        
        # 2. 创建评分结果
        high_score_result = create_mock_self_reward_result(8.0)
        low_score_result = create_mock_self_reward_result(4.0, patch="改进内容")
        
        # 3. 创建Critic报告
        critic_report = create_mock_critic_report(num_problems=1)
        
        # 4. 存储经验记录
        high_record = ExperienceRecord(
            record_id=str(uuid.uuid4()),
            session_id=session_id,
            trajectory=trajectory,
            self_reward=high_score_result,
            critic_report=critic_report,
            timestamp=datetime.now().isoformat(),
            tags=["test"]
        )
        self.store.save(high_record)
        self.created_record_ids.append(high_record.record_id)
        
        # 创建低分记录用于偏好对
        low_trajectory = create_mock_trajectory(
            user_input="完整流程测试输入",
            final_output="低质量输出",
            session_id=session_id
        )
        low_record = ExperienceRecord(
            record_id=str(uuid.uuid4()),
            session_id=session_id,
            trajectory=low_trajectory,
            self_reward=low_score_result,
            critic_report=critic_report,
            timestamp=datetime.now().isoformat(),
            tags=["test"]
        )
        self.store.save(low_record)
        self.created_record_ids.append(low_record.record_id)
        
        # 5. 验证记录已存储
        retrieved = self.store.get(high_record.record_id)
        assert retrieved is not None
        
        # 6. 获取统计信息
        stats = self.generator.get_statistics()
        assert 'total_records' in stats
        
        # 7. 检查训练就绪状态
        check_result = self.trigger.check_data_ready()
        assert check_result is not None
        assert check_result.status in [TrainingStatus.PENDING, TrainingStatus.DATA_READY]

"""
RLHF 集成模块

该模块提供与现有RLHF训练流程的集成能力，包括偏好数据生成和训练触发。
"""

from rlhf_integration.preference_generator import PreferenceGenerator, PreferencePair
from rlhf_integration.training_trigger import (
    TrainingTrigger,
    TrainingStatus,
    TriggerConfig,
    TriggerResult,
    TrainingMetrics,
    TrainingReport,
    TrainingReportGenerator,
)

__all__ = [
    "PreferenceGenerator",
    "PreferencePair",
    "TrainingTrigger",
    "TrainingStatus",
    "TriggerConfig",
    "TriggerResult",
    "TrainingMetrics",
    "TrainingReport",
    "TrainingReportGenerator",
]

"""
Memory 数据模型

定义经验记录相关的数据结构。
"""

from pydantic import BaseModel
from typing import List

from core.trajectory.models import ExecutionTrajectory
from core.self_reward.models import SelfRewardResult
from core.critic.models import CriticReport


class ExperienceRecord(BaseModel):
    """经验记录"""
    record_id: str
    session_id: str
    trajectory: ExecutionTrajectory
    self_reward: SelfRewardResult
    critic_report: CriticReport
    timestamp: str
    tags: List[str] = []

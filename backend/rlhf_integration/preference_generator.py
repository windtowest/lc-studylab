"""
偏好数据生成器

从经验记录中提取偏好数据对，用于训练奖励模型。
支持批量导出为与现有RM训练兼容的格式。
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path

from pydantic import BaseModel

from core.memory.experience_store import ExperienceStore
from core.memory.models import ExperienceRecord

logger = logging.getLogger(__name__)


class PreferencePair(BaseModel):
    """偏好数据对"""
    prompt: str                              # 用户输入/提示
    chosen: str                              # 高质量响应
    rejected: str                            # 低质量响应
    chosen_score: float                      # chosen响应的评分
    rejected_score: float                    # rejected响应的评分
    source_records: Tuple[str, str]          # (chosen_record_id, rejected_record_id)


class PreferenceGenerator:
    """
    偏好数据生成器
    
    从经验存储中提取高分和低分响应，构造偏好数据对用于RM训练。
    """
    
    def __init__(
        self, 
        experience_store: ExperienceStore,
        chosen_threshold: float = 7.0,
        rejected_threshold: float = 6.0
    ):
        """
        初始化偏好数据生成器
        
        Args:
            experience_store: 经验存储实例
            chosen_threshold: chosen响应的最低评分阈值（>=此分数或来自patch修正）
            rejected_threshold: rejected响应的最高评分阈值（<此分数）
        """
        self.store = experience_store
        self.chosen_threshold = chosen_threshold
        self.rejected_threshold = rejected_threshold
    
    def generate_pairs(self, limit: Optional[int] = None) -> List[PreferencePair]:
        """
        生成偏好数据对
        
        从经验存储中查询高分和低分记录，匹配相似输入构造偏好对。
        
        Args:
            limit: 生成数量限制，None表示不限制
            
        Returns:
            偏好数据对列表
        """
        # 查询高分记录（chosen候选）
        chosen_records = self._get_chosen_candidates()
        
        # 查询低分记录（rejected候选）
        rejected_records = self._get_rejected_candidates()
        
        if not chosen_records or not rejected_records:
            logger.warning(
                f"数据不足: chosen={len(chosen_records)}, rejected={len(rejected_records)}"
            )
            return []
        
        # 匹配生成偏好对
        pairs = self._match_pairs(chosen_records, rejected_records, limit)
        
        logger.info(f"生成偏好数据对: {len(pairs)} 条")
        return pairs
    
    def _get_chosen_candidates(self) -> List[ExperienceRecord]:
        """
        获取chosen候选记录
        
        包括：
        1. 评分 >= chosen_threshold 的记录
        2. 有patch修正的记录（patch作为chosen）
        """
        # 查询高分记录
        high_score_records = self.store.query(
            min_score=self.chosen_threshold,
            limit=1000
        )
        
        # 查询有patch的低分记录（patch可作为chosen）
        low_score_with_patch = self.store.query(
            max_score=self.rejected_threshold,
            limit=1000
        )
        # 过滤出有patch的记录
        patch_records = [
            r for r in low_score_with_patch 
            if r.self_reward.patch is not None and r.self_reward.patch.strip()
        ]
        
        return high_score_records + patch_records
    
    def _get_rejected_candidates(self) -> List[ExperienceRecord]:
        """获取rejected候选记录（评分 < rejected_threshold）"""
        return self.store.query(
            max_score=self.rejected_threshold,
            limit=1000
        )
    
    def _match_pairs(
        self, 
        chosen_records: List[ExperienceRecord],
        rejected_records: List[ExperienceRecord],
        limit: Optional[int]
    ) -> List[PreferencePair]:
        """
        匹配chosen和rejected记录生成偏好对
        
        匹配策略：
        1. 优先匹配相同session_id的记录
        2. 其次匹配相似user_input的记录
        """
        pairs = []
        used_rejected_ids = set()
        
        for chosen_record in chosen_records:
            if limit is not None and len(pairs) >= limit:
                break
            
            # 确定chosen响应内容
            chosen_response = self._get_chosen_response(chosen_record)
            chosen_score = self._get_chosen_score(chosen_record)
            
            # 查找匹配的rejected记录
            matched_rejected = self._find_matching_rejected(
                chosen_record, 
                rejected_records, 
                used_rejected_ids
            )
            
            if matched_rejected is None:
                continue
            
            used_rejected_ids.add(matched_rejected.record_id)
            
            pair = PreferencePair(
                prompt=chosen_record.trajectory.user_input,
                chosen=chosen_response,
                rejected=matched_rejected.trajectory.final_output,
                chosen_score=chosen_score,
                rejected_score=matched_rejected.self_reward.score,
                source_records=(chosen_record.record_id, matched_rejected.record_id)
            )
            pairs.append(pair)
        
        return pairs
    
    def _get_chosen_response(self, record: ExperienceRecord) -> str:
        """
        获取chosen响应内容
        
        如果记录有patch且评分低于阈值，使用patch作为chosen；
        否则使用原始final_output。
        """
        if (record.self_reward.score < self.chosen_threshold 
            and record.self_reward.patch is not None 
            and record.self_reward.patch.strip()):
            return record.self_reward.patch
        return record.trajectory.final_output
    
    def _get_chosen_score(self, record: ExperienceRecord) -> float:
        """
        获取chosen响应的评分
        
        如果使用patch作为chosen，返回一个假定的高分（patch被认为是改进后的版本）
        """
        if (record.self_reward.score < self.chosen_threshold 
            and record.self_reward.patch is not None 
            and record.self_reward.patch.strip()):
            # patch被认为是改进后的版本，给予高于阈值的分数
            return self.chosen_threshold + 1.0
        return record.self_reward.score
    
    def _find_matching_rejected(
        self,
        chosen_record: ExperienceRecord,
        rejected_records: List[ExperienceRecord],
        used_ids: set
    ) -> Optional[ExperienceRecord]:
        """
        查找与chosen记录匹配的rejected记录
        
        匹配优先级：
        1. 相同session_id
        2. 相同user_input
        3. 相似user_input（简单的词重叠匹配）
        """
        chosen_input = chosen_record.trajectory.user_input.lower()
        chosen_session = chosen_record.session_id
        
        best_match = None
        best_score = -1
        
        for rejected in rejected_records:
            if rejected.record_id in used_ids:
                continue
            
            # 不能是同一条记录
            if rejected.record_id == chosen_record.record_id:
                continue
            
            match_score = 0
            
            # 相同session_id加分
            if rejected.session_id == chosen_session:
                match_score += 10
            
            # 相同user_input加分
            rejected_input = rejected.trajectory.user_input.lower()
            if rejected_input == chosen_input:
                match_score += 20
            else:
                # 计算词重叠相似度
                similarity = self._compute_similarity(chosen_input, rejected_input)
                match_score += similarity * 5
            
            if match_score > best_score:
                best_score = match_score
                best_match = rejected
        
        # 只有当匹配分数足够高时才返回
        if best_score >= 5:
            return best_match
        return None
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（基于词重叠）
        
        Returns:
            0-1之间的相似度分数
        """
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def export_for_rm_training(self, output_path: str) -> int:
        """
        导出为RM训练格式
        
        导出格式与现有RM训练兼容：
        {
            "prompt": str,
            "chosen": str,
            "rejected": str,
            "metadata": {
                "chosen_score": float,
                "rejected_score": float,
                "source": "self_reward_agent"
            }
        }
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            导出的数据条数
        """
        pairs = self.generate_pairs()
        
        if not pairs:
            logger.warning("没有可导出的偏好数据")
            return 0
        
        # 转换为RM训练格式
        export_data = []
        for pair in pairs:
            item = {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "metadata": {
                    "chosen_score": pair.chosen_score,
                    "rejected_score": pair.rejected_score,
                    "source": "self_reward_agent",
                    "source_records": list(pair.source_records)
                }
            }
            export_data.append(item)
        
        # 确保输出目录存在
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"导出偏好数据到 {output_path}: {len(export_data)} 条")
        return len(export_data)
    
    def get_statistics(self) -> dict:
        """
        获取偏好数据统计信息
        
        Returns:
            统计信息字典
        """
        # 获取总记录数
        total_records = self.store.count()
        
        # 获取高分记录数
        high_score_count = self.store.count({"min_score": self.chosen_threshold})
        
        # 获取低分记录数
        low_score_count = self.store.count({"max_score": self.rejected_threshold})
        
        # 获取有patch的低分记录
        low_score_records = self.store.query(
            max_score=self.rejected_threshold,
            limit=1000
        )
        patch_count = sum(
            1 for r in low_score_records 
            if r.self_reward.patch is not None and r.self_reward.patch.strip()
        )
        
        # 生成偏好对并统计
        pairs = self.generate_pairs()
        
        # 计算平均分数差
        avg_score_diff = 0.0
        if pairs:
            score_diffs = [p.chosen_score - p.rejected_score for p in pairs]
            avg_score_diff = sum(score_diffs) / len(score_diffs)
        
        return {
            "total_records": total_records,
            "high_score_records": high_score_count,
            "low_score_records": low_score_count,
            "records_with_patch": patch_count,
            "chosen_threshold": self.chosen_threshold,
            "rejected_threshold": self.rejected_threshold,
            "generated_pairs": len(pairs),
            "average_score_difference": round(avg_score_diff, 2),
            "timestamp": datetime.now().isoformat()
        }

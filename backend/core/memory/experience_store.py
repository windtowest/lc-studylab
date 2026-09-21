"""
经验存储模块

提供经验记录的持久化存储、检索和导出功能。
使用PostgreSQL作为后端存储。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json

from config.settings import settings
from core.memory.models import ExperienceRecord
from core.critic.models import ProblemType

logger = logging.getLogger(__name__)


class ExperienceStore:
    """
    经验存储类
    
    提供经验记录的CRUD操作、多条件查询、相似检索和数据导出功能。
    """
    
    _connection_pool: Optional[pool.ThreadedConnectionPool] = None
    
    def __init__(self, db_uri: str = None):
        """
        初始化经验存储
        
        Args:
            db_uri: PostgreSQL连接URI，如果为None则从settings读取
        """
        self.db_uri = db_uri or settings.db_uri
        self._ensure_pool()
        self._ensure_table()
    
    def _ensure_pool(self):
        """确保连接池已初始化"""
        if ExperienceStore._connection_pool is None:
            try:
                ExperienceStore._connection_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=settings.db_pool_size,
                    dsn=self.db_uri
                )
                logger.info("数据库连接池初始化成功")
            except Exception as e:
                logger.error(f"数据库连接池初始化失败: {e}")
                raise
    
    def _get_connection(self):
        """从连接池获取连接"""
        if ExperienceStore._connection_pool is None:
            self._ensure_pool()
        return ExperienceStore._connection_pool.getconn()
    
    def _put_connection(self, conn):
        """归还连接到连接池"""
        if ExperienceStore._connection_pool is not None:
            ExperienceStore._connection_pool.putconn(conn)
    
    def _ensure_table(self):
        """确保数据库表已创建"""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            logger.warning(f"Schema文件不存在: {schema_path}")
            return
        
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    cur.execute(f.read())
            conn.commit()
            logger.info("数据库表结构已确保存在")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"创建数据库表失败: {e}")
            raise
        finally:
            if conn:
                self._put_connection(conn)
    
    def save(self, record: ExperienceRecord) -> str:
        """
        保存经验记录
        
        Args:
            record: 经验记录对象
            
        Returns:
            record_id: 记录ID
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO experience_records (
                        record_id, session_id, trajectory, self_reward, 
                        critic_report, timestamp, tags, user_input, 
                        final_output, score
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (record_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        trajectory = EXCLUDED.trajectory,
                        self_reward = EXCLUDED.self_reward,
                        critic_report = EXCLUDED.critic_report,
                        timestamp = EXCLUDED.timestamp,
                        tags = EXCLUDED.tags,
                        user_input = EXCLUDED.user_input,
                        final_output = EXCLUDED.final_output,
                        score = EXCLUDED.score
                """, (
                    record.record_id,
                    record.session_id,
                    Json(record.trajectory.model_dump()),
                    Json(record.self_reward.model_dump()),
                    Json(record.critic_report.model_dump()),
                    record.timestamp,
                    record.tags,
                    record.trajectory.user_input,
                    record.trajectory.final_output,
                    record.self_reward.score
                ))
            conn.commit()
            logger.info(f"保存经验记录成功: {record.record_id}")
            return record.record_id
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"保存经验记录失败: {e}")
            raise
        finally:
            if conn:
                self._put_connection(conn)
    
    def get(self, record_id: str) -> Optional[ExperienceRecord]:
        """
        获取单条经验记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            经验记录对象，如果不存在则返回None
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT record_id, session_id, trajectory, self_reward,
                           critic_report, timestamp, tags
                    FROM experience_records
                    WHERE record_id = %s
                """, (record_id,))
                row = cur.fetchone()
                
            if row is None:
                return None
            
            return self._row_to_record(row)
        except Exception as e:
            logger.error(f"获取经验记录失败: {e}")
            raise
        finally:
            if conn:
                self._put_connection(conn)
    
    def query(
        self,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        problem_types: Optional[List[ProblemType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ExperienceRecord]:
        """
        多条件查询经验记录
        
        Args:
            min_score: 最低评分
            max_score: 最高评分
            problem_types: 问题类型列表
            start_time: 开始时间
            end_time: 结束时间
            tags: 标签列表
            session_id: 会话ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            经验记录列表
        """
        conn = None
        try:
            conn = self._get_connection()
            
            conditions = []
            params = []
            
            if min_score is not None:
                conditions.append("score >= %s")
                params.append(min_score)
            
            if max_score is not None:
                conditions.append("score <= %s")
                params.append(max_score)
            
            if start_time is not None:
                conditions.append("timestamp >= %s")
                params.append(start_time)
            
            if end_time is not None:
                conditions.append("timestamp <= %s")
                params.append(end_time)
            
            if session_id is not None:
                conditions.append("session_id = %s")
                params.append(session_id)
            
            if tags is not None and len(tags) > 0:
                conditions.append("tags && %s")
                params.append(tags)
            
            if problem_types is not None and len(problem_types) > 0:
                # 查询包含指定问题类型的记录
                type_values = [pt.value for pt in problem_types]
                conditions.append("""
                    EXISTS (
                        SELECT 1 FROM jsonb_array_elements(critic_report->'problems') AS p
                        WHERE p->>'type' = ANY(%s)
                    )
                """)
                params.append(type_values)
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            query = f"""
                SELECT record_id, session_id, trajectory, self_reward,
                       critic_report, timestamp, tags
                FROM experience_records
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            
            return [self._row_to_record(row) for row in rows]
        except Exception as e:
            logger.error(f"查询经验记录失败: {e}")
            raise
        finally:
            if conn:
                self._put_connection(conn)
    
    def search_similar(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[ExperienceRecord]:
        """
        检索相似历史经验
        
        基于用户输入的全文检索，返回最相似的历史经验记录。
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            相似经验记录列表
        """
        conn = None
        try:
            conn = self._get_connection()
            
            # 使用PostgreSQL全文检索
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT record_id, session_id, trajectory, self_reward,
                           critic_report, timestamp, tags,
                           ts_rank(to_tsvector('simple', user_input), 
                                   plainto_tsquery('simple', %s)) AS rank
                    FROM experience_records
                    WHERE to_tsvector('simple', user_input) @@ plainto_tsquery('simple', %s)
                       OR to_tsvector('simple', final_output) @@ plainto_tsquery('simple', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """, (query, query, query, top_k))
                rows = cur.fetchall()
            
            return [self._row_to_record(row) for row in rows]
        except Exception as e:
            logger.error(f"检索相似经验失败: {e}")
            raise
        finally:
            if conn:
                self._put_connection(conn)
    
    def export(
        self, 
        format: str = "json",
        filters: dict = None
    ) -> str:
        """
        导出经验数据
        
        Args:
            format: 导出格式，支持 "json"
            filters: 过滤条件字典
            
        Returns:
            导出的数据字符串
        """
        # 应用过滤条件查询
        query_params = filters or {}
        records = self.query(**query_params)
        
        if format == "json":
            data = [record.model_dump() for record in records]
            return json.dumps(data, ensure_ascii=False, indent=2, default=str)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def count(self, filters: dict = None) -> int:
        """
        获取记录总数
        
        Args:
            filters: 过滤条件字典
            
        Returns:
            记录数量
        """
        conn = None
        try:
            conn = self._get_connection()
            
            conditions = []
            params = []
            
            if filters:
                if filters.get("min_score") is not None:
                    conditions.append("score >= %s")
                    params.append(filters["min_score"])
                
                if filters.get("max_score") is not None:
                    conditions.append("score <= %s")
                    params.append(filters["max_score"])
                
                if filters.get("session_id") is not None:
                    conditions.append("session_id = %s")
                    params.append(filters["session_id"])
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            query = f"SELECT COUNT(*) FROM experience_records {where_clause}"
            
            with conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchone()
            
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"获取记录数量失败: {e}")
            raise
        finally:
            if conn:
                self._put_connection(conn)
    
    def delete(self, record_id: str) -> bool:
        """
        删除经验记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            是否删除成功
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM experience_records WHERE record_id = %s",
                    (record_id,)
                )
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"删除经验记录失败: {e}")
            raise
        finally:
            if conn:
                self._put_connection(conn)
    
    def _row_to_record(self, row: dict) -> ExperienceRecord:
        """将数据库行转换为ExperienceRecord对象"""
        from core.trajectory.models import ExecutionTrajectory
        from core.self_reward.models import SelfRewardResult
        from core.critic.models import CriticReport
        
        # 处理timestamp格式
        timestamp = row['timestamp']
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        
        return ExperienceRecord(
            record_id=row['record_id'],
            session_id=row['session_id'],
            trajectory=ExecutionTrajectory.model_validate(row['trajectory']),
            self_reward=SelfRewardResult.model_validate(row['self_reward']),
            critic_report=CriticReport.model_validate(row['critic_report']),
            timestamp=timestamp,
            tags=row['tags'] or []
        )
    
    @classmethod
    def close_pool(cls):
        """关闭连接池"""
        if cls._connection_pool is not None:
            cls._connection_pool.closeall()
            cls._connection_pool = None
            logger.info("数据库连接池已关闭")

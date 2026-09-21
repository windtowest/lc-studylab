-- Experience Records 表结构
-- 用于存储Agent执行的完整经验记录，包括轨迹、评分和反思报告

-- 创建experience_records表
CREATE TABLE IF NOT EXISTS experience_records (
    -- 主键
    record_id VARCHAR(64) PRIMARY KEY,
    
    -- 会话标识
    session_id VARCHAR(64) NOT NULL,
    
    -- 轨迹数据（JSON格式存储ExecutionTrajectory）
    trajectory JSONB NOT NULL,
    
    -- Self-Reward评分结果（JSON格式存储SelfRewardResult）
    self_reward JSONB NOT NULL,
    
    -- Critic反思报告（JSON格式存储CriticReport）
    critic_report JSONB NOT NULL,
    
    -- 时间戳
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- 标签（用于分类和检索）
    tags TEXT[] DEFAULT '{}',
    
    -- 用户输入（冗余存储，便于全文检索）
    user_input TEXT NOT NULL,
    
    -- 最终输出（冗余存储，便于全文检索）
    final_output TEXT NOT NULL,
    
    -- Self-Reward总分（冗余存储，便于索引查询）
    score FLOAT NOT NULL,
    
    -- 创建时间
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- 更新时间
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引

-- 评分索引（用于按评分范围查询）
CREATE INDEX IF NOT EXISTS idx_experience_records_score 
ON experience_records (score);

-- 时间索引（用于按时间范围查询）
CREATE INDEX IF NOT EXISTS idx_experience_records_timestamp 
ON experience_records (timestamp);

-- 会话索引（用于按会话查询）
CREATE INDEX IF NOT EXISTS idx_experience_records_session_id 
ON experience_records (session_id);

-- 标签索引（用于按标签查询，使用GIN索引支持数组查询）
CREATE INDEX IF NOT EXISTS idx_experience_records_tags 
ON experience_records USING GIN (tags);

-- 问题类型索引（从critic_report中提取问题类型，使用GIN索引）
CREATE INDEX IF NOT EXISTS idx_experience_records_problem_types 
ON experience_records USING GIN ((critic_report->'problems'));

-- 全文检索索引（用于搜索相似历史经验）
CREATE INDEX IF NOT EXISTS idx_experience_records_user_input_fts 
ON experience_records USING GIN (to_tsvector('simple', user_input));

CREATE INDEX IF NOT EXISTS idx_experience_records_final_output_fts 
ON experience_records USING GIN (to_tsvector('simple', final_output));

-- 复合索引（评分+时间，用于常见查询场景）
CREATE INDEX IF NOT EXISTS idx_experience_records_score_timestamp 
ON experience_records (score, timestamp DESC);

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_experience_records_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 创建触发器
DROP TRIGGER IF EXISTS trigger_update_experience_records_updated_at ON experience_records;
CREATE TRIGGER trigger_update_experience_records_updated_at
    BEFORE UPDATE ON experience_records
    FOR EACH ROW
    EXECUTE FUNCTION update_experience_records_updated_at();

-- 添加注释
COMMENT ON TABLE experience_records IS '经验记录表，存储Agent执行的完整经验数据';
COMMENT ON COLUMN experience_records.record_id IS '记录唯一标识';
COMMENT ON COLUMN experience_records.session_id IS '会话标识';
COMMENT ON COLUMN experience_records.trajectory IS '执行轨迹（JSON格式）';
COMMENT ON COLUMN experience_records.self_reward IS 'Self-Reward评分结果（JSON格式）';
COMMENT ON COLUMN experience_records.critic_report IS 'Critic反思报告（JSON格式）';
COMMENT ON COLUMN experience_records.timestamp IS '记录时间戳';
COMMENT ON COLUMN experience_records.tags IS '标签数组';
COMMENT ON COLUMN experience_records.user_input IS '用户输入（用于全文检索）';
COMMENT ON COLUMN experience_records.final_output IS '最终输出（用于全文检索）';
COMMENT ON COLUMN experience_records.score IS 'Self-Reward总分（用于索引查询）';

#!/usr/bin/env python3
"""测试 PostgreSQL 连接和 checkpointer"""

from langgraph.checkpoint.postgres import PostgresSaver
from config.settings import settings

print(f"连接到: {settings.db_uri}")

# 测试连接
with PostgresSaver.from_conn_string(settings.db_uri) as checkpointer:
    print("PostgreSQL 连接成功！")
    
    # 初始化表结构
    checkpointer.setup()
    print("表结构初始化成功！")
    
    # 查询表
    with checkpointer.conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()
        
        print(f"\n数据库中的表 ({len(tables)} 个):")
        for table in tables:
            print(f"   - {table}")

print("\n所有测试通过！")

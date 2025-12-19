#!/usr/bin/env python3
"""检查 PostgreSQL 连接状态"""

import psycopg
from config.settings import settings

print("检查 PostgreSQL 连接状态...\n")

try:
    # 连接到数据库
    with psycopg.connect(settings.db_uri) as conn:
        with conn.cursor() as cur:
            # 查询当前数据库的所有连接
            cur.execute("""
                SELECT 
                    pid,
                    usename,
                    application_name,
                    client_addr,
                    state,
                    state_change,
                    query_start,
                    query
                FROM pg_stat_activity
                WHERE datname = 'langchain'
                ORDER BY state_change DESC;
            """)
            
            connections = cur.fetchall()
            
            print(f"当前数据库连接数: {len(connections)}\n")
            
            for conn_info in connections:
                pid, user, app, addr, state, state_change, query_start, query = conn_info
                print(f"PID: {pid}")
                print(f"  用户: {user}")
                print(f"  应用: {app or 'N/A'}")
                print(f"  地址: {addr or 'localhost'}")
                print(f"  状态: {state}")
                print(f"  最后活动: {state_change}")
                print()
            
            # 查询连接限制
            cur.execute("SHOW max_connections;")
            max_conn = cur.fetchone()[0]
            print(f"最大连接数限制: {max_conn}")
            
            # 查询空闲连接超时设置
            cur.execute("SHOW idle_in_transaction_session_timeout;")
            timeout = cur.fetchone()[0]
            print(f"空闲事务超时: {timeout}")
            
except Exception as e:
    print(f"错误: {e}")

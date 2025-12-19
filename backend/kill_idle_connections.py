#!/usr/bin/env python3
"""关闭空闲的 PostgreSQL 连接"""

import psycopg
from config.settings import settings

print("清理空闲的 PostgreSQL 连接...\n")

try:
    with psycopg.connect(settings.db_uri) as conn:
        with conn.cursor() as cur:
            # 查找空闲超过 5 分钟的连接（排除当前连接）
            cur.execute("""
                SELECT 
                    pid,
                    usename,
                    application_name,
                    state,
                    state_change
                FROM pg_stat_activity
                WHERE datname = 'langchain'
                  AND state = 'idle'
                  AND state_change < NOW() - INTERVAL '5 minutes'
                  AND pid != pg_backend_pid();
            """)
            
            idle_connections = cur.fetchall()
            
            if not idle_connections:
                print("没有需要清理的空闲连接")
            else:
                print(f"找到 {len(idle_connections)} 个空闲连接:\n")
                
                for conn_info in idle_connections:
                    pid, user, app, state, state_change = conn_info
                    print(f"PID {pid}: {user} - {app or 'N/A'} (空闲自 {state_change})")
                
                # 询问是否关闭
                response = input("\n是否关闭这些连接? (y/N): ")
                
                if response.lower() == 'y':
                    for conn_info in idle_connections:
                        pid = conn_info[0]
                        cur.execute(f"SELECT pg_terminate_backend({pid});")
                        print(f"已关闭连接 PID {pid}")
                    
                    conn.commit()
                    print(f"\n成功关闭 {len(idle_connections)} 个连接")
                else:
                    print("取消操作")
                    
except Exception as e:
    print(f"错误: {e}")

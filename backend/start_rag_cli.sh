#!/bin/bash
# RAG CLI 启动脚本

# 确保脚本从 backend 目录运行
cd "$(dirname "$0")"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行: python -m venv venv"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "⚠️  .env 文件不存在，请先配置环境变量"
    echo "   参考 env.example 创建 .env 文件"
    exit 1
fi

# 运行 RAG CLI
echo "🚀 启动 RAG CLI..."
echo ""
python scripts/rag_cli.py "$@"


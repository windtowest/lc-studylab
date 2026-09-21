#!/bin/bash
# LC-StudyLab 后端服务启动脚本

echo "🚀 启动 LC-StudyLab 后端服务..."
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 未找到虚拟环境，请先运行: python -m venv venv"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，从 env.example 复制..."
    cp env.example .env
    echo "✅ 已创建 .env 文件，请编辑并填写必要的配置"
    echo ""
fi

# 启动服务器
echo "📡 启动 FastAPI 服务器..."
echo "   访问 API 文档: http://localhost:8000/docs"
echo "   按 Ctrl+C 停止服务器"
echo ""

python api/http_server.py


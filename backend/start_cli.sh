#!/bin/bash
# LC-StudyLab CLI 演示工具启动脚本

echo "🎓 启动 LC-StudyLab CLI 演示工具..."
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

# 启动 CLI
python scripts/demo_cli.py


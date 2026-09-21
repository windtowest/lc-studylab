#!/bin/bash
# 测试增强的流式输出功能

echo "🧪 测试增强的 SSE 流式输出"
echo "================================"
echo ""

# 检查后端是否运行
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ 后端未运行，请先启动后端服务:"
    echo "   cd backend && ./start_server.sh"
    exit 1
fi

echo "✓ 后端服务正在运行"
echo ""

# 激活虚拟环境
source venv/bin/activate

# 运行测试脚本
python scripts/test_enhanced_stream.py

# 保存退出码
EXIT_CODE=$?

echo ""
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试完成 - 所有测试通过"
else
    echo "⚠️  测试完成 - 部分测试失败"
fi

exit $EXIT_CODE


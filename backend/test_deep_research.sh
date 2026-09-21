#!/bin/bash
# DeepAgent 深度研究测试脚本
# 快速启动测试

# 设置颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}     DeepAgent 深度研究功能测试${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  虚拟环境不存在，请先创建虚拟环境${NC}"
    exit 1
fi

# 激活虚拟环境
echo -e "${GREEN}✓ 激活虚拟环境...${NC}"
source venv/bin/activate

# 运行测试
echo -e "${GREEN}✓ 运行测试脚本...${NC}"
echo ""
python scripts/test_deep_research.py

# 显示测试结果
exit_code=$?
echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✅ 测试完成！${NC}"
else
    echo -e "${YELLOW}⚠️  测试失败，退出码: $exit_code${NC}"
fi

exit $exit_code


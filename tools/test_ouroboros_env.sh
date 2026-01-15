#!/bin/bash
# 测试 Ouroboros 环境是否配置正确

echo "========================================="
echo "Ouroboros 环境检查"
echo "========================================="
echo ""

# 检查 Python
echo "1. 检查 Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python 已安装: $PYTHON_VERSION"
else
    echo "✗ Python 未找到"
    exit 1
fi

# 检查 Ouroboros 目录
echo ""
echo "2. 检查 Ouroboros 目录..."
if [ -d "Ouroboros" ]; then
    echo "✓ Ouroboros 目录存在"
else
    echo "✗ Ouroboros 目录不存在"
    exit 1
fi

# 检查 Ouroboros inference 脚本
echo ""
echo "3. 检查 Ouroboros 推理脚本..."
if [ -f "Ouroboros/rgb2x/inference.py" ]; then
    echo "✓ inference.py 存在"
else
    echo "✗ inference.py 不存在"
    exit 1
fi

# 检查场景 001
echo ""
echo "4. 检查场景 001..."
if [ -d "data/waymo/processed/training/001/images" ]; then
    IMG_COUNT=$(ls data/waymo/processed/training/001/images/*.jpg 2>/dev/null | wc -l)
    echo "✓ 场景 001 存在，包含 $IMG_COUNT 张图像"
else
    echo "✗ 场景 001 不存在"
    exit 1
fi

# 检查 GPU
echo ""
echo "5. 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU 驱动已安装"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠ 未检测到 NVIDIA GPU（可能影响性能）"
fi

# 检查处理脚本
echo ""
echo "6. 检查处理脚本..."
if [ -f "tools/generate_ouroboros_priors_waymo.py" ]; then
    echo "✓ 处理脚本已创建"
else
    echo "✗ 处理脚本不存在"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ 所有检查通过！环境配置正确。"
echo "========================================="
echo ""
echo "运行以下命令开始处理："
echo "  ./tools/run_ouroboros_scene001.sh"
echo ""

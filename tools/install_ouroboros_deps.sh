#!/bin/bash
# 快速安装 Ouroboros 依赖（如果环境已存在但缺少包）

set -e

echo "========================================="
echo "安装 Ouroboros 缺失的依赖包"
echo "========================================="
echo ""

# 检查是否在正确的环境中
if [[ "$CONDA_DEFAULT_ENV" != "ouroboros" ]]; then
    echo "警告：当前不在 ouroboros 环境中"
    echo "请先运行: conda activate ouroboros"
    echo ""
    read -p "是否尝试激活环境？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ouroboros
    else
        exit 1
    fi
fi

echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 核心依赖列表
PACKAGES=(
    "diffusers==0.20.0"
    "transformers==4.37.2"
    "accelerate==0.27.2"
    "torch==2.2.0"
    "torchvision==0.17.0"
    "safetensors==0.5.3"
    "huggingface-hub==0.25.0"
    "pillow"
    "numpy"
    "tqdm"
)

echo "安装核心依赖包..."
echo ""

for pkg in "${PACKAGES[@]}"; do
    echo "正在安装: $pkg"
    pip install "$pkg" --no-deps 2>/dev/null || pip install "$pkg"
done

echo ""
echo "========================================="
echo "✓ 依赖安装完成"
echo "========================================="
echo ""
echo "验证安装："
python3 -c "
try:
    import diffusers
    import transformers
    import torch
    print('✓ diffusers:', diffusers.__version__)
    print('✓ transformers:', transformers.__version__)
    print('✓ torch:', torch.__version__)
    print('✓ CUDA available:', torch.cuda.is_available())
    print('')
    print('所有核心依赖已正确安装！')
except ImportError as e:
    print('✗ 导入失败:', e)
    exit(1)
"

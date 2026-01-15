#!/bin/bash
# Docker 环境中安装 Ouroboros 依赖

set -e

echo "========================================="
echo "Docker 环境 - 安装 Ouroboros 依赖"
echo "========================================="
echo ""

# 检查是否在 Docker 中
if [ -f /.dockerenv ]; then
    echo "✓ 检测到 Docker 环境"
else
    echo "⚠ 未检测到 Docker 环境，但继续执行..."
fi

echo ""
echo "当前目录: $(pwd)"
echo "Python 路径: $(which python3)"
echo "Python 版本: $(python3 --version)"
echo ""

# 升级 pip
echo "步骤 1/3: 升级 pip..."
python3 -m pip install --upgrade pip -q

# 安装核心依赖
echo "步骤 2/3: 安装核心依赖..."
python3 -m pip install -q \
    diffusers==0.20.0 \
    transformers==4.37.2 \
    accelerate==0.27.2 \
    safetensors==0.5.3 \
    huggingface-hub \
    pillow \
    numpy \
    tqdm

# 验证安装
echo "步骤 3/3: 验证安装..."
echo ""

python3 << 'PYEOF'
import sys
try:
    import diffusers
    import transformers
    import accelerate
    import torch
    
    print("✓ diffusers:", diffusers.__version__)
    print("✓ transformers:", transformers.__version__)
    print("✓ accelerate:", accelerate.__version__)
    print("✓ torch:", torch.__version__)
    print("✓ CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("✓ GPU:", torch.cuda.get_device_name(0))
    
    print("\n✅ 所有依赖已成功安装！")
    sys.exit(0)
    
except ImportError as e:
    print(f"\n✗ 导入失败: {e}")
    print("\n请手动安装缺失的包：")
    print("  pip install diffusers transformers accelerate")
    sys.exit(1)
PYEOF

echo ""
echo "========================================="
echo "安装完成！现在可以运行 Ouroboros 了"
echo "========================================="

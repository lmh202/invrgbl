#!/bin/bash
# Docker 环境下一键安装依赖并运行 Ouroboros（场景 001）

set -e

echo "========================================="
echo "Docker - Ouroboros 先验生成（场景 001）"
echo "========================================="
echo ""

# 检查并安装依赖
echo "检查依赖..."
if ! python3 -c "import diffusers" 2>/dev/null; then
    echo ""
    echo "⚠ 缺少依赖包，开始安装..."
    echo ""
    
    pip install -q diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 safetensors
    
    echo ""
    echo "✓ 依赖安装完成"
else
    echo "✓ 依赖已安装"
fi

echo ""
echo "验证环境..."
python3 << 'PYEOF'
import diffusers
import transformers
import torch
print(f"✓ diffusers: {diffusers.__version__}")
print(f"✓ transformers: {transformers.__version__}")
print(f"✓ torch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
PYEOF

echo ""
echo "========================================="
echo "开始处理场景 001（995 张图像）"
echo "========================================="
echo ""

# 运行 Ouroboros
python3 tools/generate_ouroboros_priors_waymo.py \
    --scene_idx 001 \
    --checkpoint Shanlin/Ouroboros \
    --modalities normals albedo irradiance roughness metallicity \
    --noise gaussian \
    --seed 0 \
    --skip_existing

echo ""
echo "========================================="
echo "✓ 处理完成！"
echo "========================================="
echo ""
echo "输出位置："
echo "  data/waymo/processed/training/001/albedo_ouroboros/"
echo "  data/waymo/processed/training/001/normal_ouroboros/"
echo "  data/waymo/processed/training/001/rough_ouroboros/"
echo "  data/waymo/processed/training/001/metallic_ouroboros/"
echo "  data/waymo/processed/training/001/irradiance_ouroboros/"
echo ""

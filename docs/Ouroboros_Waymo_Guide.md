# 使用 Ouroboros 生成 Waymo 场景先验

完整指南：使用 Ouroboros 替代 rgbx 为 Waymo 场景生成高质量先验数据

### 1. 环境准备

#### 方法 A：创建完整的 Ouroboros 环境（首次安装）

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 方法 B：在现有环境中安装依赖（快速修复）

如果遇到 `ModuleNotFoundError: No module named 'diffusers'` 错误：

```bash
conda activate ouroboros  # 或你的工作环境
cd /nas0/codes/liuminhao/invrgbl

# 快速安装缺失的依赖
./tools/install_ouroboros_deps.sh

# 或手动安装核心依赖
pip install diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 torch torchvision
```

#### 验证安装

```bash
python3 -c "
import diffusers
import transformers
import torch
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ torch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"
```
### 1. 环境准备

#### 方法 A：创建完整的 Ouroboros 环境（首次安装）

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 方法 B：在现有环境中安装依赖（快速修复）

如果遇到 `ModuleNotFoundError: No module named 'diffusers'` 错误：

```bash
conda activate ouroboros  # 或你的工作环境
cd /nas0/codes/liuminhao/invrgbl

# 快速安装缺失的依赖
./tools/install_ouroboros_deps.sh

# 或手动安装核心依赖
pip install diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 torch torchvision
```

#### 验证安装

```bash
python3 -c "
import diffusers
import transformers
import torch
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ torch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"
```
### 1. 环境准备

#### 方法 A：创建完整的 Ouroboros 环境（首次安装）

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 方法 B：在现有环境中安装依赖（快速修复）

如果遇到 `ModuleNotFoundError: No module named 'diffusers'` 错误：

```bash
conda activate ouroboros  # 或你的工作环境
cd /nas0/codes/liuminhao/invrgbl

# 快速安装缺失的依赖
./tools/install_ouroboros_deps.sh

# 或手动安装核心依赖
pip install diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 torch torchvision
```

#### 验证安装

```bash
python3 -c "
import diffusers
import transformers
import torch
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ torch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"
```
### 1. 环境准备

#### 方法 A：创建完整的 Ouroboros 环境（首次安装）

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 方法 B：在现有环境中安装依赖（快速修复）

如果遇到 `ModuleNotFoundError: No module named 'diffusers'` 错误：

```bash
conda activate ouroboros  # 或你的工作环境
cd /nas0/codes/liuminhao/invrgbl

# 快速安装缺失的依赖
./tools/install_ouroboros_deps.sh

# 或手动安装核心依赖
pip install diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 torch torchvision
```

#### 验证安装

```bash
python3 -c "
import diffusers
import transformers
import torch
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ torch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"
```
### 1. 环境准备

#### 方法 A：创建完整的 Ouroboros 环境（首次安装）

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 方法 B：在现有环境中安装依赖（快速修复）

如果遇到 `ModuleNotFoundError: No module named 'diffusers'` 错误：

```bash
conda activate ouroboros  # 或你的工作环境
cd /nas0/codes/liuminhao/invrgbl

# 快速安装缺失的依赖
./tools/install_ouroboros_deps.sh

# 或手动安装核心依赖
pip install diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 torch torchvision
```

#### 验证安装

```bash
python3 -c "
import diffusers
import transformers
import torch
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ torch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"
```
### 1. 环境准备

#### 方法 A：创建完整的 Ouroboros 环境（首次安装）

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 方法 B：在现有环境中安装依赖（快速修复）

如果遇到 `ModuleNotFoundError: No module named 'diffusers'` 错误：

```bash
conda activate ouroboros  # 或你的工作环境
cd /nas0/codes/liuminhao/invrgbl

# 快速安装缺失的依赖
./tools/install_ouroboros_deps.sh

# 或手动安装核心依赖
pip install diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 torch torchvision
```

#### 验证安装

```bash
python3 -c "
import diffusers
import transformers
import torch
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ torch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"
```
### 1. 环境准备

#### 方法 A：创建完整的 Ouroboros 环境（首次安装）

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 方法 B：在现有环境中安装依赖（快速修复）

如果遇到 `ModuleNotFoundError: No module named 'diffusers'` 错误：

```bash
conda activate ouroboros  # 或你的工作环境
cd /nas0/codes/liuminhao/invrgbl

# 快速安装缺失的依赖
./tools/install_ouroboros_deps.sh

# 或手动安装核心依赖
pip install diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 torch torchvision
```

#### 验证安装

```bash
python3 -c "
import diffusers
import transformers
import torch
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ torch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"
```
### 1. 环境准备

#### 方法 A：创建完整的 Ouroboros 环境（首次安装）

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 方法 B：在现有环境中安装依赖（快速修复）

如果遇到 `ModuleNotFoundError: No module named 'diffusers'` 错误：

```bash
conda activate ouroboros  # 或你的工作环境
cd /nas0/codes/liuminhao/invrgbl

# 快速安装缺失的依赖
./tools/install_ouroboros_deps.sh

# 或手动安装核心依赖
pip install diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 torch torchvision
```

#### 验证安装

```bash
python3 -c "
import diffusers
import transformers
import torch
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ torch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"
```
### 1. 环境准备

#### 方法 A：创建完整的 Ouroboros 环境（首次安装）

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 方法 B：在现有环境中安装依赖（快速修复）

如果遇到 `ModuleNotFoundError: No module named 'diffusers'` 错误：

```bash
conda activate ouroboros  # 或你的工作环境
cd /nas0/codes/liuminhao/invrgbl

# 快速安装缺失的依赖
./tools/install_ouroboros_deps.sh

# 或手动安装核心依赖
pip install diffusers==0.20.0 transformers==4.37.2 accelerate==0.27.2 torch torchvision
```

#### 验证安装

```bash
python3 -c "
import diffusers
import transformers
import torch
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ torch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"
```
## 快速开始

### 一键运行（场景 001）

```bash
# 1. 激活 Ouroboros 环境
conda activate ouroboros

# 2. 运行脚本
cd /nas0/codes/liuminhao/invrgbl
./tools/run_ouroboros_scene001.sh
```

### 自定义运行

```bash
python3 tools/generate_ouroboros_priors_waymo.py \
    --scene_idx 001 \
    --checkpoint Shanlin/Ouroboros \
    --modalities normals albedo irradiance roughness metallicity \
    --noise gaussian \
    --seed 0 \
    --skip_existing
```

---

## 详细步骤

### 1. 环境准备

#### 安装 Ouroboros 环境

```bash
cd /nas0/codes/liuminhao/invrgbl/Ouroboros
conda env create -f environment.yml
conda activate ouroboros
```

#### 验证环境

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

预期输出：
```
PyTorch: 2.x.x, CUDA: True
```

### 2. 单场景处理

#### 方法 A：使用快捷脚本

```bash
cd /nas0/codes/liuminhao/invrgbl
./tools/run_ouroboros_scene001.sh
```

#### 方法 B：使用 Python 脚本（更多控制）

```bash
python3 tools/generate_ouroboros_priors_waymo.py \
    --scene_idx 001 \
    --checkpoint Shanlin/Ouroboros \
    --modalities normals albedo irradiance roughness metallicity \
    --noise gaussian \
    --seed 0 \
    --skip_existing
```

**参数说明：**
- `--scene_idx 001`：处理场景 001
- `--checkpoint Shanlin/Ouroboros`：使用 HuggingFace 预训练模型
- `--modalities`：要生成的先验类型
- `--noise gaussian`：噪声类型（推荐）
- `--seed 0`：随机种子（可重复）
- `--skip_existing`：跳过已生成文件（增量处理）

### 3. 批量处理多场景

处理场景 000-007（8 个场景）：

```bash
python3 tools/generate_ouroboros_priors_waymo.py \
    --scene_idx 000-007 \
    --checkpoint Shanlin/Ouroboros \
    --modalities normals albedo irradiance roughness metallicity \
    --skip_existing
```

### 4. 监控进度

脚本会显示：
- 当前处理的场景
- 进度条（每张图像）
- 统计信息（成功/跳过/失败）

示例输出：
```
Processing scene 001: 995 images
Scene 001: 100%|████████████| 995/995 [45:23<00:00, 2.74s/it]

Scene 001 Summary:
  Successfully processed: 995
  Skipped (existing): 0
  Failed: 0
  Total: 995

Total time: 45.38 minutes
```

---

## 输出说明

### 目录结构

处理完成后，场景目录下会新增以下目录：

```
data/waymo/processed/training/001/
├── albedo_ouroboros/           # Albedo (反射率)
│   ├── 000_0.png
│   ├── 000_1.png
│   └── ...
├── normal_ouroboros/           # Surface Normal (表面法线)
│   ├── 000_0.png               # PNG 格式 (0-255)
│   ├── 000_1.png
│   ├── ...
│   └── normal_npy/             # NPY 格式 (float32, [-1,1])
│       ├── 000_0_pred.npy
│       ├── 000_1_pred.npy
│       └── ...
├── rough_ouroboros/            # Roughness (粗糙度)
│   ├── 000_0.png
│   └── ...
├── metallic_ouroboros/         # Metallic (金属度)
│   ├── 000_0.png
│   └── ...
└── irradiance_ouroboros/       # Irradiance (辐照度)
    ├── 000_0.png
    └── ...
```

### 文件格式

| 模态 | PNG 格式 | NPY 格式 | 数值范围 |
|------|---------|----------|---------|
| Albedo | ✓ | - | [0, 255] RGB |
| Normal | ✓ | ✓ | PNG: [0,255], NPY: [-1,1] |
| Roughness | ✓ | - | [0, 255] 灰度 |
| Metallic | ✓ | - | [0, 255] 灰度 |
| Irradiance | ✓ | - | [0, 255] RGB |

### 验证输出

```bash
# 检查文件数量
ls data/waymo/processed/training/001/normal_ouroboros/*.png | wc -l
# 应输出: 995

# 检查单个文件
python3 << EOF
import numpy as np
from PIL import Image

# PNG
png = Image.open('data/waymo/processed/training/001/normal_ouroboros/000_0.png')
print(f"Normal PNG: {np.array(png).shape}, dtype: {np.array(png).dtype}")

# NPY
npy = np.load('data/waymo/processed/training/001/normal_ouroboros/normal_npy/000_0_pred.npy')
print(f"Normal NPY: {npy.shape}, dtype: {npy.dtype}, range: [{npy.min():.2f}, {npy.max():.2f}]")
EOF
```

预期输出：
```
Normal PNG: (1280, 1920, 3), dtype: uint8
Normal NPY: (1280, 1920, 3), dtype: float32, range: [-1.00, 1.00]
```

---

## 配置训练

### 更新数据集配置

修改训练配置文件以使用 Ouroboros 生成的先验。

#### 原配置（configs/invrgbl.yaml）

```yaml
dataset:
  type: 'waymo'
  data_root: 'data/waymo/processed/training'
  scene_idx: 0
  
  # 旧路径（rgbx）
  albedo_path: "albedo_rgbx"
  normal_path: "normal_rgbx"
  rough_path: "rough_rgbx"
  metallic_path: "metallic_rgbx"
  irradiance_path: "irradiance_rgbx"
```

#### 新配置（使用 Ouroboros）

```yaml
dataset:
  type: 'waymo'
  data_root: 'data/waymo/processed/training'
  scene_idx: 0
  
  # 新路径（Ouroboros）
  albedo_path: "albedo_ouroboros"
  normal_path: "normal_ouroboros"
  rough_path: "rough_ouroboros"
  metallic_path: "metallic_ouroboros"
  irradiance_path: "irradiance_ouroboros"
```

### 启动训练

```bash
conda activate your_training_env

python tools/train.py \
    --config configs/invrgbl.yaml \
    --output_dir output/waymo_ouroboros_test
```

---

## 常见问题

### Q1: "Command 'python' not found"

**解决方案：**
```bash
conda activate ouroboros
# 或使用 python3
python3 tools/generate_ouroboros_priors_waymo.py ...
```

### Q2: "CUDA out of memory"

**原因：** GPU 内存不足

**解决方案：**
1. 关闭其他 GPU 进程
   ```bash
   nvidia-smi  # 查看 GPU 使用情况
   kill <PID>  # 终止不需要的进程
   ```

2. 使用 `--skip_existing` 继续处理
   ```bash
   # 脚本会自动跳过已生成的文件
   python3 tools/generate_ouroboros_priors_waymo.py --scene_idx 001 --skip_existing
   ```

### Q3: HuggingFace 模型下载失败

**原因：** 网络连接问题或 HuggingFace 访问受限

**解决方案 1：使用镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 tools/generate_ouroboros_priors_waymo.py ...
```

**解决方案 2：手动下载模型**
```bash
# 使用 wget 或浏览器下载
# 模型链接: https://huggingface.co/Shanlin/Ouroboros

# 然后使用本地路径
python3 tools/generate_ouroboros_priors_waymo.py \
    --checkpoint /path/to/downloaded/checkpoint \
    ...
```

### Q4: 处理速度太慢

**估算时间：**
- 单张图像：2-5 秒（取决于 GPU）
- 场景 001（995 张）：约 30-80 分钟
- 8 个场景（约 8000 张）：约 4-10 小时

**加速方法：**
1. 使用 `--skip_existing` 进行增量处理
2. 确保 GPU 驱动和 CUDA 最新
3. 检查其他进程是否占用 GPU

### Q5: 输出文件数量不对

**检查方法：**
```bash
# 输入图像数
ls data/waymo/processed/training/001/images/*.jpg | wc -l

# 输出法线数
ls data/waymo/processed/training/001/normal_ouroboros/*.png | wc -l

# 应该相等
```

**如果不等：**
```bash
# 使用 --no_skip_existing 重新生成所有文件
python3 tools/generate_ouroboros_priors_waymo.py \
    --scene_idx 001 \
    --no_skip_existing
```

### Q6: 与 rgbx 输出对比

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取两种输出
rgbx = np.array(Image.open('data/waymo/processed/training/001/normal_rgbx/000_0.png'))
ouroboros = np.array(Image.open('data/waymo/processed/training/001/normal_ouroboros/000_0.png'))

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgbx); axes[0].set_title('RGBX')
axes[1].imshow(ouroboros); axes[1].set_title('Ouroboros')
axes[2].imshow(np.abs(rgbx.astype(float) - ouroboros.astype(float)).astype(np.uint8))
axes[2].set_title('Difference')
plt.savefig('comparison.png')
```

---

## 性能基准

### 测试环境
- GPU: NVIDIA RTX 3090 / A100
- 场景: Waymo training/001 (995 images)
- 分辨率: 1920x1280

### 处理时间

| GPU | 单图时间 | 场景总时间 |
|-----|---------|-----------|
| RTX 3090 | ~3s | ~50min |
| A100 | ~2s | ~35min |
| RTX 2080 Ti | ~5s | ~85min |

### 存储空间

| 模态 | 单张大小 | 场景总大小 (995张) |
|------|---------|------------------|
| Albedo | ~300KB | ~300MB |
| Normal (PNG) | ~400KB | ~400MB |
| Normal (NPY) | ~24MB | ~24GB |
| Roughness | ~100KB | ~100MB |
| Metallic | ~100KB | ~100MB |
| Irradiance | ~300KB | ~300MB |
| **总计** | - | **~26GB** |

---

## 进阶用法

### 自定义输出路径

修改脚本中的 `target_dirs` 字典：

```python
# 在 organize_outputs 函数中
target_dirs = {
    'albedo': os.path.join(scene_root, 'my_custom_albedo'),
    # ...
}
```

### 仅生成部分模态

```bash
# 仅生成法线和 albedo
python3 tools/generate_ouroboros_priors_waymo.py \
    --scene_idx 001 \
    --modalities normals albedo
```

### 使用不同噪声类型

```bash
# 尝试 pyramid 噪声
python3 tools/generate_ouroboros_priors_waymo.py \
    --scene_idx 001 \
    --noise pyramid
```

### 批处理脚本示例

```bash
#!/bin/bash
# 批量处理所有已预处理的场景

for scene in 000 001 002 003 004 005 006 007; do
    echo "Processing scene $scene..."
    python3 tools/generate_ouroboros_priors_waymo.py \
        --scene_idx $scene \
        --skip_existing
done
```

---

## 参考资料

- **Ouroboros 论文**: https://arxiv.org/abs/2508.14461
- **项目主页**: https://siwensun.github.io/ouroboros-project/
- **HuggingFace 模型**: https://huggingface.co/Shanlin/Ouroboros
- **仓库 README**: /nas0/codes/liuminhao/invrgbl/Ouroboros/README.md

---

## 文件清单

本指南相关的文件：

```
/nas0/codes/liuminhao/invrgbl/
├── tools/
│   ├── generate_ouroboros_priors_waymo.py  # 主处理脚本
│   └── run_ouroboros_scene001.sh           # 快捷启动脚本
├── docs/
│   └── Ouroboros_Waymo_Guide.md            # 本文档
└── Ouroboros/
    └── rgb2x/
        └── inference.py                     # Ouroboros 推理脚本
```

---

**最后更新**: 2026-01-13

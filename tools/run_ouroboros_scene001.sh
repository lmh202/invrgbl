#!/bin/bash
# 使用 Ouroboros 为 Waymo 场景 001 生成先验

set -e

SCENE_IDX="001"
CHECKPOINT="Shanlin/Ouroboros"
MODALITIES="normals albedo irradiance roughness metallicity"

echo "==========================================="
echo "Ouroboros Prior Generation for Waymo"
echo "==========================================="
echo "Scene: $SCENE_IDX"
echo "Checkpoint: $CHECKPOINT"
echo ""

python3 tools/generate_ouroboros_priors_waymo.py \
    --scene_idx "$SCENE_IDX" \
    --checkpoint "$CHECKPOINT" \
    --modalities $MODALITIES \
    --noise gaussian \
    --seed 0 \
    --skip_existing

echo ""
echo "Processing complete!"
echo "Output saved to: data/waymo/processed/training/$SCENE_IDX/*_ouroboros/"

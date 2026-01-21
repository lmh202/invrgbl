#!/usr/bin/env python3
"""
测试单个光照条件的重光照效果
"""
import torch
import numpy as np
from PIL import Image
import sys
import os

# 添加项目路径
sys.path.insert(0, '/data0')

from tools.relight import PRESET_SUN_DIRECTIONS, normalize_sun_direction

def compare_images(img1_path, img2_path):
    """比较两张图片的差异"""
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))
    
    if img1.shape != img2.shape:
        print(f"⚠️  Images have different shapes: {img1.shape} vs {img2.shape}")
        return
    
    diff = np.abs(img1.astype(float) - img2.astype(float))
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"  Mean difference: {mean_diff:.2f}")
    print(f"  Max difference: {max_diff:.2f}")
    print(f"  Pixels with difference > 1: {np.sum(diff > 1)} / {diff.size}")
    print(f"  Pixels with difference > 10: {np.sum(diff > 10)} / {diff.size}")
    
    if mean_diff < 0.1:
        print("  ⚠️  WARNING: Images are almost identical! Relighting may not be working.")
    else:
        print("  ✓ Images are different - relighting appears to be working.")

if __name__ == "__main__":
    results_dir = "/data0/output/tester/test/relight_results"
    
    # 测试两个明显不同的光照条件
    preset1 = "top"
    preset2 = "front_top"
    
    img1_path = os.path.join(results_dir, f"{preset1}_rgb.png")
    img2_path = os.path.join(results_dir, f"{preset2}_rgb.png")
    
    if not os.path.exists(img1_path):
        print(f"Error: {img1_path} not found")
        sys.exit(1)
    if not os.path.exists(img2_path):
        print(f"Error: {img2_path} not found")
        sys.exit(1)
    
    print(f"Comparing {preset1} vs {preset2}:")
    print(f"  {preset1} sun direction: {normalize_sun_direction(PRESET_SUN_DIRECTIONS[preset1], torch.device('cpu')).numpy()}")
    print(f"  {preset2} sun direction: {normalize_sun_direction(PRESET_SUN_DIRECTIONS[preset2], torch.device('cpu')).numpy()}")
    print()
    
    compare_images(img1_path, img2_path)


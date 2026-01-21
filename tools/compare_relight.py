#!/usr/bin/env python3
"""
详细对比重光照结果的脚本
"""
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    """加载图片"""
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    return np.array(img)

def compare_two_images(img1, img2, name1, name2):
    """对比两张图片"""
    if img1 is None or img2 is None:
        print(f"⚠️  Cannot compare: one or both images missing")
        return
    
    if img1.shape != img2.shape:
        print(f"⚠️  Images have different shapes: {img1.shape} vs {img2.shape}")
        return
    
    # 计算差异
    diff = np.abs(img1.astype(float) - img2.astype(float))
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    std_diff = np.std(diff)
    
    # 统计不同像素
    diff_pixels = np.sum(diff > 1)
    diff_pixels_10 = np.sum(diff > 10)
    diff_pixels_50 = np.sum(diff > 50)
    
    total_pixels = diff.size
    
    print(f"\n{'='*60}")
    print(f"Comparing: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"Image shape: {img1.shape}")
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Max difference: {max_diff:.2f}")
    print(f"Std difference: {std_diff:.4f}")
    print(f"\nPixel differences:")
    print(f"  > 1:   {diff_pixels:8d} / {total_pixels:8d} ({100*diff_pixels/total_pixels:.2f}%)")
    print(f"  > 10:  {diff_pixels_10:8d} / {total_pixels:8d} ({100*diff_pixels_10/total_pixels:.2f}%)")
    print(f"  > 50:  {diff_pixels_50:8d} / {total_pixels:8d} ({100*diff_pixels_50/total_pixels:.2f}%)")
    
    if mean_diff < 0.01:
        print(f"\n❌ CRITICAL: Images are IDENTICAL! Relighting is NOT working!")
        return False
    elif mean_diff < 1.0:
        print(f"\n⚠️  WARNING: Images are very similar. Relighting effect may be weak.")
        return True
    else:
        print(f"\n✓ Images are different - relighting appears to be working.")
        return True

def create_diff_visualization(img1, img2, name1, name2, output_path):
    """创建差异可视化"""
    if img1 is None or img2 is None:
        return
    
    if img1.shape != img2.shape:
        return
    
    diff = np.abs(img1.astype(float) - img2.astype(float))
    diff_normalized = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(name1, fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title(name2, fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 差异图（灰度）
    diff_gray = np.mean(diff_normalized, axis=2) if len(diff_normalized.shape) == 3 else diff_normalized
    axes[1, 0].imshow(diff_gray, cmap='hot')
    axes[1, 0].set_title('Difference (Hot Colormap)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 差异图（RGB）
    if len(diff_normalized.shape) == 3:
        axes[1, 1].imshow(diff_normalized)
    else:
        axes[1, 1].imshow(diff_normalized, cmap='gray')
    axes[1, 1].set_title('Difference (RGB)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved difference visualization to: {output_path}")
    plt.close()

def main():
    results_dir = "/data0/output/tester/test/relight_results"
    
    # 测试几个明显不同的光照条件
    test_pairs = [
        ("top", "front_top"),
        ("top", "back_top"),
        ("morning", "evening"),
        ("front_top", "back_top"),
    ]
    
    print("="*60)
    print("Relighting Comparison Analysis")
    print("="*60)
    
    all_working = True
    
    for preset1, preset2 in test_pairs:
        img1_path = os.path.join(results_dir, f"{preset1}_rgb.png")
        img2_path = os.path.join(results_dir, f"{preset2}_rgb.png")
        
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)
        
        is_working = compare_two_images(img1, img2, preset1, preset2)
        all_working = all_working and is_working
        
        # 创建差异可视化
        if img1 is not None and img2 is not None:
            diff_output = os.path.join(results_dir, f"diff_{preset1}_vs_{preset2}.png")
            create_diff_visualization(img1, img2, preset1, preset2, diff_output)
    
    print("\n" + "="*60)
    if all_working:
        print("✓ SUMMARY: Relighting appears to be working (some differences detected)")
    else:
        print("❌ SUMMARY: Relighting is NOT working (images are identical)")
    print("="*60)
    
    # 检查 PBR 结果
    print("\n" + "="*60)
    print("Checking PBR results (may show more visible differences)")
    print("="*60)
    
    preset1, preset2 = "top", "front_top"
    pbr1_path = os.path.join(results_dir, f"{preset1}_pbr.png")
    pbr2_path = os.path.join(results_dir, f"{preset2}_pbr.png")
    
    pbr1 = load_image(pbr1_path)
    pbr2 = load_image(pbr2_path)
    
    if pbr1 is not None and pbr2 is not None:
        compare_two_images(pbr1, pbr2, f"{preset1}_pbr", f"{preset2}_pbr")
        diff_output = os.path.join(results_dir, f"diff_{preset1}_vs_{preset2}_pbr.png")
        create_diff_visualization(pbr1, pbr2, f"{preset1}_pbr", f"{preset2}_pbr", diff_output)

if __name__ == "__main__":
    main()


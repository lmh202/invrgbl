#!/usr/bin/env python3
"""
可视化重光照对比结果的工具
"""
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def create_comparison_grid(image_paths, labels, output_path, title="Relighting Comparison"):
    """创建对比网格图"""
    n_images = len(image_paths)
    if n_images == 0:
        print("No images to compare")
        return
    
    # 计算网格大小
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(rows, cols)
    
    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(label, fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'Not found:\n{label}', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison grid to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize relighting comparison results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing relight results")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for comparison image (default: results_dir/comparison_visual.png)")
    parser.add_argument("--type", type=str, default="rgb", 
                       choices=["rgb", "pbr", "albedo", "normal", "roughness", "depth", "sun_visibility"],
                       help="Type of image to compare")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # 预设的光照条件（按顺序）
    lighting_presets = [
        "top", "front_top", "back_top", "left_top", "right_top",
        "morning", "noon", "evening", "low_front", "low_back"
    ]
    
    # 查找所有可用的光照条件
    available_lightings = []
    for preset in lighting_presets:
        img_path = results_dir / f"{preset}_{args.type}.png"
        if img_path.exists():
            available_lightings.append(preset)
    
    if len(available_lightings) == 0:
        print(f"Error: No {args.type} images found in {results_dir}")
        print(f"Looking for files like: {lighting_presets[0]}_{args.type}.png")
        return
    
    print(f"Found {len(available_lightings)} lighting conditions: {', '.join(available_lightings)}")
    
    # 收集图片路径和标签
    image_paths = []
    labels = []
    
    for preset in available_lightings:
        img_path = results_dir / f"{preset}_{args.type}.png"
        image_paths.append(str(img_path))
        labels.append(preset.replace('_', ' ').title())
    
    # 输出路径
    if args.output is None:
        output_path = results_dir / f"visual_comparison_{args.type}.png"
    else:
        output_path = Path(args.output)
    
    # 创建对比图
    create_comparison_grid(
        image_paths, 
        labels, 
        str(output_path),
        title=f"Relighting Comparison - {args.type.upper()}"
    )
    
    print(f"\n✓ Comparison visualization saved to: {output_path}")
    print(f"\nTo view the image:")
    print(f"  python -m PIL.Image.open '{output_path}'")

if __name__ == "__main__":
    main()


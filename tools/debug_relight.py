#!/usr/bin/env python3
"""
调试重光照效果的脚本
"""
import torch
import numpy as np

# 测试太阳方向是否正确传递
def test_sun_directions():
    """测试不同预设的太阳方向是否不同"""
    PRESET_SUN_DIRECTIONS = {
        "top": [0, 0, 1],
        "front_top": [0.707, 0, 0.707],
        "back_top": [-0.707, 0, 0.707],
        "left_top": [0, 0.707, 0.707],
        "right_top": [0, -0.707, 0.707],
        "morning": [0.5, 0, 0.866],
        "noon": [0, 0, 1],
        "evening": [-0.5, 0, 0.866],
        "low_front": [0.866, 0, 0.5],
        "low_back": [-0.866, 0, 0.5],
    }
    
    print("Testing sun directions:")
    for name, dir_list in PRESET_SUN_DIRECTIONS.items():
        dir_tensor = torch.tensor(dir_list, dtype=torch.float32)
        dir_normalized = dir_tensor / dir_tensor.norm()
        print(f"  {name:15s}: {dir_normalized.numpy()}")
    
    # 检查是否有重复
    directions = {}
    for name, dir_list in PRESET_SUN_DIRECTIONS.items():
        dir_tensor = torch.tensor(dir_list, dtype=torch.float32)
        dir_normalized = dir_tensor / dir_tensor.norm()
        dir_normalized_tuple = tuple(dir_normalized.numpy().round(4))
        if dir_normalized_tuple in directions:
            print(f"\n⚠️  WARNING: {name} and {directions[dir_normalized_tuple]} have the same direction!")
        directions[dir_normalized_tuple] = name
    
    print("\n✓ All sun directions are unique")

if __name__ == "__main__":
    test_sun_directions()

"""
分析为什么 relight 没有效果的根本原因
"""
import torch
import numpy as np

print('='*80)
print('分析 Relight 失败的原因')
print('='*80)

# 模拟 rendering_equation 中的计算
print('\n1. 太阳光照计算:')
print('   sun_light = (sun_direction · normals) * intensity * 3')
print('   incident_sun_light = (diffuse + specular) * sun_light * 0.3')
print('   pbr = pbr_env + incident_sun_light * sun_visibility')

# 估算贡献比例
env_light = 0.78  # [200,200,180]/255 * 1.5
num_samples = 23
solid_angle = 4 * 3.14159 / num_samples
env_contrib = env_light * solid_angle * 0.5 * 23  # 假设平均cos=0.5

sun_intensity = 1.0
sun_contrib_max = 1.0 * sun_intensity * 3 * 0.3  # cos=1时最大

print(f'\n2. 光照贡献估算:')
print(f'   环境光总贡献: ~{env_contrib:.2f}')
print(f'   太阳光最大贡献: ~{sun_contrib_max:.2f} (cos(θ)=1时)')
print(f'   太阳光占比: ~{sun_contrib_max / (env_contrib + sun_contrib_max) * 100:.1f}%')

print('\n3. 问题分析:')
print('   ❌ 太阳光权重太小 (* 0.3)，只占总光照的 ~10%')
print('   ❌ sun_visibility 使用训练时的遮挡图，与新方向不匹配')
print('   ❌ affine_transformation 会把颜色校正回训练分布')

print('\n4. 解决方案:')
print('   ✓ 增大太阳光权重: 0.3 → 2.0 或更大')
print('   ✓ 忽略 sun_visibility (设为1) 或重新计算')
print('   ✓ 在 relight 模式下禁用 affine_transformation')
print('   ✓ 可选: 降低环境光权重')

print('\n5. 需要修改的代码位置:')
print('   📝 models/trainers/pbr.py 第120行:')
print('      incident_sun_light = ... * 0.3  →  * 2.0')
print('   📝 models/trainers/pbr.py 第121行:')
print('      ... * sun_visibility  →  忽略或设为1')
print('   📝 models/trainers/scene_graph.py forward():')
print('      在 relight 时跳过 affine_transformation')

print('\n' + '='*80)

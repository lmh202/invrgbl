"""
Direct test of rendering_equation to see if relight_sun_dir works
"""
import torch
import sys
sys.path.append('.')
from models.trainers.pbr import rendering_equation

# Create dummy data
batch_size = 100
base_color = torch.rand(batch_size, 3).cuda()
roughness = torch.rand(batch_size, 1).cuda() * 0.5 + 0.3
normals = torch.tensor([[0.0, 0.0, 1.0]]).cuda().repeat(batch_size, 1)  # All pointing up
normals = normals / normals.norm(dim=-1, keepdim=True)
viewdirs = torch.tensor([[0.0, 0.0, -1.0]]).cuda().repeat(batch_size, 1)  # Looking down

# Create dummy incident directions (sky dome sampling)
num_samples = 24
incident_dirs = torch.randn(batch_size, num_samples, 3).cuda()
incident_dirs = incident_dirs / incident_dirs.norm(dim=-1, keepdim=True)
incident_areas = torch.ones(batch_size, num_samples, 1).cuda() * (4 * 3.14159 / num_samples)
visibility = torch.ones(batch_size, num_samples + 1, 1).cuda()  # +1 for sun
incidents = torch.zeros(batch_size, (3+1)**2, 3).cuda()  # SH coefficients

# Dummy sun visibility
sun_visibility = torch.ones(batch_size, 1).cuda()

# Dummy env light (mock object)
class DummyEnvLight:
    def __init__(self):
        self.sun_intensity = torch.tensor([1.0, 1.0, 0.9]).cuda()
    def direct_light(self, dirs, step):
        return torch.ones(dirs.shape[0], dirs.shape[1], 3).cuda() * 0.5

env_light = DummyEnvLight()

print('Testing rendering_equation with different sun directions:\n')

# Test 1: No relight (use original sun direction from incident_dirs)
print('Test 1: Original (relight_sun_dir=None)')
pbr1, extras1 = rendering_equation(
    base_color=base_color, roughness=roughness, normals=normals, viewdirs=viewdirs,
    incidents=incidents, direct_light_env_light=env_light,
    incident_dirs=incident_dirs.clone(), incident_areas=incident_areas,
    visibility_precompute=visibility, sun_visibility=sun_visibility,
    xyz=torch.zeros(batch_size, 3).cuda(), step=1000,
    relight_sun_dir=None
)
print(f'  PBR mean: {pbr1.mean().item():.6f}')
print(f'  Sun contribution: {extras1["incident_sun_light"].mean().item():.6f}')

# Test 2: Sun from top [0, 0, 1] - should give maximum Lambert for upward normals
print('\nTest 2: Sun from top [0, 0, 1]')
sun_top = torch.tensor([0.0, 0.0, 1.0]).cuda()
pbr2, extras2 = rendering_equation(
    base_color=base_color, roughness=roughness, normals=normals, viewdirs=viewdirs,
    incidents=incidents, direct_light_env_light=env_light,
    incident_dirs=incident_dirs.clone(), incident_areas=incident_areas,
    visibility_precompute=visibility, sun_visibility=sun_visibility,
    xyz=torch.zeros(batch_size, 3).cuda(), step=1000,
    relight_sun_dir=sun_top
)
print(f'  PBR mean: {pbr2.mean().item():.6f}')
print(f'  Sun contribution: {extras2["incident_sun_light"].mean().item():.6f}')
print(f'  Expected Lambert term (n·l): {(normals * sun_top).sum(dim=-1).mean().item():.6f}')

# Test 3: Sun from front [1, 0, 0] - should give zero Lambert for upward normals
print('\nTest 3: Sun from front [1, 0, 0]')
sun_front = torch.tensor([1.0, 0.0, 0.0]).cuda()
pbr3, extras3 = rendering_equation(
    base_color=base_color, roughness=roughness, normals=normals, viewdirs=viewdirs,
    incidents=incidents, direct_light_env_light=env_light,
    incident_dirs=incident_dirs.clone(), incident_areas=incident_areas,
    visibility_precompute=visibility, sun_visibility=sun_visibility,
    xyz=torch.zeros(batch_size, 3).cuda(), step=1000,
    relight_sun_dir=sun_front
)
print(f'  PBR mean: {pbr3.mean().item():.6f}')
print(f'  Sun contribution: {extras3["incident_sun_light"].mean().item():.6f}')
print(f'  Expected Lambert term (n·l): {(normals * sun_front).sum(dim=-1).mean().item():.6f}')

# Check differences
print('\n' + '='*60)
diff_12 = (pbr1 - pbr2).abs().mean().item()
diff_13 = (pbr1 - pbr3).abs().mean().item()
diff_23 = (pbr2 - pbr3).abs().mean().item()
print(f'Differences:')
print(f'  |PBR1 - PBR2|: {diff_12:.6f}')
print(f'  |PBR1 - PBR3|: {diff_13:.6f}')
print(f'  |PBR2 - PBR3|: {diff_23:.6f}')

if diff_23 > 0.01:
    print('\n✅ Relight is working in rendering_equation!')
else:
    print('\n❌ Relight NOT working in rendering_equation!')

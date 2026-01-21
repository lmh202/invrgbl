#!/usr/bin/env python
"""
Relighting Script

This script renders the scene with different sun directions to create relighting effects.
It supports both single frame rendering and video rendering.

Usage:
    # Render single frame with different lightings (default: all presets)
    python tools/relight.py --resume_from output/waymo_dynamic1/ckpts/model_30000.pth
    
    # Render single frame with specific lighting preset
    python tools/relight.py --resume_from output/waymo_dynamic1/ckpts/model_30000.pth --lighting top
    
    # Render video with specific lighting
    python tools/relight.py --resume_from output/waymo_dynamic1/ckpts/model_30000.pth --render_video --lighting top
    
    # Render with custom sun direction (x, y, z in Waymo coordinate system)
    python tools/relight.py --resume_from output/waymo_dynamic1/ckpts/model_30000.pth --custom_sun 1.0 0.0 0.5
    
    # Render specific frame with custom output directory
    python tools/relight.py --resume_from output/waymo_dynamic1/ckpts/model_30000.pth --frame_idx 10 --output_dir ./my_relight_results
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
import logging
import imageio
import gc
import copy
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined sun directions (normalized)
# Waymo coordinate system: x=forward, y=left, z=up
PRESET_SUN_DIRECTIONS = {
    "original": None,  # Use original sun direction from training
    "top": [0.0, 0.0, 1.0],  # Directly from above
    "front_top": [0.707, 0.0, 0.707],  # 45° from front
    "back_top": [-0.707, 0.0, 0.707],  # 45° from back
    "left_top": [0.0, 0.707, 0.707],  # 45° from left
    "right_top": [0.0, -0.707, 0.707],  # 45° from right
    "morning": [0.866, 0.0, 0.5],  # 30° elevation from front (sunrise)
    "noon": [0.0, 0.0, 1.0],  # Noon, sun directly overhead
    "evening": [-0.866, 0.0, 0.5],  # 30° elevation from back (sunset)
    "low_front": [0.94, 0.0, 0.34],  # 20° elevation from front
    "low_back": [-0.94, 0.0, 0.34],  # 20° elevation from back
}


def normalize_sun_direction(sun_dir: List[float], device: torch.device) -> Optional[torch.Tensor]:
    """Normalize sun direction vector and move to device.
    
    Returns:
        A normalized 3D vector tensor of shape (3,) or None
    """
    if sun_dir is None:
        return None
    sun_tensor = torch.tensor(sun_dir, dtype=torch.float32, device=device)
    # Ensure it's a 1D vector of shape (3,)
    if sun_tensor.dim() > 1:
        sun_tensor = sun_tensor.squeeze()
    # Normalize
    norm = sun_tensor.norm()
    if norm < 1e-8:
        logger.warning(f"Sun direction norm too small: {norm}, using default [0, 0, 1]")
        sun_tensor = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    else:
        sun_tensor = sun_tensor / norm
    return sun_tensor


def setup_trainer_and_dataset(args):
    """Setup trainer and dataset from checkpoint (similar to eval.py)."""
    # Set CUDA device if specified
    cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_device is not None:
        logger.info(f'Using CUDA_VISIBLE_DEVICES={cuda_device}')
        # Set the device index
        if torch.cuda.is_available():
            device_id = int(cuda_device.split(',')[0]) if cuda_device else 0
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration from config.yaml in the same directory as checkpoint
    checkpoint_dir = os.path.dirname(os.path.abspath(args.resume_from))
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        # Try alternative paths
        alt_paths = [
            os.path.join(os.path.dirname(checkpoint_dir), 'config.yaml'),
            os.path.join(checkpoint_dir, '..', 'config.yaml'),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                config_path = alt_path
                logger.info(f'Found config at alternative path: {config_path}')
                break
        else:
            raise FileNotFoundError(f'Config file not found. Tried: {config_path} and alternatives')
    
    logger.info(f'Loading config from: {config_path}')
    cfg = OmegaConf.load(config_path)
    
    # Set device based on CUDA_VISIBLE_DEVICES or default
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is not None:
        logger.info(f'CUDA_VISIBLE_DEVICES={cuda_visible}')
        # When CUDA_VISIBLE_DEVICES is set, PyTorch will map it to cuda:0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f'Using device: {device}')
    
    # Clear GPU cache before loading dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()  # Call twice to ensure cleanup
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f'GPU memory before dataset loading: allocated={mem_allocated:.2f} GB, reserved={mem_reserved:.2f} GB')
        
        # If too much memory is already allocated, warn user
        if mem_reserved > 5.0:
            logger.warning(f'GPU memory is already heavily used ({mem_reserved:.2f} GB reserved). '
                          f'Consider freeing GPU memory before running this script.')
    
    # Setup dataset - use CPU device to avoid OOM during initialization
    logger.info('Setting up dataset on CPU (will move to GPU only when needed)...')
    
    # Create a copy of data config and modify device to CPU
    data_cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    
    # Force CPU device in pixel_source config (use string, not torch.device)
    if hasattr(data_cfg_copy, 'pixel_source') and data_cfg_copy.pixel_source is not None:
        pixel_source_cfg = OmegaConf.to_container(data_cfg_copy.pixel_source, resolve=True)
        pixel_source_cfg['device'] = 'cpu'  # Use string, not torch.device
        data_cfg_copy.pixel_source = OmegaConf.create(pixel_source_cfg)
    
    # Force CPU device in lidar_source config (use string, not torch.device)
    if hasattr(data_cfg_copy, 'lidar_source') and data_cfg_copy.lidar_source is not None:
        lidar_source_cfg = OmegaConf.to_container(data_cfg_copy.lidar_source, resolve=True)
        lidar_source_cfg['device'] = 'cpu'  # Use string, not torch.device
        data_cfg_copy.lidar_source = OmegaConf.create(lidar_source_cfg)
    
    # Set preload_device to CPU (this is what self.device property returns)
    if hasattr(data_cfg_copy, 'preload_device'):
        data_cfg_copy.preload_device = 'cpu'
    else:
        OmegaConf.set_struct(data_cfg_copy, False)
        data_cfg_copy.preload_device = 'cpu'
        OmegaConf.set_struct(data_cfg_copy, True)
    
    # Also set top-level device to CPU for compatibility
    if hasattr(data_cfg_copy, 'device'):
        data_cfg_copy.device = 'cpu'
    else:
        OmegaConf.set_struct(data_cfg_copy, False)
        data_cfg_copy.device = 'cpu'
        OmegaConf.set_struct(data_cfg_copy, True)
    
    logger.info('Data config device settings:')
    logger.info(f'  preload_device: {data_cfg_copy.get("preload_device", "not set")}')
    logger.info(f'  pixel_source.device: {data_cfg_copy.pixel_source.get("device", "not set")}')
    logger.info(f'  lidar_source.device: {data_cfg_copy.lidar_source.get("device", "not set")}')
    
    try:
        dataset = DrivingDataset(data_cfg=data_cfg_copy)
        logger.info(f'Dataset loaded successfully. Dataset device property: {dataset.device}')
        
    except RuntimeError as e:
        error_msg = str(e)
        if 'cusolver' in error_msg or 'CUDA' in error_msg or 'out of memory' in error_msg:
            logger.error(f'CUDA error during dataset loading: {error_msg}')
            logger.error('GPU memory issue detected. Trying to clear GPU cache and retry...')
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            
            # Restore original method
            if original_build_data_source is not None:
                from datasets.driving_dataset import DrivingDataset as DD
                DD.build_data_source = original_build_data_source
            
            logger.error('Please try:')
            logger.error('  1. Restart Python process to clear GPU memory')
            logger.error('  2. Run: python -c "import torch; torch.cuda.empty_cache()"')
            logger.error('  3. Check if other processes are using GPU: nvidia-smi')
            raise
        else:
            if original_build_data_source is not None:
                from datasets.driving_dataset import DrivingDataset as DD
                DD.build_data_source = original_build_data_source
            raise
    except Exception as e:
        # Restore original method on any error
        if original_build_data_source is not None:
            from datasets.driving_dataset import DrivingDataset as DD
            DD.build_data_source = original_build_data_source
        logger.error(f'Failed to load dataset: {e}')
        import traceback
        traceback.print_exc()
        raise
    
    # Setup trainer (same as eval.py)
    logger.info('Setting up trainer...')
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )
    
    # Load model weights from checkpoint
    checkpoint_path = os.path.abspath(args.resume_from)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')
    
    logger.info(f'Loading model weights from: {checkpoint_path}')
    trainer.resume_from_checkpoint(
        ckpt_path=checkpoint_path,
        load_only_model=True
    )
    trainer.set_eval()
    logger.info(f'Model loaded successfully! Step: {trainer.step}')
    
    # Verify PBR mode is enabled
    if not trainer.pbr:
        logger.warning('PBR mode is not enabled! Relighting may not work correctly.')
    
    return trainer, dataset, cfg, device


def render_single_frame_with_sun(
    trainer,
    dataset,
    device: torch.device,
    frame_idx: int,
    sun_direction: Optional[torch.Tensor] = None,
    data_source: str = "full"
) -> Dict[str, np.ndarray]:
    """
    Render a single frame with specified sun direction.
    
    Args:
        trainer: The trained model
        dataset: Dataset object
        device: Torch device
        frame_idx: Frame index to render
        sun_direction: Optional custom sun direction (normalized tensor)
        data_source: "full" or "test" set
        
    Returns:
        Dictionary of rendered images as numpy arrays
    """
    # Select data source
    if data_source == "test" and dataset.test_image_set is not None:
        source = dataset.test_image_set
    else:
        source = dataset.full_image_set
    
    # Clamp frame index
    frame_idx = min(frame_idx, len(source) - 1)
    
    # Get data for the frame using the same interface as eval.py
    camera_downscale = 1  # No downscale during evaluation
    image_infos, cam_infos = source.get_image(frame_idx, camera_downscale)
    
    # Move tensors to GPU
    for k, v in image_infos.items():
        if isinstance(v, torch.Tensor):
            image_infos[k] = v.to(device, non_blocking=True)
    for k, v in cam_infos.items():
        if isinstance(v, torch.Tensor):
            cam_infos[k] = v.to(device, non_blocking=True)
    
    # Set current frame for trainer based on normalized time
    normed_time = image_infos["normed_time"].flatten()[0]
    trainer.cur_frame = torch.argmin(
        torch.abs(trainer.normalized_timestamps - normed_time)
    )
    
    # For evaluation: inform all models about the current test state
    for model in trainer.models.values():
        if hasattr(model, 'in_test_set'):
            model.in_test_set = trainer.in_test_set
    
    # Set current frame for RigidNodes models (required for get_xyz)
    for class_name in trainer.gaussian_classes.keys():
        model = trainer.models[class_name]
        if hasattr(model, 'set_cur_frame'):
            model.set_cur_frame(trainer.cur_frame.item())
    
    # If custom sun direction is provided, we need to clear visibility cache
    # and let collect_gaussians recalculate with new sun direction
    if sun_direction is not None:
        # Ensure sun_direction is a 1D tensor of shape (3,)
        if sun_direction.dim() > 1:
            sun_direction = sun_direction.squeeze()
        if sun_direction.shape[0] != 3:
            raise ValueError(f"Sun direction must be a 3D vector, got shape {sun_direction.shape}")
        
        cur_frame = trainer.cur_frame.item()
        # Clear cached visibility for this frame to force recalculation
        # IMPORTANT: Cache key is only based on cur_frame, not sun_direction
        # So we must clear cache before each render with different sun direction
        if cur_frame in trainer._visibility_tracings_list:
            del trainer._visibility_tracings_list[cur_frame]
            logger.debug(f"Cleared visibility cache for frame {cur_frame}")
        if cur_frame in trainer._incident_dirs_list:
            del trainer._incident_dirs_list[cur_frame]
            logger.debug(f"Cleared incident_dirs cache for frame {cur_frame}")
        if cur_frame in trainer._incident_areas_list:
            del trainer._incident_areas_list[cur_frame]
            logger.debug(f"Cleared incident_areas cache for frame {cur_frame}")
        
        logger.info(f"Using custom sun direction: {sun_direction.cpu().numpy()}")
    
    # Forward pass with custom sun direction
    with torch.no_grad():
        # Process camera
        processed_cam = trainer.process_camera(
            camera_infos=cam_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=False
        )
        
        # Collect gaussians with custom sun direction
        # The update_visibility will be called inside collect_gaussians
        # CRITICAL: Must set update=True to force recalculation with new sun_direction
        # because cache key doesn't include sun_direction
        force_update = sun_direction is not None
        if force_update:
            logger.debug(f"Force updating visibility with sun_direction: {sun_direction.cpu().numpy()}")
        
        gs = trainer.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0],
            sun_direction=sun_direction,
            update=force_update  # Always update when sun_direction is provided
        )
        
        # Render gaussians
        # Important: pass Sky model as direct_light_env_light for PBR rendering
        outputs, _ = trainer.render_gaussians(
            gs=gs,
            cam=processed_cam,
            direct_light_env_light=trainer.models['Sky'],
            near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=trainer.render_cfg.get('radius_clip', 0.),
        )
        
        # Render sky and compose final image
        sky_model = trainer.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # Affine transformation
        outputs["rgb"] = trainer.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), 
            image_infos
        )
    
    # Collect results
    results = {}
    
    # RGB output
    if "rgb" in outputs:
        rgb_np = outputs["rgb"].cpu().numpy()
        # Handle different tensor shapes: (H, W, C) or (C, H, W)
        if rgb_np.shape[0] == 3 and len(rgb_np.shape) == 3:
            rgb_np = rgb_np.transpose(1, 2, 0)
        results["rgb"] = rgb_np.clip(0, 1)
    
    # PBR output
    if "rendered_pbr" in outputs:
        pbr_np = outputs["rendered_pbr"].cpu().numpy()
        # Handle different tensor shapes: (H, W, C) or (C, H, W)
        if pbr_np.shape[0] == 3 and len(pbr_np.shape) == 3:
            pbr_np = pbr_np.transpose(1, 2, 0)
        results["pbr"] = pbr_np.clip(0, 1)
    
    # Albedo
    if "rendered_albedos" in outputs:
        albedo_np = outputs["rendered_albedos"].cpu().numpy()
        if albedo_np.shape[0] == 3 and len(albedo_np.shape) == 3:
            albedo_np = albedo_np.transpose(1, 2, 0)
        results["albedo"] = albedo_np.clip(0, 1)
    
    # Normal
    if "rendered_normal" in outputs:
        normal = outputs["rendered_normal"].cpu().numpy()
        if normal.shape[0] == 3 and len(normal.shape) == 3:
            normal = normal.transpose(1, 2, 0)
        # Convert from [-1, 1] to [0, 1] for visualization
        results["normal"] = (normal + 1) / 2
    
    # Roughness
    if "rendered_roughness" in outputs:
        roughness = outputs["rendered_roughness"].cpu().numpy()
        results["roughness"] = np.repeat(roughness, 3, axis=-1)
    
    # Sun visibility
    if "rendered_sun_visibility" in outputs:
        sun_vis = outputs["rendered_sun_visibility"].cpu().numpy()
        results["sun_visibility"] = np.repeat(sun_vis, 3, axis=-1)
    
    # Diffuse light
    if "diffuse_light" in outputs:
        diffuse = outputs["diffuse_light"].cpu().numpy()
        diffuse = diffuse / (diffuse.max() + 1e-8)  # Normalize
        results["diffuse_light"] = diffuse
    
    # Ground truth
    if "pixels" in image_infos:
        results["gt"] = image_infos["pixels"].cpu().numpy()
    
    # Depth
    if "depth" in outputs:
        depth = outputs["depth"].cpu().numpy()
        # Normalize for visualization
        depth_vis = depth / (depth.max() + 1e-8)
        results["depth"] = np.repeat(depth_vis, 3, axis=-1)
    
    return results


def render_relight_comparison(
    trainer,
    dataset,
    device: torch.device,
    frame_idx: int,
    output_dir: str,
    lighting_presets: List[str] = None
):
    """Render comparison images with different lighting conditions."""
    os.makedirs(output_dir, exist_ok=True)
    
    if lighting_presets is None:
        lighting_presets = list(PRESET_SUN_DIRECTIONS.keys())
    
    logger.info(f"Rendering frame {frame_idx} with {len(lighting_presets)} lighting conditions")
    
    all_results = {}
    
    for preset_name in tqdm(lighting_presets, desc="Rendering lightings"):
        sun_dir_list = PRESET_SUN_DIRECTIONS.get(preset_name)
        sun_dir = normalize_sun_direction(sun_dir_list, device)
        
        results = render_single_frame_with_sun(
            trainer=trainer,
            dataset=dataset,
            device=device,
            frame_idx=frame_idx,
            sun_direction=sun_dir
        )
        
        all_results[preset_name] = results
        
        # Save individual images
        for key, img in results.items():
            # Ensure image is in (H, W, C) format
            if len(img.shape) == 2:
                img = img[..., np.newaxis]
            elif img.shape[0] < img.shape[-1] and len(img.shape) == 3:
                # Likely (C, H, W), convert to (H, W, C)
                img = img.transpose(1, 2, 0)
            
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            save_path = os.path.join(output_dir, f"{preset_name}_{key}.png")
            imageio.imwrite(save_path, img_uint8)
        
        logger.info(f"  {preset_name}: saved {len(results)} images")
    
    # Create comparison grid
    create_comparison_grid(all_results, output_dir)
    
    return all_results


def create_comparison_grid(
    all_results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str
):
    """Create a grid image comparing different lighting conditions."""
    if not all_results:
        return
        
    # Get common keys across all results
    common_keys = set(all_results[list(all_results.keys())[0]].keys())
    for results in all_results.values():
        common_keys &= set(results.keys())
    
    preset_names = list(all_results.keys())
    
    for img_type in common_keys:
        images = []
        for preset_name in preset_names:
            if img_type in all_results[preset_name]:
                images.append(all_results[preset_name][img_type])
        
        if len(images) > 0:
            # Create grid
            n_cols = min(4, len(images))
            n_rows = (len(images) + n_cols - 1) // n_cols
            
            h, w = images[0].shape[:2]
            channels = images[0].shape[2] if len(images[0].shape) > 2 else 1
            
            if channels == 1:
                grid = np.zeros((n_rows * h, n_cols * w), dtype=np.float32)
            else:
                grid = np.zeros((n_rows * h, n_cols * w, channels), dtype=np.float32)
            
            for idx, img in enumerate(images):
                row = idx // n_cols
                col = idx % n_cols
                if len(img.shape) == 2:
                    img = img[..., np.newaxis]
                if channels == 1 and img.shape[-1] == 1:
                    grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img.squeeze()
                else:
                    grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img
            
            grid_uint8 = (grid * 255).astype(np.uint8)
            grid_path = os.path.join(output_dir, f"comparison_{img_type}.png")
            imageio.imwrite(grid_path, grid_uint8)
            logger.info(f"Created comparison grid: {grid_path}")


def render_relight_video(
    trainer,
    dataset,
    device: torch.device,
    output_dir: str,
    sun_direction: Optional[torch.Tensor],
    lighting_name: str,
    fps: int = 10,
    max_frames: int = None
):
    """Render a video with specified lighting condition."""
    os.makedirs(output_dir, exist_ok=True)
    
    data_source = dataset.full_image_set
    num_frames = len(data_source)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
    
    logger.info(f"Rendering video with {num_frames} frames, lighting: {lighting_name}")
    
    # Video writers
    video_path = os.path.join(output_dir, f"relight_{lighting_name}.mp4")
    writer = imageio.get_writer(video_path, fps=fps)
    
    pbr_video_path = os.path.join(output_dir, f"relight_{lighting_name}_pbr.mp4")
    pbr_writer = None
    
    for frame_idx in tqdm(range(num_frames), desc=f"Rendering {lighting_name}"):
        results = render_single_frame_with_sun(
            trainer=trainer,
            dataset=dataset,
            device=device,
            frame_idx=frame_idx,
            sun_direction=sun_direction
        )
        
        # Write RGB
        if "rgb" in results:
            rgb = results["rgb"]
            # Ensure (H, W, C) format
            if len(rgb.shape) == 2:
                rgb = rgb[..., np.newaxis]
            elif rgb.shape[0] < rgb.shape[-1] and len(rgb.shape) == 3:
                rgb = rgb.transpose(1, 2, 0)
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.append_data(rgb_uint8)
        
        # Write PBR
        if "pbr" in results:
            if pbr_writer is None:
                pbr_writer = imageio.get_writer(pbr_video_path, fps=fps)
            pbr = results["pbr"]
            # Ensure (H, W, C) format
            if len(pbr.shape) == 2:
                pbr = pbr[..., np.newaxis]
            elif pbr.shape[0] < pbr.shape[-1] and len(pbr.shape) == 3:
                pbr = pbr.transpose(1, 2, 0)
            pbr_uint8 = (np.clip(pbr, 0, 1) * 255).astype(np.uint8)
            pbr_writer.append_data(pbr_uint8)
    
    writer.close()
    if pbr_writer is not None:
        pbr_writer.close()
    
    logger.info(f"Video saved to: {video_path}")
    if pbr_writer is not None:
        logger.info(f"PBR video saved to: {pbr_video_path}")


def main(args):
    # Setup
    try:
        trainer, dataset, cfg, device = setup_trainer_and_dataset(args)
    except Exception as e:
        logger.error(f"Failed to setup trainer and dataset: {e}")
        raise
    
    # Determine output directory
    checkpoint_dir = os.path.dirname(os.path.abspath(args.resume_from))
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(checkpoint_dir, 'relight_results')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Handle custom sun direction
    if args.custom_sun is not None:
        custom_sun_dir = normalize_sun_direction(args.custom_sun, device)
        logger.info(f"Using custom sun direction: {args.custom_sun} -> {custom_sun_dir.cpu().numpy()}")
    else:
        custom_sun_dir = None
    
    # Determine lighting presets to use
    if args.lighting == "all":
        lighting_presets = list(PRESET_SUN_DIRECTIONS.keys())
    elif args.lighting in PRESET_SUN_DIRECTIONS:
        lighting_presets = [args.lighting]
    else:
        lighting_presets = ["original", "top", "front_top", "back_top"]
    
    # Render video or single frame
    if args.render_video:
        # Render video with specified lighting
        if custom_sun_dir is not None:
            sun_dir = custom_sun_dir
            lighting_name = "custom"
        else:
            sun_dir_list = PRESET_SUN_DIRECTIONS.get(args.lighting)
            if sun_dir_list is None and args.lighting != "original":
                logger.warning(f"Unknown lighting preset: {args.lighting}, using 'original'")
                sun_dir_list = None
            sun_dir = normalize_sun_direction(sun_dir_list, device)
            lighting_name = args.lighting
        
        logger.info(f"Rendering video with lighting: {lighting_name}")
        if sun_dir is not None:
            logger.info(f"Sun direction: {sun_dir.cpu().numpy()}")
        
        render_relight_video(
            trainer=trainer,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            sun_direction=sun_dir,
            lighting_name=lighting_name,
            fps=args.fps,
            max_frames=args.max_frames
        )
    else:
        # Render single frame comparison
        frame_idx = args.frame_idx
        if frame_idx is None:
            # Use middle frame by default
            frame_idx = len(dataset.full_image_set) // 2
        
        if custom_sun_dir is not None:
            # Render with custom sun direction only
            results = render_single_frame_with_sun(
                trainer=trainer,
                dataset=dataset,
                device=device,
                frame_idx=frame_idx,
                sun_direction=custom_sun_dir
            )
            
            for key, img in results.items():
                # Ensure image is in (H, W, C) format
                if len(img.shape) == 2:
                    img = img[..., np.newaxis]
                elif img.shape[0] < img.shape[-1] and len(img.shape) == 3:
                    # Likely (C, H, W), convert to (H, W, C)
                    img = img.transpose(1, 2, 0)
                
                img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                save_path = os.path.join(output_dir, f"custom_{key}.png")
                imageio.imwrite(save_path, img_uint8)
                logger.info(f"Saved: {save_path}")
        else:
            # Render comparison with multiple lightings
            render_relight_comparison(
                trainer=trainer,
                dataset=dataset,
                device=device,
                frame_idx=frame_idx,
                output_dir=output_dir,
                lighting_presets=lighting_presets
            )
    
    logger.info(f"\n✓ Relighting complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relighting script for rendering with different sun directions")
    
    # Required arguments
    parser.add_argument("--resume_from", required=True, type=str,
                        help="Path to checkpoint file (e.g., output/xxx/checkpoint_final.pth)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: checkpoint_dir/relight_results)")
    
    # Lighting options
    parser.add_argument("--lighting", type=str, default="all",
                        choices=list(PRESET_SUN_DIRECTIONS.keys()) + ["all"],
                        help="Preset lighting condition to use")
    parser.add_argument("--custom_sun", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Custom sun direction vector [x, y, z]")
    
    # Frame options
    parser.add_argument("--frame_idx", type=int, default=None,
                        help="Frame index to render (default: middle frame)")
    
    # Video options
    parser.add_argument("--render_video", action="store_true",
                        help="Render video instead of single frame")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video frame rate (default: 10)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to render")
    
    args = parser.parse_args()
    
    main(args)

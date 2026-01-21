#!/usr/bin/env python
"""Simple relight evaluation script."""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import imageio
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--frames", default="0,10,20,30")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "relight_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_indices = [int(x.strip()) for x in args.frames.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Get relight config
    render_config = cfg.trainer.render.render_relight
    light_directions = render_config.light_directions
    
    print(f"Output: {output_dir}")
    print(f"Frames: {frame_indices}")
    print(f"Lights: {len(light_directions)} directions")
    for i, d in enumerate(light_directions):
        print(f"  Light {i}: {d}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = DrivingDataset(data_cfg=cfg.data)
    test_dataset = dataset.test_image_set
    if test_dataset is None:
        print("No test dataset, using main dataset")
        test_dataset = dataset
    print(f"Dataset: {len(test_dataset)} frames")
    
    # Create model
    print("Creating model...")
    model_class = import_str(cfg.model.name)
    model = model_class(cfg.model).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded!")
    
    # Create output dirs
    original_dir = output_dir / "original"
    original_dir.mkdir(exist_ok=True)
    light_dirs = []
    for i in range(len(light_directions)):
        d = output_dir / f"light_{i}"
        d.mkdir(exist_ok=True)
        light_dirs.append(d)
    
    # Render frames
    print(f"\nRendering {len(frame_indices)} frames...")
    for frame_idx in tqdm(frame_indices):
        data = test_dataset[frame_idx]
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).unsqueeze(0)
        
        # Original
        with torch.no_grad():
            outputs = model(data, global_step=999999)
        rendered = outputs.get('rendered_pbr', outputs.get('rgb', outputs['rendered']))
        original_np = rendered[0].cpu().numpy()
        if original_np.shape[0] == 3:
            original_np = original_np.transpose(1, 2, 0)
        original_np = np.clip(original_np, 0, 1)
        original_np = (original_np * 255).astype(np.uint8)
        imageio.imwrite(original_dir / f"frame_{frame_idx:04d}.png", original_np)
        
        # Relit versions
        for i, light_dir in enumerate(light_directions):
            light_tensor = torch.tensor(light_dir, dtype=torch.float32, device=device)
            light_tensor = light_tensor / (torch.norm(light_tensor) + 1e-8)
            light_tensor = light_tensor.unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(data, relight_sun_dir=light_tensor, global_step=999999)
            
            rendered = outputs.get('rendered_pbr', outputs.get('rgb', outputs['rendered']))
            relit_np = rendered[0].cpu().numpy()
            if relit_np.shape[0] == 3:
                relit_np = relit_np.transpose(1, 2, 0)
            relit_np = np.clip(relit_np, 0, 1)
            relit_np = (relit_np * 255).astype(np.uint8)
            imageio.imwrite(light_dirs[i] / f"frame_{frame_idx:04d}.png", relit_np)
    
    print(f"\nDone! Results saved to:")
    print(f"  Original: {original_dir}")
    for i, d in enumerate(light_dirs):
        print(f"  Light {i}: {d}")

if __name__ == "__main__":
    main()

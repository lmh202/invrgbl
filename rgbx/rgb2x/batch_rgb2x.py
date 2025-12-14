#!/usr/bin/env python3
"""Offline RGB->X batch converter.

This script mirrors the Gradio demo (`gradio_demo_rgb2x.py`) but processes an
entire directory of images and stores the generated AOVs alongside the RGB
frames. Typical usage:

```
python rgbx/rgb2x/batch_rgb2x.py \
    --input-dir data/waymo/processed/training/000/images \
    --aovs albedo normal roughness metallic irradiance
```

The outputs will be placed under the parent folder of `--input-dir`, inside
`<aov>_rgbx/` directories (e.g., `albedo_rgbx/000_0.png`). Optionally normals
can also be serialized as NumPy arrays for downstream training.
"""

import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from diffusers import DDIMScheduler
from tqdm import tqdm

from load_image import load_exr_image, load_ldr_image
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    level=os.environ.get("RGB2X_LOGLEVEL", "INFO"),
)
LOGGER = logging.getLogger("rgb2x_batch")

AOV_PROMPTS: Dict[str, str] = {
    "albedo": "Albedo (diffuse basecolor)",
    "normal": "Camera-space Normal",
    "roughness": "Roughness",
    "metallic": "Metallicness",
    "irradiance": "Irradiance (diffuse lighting)",
}

AOV_DIR_MAP: Dict[str, str] = {
    "albedo": "albedo_rgbx",
    "normal": "normal_rgbx",
    "roughness": "rough_rgbx",
    "metallic": "metallic_rgbx",
    "irradiance": "irradiance_rgbx",
}

AOV_EXT_MAP: Dict[str, str] = {
    "albedo": ".jpg",
    "roughness": ".jpg",
    "irradiance": ".jpg",
    "normal": ".png",
    "metallic": ".png",
}

AOV_SEED_OFFSETS: Dict[str, int] = {
    name: (idx + 1) * 9973 for idx, name in enumerate(AOV_PROMPTS.keys())
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".exr"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch RGB->X converter")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/waymo/processed/training/000/images"),
        help="Directory that contains RGB frames (jpg/png/exr).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination root. Defaults to parent of --input-dir.",
    )
    parser.add_argument(
        "--aovs",
        nargs="+",
        default=["albedo", "normal", "roughness", "metallic", "irradiance"],
        help="List of AOV channels to generate.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Diffusion steps per AOV (higher = slower but potentially better).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. A different offset is used per frame.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1000,
        help="Resize the longer image side to this value before inference (>=8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for the diffusion pipeline.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files even if they already exist.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for albedo/roughness/irradiance outputs.",
    )
    parser.add_argument(
        "--save-normal-npy",
        action="store_true",
        help="Additionally store normals as float32 npy files in normals/normal_npy.",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="zheng95z/rgb-to-x",
        help="HuggingFace repo or local path for the RGB->X pipeline.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("rgbx/rgb2x/model_cache"),
        help="Where to cache the downloaded model weights.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Torch dtype for the pipeline weights.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N frames (useful for quick tests).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Avoid downloading models and rely on the local HuggingFace cache.",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()
    ]
    if not files:
        raise RuntimeError(f"No supported images found inside {input_dir}")
    return files


def prepare_photo_tensor(
    photo: torch.Tensor, max_side: int
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    _, old_height, old_width = photo.shape
    if max_side is None or max_side <= 0:
        target_height, target_width = old_height, old_width
    else:
        ratio = old_height / max(old_width, 1)
        if old_height >= old_width:
            target_height = min(max_side, old_height)
            target_width = int(target_height / ratio)
        else:
            target_width = min(max_side, old_width)
            target_height = int(target_width * ratio)

    target_width = max(8, (target_width // 8) * 8)
    target_height = max(8, (target_height // 8) * 8)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Computed invalid resize dimensions")

    if target_width == old_width and target_height == old_height:
        resized = photo
    else:
        resized = T.Resize((target_height, target_width))(photo)
    return resized, (old_height, old_width), (target_height, target_width)


def load_photo(path: Path, device: torch.device) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".exr":
        tensor = load_exr_image(str(path), tonemaping=True, clamp=True)
    else:
        tensor = load_ldr_image(str(path), from_srgb=True)
    return tensor.to(device)


def ensure_output_dirs(root: Path, aovs: Iterable[str]) -> Dict[str, Path]:
    dirs = {}
    for aov in aovs:
        sub_dir = AOV_DIR_MAP.get(aov, f"{aov}_rgbx")
        dst = root / sub_dir
        dst.mkdir(parents=True, exist_ok=True)
        dirs[aov] = dst
    return dirs


def save_image(image, path: Path, quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=quality)
    else:
        image.save(path)


def maybe_save_normal_npy(image, npy_dir: Path, stem: str):
    npy_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0
    np.save(npy_dir / f"{stem}_pred.npy", arr)


def main():
    args = parse_args()
    device = torch.device(args.device)

    missing_prompts = set(args.aovs) - set(AOV_PROMPTS.keys())
    if missing_prompts:
        raise ValueError(f"Unsupported AOVs requested: {sorted(missing_prompts)}")

    input_dir = args.input_dir
    output_root = args.output_root or input_dir.parent
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    total_images = len(images)
    if args.limit is not None:
        images = images[: args.limit]
        LOGGER.info(
            "Limiting processing to %d/%d frames", len(images), total_images
        )
    else:
        LOGGER.info("Found %d frames under %s", total_images, input_dir)
    output_dirs = ensure_output_dirs(output_root, args.aovs)
    normal_npy_dir = output_root / "normals" / "normal_npy"

    torch_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        args.model_repo,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    amp_ctx = (
        torch.cuda.amp.autocast
        if device.type == "cuda" and args.precision == "fp16"
        else nullcontext
    )

    with torch.inference_mode():
        for idx, img_path in enumerate(tqdm(images, desc="RGB->X", unit="img")):
            seed = args.seed + idx
            photo = load_photo(img_path, device)
            photo, (old_h, old_w), (new_h, new_w) = prepare_photo_tensor(
                photo, args.max_side
            )

            for aov in args.aovs:
                dst_dir = output_dirs[aov]
                ext = AOV_EXT_MAP.get(aov, ".png")
                dst_path = dst_dir / f"{img_path.stem}{ext}"
                if dst_path.exists() and not args.overwrite:
                    continue

                gen_seed = seed * 1000 + AOV_SEED_OFFSETS.get(aov, 0)
                generator = torch.Generator(device=device).manual_seed(gen_seed)
                prompt = AOV_PROMPTS[aov]
                with amp_ctx():
                    result = pipe(
                        prompt=prompt,
                        photo=photo,
                        num_inference_steps=args.num_inference_steps,
                        height=new_h,
                        width=new_w,
                        generator=generator,
                        required_aovs=[aov],
                    ).images[0][0]

                result = T.Resize((old_h, old_w))(result)
                save_image(result, dst_path, args.jpeg_quality)

                if aov == "normal" and args.save_normal_npy:
                    maybe_save_normal_npy(result, normal_npy_dir, img_path.stem)

    LOGGER.info("Done. Outputs written under %s", output_root)


if __name__ == "__main__":
    main()

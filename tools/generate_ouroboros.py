#!/usr/bin/env python3
"""
使用 Ouroboros 为 Waymo 场景批量生成先验（albedo, normal, roughness, metallic, irradiance）。

改动点：
- 每张输入图像 images/{frame}_{cam}.jpg 处理后，结果直接保存到场景目录下：
  albedo_ouroboros/{frame}_{cam}.{ext}
  normal_ouroboros/{frame}_{cam}.{ext}
  rough_ouroboros/{frame}_{cam}.{ext}
  metallic_ouroboros/{frame}_{cam}.{ext}
  irradiance_ouroboros/{frame}_{cam}.{ext}
- 同时生成 normals 的 npy：normal_ouroboros/normal_npy/{frame}_{cam}_pred.npy
- Ouroboros 原脚本输出写入“每张图独立的临时目录”，随后立刻搬运/转换到目标目录并清理临时目录。
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import subprocess
import time
import shutil
import tempfile


def get_image_files(images_dir: str) -> List[str]:
    """获取所有图像文件（按 frame_cam 排序）"""
    return sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')])


def _target_dirs(scene_root: str) -> Dict[str, str]:
    """场景内各模态输出目录"""
    return {
        'albedo': os.path.join(scene_root, 'albedo_ouroboros'),
        'normals': os.path.join(scene_root, 'normal_ouroboros'),
        'normals_npy': os.path.join(scene_root, 'normal_ouroboros', 'normal_npy'),
        'roughness': os.path.join(scene_root, 'rough_ouroboros'),
        'metallicity': os.path.join(scene_root, 'metallic_ouroboros'),
        'irradiance': os.path.join(scene_root, 'irradiance_ouroboros'),
    }


def ensure_target_dirs(scene_root: str) -> Dict[str, str]:
    dirs = _target_dirs(scene_root)
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs


def save_normals_npy_from_image(normal_img_path: str, npy_path: str) -> None:
    """
    将法线RGB图转为 [-1,1] 的 float32 array 并保存为 npy。
    注意：如果你用 jpg 会有压缩伪影，建议用 png。
    """
    import numpy as np
    from PIL import Image

    img = Image.open(normal_img_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    np.save(npy_path, arr)


def _convert_and_save(src_path: str, dst_path: str) -> None:
    """
    将 Ouroboros 输出（通常为 png）保存为目标格式（jpg/png）。
    - jpg：RGB 用 RGB 保存，单通道用 L 保存
    - png：保持无损
    """
    from PIL import Image

    dst_ext = Path(dst_path).suffix.lower()
    img = Image.open(src_path)

    if dst_ext in [".jpg", ".jpeg"]:
        # 尽量保持合理的通道
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode == "I;16":
            # 16bit图转8bit（粗略），如果你发现不对建议改用png保存
            img = img.convert("I")
        # 若是单通道
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # JPEG 质量可调
        img.save(dst_path, quality=95, subsampling=0)
    else:
        # png / 其他：直接保存
        img.save(dst_path)


def run_ouroboros_inference_one(
    scene_root: str,
    checkpoint: str,
    input_image: str,
    modalities: List[str],
    save_ext: str,
    seed: int = 0,
    noise: str = "gaussian",
) -> Tuple[bool, str]:
    """
    对单张图像调用 Ouroboros inference，并将结果“直接落盘”到 *_ouroboros 目录。

    返回 (success, message)
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inference_script = os.path.join(repo_root, "Ouroboros", "rgb2x", "inference.py")
    if not os.path.exists(inference_script):
        return False, f"Ouroboros inference script not found: {inference_script}"

    # 目标输出目录
    dirs = ensure_target_dirs(scene_root)
    frame_cam_stem = Path(input_image).stem  # e.g. "000_0"
    save_ext = save_ext.lstrip(".").lower()

    # 每张图独立临时目录，避免相互覆盖
    tmp_root = tempfile.mkdtemp(prefix="ouroboros_tmp_", dir=scene_root)

    cmd = [
        sys.executable,
        inference_script,
        f"--checkpoint={checkpoint}",
        "--condition", "rgb",
        f"--noise={noise}",
        f"--seed={seed}",
        f"--input_rgb_path={input_image}",
        f"--output_dir={tmp_root}",
        "--modality",
    ]
    cmd.extend(modalities)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return False, result.stderr or result.stdout or "Unknown error"

    # Ouroboros 原生输出：tmp_root/<modality>/<modality>.png
    # 我们需要搬运到：scene_root/*_ouroboros/{frame_cam}.{save_ext}
    dst_map = {
        "albedo": os.path.join(dirs["albedo"], f"{frame_cam_stem}.{save_ext}"),
        "normals": os.path.join(dirs["normals"], f"{frame_cam_stem}.{save_ext}"),
        "roughness": os.path.join(dirs["roughness"], f"{frame_cam_stem}.{save_ext}"),
        "metallicity": os.path.join(dirs["metallicity"], f"{frame_cam_stem}.{save_ext}"),
        "irradiance": os.path.join(dirs["irradiance"], f"{frame_cam_stem}.{save_ext}"),
    }

    # 逐模态处理
    for m in modalities:
        # 只处理我们关心且在 dst_map 中的键
        if m not in dst_map:
            continue

        src_path = os.path.join(tmp_root, m, f"{m}.png")
        if not os.path.exists(src_path):
            # 有的实现可能输出 jpg 或别的扩展，这里做一个兜底搜索
            cand = list(Path(tmp_root).glob(f"{m}/{m}.*"))
            if cand:
                src_path = str(cand[0])
            else:
                shutil.rmtree(tmp_root, ignore_errors=True)
                return False, f"Missing output for modality '{m}': expected {tmp_root}/{m}/{m}.png"

        dst_path = dst_map[m]
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        _convert_and_save(src_path, dst_path)

    # normals npy
    normal_img_path = dst_map.get("normals")
    if normal_img_path and os.path.exists(normal_img_path):
        npy_path = os.path.join(dirs["normals_npy"], f"{frame_cam_stem}_pred.npy")
        try:
            save_normals_npy_from_image(normal_img_path, npy_path)
        except Exception as e:
            # 不致命
            print(f"Warning: failed to write normals npy for {frame_cam_stem}: {e}")

    # 清理临时目录
    shutil.rmtree(tmp_root, ignore_errors=True)
    return True, ""


def process_scene(
    scene_idx: str,
    checkpoint: str,
    modalities: List[str],
    save_ext: str = "jpg",
    noise: str = "gaussian",
    seed: int = 0,
    skip_existing: bool = True,
    max_images: int = 0,
):
    """处理单个 Waymo 场景。max_images>0 时仅处理前 N 张用于快速验证。"""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_root = os.path.join(repo_root, "data", "waymo", "processed", "training", scene_idx)
    images_dir = os.path.join(scene_root, "images")

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return

    image_files = get_image_files(images_dir)
    print(f"\nProcessing scene {scene_idx}: {len(image_files)} images")

    success_count = 0
    skip_count = 0
    fail_count = 0

    # 用 normals 作为“是否已处理”的判据
    dirs = ensure_target_dirs(scene_root)
    for img_file in tqdm(image_files, desc=f"Scene {scene_idx}"):
        frame_cam_stem = img_file[:-4]  # remove .jpg
        input_image = os.path.join(images_dir, img_file)

        if max_images and (success_count + skip_count) >= max_images:
            break

        if skip_existing:
            check_file = os.path.join(dirs["normals"], f"{frame_cam_stem}.{save_ext.lstrip('.')}")
            if os.path.exists(check_file):
                skip_count += 1
                continue

        ok, err = run_ouroboros_inference_one(
            scene_root=scene_root,
            checkpoint=checkpoint,
            input_image=input_image,
            modalities=modalities,
            save_ext=save_ext,
            seed=seed,
            noise=noise,
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1
            print(f"\nError processing {input_image}:\n{err}\n")

    print(f"\nScene {scene_idx} Summary:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped (existing): {skip_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total images in images/: {len(image_files)}")


def main():
    parser = argparse.ArgumentParser(description="使用 Ouroboros 为 Waymo 场景批量生成先验（直接落盘到 *_ouroboros）")
    parser.add_argument('--scene_idx', type=str, default='001',
                        help="单个场景如 001，或范围如 000-007")
    parser.add_argument('--checkpoint', type=str, default='./ckpts',
                        help="Ouroboros 权重目录（包含 model_index.json 的目录）")
    parser.add_argument('--modalities', nargs='+',
                        default=['normals', 'albedo', 'irradiance', 'roughness', 'metallicity'])
    parser.add_argument('--save_ext', type=str, default='jpg',
                        help="输出图像扩展名：jpg 或 png（建议 normals/roughness 用 png）")
    parser.add_argument('--noise', type=str, default='gaussian')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_images', type=int, default=0, help='>0 时仅处理前 N 张用于快速验证')
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false')

    args = parser.parse_args()

    if '-' in args.scene_idx:
        start, end = args.scene_idx.split('-')
        scene_indices = [f"{i:03d}" for i in range(int(start), int(end) + 1)]
    else:
        scene_indices = [args.scene_idx]

    print("=" * 80)
    print("Ouroboros Prior Generation for Waymo Scenes")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Modalities: {args.modalities}")
    print(f"Save ext: {args.save_ext}")
    print(f"Scenes: {scene_indices}")
    print(f"Max images: {args.max_images if args.max_images else 'all'}")
    print("=" * 80)

    start_time = time.time()
    for scene_idx in scene_indices:
        process_scene(
            scene_idx=scene_idx,
            checkpoint=args.checkpoint,
            modalities=args.modalities,
            save_ext=args.save_ext,
            noise=args.noise,
            seed=args.seed,
            skip_existing=args.skip_existing,
            max_images=args.max_images,
        )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.2f} minutes")


if __name__ == '__main__':
    main()

# Preparing Waymo Dataset
## 1. Register on Waymo Open Dataset

#### Sign Up for a Waymo Open Dataset Account and Install gcloud SDK

To download the Waymo dataset, you need to register an account at [Waymo Open Dataset](https://waymo.com/open/). You also need to install gcloud SDK and authenticate your account. Please refer to [this page](https://cloud.google.com/sdk/docs/install) for more details.

#### Set Up the Data Directory

Once you've registered and installed the gcloud SDK, create a directory to house the raw data:

```shell
# Create the data directory or create a symbolic link to the data directory
mkdir -p ./data/waymo/raw   
mkdir -p ./data/waymo/processed 
```

## 2. Download the raw data
For the Waymo Open Dataset, we first organize the scene names alphabetically and store them in `data/waymo_train_list.txt`. The scene index is then determined by the line number minus one.

For example, to obtain the 23th, 114th, and 788th scenes from the Waymo Open Dataset, execute:

```shell
python datasets/waymo/waymo_download.py \
    --target_dir ./data/waymo/raw \
    --scene_ids 23 114 327 621 703 172 552 788
```

You can also provide a split file (e.g. `data/waymo_example_scenes.txt`) to download a batch of scenes at once:

```shell
python datasets/waymo/waymo_download.py \
    --target_dir ./data/waymo/raw \
    --split_file data/waymo_example_scenes.txt
```

If you wish to run experiments on different scenes, please specify your own list of scenes.

<details>
<summary>If this script doesn't work due to network issues, also consider manual download:</summary>

Download the [scene flow version](https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) of Waymo.

![Waymo Dataset Download Page](https://github.com/user-attachments/assets/a1737699-e792-4fa0-bb68-0ab1813f1088)

> **Note**: Ensure you're downloading the scene flow version to avoid errors.

</details>

</details>

## 3. Preprocess the data
After downloading the raw dataset, you'll need to preprocess this compressed data to extract and organize various components.

#### Install Waymo Development Toolkit
```shell
pip install waymo-open-dataset-tf-2-11-0==1.6.0
```

#### Running the preprocessing script
To preprocess specific scenes of the dataset, use the following command:
```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_ids 23 114 327 621 703 172 552 788 \
    --workers 8 \
    --process_keys images lidar calib pose dynamic_masks objects
```
Alternatively, preprocess a batch of scenes by providing the split file:
```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --split_file data/waymo_example_scenes.txt \
    --workers 8 \
    --process_keys images lidar calib pose dynamic_masks objects
```
The extracted data will be stored in the `data/waymo/processed` directory.

## 4. Extract Masks

To generate:

- **sky masks (required)** 
- fine dynamic masks (optional)

Follow these steps:

#### Install `SegFormer` (Skip if already installed)

:warning: SegFormer relies on `mmcv-full=1.2.7`, which relies on `pytorch=1.8` (pytorch<1.9). Hence, a seperate conda env is required.

```shell
#-- Set conda env
conda create -n segformer python=3.8
conda activate segformer
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#-- Install mmcv-full
pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf
pip install mmcv-full==1.2.7 --no-cache-dir

#-- Clone and install segformer
git clone https://github.com/NVlabs/SegFormer
cd SegFormer
pip install .
```

Download the pretrained model `segformer.b5.1024x1024.city.160k.pth` from the google_drive / one_drive links in https://github.com/NVlabs/SegFormer#evaluation .

Remember the location where you download into, and pass it to the script in the next step with `--checkpoint` .


#### Run Mask Extraction Script

```shell
conda activate segformer
segformer_path=/data0/SegFormer

python datasets/tools/extract_masks.py \
    --data_root data/waymo/processed/training \
    --segformer_path /data0/SegFormer \
    --checkpoint ./data/waymo/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/waymo_example_scenes.txt \
    --process_dynamic_mask
```
Replace `/pathtosegformer` with the actual path to your Segformer installation.

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 5. Normal & Material Prior

InvRGB+L consumes per-frame priors for surface normal, diffuse albedo, and micro-roughness. We obtain these maps by running an offline geometric reconstructor such as **GeoWizard** (RGB-X checkpoints) on top of the RGB frames stored in `data/waymo/processed/training/<scene>/images`.

### 5.1 Environment & Inputs

1. Clone the GeoWizard / RGB-X repository (or the tool you prefer) to a separate workspace and follow its installation guide. We usually run it inside a dedicated conda env (PyTorch 2.1 + CUDA 12 works well).
2. Copy or symlink the Waymo scene you want to process (e.g. `000`, `023`, …) from `data/waymo/processed/training/`. Each scene already contains undistorted RGB frames for the 5 surrounding cameras (`*_0.jpg` … `*_4.jpg`).
3. Export the scene list you want to process:

```shell
export SCENE=000
export SCENE_ROOT=data/waymo/processed/training/${SCENE}
export GEO_OUT=/data0/geowizard_outputs/${SCENE}
mkdir -p "$GEO_OUT"
```

### 5.2 Run GeoWizard & RGB-X

Feed the RGB frames into GeoWizard (replace the actual command with the one provided by the toolchain):

```shell
python geowizard/geowizard/run_infer.py \
    --input_dir /data0/data/waymo/processed/training/001/images \
    --output_dir /data0/data/waymo/processed/training/001/normal_prior \
    --ensemble_size 3 \
    --denoise_steps 10 \
    --seed 0 \
    --domain "outdoor"
```

GeoWizard writes per-frame PNG/EXR maps (e.g. `000_0_normal.png`, `000_0_albedo.png`, …). The filenames must embed the **frame index** (three digits) and **camera id** (0–4) so they can be matched later.

```shell
python rgbx/rgb2x/batch_rgb2x.py
```

### 5.3 Repack to InvRGB+L layout

The training dataloader (`datasets/base/pixel_source.py`) expects:

```
data/waymo/processed/training/<scene>/
├─ normals/normal_npy/        # float32 .npy normals
├─ albedo_rgbx/               # uint8 .jpg diffuse priors
├─ rough_rgbx/                # uint8 .jpg roughness priors
└─ visibility/                # optional sun-visibility/shadow hint
```

Use the helper snippet below to convert GeoWizard outputs to the expected naming convention (run once per scene). It converts normals to camera-space float arrays in `[-1, 1]`, masks out the sky, and stores albedo/roughness JPEGs with the correct filenames:

```shell
python - <<'PY'
import os, glob
import numpy as np
from PIL import Image

scene = os.environ.get("SCENE", "000")
scene_root = f"data/waymo/processed/training/{scene}"
geo_out = os.environ.get("GEO_OUT", f"/data0/geowizard_outputs/{scene}")

normal_dst = os.path.join(scene_root, "normals", "normal_npy")
albedo_dst = os.path.join(scene_root, "albedo_rgbx")
rough_dst = os.path.join(scene_root, "rough_rgbx")
os.makedirs(normal_dst, exist_ok=True)
os.makedirs(albedo_dst, exist_ok=True)
os.makedirs(rough_dst, exist_ok=True)

sky_mask_cache = {}

def sky_mask(frame, cam):
    key = (frame, cam)
    if key not in sky_mask_cache:
        path = os.path.join(scene_root, "sky_masks", f"{frame:03d}_{cam}.png")
        sky_mask_cache[key] = (np.array(Image.open(path)) < 128)[..., None]
    return sky_mask_cache[key]

for normal_path in sorted(glob.glob(os.path.join(geo_out, "normals", "*.png"))):
    stem = os.path.basename(normal_path).split(".")[0]
    frame_str, cam_str, *_ = stem.split("_")
    frame = int(frame_str)
    cam = int(cam_str)
    normal = np.array(Image.open(normal_path)).astype(np.float32) / 127.5 - 1.0
    normal = normal * sky_mask(frame, cam)  # zero out sky
    np.save(os.path.join(normal_dst, f"{frame:03d}_{cam}_pred.npy"), normal)

for rgb_dir, dst in [("albedo", albedo_dst), ("roughness", rough_dst)]:
    src_dir = os.path.join(geo_out, rgb_dir)
    if not os.path.isdir(src_dir):
        continue
    for img_path in glob.glob(os.path.join(src_dir, "*.png")):
        stem = os.path.basename(img_path).split(".")[0]
        frame_str, cam_str, *_ = stem.split("_")
        Image.open(img_path).save(os.path.join(dst, f"{frame_str}_{cam_str}.jpg"), quality=95)
PY
```

> Notes
> - Normals are stored in **camera space**. The loader automatically rotates them to world coordinates using the per-frame extrinsics (`cam_to_worlds`).
> - Keep the resolution identical to the RGB frames (1920×1280 for Waymo). The dataloader will downscale consistently.
> - If GeoWizard outputs roughness in [0, 1] EXR format, normalize to `[0, 255]` before saving JPGs.

Once these files exist, set `trainer.losses.gt_normal/albedo/roughness` weights as needed in `configs/invrgbl*.yaml` and start training.

## 6. LiDAR Intensity Map

LiDAR intensity maps encourage the model to match the active laser reflectance measured by Waymo. We generate sparse intensity images by projecting LiDAR points (already stored in `lidar/*.bin`) into each camera view.

### 6.1 Input data

- `lidar/{frame:03d}.bin`: packed float32 arrays with columns `[origin(3), point(3), flow(4), ground_flag(1), intensity(1), elongation(1), laser_id(1)]`.
- `intrinsics/<cam>.txt`: focal lengths, principal points, and distortion (9 numbers).
- `extrinsics/<cam>.txt`: 4×4 camera-to-vehicle matrices.
- (Optional) `sky_masks` / dynamic masks to zero-out invalid regions.

### 6.2 Projection script

Create an `intensity/` folder inside each scene and run the following helper to rasterize intensities (uses the same naming pattern as other modalities):

```shell
python - <<'PY'
import os, glob
import numpy as np
from PIL import Image

scene = os.environ.get("SCENE", "000")
scene_root = f"data/waymo/processed/training/{scene}"
img_sample = sorted(glob.glob(os.path.join(scene_root, "images", f"*_0.jpg")))[0]
H, W = Image.open(img_sample).size[1], Image.open(img_sample).size[0]

def load_intr(cam):
    vals = np.loadtxt(os.path.join(scene_root, "intrinsics", f"{cam}.txt"))
    fx, fy, cx, cy = vals[:4]
    k1, k2, p1, p2, k3 = vals[4:9]
    return fx, fy, cx, cy, (k1, k2, p1, p2, k3)

def load_extr(cam):
    return np.loadtxt(os.path.join(scene_root, "extrinsics", f"{cam}.txt")).reshape(4, 4)

def project(points, extr, intr):
    fx, fy, cx, cy, _ = intr
    Rt = np.linalg.inv(extr)  # world -> camera
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cam = (Rt @ pts_h.T).T[:, :3]
    mask = cam[:, 2] > 1e-2
    cam = cam[mask]
    uv = np.zeros_like(cam[:, :2])
    uv[:, 0] = fx * cam[:, 0] / cam[:, 2] + cx
    uv[:, 1] = fy * cam[:, 1] / cam[:, 2] + cy
    return uv, cam[:, 2], mask

intensity_dir = os.path.join(scene_root, "intensity")
os.makedirs(intensity_dir, exist_ok=True)

lidar_frames = sorted(glob.glob(os.path.join(scene_root, "lidar", "*.bin")))

for lidar_path in lidar_frames:
    frame = int(os.path.basename(lidar_path).split(".")[0])
    cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 14)
    pts = cloud[:, 3:6]
    refl = cloud[:, 11]
    for cam in range(5):
        uv, depth, mask = project(pts, load_extr(cam), load_intr(cam))
        canvas = np.zeros((H, W, 1), dtype=np.float32)
        valid = (
            (uv[:, 0] >= 0) & (uv[:, 0] < W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H)
        )
        idx = np.round(uv[valid]).astype(int)
        canvas[idx[:, 1], idx[:, 0], 0] = np.maximum(
            canvas[idx[:, 1], idx[:, 0], 0], refl[mask][valid]
        )
        out_path = os.path.join(intensity_dir, f"{frame:03d}_{cam}.npy")
        np.save(out_path, canvas)
PY
```

Feel free to normalize or log-scale the values afterwards (we typically divide by 255 so the map stays in `[0, 1]`). The loader automatically resizes and sparsifies them using a COO representation, so storing dense float32 arrays is fine.

### 6.3 Verification checklist

1. `intensity/{frame}_{cam}.npy` exists for every frame that will be sampled during training.
2. Values are non-negative and mostly zero except where LiDAR returns exist. You can visualize them quickly: `python -c "import numpy as np; import matplotlib.pyplot as plt; plt.imshow(np.load('...')); plt.show()"`.
3. During training, watch for the `intensity_loss` key to ensure the supervision is picked up.

After these two steps (material priors + intensity maps) the data folder for a scene is fully populated and ready for InvRGB+L training.

## 7. Stable RGB→X Batch Conversion

We provide an automated wrapper around `StableDiffusionAOVMatEstPipeline` to generate albedo/normal/etc. priors directly from the Waymo RGB frames. The script stores its outputs next to the original `images/` folder so the trainers can consume them immediately.

1. Enter the docker container and activate the same `rgbx` environment used for the demo:

    ```shell
    docker exec -it LMH_invrgbl /bin/bash
    source /opt/conda/etc/profile.d/conda.sh
    conda activate rgbx
    cd /data0   # repo lives here inside the container
    ```

2. (Optional) If you need internet access for HuggingFace downloads, export the provided proxy endpoints before running the script:

    ```shell
    export http_proxy="http://127.0.0.1:7890"
    export https_proxy="http://127.0.0.1:7890"
    export all_proxy="socks5://127.0.0.1:7891"
    ```

3. Run the batch converter. The example below processes a single frame for smoke-testing, generates albedo + normals, saves fp16 results, and keeps everything offline if the cache already exists:

    ```shell
    python rgbx/rgb2x/batch_rgb2x.py \
        --input-dir data/waymo/processed/training/000/images \
        --aovs albedo normal roughness metallic irradiance \
        --num-inference-steps 5 \
        --max-side 512 \
        --save-normal-npy \
        --precision fp16 \
        --local-files-only
    ```

    Key flags:

    - `--limit N`: only converts the first *N* frames; omit it for the entire sequence (990 frames per scene).
    - `--save-normal-npy`: adds `normals/normal_npy/{frame}_pred.npy` that match the PNG normals but stay in `[-1, 1]` float space.
    - `--local-files-only`: ensures the diffusion pipeline loads solely from `rgbx/rgb2x/model_cache` (set `--cache-dir` if you store weights elsewhere). Drop the flag to allow downloads.

4. Check the outputs under `data/waymo/processed/training/<scene>/`:

    - `albedo_rgbx/{frame}_{cam}.jpg`
    - `normal_rgbx/{frame}_{cam}.png` + `normals/normal_npy/{frame}_{cam}_pred.npy`
    - `rough_rgbx`, `metallic_rgbx`, `irradiance_rgbx` when requested

   The script skips existing files unless `--overwrite` is passed, so it can be resumed safely.

Once these RGB→X priors exist alongside the LiDAR intensity maps, the dataset contains every supervision used by InvRGB+L.

# Snap-to-3D: Multi-View Object Reconstruction from Smartphone Images

Final project for the 2025-2026 Postgraduate course on Artificial Intelligence with Deep Learning, [UPC School](https://www.talent.upc.edu/).

**Authors:** Marc Castellana, Maria Bertolín, Marc Borràs, Gerard Rosell
**Advisor:** Pablo Vega
**Repository:** [github.com/Final-Project-Snap-3D](https://github.com/Final-Project-Snap-3D) <!-- TODO: exact URL -->

## Table of Contents

1. [Introduction and Motivation](#introduction-and-motivation)
2. [Pipeline Overview](#pipeline-overview)
3. [Datasets](#datasets)
4. [Experiments](#experiments)
   - 4.1 [Experiment 1: U-Net / U²-Net Binary Segmentation from Scratch](#experiment-1-u-net--u²-net-binary-segmentation-from-scratch)
   - 4.2 [Experiment 2: Loss Function Selection](#experiment-2-loss-function-selection)
   - 4.3 [Experiment 3: Data Augmentation Impact on Segmentation](#experiment-3-data-augmentation-impact-on-segmentation)
   - 4.4 [Experiment 4: YOLO26 vs U-Net for Salient Object Detection](#experiment-4-yolo26-vs-u-net-for-salient-object-detection)
   - 4.5 [Experiment 5: VGGT-Ω 3D Reconstruction from Casual Photos](#experiment-5-vggt-ω-3d-reconstruction-from-casual-photos)
   - 4.6 [Experiment 6: Mask Integration Strategy](#experiment-6-mask-integration-strategy)
5. [Final Results](#final-results)
6. [Inference API](#inference-api)
7. [How to Run](#how-to-run)
   - 7.1 [Segmentation Training (U-Net / U²-Net)](#segmentation-training-u-net--u²-net)
   - 7.2 [Segmentation Training (YOLO26)](#segmentation-training-yolo26)
   - 7.3 [Segmentation Inference](#segmentation-inference)
   - 7.4 [VGGT-Ω Inference API](#vggt-ω-inference-api)
8. [Repository Structure](#repository-structure)

---

## Introduction and Motivation

Reconstructing 3D objects from images is a fundamental problem in computer vision with direct applications in e-commerce, digital twins, cultural heritage preservation, and 3D printing. Current state-of-the-art methods either require specialized hardware (LiDAR, structured-light scanners) or produce raw point clouds that need extensive manual post-processing before being usable in tools like Blender.

**Snap-to-3D** is an end-to-end pipeline that converts casual multi-view smartphone photographs of real-world objects into Blender-ready, editable, and 3D-printable meshes (`.obj` / `.stl` / `.ply`). The pipeline has three stages: binary segmentation of the salient object, multi-view 3D reconstruction with camera pose estimation (VGGT-Ω), and mesh export via Poisson surface reconstruction.

<!-- TODO: add a high-level figure of the full pipeline (capture → segmentation → VGGT-Ω → mesh) -->

## Pipeline Overview

The system follows a modular architecture:

```
Smartphone photos (N views)
        │
        ▼
┌──────────────────┐
│   Segmentation   │  U-Net / U²-Net (from scratch) or YOLO26
│   (binary mask)  │  trained on VizWiz
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    VGGT-Omega    │  Full RGB images → depth, poses, point cloud
│  3D Backbone     │  (masks applied post-inference)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Mesh Export    │  Point cloud → Poisson surface → .obj/.stl/.ply
└──────────────────┘
```

The segmentation stage produces binary masks that isolate the object of interest. Crucially, these masks are **not** applied before feeding images to VGGT-Ω — the model relies on background context for parallax estimation, texture matching, and camera pose recovery. Instead, masks are applied post-inference to filter the depth output before unprojecting to a point cloud (Option C, see [Experiment 6](#experiment-6-mask-integration-strategy)).

## Datasets

### VizWiz (Segmentation)

Source: [vizwiz.org/tasks-and-datasets/salient-object-detection](https://vizwiz.org/tasks-and-datasets/salient-object-detection/)

Used to train and evaluate the segmentation models. Contains RGB images of variable size with polygon annotations (one JSON per split: `VizWiz_SOD_{train,val,test}_challenge.json`). Each annotation provides `Ground Truth Dimensions` `[H, W]` and a list of `Salient Object` polygons, converted to binary masks via `cv2.fillPoly`. Images without a valid annotation are filtered out automatically by the `VizWiz` dataset class.

| Split | Images |
|-------|--------|
| Train | <!-- TODO --> |
| Val   | <!-- TODO --> |
| Test  | <!-- TODO --> |

### Evaluation Benchmarks (3D Reconstruction)

<!-- TODO: fill in which datasets you actually used for 3D evaluation -->

| Dataset | Purpose | Scenes |
|---------|---------|--------|
| CO3D | Real-world multi-view objects | <!-- TODO --> |
| DTU | Controlled multi-view | <!-- TODO --> |
| MVImgNet | Large-scale multi-view | <!-- TODO --> |

## Experiments

### Experiment 1: U-Net / U²-Net Binary Segmentation from Scratch

#### Hypothesis

A U-Net architecture trained entirely from scratch (no pretrained encoder) can learn effective binary segmentation on VizWiz if equipped with Batch Normalization to stabilize gradient flow and a suitable combined loss. We expect BatchNorm to be critical when training from zero, since the network has no pretrained feature priors. We additionally evaluate a deeper **U²-Net** (nested RSU blocks with deep supervision) as the default backbone, expecting its multi-scale residual U-blocks to better capture salient objects at varying scales.

#### Experiment Setup

| Component | Details |
|-----------|---------|
| Architecture (`U`) | U-Net, `Block = Conv2d → BN → ReLU → Conv2d → BN → ReLU`. No center cropping — `padding=1` preserves spatial dims. 4-level encoder, 3-level decoder with skip connections |
| Architecture (`U2`, default) | U²-Net: RSU7/6/5/4/4F nested blocks + 6 side outputs fused into the final map (**deep supervision**) |
| Loss | `BCEDiceLoss` (0.3 × BCE + 0.7 × Dice) |
| Optimizer | AdamW (decoupled weight decay) |
| LR | 1e-3 |
| Batch size | 8–32 |
| Image size | 512 × 512 |
| Base channels (U-Net) | 32 |
| Augmentation (train) | `HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`, `RandomBrightnessContrast`, `HueSaturationValue`, `GaussianBlur`, `GaussNoise`, `Normalize` (ImageNet), `ToTensorV2` |
| Augmentation (val/test) | `Resize`, `Normalize` (ImageNet), `ToTensorV2` — deterministic only |
| Metrics | IoU, Dice (computed per-batch in the validation loop) |
| Tracking | Weights & Biases (project `snap-to-3d`) |

Key design decisions: the validation pipeline uses **only deterministic transforms** (Resize, Normalize) so metrics are comparable across epochs; for U²-Net the loss is the mean over all 7 outputs (`d0` + 6 side maps), while IoU/Dice are computed on `d0`. Training is resumable via `last.pt` (model + optimizer + epoch + best val loss).

Implementation: `src/model.py` (architectures), `src/augmentation.py` (pipelines), `src/losses.py` (loss), `src/main.py` (training loop + metrics).

#### Results

<!-- TODO: paste train/val loss curves from W&B -->
<!-- TODO: paste sample predictions (image | GT mask | predicted mask) -->

| Model | Val Dice | Val IoU | Val Loss |
|-------|----------|---------|----------|
| U-Net  | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| U²-Net | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

#### Conclusions

<!-- TODO: did BatchNorm prove essential? Did U²-Net beat U-Net? Where does the model fail (small objects, cluttered backgrounds)? Is accuracy sufficient for downstream 3D? -->

---

### Experiment 2: Loss Function Selection

#### Hypothesis

A combined `BCEDiceLoss` will outperform standalone BCE or Dice. BCE provides pixel-wise gradient signal across the whole image, while Dice directly optimizes overlap and handles the strong class imbalance (background ≫ foreground). We expect a Dice-heavy weighting to perform best given VizWiz's background-to-object imbalance.

#### Experiment Setup

Three loss configurations, all other hyperparameters fixed (same backbone, AdamW, lr=1e-3, 512×512):

| Config | Loss | Weighting |
|--------|------|-----------|
| A | BCE only (`BCEWithLogitsLoss`) | — |
| B | Dice only | — |
| C | `BCEDiceLoss` | 0.3 × BCE + 0.7 × Dice |

#### Results

<!-- TODO: comparative loss curves -->

| Config | Val Dice | Val IoU |
|--------|----------|---------|
| A (BCE) | <!-- TODO --> | <!-- TODO --> |
| B (Dice) | <!-- TODO --> | <!-- TODO --> |
| C (0.3 BCE + 0.7 Dice) | <!-- TODO --> | <!-- TODO --> |

#### Conclusions

<!-- TODO: did the combined 0.3/0.7 loss confirm the hypothesis? -->

---

### Experiment 3: Data Augmentation Impact on Segmentation

#### Hypothesis

Geometric and photometric augmentations will improve generalization on VizWiz, whose images vary widely in lighting, camera angle, and object placement. Augmentations applied jointly to image and mask (via Albumentations) should reduce overfitting versus training without augmentation.

#### Experiment Setup

Two training runs, identical except for augmentation:

| Config | Augmentation |
|--------|-------------|
| Baseline | `Resize` + `Normalize` only |
| Augmented | `HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`, `RandomBrightnessContrast`, `HueSaturationValue`, `GaussianBlur`, `GaussNoise` + `Normalize` |

Both use `BCEDiceLoss` (0.3/0.7), AdamW, lr=1e-3.

#### Results

<!-- TODO: comparative loss/Dice curves -->

| Config | Val Dice | Val IoU | Overfitting epoch |
|--------|----------|---------|-------------------|
| Baseline | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Augmented | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

#### Conclusions

<!-- TODO -->

---

### Experiment 4: YOLO26 vs U-Net for Salient Object Detection

#### Hypothesis

YOLO26 (pretrained on COCO, fine-tuned on VizWiz) should converge faster and potentially reach higher mask quality than a U-Net trained from scratch, thanks to strong pretrained features. However, YOLO treats the task as instance segmentation (detection + mask), adding localization overhead that is unnecessary for a single-class binary task.

#### Experiment Setup

| Component | U-Net / U²-Net | YOLO26 |
|-----------|----------------|--------|
| Architecture | Custom U-Net / U²-Net + BN | `yolo26s-seg` (Ultralytics) |
| Pretrained | No (from scratch) | COCO pretrained |
| Loss | `BCEDiceLoss` | YOLO built-in (box + seg + cls + dfl) |
| Optimizer | AdamW | SGD (YOLO default) |
| Image size | 512 | 512 |
| Batch size | 8–32 | 4 |
| Epochs | <!-- TODO --> | 100 (patience 20) |
| GPU | <!-- TODO --> | GTX 1060 6 GB (AMP on) |

Data conversion: VizWiz JSON → YOLO segmentation format (one normalized-polygon `.txt` per image, class `0 = salient_object`) via `src/convert_vizwiz_to_yolo.py`, which builds a sibling `data_yolo/` directory plus `vizwiz.yaml`.

**Evaluation note:** YOLO outputs one mask per detected instance. To compare against the single-mask U-Net/U²-Net output, all YOLO instance masks are merged into one binary mask (`np.maximum`) before computing Dice/IoU (`src/test_evaluation.py`). The W&B callback also logs `mAP50(M)` and a Dice/IoU proxy derived from it.

#### Results

<!-- TODO: YOLO training curves (mAP50, mAP50-95 from results.csv) -->
<!-- TODO: side-by-side prediction comparison -->

| Model | Val Dice | Val IoU | mAP50(M) | mAP50-95(M) | Training time |
|-------|----------|---------|----------|-------------|---------------|
| U-Net / U²-Net | <!-- TODO --> | <!-- TODO --> | — | — | <!-- TODO --> |
| YOLO26s | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

#### Conclusions

<!-- TODO: which model was selected for the final pipeline and why? -->

---

### Experiment 5: VGGT-Ω 3D Reconstruction from Casual Photos

#### Hypothesis

VGGT-Ω (CVPR 2026 Best Paper Finalist, VGG Oxford + Meta AI) — a feed-forward transformer that jointly predicts camera poses, depth maps, and 3D point clouds in a single forward pass — can replace the traditional COLMAP + OpenMVS pipeline for our use case. We expect faster inference, no need for sequential feature matching, and competitive reconstruction quality on casual smartphone captures.

#### Experiment Setup

| Component | Details |
|-----------|---------|
| Model | VGGT-Ω 1B (512 px), checkpoint `vggt_omega_1b_512.pt` |
| Tokenizer | DINOv3 `patch_embed` (3-channel RGB input) |
| Attention | Camera tokens, register tokens, cross-view |
| Decoding heads | Depth, camera extrinsics/intrinsics, point cloud |
| Input | N full RGB images (background **not removed** — required for parallax and camera estimation); CUDA required (`torch.autocast(device_type="cuda")`) |
| Output | Per-view depth maps, camera poses, fused 3D point cloud → Poisson mesh |
| Baseline | COLMAP (SfM) + OpenMVS (dense) |
| Metrics | Chamfer Distance, F-Score, SSIM, LPIPS |

Implementation: `vggt_omega/inference_vggt.py` (inference), `vggt_omega/models/` (VGGTOmega), `vggt_omega/visualize_predictions.py` (point cloud / mesh / depth export).

<!-- TODO: add architecture diagram -->

#### Results

<!-- TODO: quantitative comparison vs COLMAP+OpenMVS -->
<!-- TODO: qualitative point cloud / mesh visualizations + timing -->

| Method | Chamfer Distance ↓ | F-Score ↑ | SSIM ↑ | LPIPS ↓ | Time / scene |
|--------|--------------------:|----------:|-------:|--------:|-------------:|
| COLMAP + OpenMVS | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| VGGT-Ω | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

#### Conclusions

<!-- TODO -->

---

### Experiment 6: Mask Integration Strategy

#### Hypothesis

The way segmentation masks are integrated into the 3D backbone significantly affects output quality. Removing the background before inference (Option A) should degrade reconstruction because VGGT-Ω relies on background for parallax, texture matching, and camera estimation. A post-inference filtering approach (Option C) should preserve these cues while still isolating the object in the final point cloud.

#### Experiment Setup

| Option | Strategy | Description |
|--------|----------|-------------|
| A | Pre-inference mask | Zero out background pixels before feeding to VGGT-Ω |
| B | RGBM fine-tuning | Expand DINOv3 `patch_embed` Conv2d from 3→4 channels, fine-tune with concatenated mask |
| C | Post-inference filter | Feed full RGB to VGGT-Ω, apply the (YOLO-seg) object mask to the depth/point output before unprojecting |

Option A was ruled out by advisor guidance: VGGT-Ω extracts critical geometric cues from the background. Option B requires fine-tuning the backbone and was not pursued due to time constraints. **Option C** is the implemented strategy: segmentation runs independently and the mask filters the prediction post-inference (`vggt_omega/segmentation.py::add_object_masks`, wired through `inference_vggt.run_inference(seg_checkpoint=...)`).

#### Results

<!-- TODO: point cloud comparisons for Options A vs C + Chamfer/F-Score -->

| Option | Chamfer Distance ↓ | F-Score ↑ | Notes |
|--------|--------------------:|----------:|-------|
| A (pre-mask) | <!-- TODO --> | <!-- TODO --> | Camera estimation degrades on masked images |
| C (post-filter) | <!-- TODO --> | <!-- TODO --> | Preserves full geometric context |

#### Conclusions

<!-- TODO: confirm Option C as final choice and explain why -->

---

## Final Results

<!-- TODO: end-to-end pipeline results -->
<!-- TODO: final mesh visualizations (Blender renders, 3D prints if available) -->
<!-- TODO: full-pipeline metrics on test objects -->

## Inference API

A FastAPI service wraps the full reconstruction pipeline (`vggt_omega/api/main.py`). It accepts N images in a single request and returns a reconstructed mesh (`.obj` / `.stl` / `.ply`) ready for Blender or 3D printing, or optionally the raw point cloud and depth / object-mask PNGs. Inference is serialised behind a lock (single shared GPU); the endpoint validates upload count (≤ 32), per-file size (≤ 25 MB) and extension before running.

> ⚠️ Requires a CUDA GPU. The forward pass uses `torch.autocast(device_type="cuda")`, so CPU-only hosts get `503`.

Full API specification: [`vggt_omega/api/API_SPEC.md`](vggt_omega/api/API_SPEC.md)

### Quick usage

```bash
# Reconstruct a mesh from N images (with segmentation + depth export)
curl -X POST http://localhost:8000/api/v1/inference \
  -F "images=@img1.jpg" -F "images=@img2.jpg" \
  -F "export_format=mesh" -F "mesh_format=obj" \
  -F "segment=true" -F "export_depth=true"

# Download a single artifact
curl -OJ http://localhost:8000/api/v1/jobs/<job_id>/files/scene.obj

# Or the full job as a ZIP
curl -OJ http://localhost:8000/api/v1/jobs/<job_id>/archive
```

Health check: `GET /health` (reports device, CUDA availability, checkpoint presence). Interactive docs (Swagger UI): `http://localhost:8000/docs`.

## How to Run

### Segmentation Training (U-Net / U²-Net)

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download VizWiz into data/ (train/ val/ test/ + annotations/)
#    https://vizwiz.org/tasks-and-datasets/salient-object-detection/

# 3. Train (U²-Net is the default; pass --model_name U for U-Net)
python src/main.py --model_name U2 --image_size 512 --batch_size 16 --epochs 20 --lr 1e-3

# 4. Resume from a checkpoint
python src/main.py --resume checkpoints/last.pt

# 5. Visualize dataset samples (raw + augmented)
python src/data_visualization.py     # writes viz_{i}.png and viz_transformed_{i}.png
```

**Key CLI arguments** (`src/main.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `U2` | `U2` = U²-Net, `U` = U-Net |
| `--in_channels` | 3 | Input channels (RGB) |
| `--num_classes` | 1 | Output classes (binary) |
| `--base_channels` | 32 | Filters in first U-Net encoder layer |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate (AdamW) |
| `--image_size` | 512 | Resize dimension |
| `--log_image_every` | 5 | Log validation images to W&B every N epochs |
| `--train_images_dir` | `data/train` | Training images path |
| `--val_images_dir` | `data/val` | Validation images path |
| `--train_annotations` | `data/annotations/VizWiz_SOD_train_challenge.json` | Training annotations |
| `--val_annotations` | `data/annotations/VizWiz_SOD_val_challenge.json` | Validation annotations |
| `--test_images_dir` | `data/test` | Test images path (evaluated with best model after training) |
| `--checkpoint_dir` | `checkpoints` | Where `best_model.pt` / `last.pt` / `final_model.pt` are saved |
| `--resume` | `None` | Path to `last.pt` to resume training |

### Segmentation Training (YOLO26)

```bash
# 1. Install
pip install ultralytics

# 2. Convert VizWiz annotations to YOLO format (one-time → data_yolo/)
python src/convert_vizwiz_to_yolo.py

# 3. Train (tuned for GTX 1060 6 GB)
python src/train_yolo.py --model yolo26s-seg.pt --epochs 100 --batch 4 --imgsz 512

# If CUDA is not detected (torch.cuda.is_available() == False):
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

`--model` accepts `yolo26n-seg.pt` (smallest) | `yolo26s-seg.pt` | `yolo26m-seg.pt`. Raise `--batch` to 8 if VRAM allows, lower to 2 on OOM. Runs are logged to W&B and saved under `runs/yolo_vizwiz/exp/`.

### Segmentation Inference

```bash
# Unified inference: auto-detects UNet/U2Net (.pt with model_state_dict) vs YOLO (.pt)
python src/inference.py --model_path checkpoints/best_model.pt \
  --input_path path/to/images/ --output_dir outputs/ --image_size 512

# Standalone evaluation (IoU / Dice / precision / recall)
python src/test_evaluation.py --model_path checkpoints/best_model.pt \
  --images_dir data/test --annotations data/annotations/VizWiz_SOD_test_challenge.json
```

### VGGT-Ω Inference API

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r vggt_omega/api/requirements.txt

# 2. Configure checkpoint + device
export VGGT_CHECKPOINT=/path/to/vggt_omega_1b_512.pt
export VGGT_DEVICE=cuda

# 3. Start server (from repo root)
uvicorn vggt_omega.api.main:app --host 0.0.0.0 --port 8000
```

You can also run the reconstruction directly from the CLI:

```bash
python -m vggt_omega.inference_vggt -c /path/to/vggt_omega_1b_512.pt \
  -i imgA.jpg imgB.jpg imgC.jpg --output scene.obj
```

Environment variables (defaults in [`vggt_omega/api/constants.py`](vggt_omega/api/constants.py)):

| Variable | Default | Description |
|----------|---------|-------------|
| `VGGT_CHECKPOINT` | `checkpoints/vggt_omega_1b_512.pt` | Path to the VGGT-Ω checkpoint (required; not committed) |
| `VGGT_SEG_CHECKPOINT` | `yolo26s-seg.pt` | Segmentation checkpoint for object filtering |
| `VGGT_OUTPUT_DIR` | `api_outputs` | Per-job output artifacts directory |
| `VGGT_DEVICE` | `cuda` | Inference device |
| `VGGT_RESOLUTION` | `512` | Preprocessing resolution (multiple of 16) |
| `VGGT_MODE` | `balanced` | Preprocessing mode: `balanced` or `max_size` |
| `VGGT_CONF_THRES` | `20.0` | Confidence percentile for point filtering |
| `VGGT_MAX_IMAGES` | `32` | Max images per request |

## Repository Structure

```
├── data/                          # VizWiz dataset (not tracked)
│   ├── train/ val/ test/          # RGB images
│   └── annotations/               # VizWiz_SOD_{train,val,test}_challenge.json
├── data_yolo/                     # YOLO-format dataset (generated, not tracked)
├── src/
│   ├── dataset.py                 # VizWiz PyTorch Dataset (polygon → mask)
│   ├── augmentation.py            # Albumentations train / val_test pipelines
│   ├── model.py                   # U-Net + U²-Net (RSU blocks) architectures
│   ├── losses.py                  # BCEDiceLoss
│   ├── main.py                    # Training loop, metrics, checkpoints, W&B
│   ├── wandb_logger.py            # W&B logging + checkpoint serialization
│   ├── utils.py                   # TaskType enum
│   ├── inference.py               # Unified UNet / U²-Net / YOLO inference
│   ├── test_evaluation.py         # IoU / Dice / precision / recall evaluation
│   ├── data_visualization.py      # Sample visualization
│   ├── convert_vizwiz_to_yolo.py  # VizWiz JSON → YOLO format converter
│   └── train_yolo.py              # YOLO26 training + W&B callbacks
├── vggt_omega/
│   ├── inference_vggt.py          # VGGT-Ω inference (CLI + run_inference)
│   ├── segmentation.py            # Option C: post-inference mask filtering
│   ├── visualize_predictions.py   # Point cloud / mesh / depth / mask export
│   ├── models/                    # VGGTOmega model
│   ├── utils/                     # load_fn, pose_enc, ...
│   ├── api/
│   │   ├── main.py                # FastAPI application
│   │   ├── constants.py           # Env-configurable settings (checkpoints/ → root fallback)
│   │   ├── API_SPEC.md            # Full API specification
│   │   └── requirements.txt
│   └── requirements.txt
├── tools/                         # Standalone reconstruction helpers
│   ├── debug_mesh.py              # Point cloud → Poisson mesh (with diagnostics)
│   └── render_preview.py          # Quick PNG preview of the point cloud
├── checkpoints/                   # Model weights (not tracked)
│   ├── vggt_omega_1b_512.pt       # VGGT-Ω backbone (~4.3 GB)
│   ├── best_model_U2Net.pt        # Trained U²-Net segmentation
│   └── yolo26s-seg.pt             # YOLO26-seg checkpoint
├── inference_test/                # End-to-end reconstruction sandbox
│   ├── images/                    # Sample multi-view input photos
│   └── outputs/                   # Generated artifacts (predictions.pt, scene.*, preview.png — not tracked)
├── requirements.txt
└── README.md
```

---
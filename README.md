# Snap-to-3D — Final Project Report

> **Repository:** https://github.com/Final-Project-Snap-3D/SegmentationModel_PP_AI

**Snap-to-3D** is an end-to-end pipeline that takes a set of photographs of an object shot from different viewpoints and generates a 3D mesh of that object, ready to be imported into tools such as **Blender** for further work.

The project is organized into **three tasks** chained together, plus a final integration layer:

```
  Multi-view photos
        │
        ▼
┌─────────────────────┐      ┌──────────────────────────┐      ┌────────────────┐
│ 1. SEGMENTATION     │      │ 2. 3D RECONSTRUCTION     │      │ 3. MESHING     │
│ YOLO26-seg          │─────▶│ VGGT-Omega               │─────▶│ Point cloud    │
│ (binary mask of     │      │ (depth + camera poses    │      │ → 3D mesh      │
│  the object)        │      │  → point cloud of the    │      │ (.ply → mesh)  │
│                     │      │  object only)            │      │                │
└─────────────────────┘      └──────────────────────────┘      └────────────────┘
        │                              │                              │
        └──────────────┬───────────────┴──────────────────────────────┘
                       ▼
         HTTP API (FastAPI) + mobile application
         (photos as input → object/mesh as output)
```

---

## Table of contents

1. [How to run the code](#how-to-run-the-code)
2. [Task 1 — Object segmentation (Experiments 1–4)](#task-1--object-segmentation-vizwiz)
3. [Task 2 — 3D reconstruction with VGGT-Omega (Experiments 5–6)](#task-2--3d-reconstruction-with-vggt-omega)
4. [Task 3 — 3D mesh generation](#task-3--3d-mesh-generation)
5. [Integration — API and mobile app](#integration--api-and-mobile-app)
6. [Repository structure](#repository-structure)
7. [Summary of key decisions](#summary-of-key-decisions)

---

## How to run the code

### Installation

```bash
git clone https://github.com/Final-Project-Snap-3D/SegmentationModel_PP_AI.git
cd SegmentationModel_PP_AI
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If PyTorch does not detect your GPU (`torch.cuda.is_available()` returns `False`), reinstall it with CUDA support:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Dataset

Download the **VizWiz Salient Object Detection** dataset from
https://vizwiz.org/tasks-and-datasets/salient-object-detection/ and place it as:

```
data/
├── train/ , val/ , test/          # .jpg images
└── annotations/                   # one JSON per split
```

Each annotation entry has the form:

```json
"VizWiz_train_00000000.jpg": {
    "Full Screen": false,
    "Total Polygons": 1,
    "Ground Truth Dimensions": [H, W],
    "Salient Object": [[[x1, y1], [x2, y2], ...]]
}
```

`Salient Object` is a list of polygons (vertices `[x, y]`, relative to `Ground Truth Dimensions`) delimiting the object; they are converted to a binary mask with `cv2.fillPoly`. Images without a JSON entry are filtered out automatically.

To visually inspect dataset samples (original and augmented):

```bash
python src/data_visualization.py
```

### Train U-Net / U2-Net

```bash
# U2-Net (default). Use --model_name U to train the U-Net baseline instead.
python src/main.py --image_size 512 --batch_size 16 --epochs 100 --lr 1e-3

# Resume an interrupted training from the last checkpoint
python src/main.py --resume checkpoints/last.pt --epochs 100
```

Training logs (losses, IoU/Dice, validation images) go to **Weights & Biases**; checkpoints are stored in `checkpoints/` (`best_model.pt`, `last.pt`, `final_model.pt`).

To smoke-test the training loop on a handful of samples, create `data/one_image_train` / `data/one_image_val` folders with a few images and run:

```bash
python src/main.py --batch_size 1 --train_images_dir data/one_image_train --val_images_dir data/one_image_val
```

### Train YOLO26-seg

```bash
# 1. One-time dataset conversion: VizWiz JSON → YOLO polygon format (data_yolo/)
python src/convert_vizwiz_to_yolo.py

# 2. Train (defaults: yolo26s-seg, imgsz 512, batch 4, AMP — tuned for GTX 1060 6GB)
python src/train_yolo.py --epochs 100
```

### Evaluate a segmentation checkpoint

```bash
python src/test_evaluation.py --model_path checkpoints/best_model.pt \
    --images_dir data/test --annotations data/annotations/VizWiz_SOD_test_challenge.json
```

### Run the full 3D reconstruction (VGGT-Omega + YOLO)

Requires a **CUDA GPU** (the forward pass uses `torch.autocast("cuda")`) and the
VGGT-Omega checkpoint `vggt_omega_1b_512.pt` (not committed to the repo).

```bash
pip install -r vggt_omega/requirements.txt

python -m vggt_omega.inference_vggt \
    -c checkpoints/vggt_omega_1b_512.pt \
    -i images-reconstruction/*.HEIC \
    --seg-checkpoint checkpoints/best_yolo.pt \
    --morph-open --keep-largest \
    --mask-dir masks \
    -o object.ply
```

Phone HEIC/HEIF images are supported via `pillow-heif`. The output `object.ply`
is a colored point cloud of the segmented object in world coordinates.

### Run the HTTP API (used by the mobile app)

```bash
pip install -r vggt_omega/api/requirements.txt
export VGGT_CHECKPOINT=/path/to/vggt_omega_1b_512.pt
uvicorn vggt_omega.api.main:app --host 0.0.0.0 --port 8000
```

Full spec in [`vggt_omega/api/API_SPEC.md`](vggt_omega/api/API_SPEC.md); interactive docs at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -F "images=@imageA.jpg" -F "images=@imageB.jpg" \
  -F "segment=true" -F "export_masks=true"
```

### Interactive Gradio demo

```bash
python vggt_omega/demo_gradio.py --checkpoint /path/to/vggt_omega_1b_512.pt
```

---

## Task 1 — Object segmentation (VizWiz)

The goal of this task is to obtain, for every photo, a **binary mask** separating the object of interest from the background. Models were trained on the **VizWiz Salient Object Detection** dataset (RGB images with polygon annotations of the salient object, converted to binary masks with `cv2.fillPoly`). Its most important characteristic is that the images are taken by visually impaired people, so the dataset captures everyday objects in different positions, orientation, contrast, and overall low quality photos, making it suitable for our task at hand.

We iterated over three architectures, each run as its own experiment.

### Experiment 1 — U-Net (baseline)

**Hypothesis.** A U-Net trained from scratch on VizWiz should learn the salient-object-detection task well enough to serve as a baseline, and will validate that the dataset, augmentation and training loop are sound before investing in larger architectures.

**Experiment setup.**
- Own implementation from scratch (`src/model.py`, class `SegmentationModel`): classic encoder–decoder with skip connections, adapted with padding (no center-cropping) and **Batch Normalization** between convolution and ReLU (recommended when training from scratch).
- Training (`src/main.py`): **BCE + Dice** loss (weights 0.3 / 0.7), **AdamW** optimizer, 512×512 images with ImageNet normalization.
- Data augmentation with Albumentations (`src/augmentation.py`): horizontal flip, rotations, ShiftScaleRotate, brightness/contrast, HSV, blur and Gaussian noise — train only; val/test have no randomness so metrics stay comparable across epochs.
- Metrics: IoU and Dice per epoch, logging to **Weights & Biases** (loss, metrics, validation images with mask overlays) and `best`/`last` checkpoints with training *resume* support (`--resume`).

**Results.**
<!-- TODO: paste the final numbers from the W&B run -->
| Metric | Val | Test |
|---|---|---|
| IoU  | _fill in_ | _fill in_ |
| Dice | _fill in_ | _fill in_ |

**Conclusions.** The training pipeline works end-to-end and the model learns the task, but mask quality on hard VizWiz images (blur, clutter, partial objects) leaves room for improvement — which motivates trying an architecture designed specifically for salient object detection (→ Experiment 2).

### Experiment 2 — U2-Net

**Hypothesis.** U²-Net, designed specifically for *salient object detection*, should outperform the plain U-Net at similar training cost thanks to its nested RSU blocks (richer multi-scale receptive fields) and deep supervision.

**Experiment setup.**
- **U²-Net** architecture (`src/model.py`, `U2Net`): nested RSU blocks (`RSU7…RSU4F`).
- Trained with **deep supervision**: the forward pass returns 7 outputs (`d0…d6`, the fused map plus 6 side outputs) and the loss is averaged over all of them; at inference time only `d0` is used.
- Same data pipeline, loss and logging as the U-Net (selectable via `--model_name U2`), so the comparison isolates the effect of the architecture.

**Results.**
<!-- TODO: paste the final numbers from the W&B run -->
| Metric | Val | Test |
|---|---|---|
| IoU  | _fill in_ | _fill in_ |
| Dice | _fill in_ | _fill in_ |

**Conclusions.** U2-Net improves over the U-Net baseline, confirming that multi-scale features with deep supervision help on this task. However, training a salient-object model from scratch remains data-hungry; a pretrained, production-grade segmenter might extract more from the same dataset (→ Experiment 3).

### Experiment 3 — YOLO26-seg (final choice)

**Hypothesis.** A pretrained instance-segmentation model (YOLO26-seg) fine-tuned on VizWiz, treating salient object detection as single-class segmentation, should beat both from-scratch networks in mask quality and inference speed.

**Experiment setup.**
- **YOLO26 instance segmentation** from Ultralytics (`yolo26s-seg`), fine-tuned on VizWiz.
- Requires a dataset conversion step: `src/convert_vizwiz_to_yolo.py` transforms the VizWiz JSONs into YOLO format (one `.txt` per image with normalized polygons) and generates `data_yolo/` with its `vizwiz.yaml`.
- Training (`src/train_yolo.py`): imgsz 512, batch 4, AMP, early stopping (patience 20), tuned for a GTX 1060 6GB. W&B logging via custom callbacks, including an IoU/Dice *proxy* derived from `mAP50(M)` so results can be compared against U-Net/U2-Net; predicted instance masks are merged into a single binary mask per image for the comparison.

**Results.**
<!-- TODO: paste the final numbers from runs/yolo_vizwiz/exp/results.csv and W&B -->
| Metric | Value |
|---|---|
| mAP50(M)    | _fill in_ |
| mAP50-95(M) | _fill in_ |
| IoU (merged masks, test)  | _fill in_ |
| Dice (merged masks, test) | _fill in_ |

**Conclusions.** **We kept YOLO26** as the segmenter of the final pipeline (better mask quality and inference than our own networks). U2-Net remains available as an alternative, and there is a **mixed inference mode** that combines both masks with a pixel-wise logical AND (only pixels that both models consider foreground survive).

### Experiment 4 — Mask post-processing

**Hypothesis.** Raw predicted masks contain noise (small spurious regions, secondary detections) that, if kept, turns into floating "ghost" blobs in the 3D point cloud. Classic morphology should clean them without harming the main object.

**Experiment setup.** Two operations applied to the binary masks (`vggt_omega/segmentation.py`):

1. **Morphological opening** (`--morph-open`): erosion followed by dilation with an **elliptical structuring element of size 21** (configurable via `--morph-kernel`). It removes small regions and thin connections between blobs while preserving the main compact object. Moreover, by breaking those thin "bridges", it leaves spurious regions as separate connected components…
2. **…which are then discarded by keeping only the largest connected component** (`--keep-largest`): connected-component labeling with 8-connectivity (`cv2.connectedComponentsWithStats`) and selection of the largest-area component per frame. The result is a single clean region per image: the object of interest.

Kernel sizes were swept qualitatively; 21 was the smallest that reliably removed noise without eating thin parts of the object.

**Results.** Per-frame debug masks (`--masks-debug`) before/after post-processing show spurious blobs and background leaks removed; the resulting point clouds contain a single clean object without floating fragments.
<!-- TODO: optionally add a before/after mask figure and a before/after point-cloud screenshot -->

**Conclusions.** A single clean region per image is exactly what the 3D stage needs: the cleanup translates directly into point clouds free of disconnected debris, at negligible cost. The kernel size is the key hyperparameter and is exposed as a CLI flag.

---

## Task 2 — 3D reconstruction with VGGT-Omega

Given the photos (and their masks), this task generates the **3D point cloud of the object**.

### Background: VGGT-Omega

We use **VGGT-Omega 1B (512)**, a feed-forward multi-view model that, in a single pass over all images at once, predicts:

- **Camera poses** (extrinsics and intrinsics, decoded from the `pose_enc`),
- **Depth maps** per frame,
- A **depth confidence** map (`depth_conf`).

Using the intrinsics/extrinsics, the depth maps are **unprojected** into 3D points in world coordinates (`unproject_depth_map_to_point_map`), and each point's color is taken from the corresponding image pixel.

### Experiment 5 — Scene reconstruction and PLY export

**Hypothesis.** VGGT-Omega should recover consistent geometry and camera poses from a handful of casual phone photos in a single forward pass, without the per-scene optimization that classic SfM/NeRF pipelines require.

**Experiment setup.**
- CLI (`vggt_omega/inference_vggt.py`) that loads N images (JPG/PNG/**HEIC** via `pillow-heif`…), runs one forward pass and unprojects the depth maps into a colored cloud.
- Cleanup filters, all operating on the confidence array: **confidence percentile threshold** (`--conf-thres`, default 20) dropping the least reliable points; a **depth-edge filter** that detects sharp relative depth jumps (typical "contour" artifacts) and zeroes them out; optional black/white background filters and sky filter (skyseg ONNX); subsampling to `max_points` to keep the cloud manageable.
- The original upstream export was a GLB scene including the reconstructed **camera frustums**; we refactored it to a **points-only PLY** in world coordinates (the standard input for MeshLab/CloudCompare and for Task 3), removing the camera visualization entirely. The interactive **Gradio demo** (`vggt_omega/demo_gradio.py`) was ported to PLY as well.

**Results.** Consistent, correctly-posed point clouds of the full scene from a few phone photos (including mixed portrait/landscape HEIC input, which is padded to a common size).
<!-- TODO: add a screenshot of a reconstructed scene point cloud -->

**Conclusions.** The reconstruction contains the whole scene — table, background, walls — while we only want the object. Cropping the cloud by hand is not scalable: the segmentation masks from Task 1 should do it automatically (→ Experiment 6).

### Experiment 6 — Coupling masks ↔ depth map

**Hypothesis.** Running the segmenter on the *same* preprocessed frames VGGT consumes, and zeroing the depth confidence outside the mask, should remove every background point without touching the object — with no reprojection or 3D-space filtering needed.

**Experiment setup.** The core of the YOLO + VGGT integration (`vggt_omega/inference_vggt.py` + `vggt_omega/segmentation.py` + `vggt_omega/visual_util.py`):

1. YOLO runs on **the very same preprocessed images** consumed by VGGT-Omega, so the mask is pixel-aligned with the depth map (same resolution and framing).
2. The per-frame binary mask (union of detected instances + morphological opening + largest component) is stored in `predictions["object_mask"]`.
3. When building the point cloud, the depth confidence is multiplied by the mask: `conf = conf · mask`. Every background pixel drops to confidence 0 and **its 3D point is discarded** — only the object of interest remains in the cloud.

Three segmentation backends are selectable at inference time: **YOLO only** (default), **U2-Net only**, or **mixed** (pixel-wise AND of both). Additional outputs:

- **Per-frame mask PNGs** (`--mask-dir`), including a debug mode (`--masks-debug`) that saves the YOLO mask, the U2-Net mask and the combined one separately.
- **Colorized depth map** PNGs (`--depth-dir`).
- Raw tensors as `.pt` (including `object_mask`) to re-visualize without re-running inference (`vggt_omega/visualize_predictions.py`).

**Results.** Point clouds containing only the object of interest, clean enough to be handed to the meshing stage.
<!-- TODO: add before (full scene) / after (object only) point-cloud screenshots -->

**Conclusions.** Coupling through the confidence array proved to be the right abstraction: masks, confidence percentile, depth-edge and background filters all share one mechanism, and the mask travels inside the predictions dict, so every consumer (CLI, API, Gradio demo, re-visualization) gets object isolation for free.

---

## Task 3 — 3D mesh generation

The final stage converts the object's `.ply` point cloud into a **3D mesh** (triangulated surface) importable into Blender or other DCC tools.

> ⚠️ This part has been developed by another team member and is **not yet integrated into this repository**. It takes the clean point cloud produced by Task 2 as input and produces the final object mesh.
<!-- TODO: when the meshing code lands in the repo, document its method and how to run it, and add its experiment (hypothesis / setup / results / conclusions) here -->

---

## Integration — API and mobile app

The whole flow is exposed as a service consumed by the mobile application:

- **HTTP API (FastAPI)** in `vggt_omega/api/` (full spec in [`vggt_omega/api/API_SPEC.md`](vggt_omega/api/API_SPEC.md)): the `POST /api/v1/inference` endpoint accepts a variable number of images in one multipart request (up to 32; JPG/PNG/HEIC/…), runs VGGT-Omega + YOLO on the GPU server and returns download URLs for the `.ply` (and optionally the depth and mask PNGs). Configuration via environment variables (`VGGT_CHECKPOINT`, `VGGT_SEG_CHECKPOINT`, `VGGT_DEVICE`, …).
- **Mobile application**: the user photographs the object from several angles; the app sends the images to the API and receives the reconstructed 3D object (point cloud / mesh) as the result.

---

## Repository structure

```
SegmentationModel_PP_AI/
├── src/                          # Task 1: segmentation
│   ├── model.py                  #   U-Net (SegmentationModel) + U2-Net
│   ├── main.py                   #   U-Net/U2-Net training (with --resume)
│   ├── losses.py                 #   BCE + Dice
│   ├── dataset.py                #   VizWiz dataset (polygons → masks)
│   ├── augmentation.py           #   Albumentations pipelines
│   ├── wandb_logger.py           #   W&B logging + checkpoints
│   ├── test_evaluation.py        #   evaluation (IoU, Dice, precision, recall)
│   ├── data_visualization.py     #   dataset sample inspection
│   ├── convert_vizwiz_to_yolo.py #   VizWiz JSON → YOLO format
│   └── train_yolo.py             #   YOLO26-seg training + W&B
├── vggt_omega/                   # Task 2: 3D reconstruction
│   ├── inference_vggt.py         #   end-to-end CLI (VGGT + YOLO → .ply)
│   ├── segmentation.py           #   YOLO/U2-Net/mixed masks + morphology
│   ├── visual_util.py            #   confidence filtering → point cloud
│   ├── visualize_predictions.py  #   PLY / depth PNG / mask PNG export
│   ├── demo_gradio.py            #   interactive web demo
│   ├── models/, utils/           #   VGGT-Omega architecture and preprocessing
│   └── api/                      # Integration: FastAPI service
├── data/ , data_yolo/            # datasets (not tracked)
└── checkpoints/                  # weights (VGGT checkpoint not in the repo)
```

## Summary of key decisions

| Decision | Rationale |
|---|---|
| YOLO26-seg as the final segmenter (over our own U-Net/U2-Net) | Better mask quality and inference; U2-Net kept as an alternative and mixed mode (AND) |
| Morphological opening (elliptical kernel 21) + largest connected component | Clean masks: removes noise and secondary detections, keeps only the object |
| Coupling the mask through `depth_conf` (conf=0 on background) | A single filtering mechanism shared by mask, confidence, depth edges and background filters |
| Points-only PLY export (previously GLB with camera frustums) | The output is the object's cloud for meshing, not a debug scene |
| Running YOLO on the images already preprocessed by VGGT | Pixel-perfect mask ↔ depth alignment with no reprojection |
| Multipart FastAPI service | Decouples the mobile app from the GPU server |

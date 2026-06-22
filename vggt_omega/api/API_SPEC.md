# VGGT-Omega Inference API — Specification

HTTP service that wraps `vggt_omega/inference_vggt.py`. It accepts a **variable
number of images** in a single multipart request, runs the VGGT-Omega forward
pass, and returns a 3D point cloud (PLY) plus optional depth and object-mask
PNGs.

- **Base URL:** `http://<host>:8000`
- **Version:** `1.0.0`
- **Interactive docs (Swagger UI):** `GET /docs`
- **OpenAPI JSON:** `GET /openapi.json`

All tunable paths and defaults live in
[`vggt_omega/api/constants.py`](constants.py) and can be overridden with
environment variables (see [Configuration](#configuration)).

> ⚠️ **CUDA required.** The model forward pass uses
> `torch.autocast(device_type="cuda")`, so a CUDA GPU is needed. On a host
> without CUDA the inference endpoint returns `503`.

---

## Running the server

```bash
# 1. Install deps (inference + API)
pip install -r vggt_omega/requirements.txt
pip install -r vggt_omega/api/requirements.txt

# 2. Point the API at your VGGT-Omega checkpoint
export VGGT_CHECKPOINT=/path/to/vggt_omega_1b_512.pt

# 3. Launch (run from the repo root so `vggt_omega` is importable)
uvicorn vggt_omega.api.main:app --host 0.0.0.0 --port 8000
```

---

## Configuration

| Env var               | Constant            | Default                              | Description                                              |
| --------------------- | ------------------- | ------------------------------------ | -------------------------------------------------------- |
| `VGGT_CHECKPOINT`     | `VGGT_CHECKPOINT`   | `checkpoints/vggt_omega_1b_512.pt`   | Main VGGT-Omega checkpoint (not committed to the repo).  |
| `VGGT_SEG_CHECKPOINT` | `SEG_CHECKPOINT`    | `yolo26s-seg.pt`                     | YOLO-seg checkpoint for object isolation.                |
| `VGGT_OUTPUT_DIR`     | `OUTPUT_DIR`        | `api_outputs`                        | Where per-job artifacts are written.                     |
| `VGGT_DEVICE`         | `DEVICE`            | `cuda`                               | Torch device.                                            |
| `VGGT_RESOLUTION`     | `DEFAULT_RESOLUTION`| `512`                                | Default preprocessing resolution.                        |
| `VGGT_MODE`           | `DEFAULT_MODE`      | `balanced`                           | Default preprocessing mode.                              |
| `VGGT_CONF_THRES`     | `DEFAULT_CONF_THRES`| `20.0`                               | Default PLY confidence percentile filter.                |
| `VGGT_SEG_CONF`       | `DEFAULT_SEG_CONF`  | `0.25`                               | Default YOLO detection confidence.                       |
| `VGGT_MAX_IMAGES`     | `MAX_IMAGES`        | `32`                                 | Max images per request.                                  |
| `VGGT_MAX_IMAGE_BYTES`| `MAX_IMAGE_BYTES`   | `26214400` (25 MB)                   | Max size per uploaded image.                             |

Relative paths are resolved against the repository root.

---

## Endpoints

### `POST /api/v1/inference`

Run inference over the uploaded images.

- **Content-Type:** `multipart/form-data`

#### Request fields

| Field          | Type            | Required | Default      | Description                                                                 |
| -------------- | --------------- | -------- | ------------ | --------------------------------------------------------------------------- |
| `images`       | file[]          | yes      | —            | One or more images. Repeat the `images` field once per file. 1‒`MAX_IMAGES`.|
| `resolution`   | int             | no       | `512`        | Preprocessing resolution; must be a multiple of 16.                         |
| `mode`         | string          | no       | `balanced`   | `balanced` or `max_size`.                                                    |
| `conf_thres`   | float           | no       | `20.0`       | Confidence percentile threshold for PLY point filtering.                    |
| `segment`      | bool            | no       | `false`      | Run YOLO-seg; keep only the segmented object in the point cloud.            |
| `seg_conf`     | float           | no       | `0.25`       | YOLO detection confidence threshold (used when `segment=true`).            |
| `export_depth` | bool            | no       | `false`      | Also export per-frame colorized depth PNGs.                                  |
| `export_masks` | bool            | no       | `false`      | Also export per-frame object-mask PNGs (requires `segment=true`).           |

**Accepted image extensions:** `.jpg .jpeg .png .bmp .tif .tiff .webp .heic .heif`

#### Response `200 OK` — `application/json`

```json
{
  "job_id": "8f1c0b2e4a7d4f9bb0c5d6e7f8a9b0c1",
  "num_images": 3,
  "device": "cuda",
  "shapes": {
    "pose_enc":   [1, 3, 9],
    "depth":      [1, 3, 512, 512, 1],
    "depth_conf": [1, 3, 512, 512],
    "extrinsics": [1, 3, 3, 4],
    "intrinsics": [1, 3, 3, 3]
  },
  "artifacts": [
    { "name": "scene.ply",        "type": "point_cloud", "url": "/api/v1/jobs/8f1c.../files/scene.ply" },
    { "name": "depth/depth_000.png", "type": "depth_map",  "url": "/api/v1/jobs/8f1c.../files/depth/depth_000.png" }
  ],
  "archive_url": "/api/v1/jobs/8f1c.../archive"
}
```

The actual tensor `shapes` depend on image count and resolution; the values
above are illustrative. `object_mask` only appears when `segment=true`.

#### Error responses

| Status | When                                                                  |
| ------ | --------------------------------------------------------------------- |
| `413`  | An uploaded image exceeds `MAX_IMAGE_BYTES`.                           |
| `422`  | No images, too many images, bad extension, or invalid `mode`.         |
| `500`  | Inference raised an error (model/forward/export failure).             |
| `503`  | CUDA unavailable, or the VGGT / segmentation checkpoint is missing.   |

---

### `GET /api/v1/jobs/{job_id}/files/{file_path}`

Download a single artifact produced by a job (e.g. `scene.ply`,
`depth/depth_000.png`). Returns the file with its original content type.
`404` if the file does not exist; `400` on path traversal attempts.

### `GET /api/v1/jobs/{job_id}/archive`

Download **all** artifacts of a job as a single ZIP (`application/zip`).
`404` if the job is unknown.

### `GET /health`

Readiness probe. Returns device info and whether the checkpoints are present.

```json
{
  "status": "ok",
  "device": "cuda",
  "cuda_available": true,
  "vggt_checkpoint_present": true,
  "seg_checkpoint_present": true
}
```

---

## Examples

### Minimal — two images, default settings

```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -F "images=@imageA.png" \
  -F "images=@imageB.png"
```

### Full — segmentation + depth + mask export

```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -F "images=@imageA.jpg" \
  -F "images=@imageB.jpg" \
  -F "images=@imageC.heic" \
  -F "resolution=512" \
  -F "mode=balanced" \
  -F "conf_thres=20" \
  -F "segment=true" \
  -F "seg_conf=0.25" \
  -F "export_depth=true" \
  -F "export_masks=true"
```

### Download the resulting point cloud

```bash
curl -OJ http://localhost:8000/api/v1/jobs/<job_id>/files/scene.ply
# or grab everything at once:
curl -OJ http://localhost:8000/api/v1/jobs/<job_id>/archive
```

### Python client

```python
import requests

files = [("images", open(p, "rb")) for p in ["a.jpg", "b.jpg", "c.jpg"]]
data = {"segment": "true", "export_depth": "true"}
r = requests.post("http://localhost:8000/api/v1/inference", files=files, data=data)
r.raise_for_status()
job = r.json()

ply = requests.get(f"http://localhost:8000{job['artifacts'][0]['url']}")
open("scene.ply", "wb").write(ply.content)
```

---

## Notes & limitations

- **One inference at a time.** Requests are serialised with a process-wide lock
  (the model shares a single GPU and is not thread-safe). Run multiple workers /
  replicas behind a load balancer for throughput.
- **Per-request model load.** `run_inference` loads the checkpoint on every
  call (faithful to `inference_vggt.py`). For lower latency, cache the model in
  a future iteration.
- **Artifact retention.** Job outputs persist under `OUTPUT_DIR/<job_id>` until
  cleaned up; add a retention policy / cron for production.

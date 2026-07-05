# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app that runs VGGT-Omega inference on uploaded images.

Run with::

    uvicorn vggt_omega.api.main:app --host 0.0.0.0 --port 8000

The interactive docs (OpenAPI / Swagger UI) are served at ``/docs``.
"""

from __future__ import annotations

import io
import logging
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from vggt_omega.api import constants
from vggt_omega.inference_vggt import run_inference
from vggt_omega.visualize_predictions import (
    export_depth_pngs,
    export_masked_depth_pngs,
    export_mesh,
    export_object_mask_pngs,
    export_point_cloud_ply,
    to_numpy_predictions,
)

# Emit INFO logs to stdout if the host process (e.g. uvicorn) hasn't already
# configured logging, so the step-by-step inference trace is visible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("vggt_omega.api")

app = FastAPI(
    title=constants.API_TITLE,
    version=constants.API_VERSION,
    description=constants.API_DESCRIPTION,
)

# The model is not thread-safe and shares a single GPU; serialise inference so
# concurrent requests don't race on CUDA memory. FastAPI runs sync endpoints in
# a threadpool, so this only blocks other inference calls, not the event loop.
_INFERENCE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Response models (drive the auto-generated OpenAPI schema)
# ---------------------------------------------------------------------------
class ArtifactInfo(BaseModel):
    name: str
    type: str  # "mesh" | "point_cloud" | "depth_map" | "object_mask"
    url: str


class InferenceResponse(BaseModel):
    job_id: str
    num_images: int
    device: str
    shapes: dict[str, list[int]]
    artifacts: List[ArtifactInfo]
    archive_url: str


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    vggt_checkpoint_present: bool
    seg_checkpoint_present: bool
    seg_u2net_checkpoint_present: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_device_ready() -> None:
    """Mirror the CUDA guard in inference_vggt.main()."""
    if constants.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "CUDA is not available but VGGT-Omega requires a CUDA device "
                "(the forward pass uses torch.autocast(device_type='cuda')). "
                "Set VGGT_DEVICE or run on a GPU host."
            ),
        )


def _validate_uploads(images: List[UploadFile]) -> None:
    if not images:
        raise HTTPException(status_code=422, detail="At least 1 image is required.")
    if len(images) > constants.MAX_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Too many images: {len(images)} > MAX_IMAGES={constants.MAX_IMAGES}.",
        )
    for upload in images:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in constants.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported file '{upload.filename}': extension '{ext}' is "
                    f"not in {sorted(constants.ALLOWED_EXTENSIONS)}."
                ),
            )


def _save_uploads(images: List[UploadFile], dest_dir: Path) -> List[str]:
    """Persist uploads to `dest_dir`, enforcing the per-file size limit."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for index, upload in enumerate(images):
        ext = Path(upload.filename or "").suffix.lower()
        target = dest_dir / f"image_{index:03d}{ext}"
        size = 0
        with target.open("wb") as fh:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > constants.MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File '{upload.filename}' exceeds the "
                            f"{constants.MAX_IMAGE_BYTES} byte limit."
                        ),
                    )
                fh.write(chunk)
        logger.info(
            "  saved upload %d/%d: '%s' -> %s (%.1f KB)",
            index + 1,
            len(images),
            upload.filename,
            target,
            size / 1024.0,
        )
        paths.append(str(target))
    return paths


def _tensor_shapes(predictions: dict) -> dict[str, list[int]]:
    keys = (
        "pose_enc",
        "depth",
        "depth_conf",
        "extrinsics",
        "intrinsics",
        "camera_tokens",
        "registers",
        "object_mask",
    )
    shapes: dict[str, list[int]] = {}
    for key in keys:
        value = predictions.get(key)
        if isinstance(value, torch.Tensor):
            shapes[key] = list(value.shape)
    return shapes


def _resolve_job_file(job_id: str, rel_path: str) -> Path:
    """Resolve a download path, guarding against directory traversal."""
    job_dir = (constants.OUTPUT_DIR / job_id).resolve()
    target = (job_dir / rel_path).resolve()
    if not str(target).startswith(str(job_dir) + "/") and target != job_dir:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return target


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness + readiness: device, CUDA and checkpoint availability."""
    return HealthResponse(
        status="ok",
        device=constants.DEVICE,
        cuda_available=torch.cuda.is_available(),
        vggt_checkpoint_present=constants.VGGT_CHECKPOINT.is_file(),
        seg_checkpoint_present=constants.SEG_CHECKPOINT.is_file(),
        seg_u2net_checkpoint_present=constants.SEG_U2NET_CHECKPOINT.is_file(),
    )


@app.post("/api/v1/inference", response_model=InferenceResponse, tags=["inference"])
def inference(
    images: List[UploadFile] = File(
        ..., description="One or more input images (variable amount)."
    ),
    resolution: int = Form(
        constants.DEFAULT_RESOLUTION, description="Preprocessing resolution (multiple of 16)."
    ),
    mode: str = Form(
        constants.DEFAULT_MODE, description="Preprocessing mode: 'balanced' or 'max_size'."
    ),
    conf_thres: float = Form(
        constants.DEFAULT_CONF_THRES,
        description="Confidence percentile threshold for point filtering.",
    ),
    export_format: str = Form(
        "mesh",
        description=(
            "'mesh' (default) reconstructs a surface mesh ready for Blender / "
            "3D printing, or 'points' returns the raw point cloud."
        ),
    ),
    mesh_format: str = Form(
        "obj", description="File format for the mesh artifact: 'obj', 'stl' or 'ply'."
    ),
    poisson_depth: int = Form(
        9, description="Octree depth for Poisson surface reconstruction (mesh only)."
    ),
    mesh_method: str = Form(
        "poisson",
        description="Surface reconstruction method: 'poisson' (default) or 'pymeshlab' "
                    "(Screened Poisson via PyMeshLab; falls back to poisson if unavailable).",
    ),
    segment: bool = Form(
        True,
        description="If true, run mixed YOLO + U2Net segmentation and keep only the "
        "segmented object in the cloud (background dropped).",
    ),
    seg_conf: float = Form(
        constants.DEFAULT_SEG_CONF, description="YOLO detection confidence threshold."
    ),
    u2net_thres: float = Form(
        constants.DEFAULT_U2NET_THRES,
        description="Binarisation threshold for the U2Net saliency map.",
    ),
    morph_open: bool = Form(
        True,
        description="Apply morphological opening to the final mask (removes small noise regions).",
    ),
    morph_kernel: int = Form(
        constants.DEFAULT_MORPH_KERNEL,
        description="Elliptical kernel size for the morphological opening (larger = more aggressive).",
    ),
    keep_largest: bool = Form(
        True,
        description="Keep only the largest connected component per frame after masking.",
    ),
    masks_debug: bool = Form(
        True,
        description="In mixed mode, also produce per-frame YOLO / U2Net / combined "
        "debug masks (exported when export_masks=true).",
    ),
    export_depth: bool = Form(False, description="Also export per-frame depth PNGs."),
    export_masks: bool = Form(
        True, description="Also export per-frame object-mask PNGs (requires segment=true)."
    ),
) -> InferenceResponse:
    """Run VGGT-Omega inference over the uploaded images.

    Returns metadata, the output tensor shapes and download URLs for the
    generated artifacts (mesh or point cloud + optional depth / mask PNGs).
    """
    request_start = time.perf_counter()
    logger.info(
        "=== /api/v1/inference received | images=%d | resolution=%d | mode=%s | "
        "export_format=%s | mesh_format=%s | mesh_method=%s | segment=%s | "
        "seg_conf=%.2f | u2net_thres=%.2f | morph_open=%s | morph_kernel=%d | "
        "keep_largest=%s | masks_debug=%s | export_depth=%s | export_masks=%s",
        len(images),
        resolution,
        mode,
        export_format,
        mesh_format,
        mesh_method,
        segment,
        seg_conf,
        u2net_thres,
        morph_open,
        morph_kernel,
        keep_largest,
        masks_debug,
        export_depth,
        export_masks,
    )

    logger.info("Validating device readiness and uploaded files...")
    _check_device_ready()
    _validate_uploads(images)

    if mode not in ("balanced", "max_size"):
        raise HTTPException(
            status_code=422, detail="mode must be 'balanced' or 'max_size'."
        )
    if export_format not in ("mesh", "points"):
        raise HTTPException(
            status_code=422, detail="export_format must be 'mesh' or 'points'."
        )
    if export_format == "mesh" and mesh_format not in ("obj", "stl", "ply"):
        raise HTTPException(
            status_code=422, detail="mesh_format must be 'obj', 'stl' or 'ply'."
        )
    if mesh_method not in ("poisson", "pymeshlab"):
        raise HTTPException(
            status_code=422, detail="mesh_method must be 'poisson' or 'pymeshlab'."
        )
    if not constants.VGGT_CHECKPOINT.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"VGGT checkpoint not found at {constants.VGGT_CHECKPOINT}.",
        )
    if segment and not constants.SEG_CHECKPOINT.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"YOLO segmentation checkpoint not found at {constants.SEG_CHECKPOINT}.",
        )
    if segment and not constants.SEG_U2NET_CHECKPOINT.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"U2Net segmentation checkpoint not found at {constants.SEG_U2NET_CHECKPOINT}.",
        )

    job_id = uuid.uuid4().hex
    job_dir = constants.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Job %s: created output directory %s", job_id, job_dir)

    # Persist the original uploads inside the job's output folder so they are
    # kept alongside the generated artifacts (and available in the ZIP archive).
    inputs_dir = job_dir / "inputs"
    logger.info("Job %s: saving %d original image(s) to %s", job_id, len(images), inputs_dir)
    try:
        image_paths = _save_uploads(images, inputs_dir)
    except HTTPException:
        # e.g. a file over the size limit: drop the partially written job dir.
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    logger.info(
        "Job %s: acquiring inference lock and running VGGT-Omega on %d image(s)...",
        job_id,
        len(image_paths),
    )
    with _INFERENCE_LOCK:
        infer_start = time.perf_counter()
        try:
            predictions = run_inference(
                checkpoint_path=str(constants.VGGT_CHECKPOINT),
                image_names=image_paths,
                image_resolution=resolution,
                mode=mode,
                device=constants.DEVICE,
                seg_checkpoint=str(constants.SEG_CHECKPOINT) if segment else None,
                seg_u2net_checkpoint=(
                    str(constants.SEG_U2NET_CHECKPOINT) if segment else None
                ),
                seg_conf=seg_conf,
                u2net_thres=u2net_thres,
                masks_debug=masks_debug,
                morph_open=morph_open,
                morph_kernel=morph_kernel,
                keep_largest=keep_largest,
            )
        except Exception as exc:  # noqa: BLE001 - surface inference errors as 500
            logger.exception("Job %s: inference failed: %s", job_id, exc)
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500, detail=f"Inference failed: {exc}"
            ) from exc
    logger.info(
        "Job %s: inference finished (%.2fs)", job_id, time.perf_counter() - infer_start
    )

    shapes = _tensor_shapes(predictions)
    logger.info("Job %s: prediction tensor shapes: %s", job_id, shapes)
    logger.info("Job %s: converting predictions to numpy...", job_id)
    predictions_np = to_numpy_predictions(predictions)

    artifacts: List[ArtifactInfo] = []

    # Save the depth maps with the object mask applied *before* generating the
    # 3D file, so the exact (masked) depth feeding the reconstruction is kept.
    if "object_mask" in predictions_np:
        masked_depth_dir = job_dir / "depth_masked"
        logger.info(
            "Job %s: exporting masked depth PNGs to %s (pre-3D)...",
            job_id,
            masked_depth_dir,
        )
        export_masked_depth_pngs(predictions_np, str(masked_depth_dir))
        for png in sorted(masked_depth_dir.glob("*.png")):
            artifacts.append(
                ArtifactInfo(
                    name=f"depth_masked/{png.name}",
                    type="depth_map",
                    url=f"/api/v1/jobs/{job_id}/files/depth_masked/{png.name}",
                )
            )

    try:
        if export_format == "mesh":
            scene_name = f"scene.{mesh_format}"
            logger.info(
                "Job %s: exporting mesh '%s' (method=%s, poisson_depth=%d, "
                "conf_thres=%.1f)...",
                job_id,
                scene_name,
                mesh_method,
                poisson_depth,
                conf_thres,
            )
            export_start = time.perf_counter()
            export_mesh(
                predictions_np,
                str(job_dir / scene_name),
                conf_thres=conf_thres,
                poisson_depth=poisson_depth,
                method=mesh_method,
            )
            logger.info(
                "Job %s: mesh exported (%.2fs)",
                job_id,
                time.perf_counter() - export_start,
            )
            artifacts.append(
                ArtifactInfo(
                    name=scene_name,
                    type="mesh",
                    url=f"/api/v1/jobs/{job_id}/files/{scene_name}",
                )
            )
            # export_mesh always writes companion .ply/.glb copies alongside the
            # requested format so the vertex colors are viewable regardless of
            # what was requested (.obj drops them, .ply/.glb keep them).
            for companion_ext in (".ply", ".glb"):
                companion_name = f"scene{companion_ext}"
                if companion_name != scene_name and (job_dir / companion_name).exists():
                    artifacts.append(
                        ArtifactInfo(
                            name=companion_name,
                            type="mesh",
                            url=f"/api/v1/jobs/{job_id}/files/{companion_name}",
                        )
                    )
        else:
            ply_path = job_dir / "scene.ply"
            logger.info(
                "Job %s: exporting point cloud 'scene.ply' (conf_thres=%.1f)...",
                job_id,
                conf_thres,
            )
            export_start = time.perf_counter()
            export_point_cloud_ply(predictions_np, str(ply_path), conf_thres=conf_thres)
            logger.info(
                "Job %s: point cloud exported (%.2fs)",
                job_id,
                time.perf_counter() - export_start,
            )
            artifacts.append(
                ArtifactInfo(
                    name="scene.ply",
                    type="point_cloud",
                    url=f"/api/v1/jobs/{job_id}/files/scene.ply",
                )
            )
    except ValueError as exc:
        # Actionable reconstruction failures (e.g. too few points after
        # segmentation / conf_thres filtering) -> 422 so the client can retry
        # with different settings instead of seeing an opaque 500.
        logger.warning("Job %s: export failed (unprocessable): %s", job_id, exc)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface export errors as 500
        logger.exception("Job %s: export failed: %s", job_id, exc)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    if export_depth:
        depth_dir = job_dir / "depth"
        logger.info("Job %s: exporting depth PNGs to %s...", job_id, depth_dir)
        export_depth_pngs(predictions_np, str(depth_dir))
        for png in sorted(depth_dir.glob("*.png")):
            artifacts.append(
                ArtifactInfo(
                    name=f"depth/{png.name}",
                    type="depth_map",
                    url=f"/api/v1/jobs/{job_id}/files/depth/{png.name}",
                )
            )

    if export_masks and "object_mask" in predictions_np:
        mask_dir = job_dir / "masks"
        logger.info("Job %s: exporting object-mask PNGs to %s...", job_id, mask_dir)
        export_object_mask_pngs(predictions_np, str(mask_dir))
        for png in sorted(mask_dir.glob("*.png")):
            artifacts.append(
                ArtifactInfo(
                    name=f"masks/{png.name}",
                    type="object_mask",
                    url=f"/api/v1/jobs/{job_id}/files/masks/{png.name}",
                )
            )

    logger.info(
        "=== Job %s: done | %d artifact(s): %s | total %.2fs",
        job_id,
        len(artifacts),
        [a.name for a in artifacts],
        time.perf_counter() - request_start,
    )
    return InferenceResponse(
        job_id=job_id,
        num_images=len(image_paths),
        device=constants.DEVICE,
        shapes=shapes,
        artifacts=artifacts,
        archive_url=f"/api/v1/jobs/{job_id}/archive",
    )


@app.get("/api/v1/jobs/{job_id}/files/{file_path:path}", tags=["inference"])
def download_file(job_id: str, file_path: str) -> FileResponse:
    """Download a single artifact produced by a previous inference job."""
    target = _resolve_job_file(job_id, file_path)
    return FileResponse(path=str(target), filename=target.name)


@app.get("/api/v1/jobs/{job_id}/archive", tags=["inference"])
def download_archive(job_id: str) -> StreamingResponse:
    """Download all artifacts of a job as a single ZIP archive."""
    job_dir = (constants.OUTPUT_DIR / job_id).resolve()
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(job_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(job_dir)))
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )

"""Object segmentation for VGGT-Omega point clouds.

Supports three modes selected automatically from the checkpoints provided:

- **YOLO only** – pass ``yolo_checkpoint`` only.
- **U2Net only** – pass ``u2net_checkpoint`` only.
- **Mixed (AND)** – pass both; the final mask keeps only pixels where *both*
  models agree the pixel is foreground (element-wise AND).

``predictions_to_point_cloud`` then drops every non-object point by zeroing its
depth confidence, so only the object of interest is left in the cloud.
"""

import os
import sys

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_morphological_open(masks: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply morphological opening (erode then dilate) to each frame in *masks*.

    Args:
        masks: ``(S, H, W)`` binary float32 array.
        kernel_size: side length of the square structuring element (must be odd).

    Returns:
        Opened ``(S, H, W)`` binary float32 array.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    opened = np.empty_like(masks)
    for i, frame in enumerate(masks):
        uint8 = (frame > 0.5).astype(np.uint8) * 255
        opened[i] = (cv2.morphologyEx(uint8, cv2.MORPH_OPEN, kernel) > 127).astype(np.float32)
    return opened


def _prepare_frame_list(frames: torch.Tensor) -> tuple[list, int, int]:
    """Convert a (S, 3, H, W) float [0,1] tensor to a BGR uint8 list plus (H, W)."""
    num_frames, _, height, width = frames.shape
    rgb = (frames.clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    frame_list = [np.ascontiguousarray(rgb[i, :, :, ::-1]) for i in range(num_frames)]
    return frame_list, height, width


def _get_yolo_masks(
    frame_list: list,
    checkpoint_path: str,
    height: int,
    width: int,
    imgsz: int = 512,
    conf: float = 0.25,
    device=None,
) -> np.ndarray:
    """Run YOLO-seg on *frame_list* and return a ``(S, H, W)`` binary float32 mask.

    Each frame's mask is the union of all detected instances.
    """
    from ultralytics import YOLO

    model = YOLO(checkpoint_path)
    results = model.predict(
        frame_list,
        imgsz=imgsz,
        conf=conf,
        device=device,
        retina_masks=True,
        verbose=False,
    )

    masks = np.zeros((len(frame_list), height, width), dtype=np.float32)
    for i, result in enumerate(results):
        if result.masks is None or len(result.masks) == 0:
            continue
        frame_mask = result.masks.data.any(dim=0).float().cpu().numpy()
        if frame_mask.shape != (height, width):
            frame_mask = cv2.resize(frame_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks[i] = (frame_mask > 0.5).astype(np.float32)

    return masks


def _get_u2net_masks(
    frame_list: list,
    checkpoint_path: str,
    height: int,
    width: int,
    device=None,
    thres: float = 0.5,
    imgsz: int = 512,
) -> np.ndarray:
    """Run U2Net on *frame_list* (BGR uint8) and return a ``(S, H, W)`` binary float32 mask.

    The model is imported from ``src/model.py`` in the project root. Preprocessing
    matches the training pipeline: resize to *imgsz*, ImageNet mean/std normalisation.
    """
    import torch.nn.functional as F

    # Import U2Net from the project's own src/ directory.
    src_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from model import U2Net  # noqa: PLC0415

    if device is None:
        device = "cpu"

    # Load checkpoint (same pattern as src/test_evaluation.py).
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        in_channels = checkpoint.get("in_channels") or 3
        out_channels = checkpoint.get("out_channels") or 1
        state_dict = checkpoint["model_state_dict"]
    else:
        in_channels = 3
        out_channels = 1
        state_dict = checkpoint

    u2net = U2Net(inChannels=in_channels, outChannels=out_channels).to(device)
    u2net.load_state_dict(state_dict)
    u2net.eval()

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    batch = []
    for bgr in frame_list:
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB [0,1]
        resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        batch.append(((resized - _MEAN) / _STD).transpose(2, 0, 1))  # (3, H, W)

    batch_tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device)  # (S, 3, imgsz, imgsz)

    with torch.inference_mode():
        outputs = u2net(batch_tensor)

    # outputs[0] is the fused finest-scale saliency map: (S, 1, imgsz, imgsz)
    probs = torch.sigmoid(outputs[0]).squeeze(1)  # (S, imgsz, imgsz)
    probs_resized = F.interpolate(
        probs.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(1)  # (S, H, W)

    return (probs_resized.cpu().numpy() > thres).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.inference_mode()
def add_object_masks(
    predictions: dict,
    yolo_checkpoint: str | None = None,
    u2net_checkpoint: str | None = None,
    imgsz: int = 512,
    conf: float = 0.25,
    u2net_thres: float = 0.5,
    device=None,
    masks_debug: bool = False,
    morph_open: bool = False,
    morph_kernel: int = 21,
) -> dict:
    """Add ``predictions["object_mask"]`` using YOLO, U2Net, or both (AND).

    Mode is determined automatically from which checkpoints are supplied:

    - *yolo_checkpoint* only → YOLO mask.
    - *u2net_checkpoint* only → U2Net mask.
    - Both → element-wise AND of the two binary masks (mixed mode).

    The resulting mask has the same leading dims and ``(H, W)`` as
    ``predictions["depth_conf"]``, with 1.0 for object pixels and 0.0 for
    background.
    """
    if yolo_checkpoint is None and u2net_checkpoint is None:
        raise ValueError("At least one of yolo_checkpoint or u2net_checkpoint must be provided.")

    images = predictions["images"]  # (1, S, 3, H, W) or (S, 3, H, W), RGB in [0, 1]
    has_batch_dim = images.ndim == 5
    frames = (images[0] if has_batch_dim else images).to(dtype=torch.float32)

    frame_list, height, width = _prepare_frame_list(frames)

    if yolo_checkpoint is not None and u2net_checkpoint is None:
        masks = _get_yolo_masks(frame_list, yolo_checkpoint, height, width, imgsz, conf, device)
    elif u2net_checkpoint is not None and yolo_checkpoint is None:
        masks = _get_u2net_masks(frame_list, u2net_checkpoint, height, width, device, u2net_thres, imgsz)
    else:
        yolo_masks = _get_yolo_masks(frame_list, yolo_checkpoint, height, width, imgsz, conf, device)
        u2net_masks = _get_u2net_masks(frame_list, u2net_checkpoint, height, width, device, u2net_thres, imgsz)
        if masks_debug:
            def _to_tensor(m: np.ndarray) -> torch.Tensor:
                t = torch.from_numpy(m)
                return t.unsqueeze(0) if has_batch_dim else t
            predictions["yolo_mask"] = _to_tensor(yolo_masks)
            predictions["u2net_mask"] = _to_tensor(u2net_masks)
        masks = ((yolo_masks > 0.5) & (u2net_masks > 0.5)).astype(np.float32)

    if morph_open:
        masks = _apply_morphological_open(masks, morph_kernel)

    mask_tensor = torch.from_numpy(masks)  # (S, H, W)
    if has_batch_dim:
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, S, H, W)
    predictions["object_mask"] = mask_tensor
    return predictions

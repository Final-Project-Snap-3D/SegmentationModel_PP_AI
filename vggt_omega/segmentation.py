"""Object segmentation for VGGT-Omega point clouds.

Runs the Ultralytics YOLO instance-segmentation model (trained under ``src/``
via ``train_yolo.py``) on the same preprocessed frames VGGT used, and stores the
resulting binary object mask inside the predictions dict.
``predictions_to_point_cloud`` then drops every non-object point by zeroing its
depth confidence, so only the object of interest is left in the cloud.
"""

import cv2
import numpy as np
import torch


@torch.inference_mode()
def add_object_masks(
    predictions: dict,
    checkpoint_path: str,
    imgsz: int = 512,
    conf: float = 0.25,
    device=None,
) -> dict:
    """Add ``predictions["object_mask"]`` from a YOLO-seg model.

    The mask has the same leading dims and (H, W) as ``predictions["depth_conf"]``
    so it aligns 1:1 with the depth maps. Each frame's mask is the union of every
    detected instance: 1.0 for object pixels, 0.0 for background.
    """
    from ultralytics import YOLO

    images = predictions["images"]  # (1, S, 3, H, W) or (S, 3, H, W), RGB in [0, 1]
    has_batch_dim = images.ndim == 5
    frames = images[0] if has_batch_dim else images  # (S, 3, H, W)
    frames = torch.as_tensor(frames, dtype=torch.float32)
    num_frames, _, height, width = frames.shape

    model = YOLO(checkpoint_path)
    # Tensor input is treated as RGB in [0, 1], so no BGR conversion is needed.
    results = model.predict(
        frames.to(device) if device is not None else frames,
        imgsz=imgsz,
        conf=conf,
        retina_masks=True,
        verbose=False,
    )

    masks = np.zeros((num_frames, height, width), dtype=np.float32)
    for i, result in enumerate(results):
        if result.masks is None or len(result.masks) == 0:
            continue
        # Union of every detected instance -> single foreground mask.
        frame_mask = result.masks.data.any(dim=0).float().cpu().numpy()
        if frame_mask.shape != (height, width):
            frame_mask = cv2.resize(frame_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks[i] = (frame_mask > 0.5).astype(np.float32)

    mask_tensor = torch.from_numpy(masks)  # (S, H, W)
    if has_batch_dim:
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, S, H, W)
    predictions["object_mask"] = mask_tensor
    return predictions

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize a predictions.pt file produced by inference_vggt.py.

It builds a PLY point cloud from the depth maps and, optionally, exports each
depth map as a colorized PNG.

Example
-------
    python -m vggt_omega.visualize_predictions \
        --predictions predictions.pt \
        --output scene.ply \
        --depth-dir depth_maps
"""

import argparse
import os

import numpy as np
import torch

from vggt_omega.visual_util import predictions_to_point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize VGGT-Omega predictions.")
    parser.add_argument(
        "-p",
        "--predictions",
        default="predictions.pt",
        help="Path to the predictions .pt file (default: predictions.pt).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="scene.ply",
        help="Output PLY point cloud path (default: scene.ply).",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=20.0,
        help="Confidence percentile threshold for point filtering (default: 20).",
    )
    parser.add_argument(
        "--depth-dir",
        default=None,
        help="If set, export each depth map as a colorized PNG into this folder.",
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="If set, export each object mask as a black/white PNG into this folder.",
    )
    return parser.parse_args()


def to_numpy_predictions(predictions: dict) -> dict:
    """Convert tensors to numpy, drop the leading batch dim, and normalize
    the key names expected by predictions_to_point_cloud."""
    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
        predictions_np[key] = value

    # predictions_to_point_cloud / unprojection use the singular key names.
    if "extrinsic" not in predictions_np and "extrinsics" in predictions_np:
        predictions_np["extrinsic"] = predictions_np["extrinsics"]
    if "intrinsic" not in predictions_np and "intrinsics" in predictions_np:
        predictions_np["intrinsic"] = predictions_np["intrinsics"]

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )
    return predictions_np


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def export_point_cloud_ply(predictions_np: dict, output_path: str, conf_thres: float = 20.0) -> None:
    """Export the depth-unprojected point cloud as a PLY file (no cameras)."""
    import trimesh

    vertices, colors = predictions_to_point_cloud(predictions_np, conf_thres=conf_thres)
    trimesh.PointCloud(vertices=vertices, colors=colors).export(output_path)


def export_depth_pngs(predictions_np: dict, depth_dir: str) -> None:
    import matplotlib.cm as cm

    os.makedirs(depth_dir, exist_ok=True)
    from PIL import Image

    depth = predictions_np["depth"][..., 0]  # (num_frames, H, W)
    colormap = cm.get_cmap("turbo")
    for i, frame in enumerate(depth):
        finite = np.isfinite(frame)
        lo = np.percentile(frame[finite], 2) if finite.any() else 0.0
        hi = np.percentile(frame[finite], 98) if finite.any() else 1.0
        norm = np.clip((frame - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        rgb = (colormap(norm)[..., :3] * 255).astype(np.uint8)
        path = os.path.join(depth_dir, f"depth_{i:03d}.png")
        Image.fromarray(rgb).save(path)
    print(f"Saved {len(depth)} depth map(s) to {depth_dir}/")


def export_object_mask_pngs(predictions_np: dict, mask_dir: str) -> None:
    """Save each frame's binary object mask(s) as black/white PNGs.

    In normal mode saves ``mask_NNN.png`` (the final object mask).
    In debug mode (when ``yolo_mask`` and ``u2net_mask`` are present alongside
    ``object_mask``) saves three files per frame:
    ``mask_NNN_yolo.png``, ``mask_NNN_u2.png``, and ``mask_NNN_mix.png``.
    """
    from PIL import Image

    final_masks = predictions_np.get("object_mask")
    if final_masks is None:
        print("No object_mask in predictions; run inference with --seg-checkpoint to produce one.")
        return

    os.makedirs(mask_dir, exist_ok=True)
    final_masks = np.asarray(final_masks)  # (num_frames, H, W)

    yolo_masks = predictions_np.get("yolo_mask")
    u2net_masks = predictions_np.get("u2net_mask")
    debug = yolo_masks is not None and u2net_masks is not None

    if debug:
        yolo_masks = np.asarray(yolo_masks)
        u2net_masks = np.asarray(u2net_masks)
        for i in range(len(final_masks)):
            def _save(arr: np.ndarray, suffix: str) -> None:
                img = (arr > 0.5).astype(np.uint8) * 255
                Image.fromarray(img).save(os.path.join(mask_dir, f"mask_{i:03d}_{suffix}.png"))
            _save(yolo_masks[i], "yolo")
            _save(u2net_masks[i], "u2")
            _save(final_masks[i], "mix")
        print(f"Saved {len(final_masks)} frame(s) × 3 debug mask(s) to {mask_dir}/")
    else:
        for i, frame in enumerate(final_masks):
            img = (frame > 0.5).astype(np.uint8) * 255
            Image.fromarray(img).save(os.path.join(mask_dir, f"mask_{i:03d}.png"))
        print(f"Saved {len(final_masks)} object mask(s) to {mask_dir}/")


def main() -> None:
    args = parse_args()

    predictions = torch.load(args.predictions, map_location="cpu")
    predictions_np = to_numpy_predictions(predictions)

    output_ext = os.path.splitext(args.output)[1].lower()
    if output_ext != ".ply":
        raise ValueError(f"Unsupported --output extension '{output_ext}'. Use .ply.")

    export_point_cloud_ply(predictions_np, args.output, conf_thres=args.conf_thres)
    print(f"Saved PLY point cloud to {args.output}")

    if args.depth_dir is not None:
        export_depth_pngs(predictions_np, args.depth_dir)

    if args.mask_dir is not None:
        export_object_mask_pngs(predictions_np, args.mask_dir)


if __name__ == "__main__":
    main()

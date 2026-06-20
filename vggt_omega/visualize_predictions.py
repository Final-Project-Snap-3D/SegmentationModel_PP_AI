#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize a predictions.pt file produced by inference_vggt.py.

It builds a GLB point cloud (with the predicted cameras) from the depth maps
and, optionally, exports each depth map as a colorized PNG.

Example
-------
    python -m vggt_omega.visualize_predictions \
        --predictions predictions.pt \
        --output scene.glb \
        --depth-dir depth_maps
"""

import argparse
import os

import numpy as np
import torch

from vggt_omega.visual_util import predictions_to_glb


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
        default="scene.glb",
        help="Output GLB scene path (default: scene.glb).",
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
    return parser.parse_args()


def to_numpy_predictions(predictions: dict) -> dict:
    """Convert tensors to numpy, drop the leading batch dim, and normalize
    the key names expected by predictions_to_glb."""
    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
        predictions_np[key] = value

    # predictions_to_glb / unprojection use the singular key names.
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


def main() -> None:
    args = parse_args()

    predictions = torch.load(args.predictions, map_location="cpu")
    predictions_np = to_numpy_predictions(predictions)

    scene = predictions_to_glb(predictions_np, conf_thres=args.conf_thres)
    scene.export(args.output)
    print(f"Saved GLB scene to {args.output}")

    if args.depth_dir is not None:
        export_depth_pngs(predictions_np, args.depth_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize a predictions.pt file produced by inference_vggt.py.

By default it reconstructs a surface mesh (Poisson) from the depth-unprojected
point cloud, ready for Blender / 3D printing. Pass --export-format points to
get the raw PLY point cloud instead. Depth/mask PNGs can also be exported.

Example
-------
    python -m vggt_omega.visualize_predictions \
        --predictions predictions.pt \
        --output scene.obj \
        --depth-dir depth_maps

    python -m vggt_omega.visualize_predictions \
        --predictions predictions.pt \
        --export-format points \
        --output scene.ply
"""

import argparse
import os

import numpy as np
import torch

from vggt_omega.visual_util import predictions_to_point_cloud

MESH_EXTENSIONS = {".obj", ".stl", ".ply"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize VGGT-Omega predictions.")
    parser.add_argument(
        "-p",
        "--predictions",
        default="predictions.pt",
        help="Path to the predictions .pt file (default: predictions.pt).",
    )
    parser.add_argument(
        "--export-format",
        choices=["mesh", "points"],
        default="mesh",
        help=(
            "Export a reconstructed surface mesh (default, for Blender / 3D "
            "printing) or the raw point cloud."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output path. For --export-format mesh: .obj/.stl/.ply "
            "(default: scene.obj). For --export-format points: .ply "
            "(default: scene.ply)."
        ),
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=20.0,
        help="Confidence percentile threshold for point filtering (default: 20).",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=9,
        help="Octree depth for Poisson surface reconstruction (default: 9; "
        "higher = more detail, slower, used only with --export-format mesh).",
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


def export_mesh(
    predictions_np: dict,
    output_path: str,
    conf_thres: float = 20.0,
    poisson_depth: int = 9,
    density_quantile: float = 0.05,
) -> None:
    """Reconstruct a surface mesh from the point cloud and export it
    (.obj/.stl/.ply) for use in Blender or 3D printing.

    Uses Poisson surface reconstruction, which needs per-point normals and
    tends to extrapolate a closed surface beyond the actual data support; the
    low-density tail of the result is trimmed away to remove that bulge, then
    any boundary holes opened up by the trim are re-closed. This gets most
    inputs to a watertight, slicer-ready mesh, but isn't a topology guarantee
    for every point cloud — sparse or noisy captures may still need a manual
    repair pass (e.g. Blender's 3D-Print Toolbox "Make Manifold") before
    printing.
    """
    import open3d as o3d

    vertices, colors = predictions_to_point_cloud(predictions_np, conf_thres=conf_thres)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    densities = np.asarray(densities)
    low_density_vertices = densities < np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(low_density_vertices)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Trimming the low-density tail above opens up boundary holes wherever the
    # source point cloud didn't actually cover the surface (e.g. the unseen
    # back side in a single/few-view capture). Close them so the result is
    # watertight and safe to send to a slicer.
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    tensor_mesh = tensor_mesh.fill_holes()
    mesh = tensor_mesh.to_legacy()

    o3d.io.write_triangle_mesh(output_path, mesh)


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
    """Save each frame's binary object mask as a black/white PNG."""
    from PIL import Image

    masks = predictions_np.get("object_mask")
    if masks is None:
        print("No object_mask in predictions; run inference with --seg-checkpoint to produce one.")
        return

    os.makedirs(mask_dir, exist_ok=True)
    masks = np.asarray(masks)  # (num_frames, H, W)
    for i, frame in enumerate(masks):
        img = (frame > 0.5).astype(np.uint8) * 255
        Image.fromarray(img).save(os.path.join(mask_dir, f"mask_{i:03d}.png"))
    print(f"Saved {len(masks)} object mask(s) to {mask_dir}/")


def main() -> None:
    args = parse_args()

    predictions = torch.load(args.predictions, map_location="cpu")
    predictions_np = to_numpy_predictions(predictions)

    output_path = args.output or ("scene.obj" if args.export_format == "mesh" else "scene.ply")
    output_ext = os.path.splitext(output_path)[1].lower()

    if args.export_format == "mesh":
        if output_ext not in MESH_EXTENSIONS:
            raise ValueError(
                f"Unsupported --output extension '{output_ext}' for --export-format mesh. "
                f"Use one of {sorted(MESH_EXTENSIONS)}."
            )
        export_mesh(
            predictions_np, output_path, conf_thres=args.conf_thres, poisson_depth=args.poisson_depth
        )
        print(f"Saved mesh to {output_path}")
    else:
        if output_ext != ".ply":
            raise ValueError(f"Unsupported --output extension '{output_ext}' for --export-format points. Use .ply.")
        export_point_cloud_ply(predictions_np, output_path, conf_thres=args.conf_thres)
        print(f"Saved PLY point cloud to {output_path}")

    if args.depth_dir is not None:
        export_depth_pngs(predictions_np, args.depth_dir)

    if args.mask_dir is not None:
        export_object_mask_pngs(predictions_np, args.mask_dir)


if __name__ == "__main__":
    main()

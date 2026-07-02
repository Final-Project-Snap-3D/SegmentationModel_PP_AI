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
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.05,
        help="Trim the low-density tail of the Poisson surface (default: 0.05; "
             "raise to 0.1–0.3 to remove more invented surface).",
    )
    parser.add_argument(
        "--outlier-std",
        type=float,
        default=2.0,
        help="Statistical outlier removal threshold (default: 2.0; lower = more "
             "aggressive; 0 to disable).",
    )
    parser.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="Skip hole-closing after Poisson trimming (avoids membranes on "
             "concave / unseen surfaces).",
    )
    parser.add_argument(
        "--smooth-iters",
        type=int,
        default=10,
        help="Taubin smoothing iterations (default: 10; 0 to disable).",
    )
    parser.add_argument(
        "--method",
        choices=["poisson", "pymeshlab"],
        default="poisson",
        help="Surface reconstruction method (default: poisson). "
             "'pymeshlab' uses Screened Poisson from PyMeshLab; falls back to poisson if unavailable.",
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
    outlier_std: float = 2.0,
    fill_holes: bool = True,
    smooth_iters: int = 10,
    method: str = "poisson",
) -> None:
    """Reconstruct a surface mesh from the point cloud and export it
    (.obj/.stl/.ply) for use in Blender or 3D printing.

    ``method="poisson"`` (default): Poisson surface reconstruction (Open3D).
    ``method="pymeshlab"``: Screened Poisson via PyMeshLab; falls back to Poisson
    automatically if pymeshlab is unavailable.
    """
    import open3d as o3d

    vertices, colors = predictions_to_point_cloud(predictions_np, conf_thres=conf_thres)
    if len(vertices) == 0:
        raise ValueError(
            "Empty point cloud: lower conf_thres to keep more points before meshing."
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    # Remove stray points (outliers) that Poisson would otherwise bridge into
    # a spike/cone toward the origin.
    if outlier_std > 0 and len(pcd.points) > 20:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=outlier_std)

    # Normal-estimation radius adapted to the cloud's scale (the previous fixed
    # 0.1 failed when the cloud was larger/smaller than that).
    pts = np.asarray(pcd.points)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    radius = max(diag * 0.02, 1e-6)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    # ── PyMeshLab branch (Screened Poisson) ──────────────────────────────────
    mesh = None
    if method == "pymeshlab":
        try:
            import pymeshlab

            pts_arr = np.asarray(pcd.points)
            nrm_arr = np.asarray(pcd.normals)
            col_arr = np.asarray(pcd.colors)  # [0, 1]
            col_rgba = np.hstack([col_arr, np.ones((len(col_arr), 1))])

            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=pts_arr,
                v_normals_matrix=nrm_arr,
                v_color_matrix=col_rgba,
            ))

            try:
                ms.generate_surface_reconstruction_screened_poisson(
                    depth=int(poisson_depth), preclean=True)
            except AttributeError:
                ms.surface_reconstruction_screened_poisson(
                    depth=int(poisson_depth), preclean=True)

            try:
                m_raw = ms.current_mesh()
                quality = m_raw.vertex_quality_array()
                if len(quality) > 0:
                    threshold = float(np.quantile(quality, density_quantile))
                    ms.compute_selection_by_condition_per_vertex(
                        condselect=f"q < {threshold}")
                    ms.meshing_remove_selected_vertices()
            except Exception:
                pass

            m = ms.current_mesh()
            v = m.vertex_matrix()
            f = m.face_matrix()

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(v.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))

            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

        except Exception as e:
            print(f"⚠ PyMeshLab no disponible/falló: {e}. Usando Poisson.")
            mesh = None
            method = "poisson"

    # ── Poisson branch (default and automatic fallback from PyMeshLab) ────────
    if method == "poisson":
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
        # back side in a single/few-view capture). Optionally close them so the
        # result is watertight and safe to send to a slicer.
        if fill_holes and len(mesh.triangles) > 0:
            tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            tensor_mesh = tensor_mesh.fill_holes()
            mesh = tensor_mesh.to_legacy()

    # ── Post-proceso común: smooth + normals + color + export ─────────────────
    # Conservar solo el componente conexo más grande (elimina geometría suelta)
    if len(mesh.triangles) > 0:
        tri_cls, cls_n, _ = mesh.cluster_connected_triangles()
        tri_cls = np.asarray(tri_cls)
        cls_n = np.asarray(cls_n)
        largest = int(cls_n.argmax())
        mesh.remove_triangles_by_mask(tri_cls != largest)
        mesh.remove_unreferenced_vertices()

    if smooth_iters > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)

    mesh.compute_vertex_normals()

    # Transferencia de color: KDTree sobre la nube original como fallback.
    if len(mesh.vertex_colors) == 0:
        from scipy.spatial import cKDTree
        pcd_pts = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)  # [0, 1]
        tree = cKDTree(pcd_pts)
        mesh_verts = np.asarray(mesh.vertices)
        # fill_holes / Taubin pueden dejar vértices NaN/inf; los reemplazamos por el
        # centroide de la nube para que el KDTree no falle.
        finite_mask = np.all(np.isfinite(mesh_verts), axis=1)
        if not finite_mask.all():
            mesh_verts = mesh_verts.copy()
            mesh_verts[~finite_mask] = pcd_pts.mean(axis=0)
        _, idx = tree.query(mesh_verts)
        mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_colors[idx])

    o3d.io.write_triangle_mesh(output_path, mesh)
    # Para ver color, abrir el .ply o el .glb (el .obj no guarda vertex_colors
    # de forma fiable: la mayoria de importadores, incluido el de Blender,
    # ignoran esa extension no estandar).
    ply_path = os.path.splitext(output_path)[0] + ".ply"
    if ply_path != output_path:
        o3d.io.write_triangle_mesh(ply_path, mesh)

    # glTF sí estandariza el color por vértice (atributo COLOR_0): el
    # importador de Blender lo detecta solo y arma el material automáticamente,
    # a diferencia del .ply (hay que configurar el shading a mano) o el .obj.
    glb_path = os.path.splitext(output_path)[0] + ".glb"
    export_glb(mesh, glb_path)


def export_glb(mesh, output_path: str) -> None:
    """Export an Open3D TriangleMesh (with vertex colors) as .glb.

    glTF's COLOR_0 vertex attribute is a standard that Blender's importer
    reads and wires into an auto-generated material, so the mesh shows its
    color immediately in Rendered/Material Preview - no manual shading setup
    needed (unlike .ply, and unlike .obj which drops the color entirely).
    """
    import trimesh

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    if len(vertex_colors) == len(vertices):
        rgba = np.hstack(
            [vertex_colors, np.ones((len(vertex_colors), 1))]
        )
        rgba = (rgba * 255).clip(0, 255).astype(np.uint8)
    else:
        rgba = None

    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba, process=False)
    # Trimesh's ColorVisuals has no material, so the exporter emits a
    # primitive with no "material" index. glTF then falls back to the spec's
    # default material (metallicFactor=1, roughnessFactor=1), which is fully
    # metallic and, without an environment/HDRI, renders near-black in
    # Blender regardless of COLOR_0 - i.e. "no color" even though the vertex
    # colors are present in the file. A diffuse (non-metallic) material lets
    # COLOR_0 actually show through.
    tmesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[255, 255, 255, 255], metallicFactor=0.0, roughnessFactor=1.0
    )
    tmesh.export(output_path)


def export_depth_pngs(predictions_np: dict, depth_dir: str) -> None:
    import matplotlib
    import matplotlib.cm as cm

    os.makedirs(depth_dir, exist_ok=True)
    from PIL import Image

    depth = predictions_np["depth"][..., 0]  # (num_frames, H, W)
    colormap = matplotlib.colormaps["turbo"]
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

    output_path = args.output or ("scene.obj" if args.export_format == "mesh" else "scene.ply")
    output_ext = os.path.splitext(output_path)[1].lower()

    if args.export_format == "mesh":
        if output_ext not in MESH_EXTENSIONS:
            raise ValueError(
                f"Unsupported --output extension '{output_ext}' for --export-format mesh. "
                f"Use one of {sorted(MESH_EXTENSIONS)}."
            )
        export_mesh(
            predictions_np, output_path,
            conf_thres=args.conf_thres,
            poisson_depth=args.poisson_depth,
            density_quantile=args.density_quantile,
            outlier_std=args.outlier_std,
            fill_holes=not args.no_fill_holes,
            smooth_iters=args.smooth_iters,
            method=args.method,
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

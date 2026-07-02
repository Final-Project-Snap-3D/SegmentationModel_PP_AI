#!/usr/bin/env python3
"""
Diagnostico + reconstruccion robusta de mesh desde un predictions.pt.

USO
---
1) Genera una vez los tensores crudos (necesita GPU, igual que la inferencia normal):

    python -m vggt_omega.inference_vggt -c vggt_omega_1b_512.pt \
        -i imgA.jpg imgB.jpg imgC.jpg --output predictions.pt

2) Lanza este script (NO necesita GPU):

    python debug_mesh.py --predictions predictions.pt --output scene.obj

Imprime cuantos puntos y triangulos quedan en cada paso, asi vemos exactamente
donde se queda sin geometria. Tambien guarda una copia en .ply (scene_mesh.ply),
que conserva los colores y se ve mejor en visores.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

# Permite ejecutar el script desde cualquier carpeta (p.ej. tools/): anade la
# raiz del repo al path para poder importar el paquete vggt_omega.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggt_omega.visualize_predictions import to_numpy_predictions
from vggt_omega.visual_util import predictions_to_point_cloud


def build_mesh(predictions_np, output_path, conf_thres=20.0, poisson_depth=9,
               density_quantile=0.05, outlier_std=2.0, do_fill_holes=True, smooth_iters=10,
               method="poisson"):
    vertices, colors = predictions_to_point_cloud(predictions_np, conf_thres=conf_thres)
    print(f"[1] Puntos en la nube: {len(vertices)}")

    if len(vertices) == 0:
        print("    ✖ Nube vacía: baja --conf-thres (prueba p.ej. --conf-thres 5) "
              "para conservar puntos.")
        return

    if len(vertices) < 100:
        print("    ⚠ Muy pocos puntos. Prueba a bajar --conf-thres (p.ej. 5) "
              "para conservar mas puntos.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    # Quita puntos sueltos (outliers): son los que provocan el pico/cono hacia
    # el origen y superficie inventada al reconstruir. --outlier-std mas bajo
    # = mas agresivo; 0 lo desactiva.
    if outlier_std > 0 and len(pcd.points) > 20:
        before = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=outlier_std)
        print(f"[2b] Outliers eliminados: {before - len(pcd.points)} (quedan {len(pcd.points)})")

    # Escala real de la nube (ya limpia) -> radio de normales adaptado.
    pts = np.asarray(pcd.points)
    extent = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(extent))
    radius = max(diag * 0.02, 1e-6)
    print(f"[2] Tamano de la nube (bbox diagonal): {diag:.4f}  ->  radio normales: {radius:.4f}")

    # Normales: hibrido KNN+radio adaptado, robusto a la escala.
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    n_valid = np.linalg.norm(np.asarray(pcd.normals), axis=1)
    print(f"[3] Normales calculadas: {np.count_nonzero(n_valid > 1e-6)}/{len(pts)} validas")

    # ── PyMeshLab branch (Screened Poisson) ──────────────────────────────────
    mesh = None
    used_method = method
    if method == "pymeshlab":
        try:
            import pymeshlab
            print("[4] Reconstruyendo con PyMeshLab (Screened Poisson)...")

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
            print(f"[5] PyMeshLab -> {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangulos")

            if len(mesh.triangles) == 0:
                print("    ✖ PyMeshLab produjo una malla vacía. Usando Poisson.")
                mesh = None
                used_method = "poisson"

        except Exception as e:
            print(f"⚠ PyMeshLab no disponible/falló: {e}. Usando Poisson.")
            mesh = None
            used_method = "poisson"

    # ── Poisson branch (default and automatic fallback from PyMeshLab) ────────
    if used_method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
        print(f"[4] Poisson -> {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangulos")

        densities = np.asarray(densities)
        low = densities < np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(low)
        print(f"[5] Tras recortar baja densidad -> {len(mesh.triangles)} triangulos")

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        print(f"[6] Tras limpieza -> {len(mesh.triangles)} triangulos")

        if len(mesh.triangles) == 0:
            print("    ✖ La mesh quedo VACIA. Suele ser por normales/escala o nube "
                  "demasiado dispersa. Prueba --conf-thres 5 o --poisson-depth 8.")
            return

        # Cerrar agujeros (la cara trasera no vista en capturas de pocas vistas).
        # Desactivalo con --no-fill-holes si genera "membranas" sobre las
        # curvas/concavidades sin datos.
        if do_fill_holes:
            t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            t = t.fill_holes()
            mesh = t.to_legacy()
            print(f"[7] Tras cerrar agujeros -> {len(mesh.triangles)} triangulos")
        else:
            print("[7] fill_holes desactivado (--no-fill-holes)")

    # ── Post-proceso común: smooth + normals + color + export ─────────────────
    # Conservar solo el componente conexo más grande (elimina geometría suelta)
    if len(mesh.triangles) > 0:
        tri_cls, cls_n, _ = mesh.cluster_connected_triangles()
        tri_cls = np.asarray(tri_cls)
        cls_n = np.asarray(cls_n)
        largest = int(cls_n.argmax())
        mesh.remove_triangles_by_mask(tri_cls != largest)
        mesh.remove_unreferenced_vertices()
        print(f"[keep-largest] -> {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangulos")

    if smooth_iters > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
        print(f"[8] Tras suavizado Taubin ({smooth_iters} iter) -> {len(mesh.triangles)} triangulos")

    mesh.compute_vertex_normals()

    # Transferencia de color: KDTree sobre la nube original como fallback.
    if len(mesh.vertex_colors) == 0:
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
    # Para ver color, abrir el .ply (el .obj no guarda vertex_colors de forma fiable)
    ply_path = os.path.splitext(output_path)[0] + ".ply"
    if ply_path != output_path:
        o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"\n✓ Método: {used_method.upper()}  |  Guardado: {output_path}  y  {ply_path}")
    print(f"  Vertices: {len(mesh.vertices)} | Triangulos: {len(mesh.triangles)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="predictions.pt")
    p.add_argument("--output", default="scene.obj")
    p.add_argument("--conf-thres", type=float, default=20.0)
    p.add_argument("--poisson-depth", type=int, default=9)
    p.add_argument("--density-quantile", type=float, default=0.05,
                   help="Recorta la cola de baja densidad del Poisson. Subir (0.1-0.3) "
                        "elimina superficie inventada.")
    p.add_argument("--outlier-std", type=float, default=2.0,
                   help="Filtro de outliers: menor = mas agresivo. 0 para desactivar.")
    p.add_argument("--no-fill-holes", action="store_true",
                   help="No cerrar agujeros (evita la membrana en curvas/concavidades).")
    p.add_argument("--smooth-iters", type=int, default=10,
                   help="Iteraciones de suavizado Taubin. 0 para desactivar.")
    p.add_argument("--method", choices=["poisson", "pymeshlab"], default="poisson",
                   help="Método de reconstrucción (default: poisson). "
                        "'pymeshlab' usa Screened Poisson de PyMeshLab; cae a poisson si no está disponible.")
    args = p.parse_args()

    predictions = torch.load(args.predictions, map_location="cpu")
    predictions_np = to_numpy_predictions(predictions)
    build_mesh(predictions_np, args.output, args.conf_thres, args.poisson_depth,
               args.density_quantile, outlier_std=args.outlier_std,
               do_fill_holes=not args.no_fill_holes,
               smooth_iters=args.smooth_iters,
               method=args.method)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Renderiza la nube de puntos de un predictions.pt a una imagen PNG, vista desde
varios angulos. NO necesita GPU ni visor 3D: abres preview.png con cualquier
visor de fotos.

USO:
    python render_preview.py --predictions predictions.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Permite ejecutar el script desde cualquier carpeta (p.ej. tools/): anade la
# raiz del repo al path para poder importar el paquete vggt_omega.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggt_omega.visualize_predictions import to_numpy_predictions
from vggt_omega.visual_util import predictions_to_point_cloud


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="predictions.pt")
    p.add_argument("--conf-thres", type=float, default=20.0)
    p.add_argument("--max-points", type=int, default=50000)
    p.add_argument("--output", default="preview.png")
    args = p.parse_args()

    predictions = torch.load(args.predictions, map_location="cpu")
    predictions_np = to_numpy_predictions(predictions)
    verts, colors = predictions_to_point_cloud(predictions_np, conf_thres=args.conf_thres)

    print(f"Puntos: {len(verts)}")
    print(f"Min XYZ: {verts.min(0)}")
    print(f"Max XYZ: {verts.max(0)}")
    print(f"Extension: {verts.max(0) - verts.min(0)}")

    # Submuestreo para que el render sea rapido
    if len(verts) > args.max_points:
        idx = np.linspace(0, len(verts) - 1, args.max_points).astype(np.int64)
        verts, colors = verts[idx], colors[idx]

    c = colors.astype(np.float32) / 255.0

    # 3 vistas desde angulos distintos
    angles = [(20, -60), (20, 30), (90, -90)]
    titles = ["Vista 1", "Vista 2", "Vista cenital"]
    fig = plt.figure(figsize=(18, 6))
    for i, ((elev, azim), title) in enumerate(zip(angles, titles), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=c, s=1, marker=".")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"\n✓ Guardado: {args.output}  (abrelo con cualquier visor de imagenes)")


if __name__ == "__main__":
    main()

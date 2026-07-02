#!/usr/bin/env bash
# run_pipeline.sh  –  Snap-to-3D: pipeline completa
#
# USO
#   bash run_pipeline.sh [CARPETA_IMÁGENES] [CARPETA_SALIDA]
#
# EJEMPLOS
#   bash run_pipeline.sh
#   bash run_pipeline.sh inference_test/images/consola inference_test/outputs/consola
#   bash run_pipeline.sh inference_test/images/mà      inference_test/outputs/ma
#
set -euo pipefail

# ── Checkpoints ───────────────────────────────────────────────────────────────
CHECKPOINT="checkpoints/vggt_omega_1b_512.pt"
SEG_CHECKPOINT="checkpoints/yolo26s-seg.pt"

# ── Rutas (primer y segundo argumento, o valores por defecto) ─────────────────
IMAGES_DIR="${1:-inference_test/images/consola}"
OUTPUT_DIR="${2:-inference_test/outputs}"

# ── Parámetros de segmentación ────────────────────────────────────────────────
MORPH_KERNEL=21          # kernel morfológico para limpiar la máscara

# ── Parámetros de mesh ────────────────────────────────────────────────────────
METHOD="pymeshlab"       # método de reconstrucción: poisson | pymeshlab
CONF_THRES=20.0          # percentil de confianza para filtrar puntos
POISSON_DEPTH=9          # profundidad del octree Poisson
DENSITY_QUANTILE=0.05    # recorta la cola de baja densidad
OUTLIER_STD=2.0          # outlier removal (0 = desactivado)
SMOOTH_ITERS=10          # iteraciones Taubin (0 = sin suavizado)
# ─────────────────────────────────────────────────────────────────────────────

# Recolecta imágenes de la carpeta (jpg / jpeg / png, case-insensitive)
IMAGES=()
while IFS= read -r -d '' f; do
    IMAGES+=("$f")
done < <(find "$IMAGES_DIR" -maxdepth 1 -type f \
         \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
         -print0 | sort -z)

if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "✗  No se encontraron imágenes en: $IMAGES_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════"
echo "  Snap-to-3D  ·  pipeline completa"
printf "  Imágenes : %d  en  %s\n" "${#IMAGES[@]}" "$IMAGES_DIR"
echo "  Salida   : $OUTPUT_DIR"
echo "  Método   : $METHOD"
echo "════════════════════════════════════════════════════════"
echo ""

# ── 1. Inferencia VGGT-Ω + segmentación  (necesita GPU) ──────────────────────
echo "▶ [1/3] Inferencia VGGT-Ω + segmentación..."
python -m vggt_omega.inference_vggt \
    -c  "$CHECKPOINT" \
    -i  "${IMAGES[@]}" \
    --seg-checkpoint "$SEG_CHECKPOINT" \
    --morph-open \
    --morph-kernel  "$MORPH_KERNEL" \
    --keep-largest \
    --output "$OUTPUT_DIR/predictions.pt"

# ── 2. Mesh: reconstrucción + Taubin + color de vértice  (sin GPU para Poisson)
echo ""
echo "▶ [2/3] Reconstrucción de mesh (${METHOD})..."

python -m vggt_omega.visualize_predictions \
    -p  "$OUTPUT_DIR/predictions.pt" \
    -o  "$OUTPUT_DIR/scene.obj" \
    --method            "$METHOD" \
    --conf-thres        "$CONF_THRES" \
    --poisson-depth     "$POISSON_DEPTH" \
    --density-quantile  "$DENSITY_QUANTILE" \
    --outlier-std       "$OUTLIER_STD" \
    --smooth-iters      "$SMOOTH_ITERS" \
    --depth-dir         "$OUTPUT_DIR/depth_maps" \
    --mask-dir          "$OUTPUT_DIR/masks"

# ── 3. Preview PNG de la nube de puntos  (sin GPU) ───────────────────────────
echo ""
echo "▶ [3/3] Preview de la nube de puntos..."
python tools/render_preview.py \
    --predictions "$OUTPUT_DIR/predictions.pt" \
    --output      "$OUTPUT_DIR/preview.png"

echo ""
echo "════════════════════════════════════════════════════════"
echo "✓  Pipeline completada. Artefactos en: $OUTPUT_DIR/"
echo ""
echo "  predictions.pt     tensores crudos (reutilizables sin GPU)"
echo "  scene.ply          mesh con color de vértice  ← abrir en visor 3D"
echo "  scene.glb          mesh con color  ← importar en Blender (color automático)"
echo "  scene.obj          mesh para Blender (sin color: el .obj no lo soporta)"
echo "  preview.png        preview rápida de la nube"
echo "  depth_maps/        mapas de profundidad por frame"
echo "  masks/             máscaras de segmentación"
echo "════════════════════════════════════════════════════════"

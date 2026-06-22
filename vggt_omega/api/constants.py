# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration constants for the VGGT-Omega inference API.

Every value can be overridden with an environment variable so the same code
runs unchanged on a laptop, a CI box or a GPU server. Paths are resolved
relative to the repository root unless an absolute path is given.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# vggt_omega/api/constants.py -> repo root is two levels up from this file.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | os.PathLike) -> Path:
    """Resolve `path` against the repo root when it is not absolute."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


# Main VGGT-Omega checkpoint (e.g. vggt_omega_1b_512.pt). This file is large
# and is NOT committed to the repo: point VGGT_CHECKPOINT at wherever you keep
# it, or drop it under checkpoints/ with the default name.
VGGT_CHECKPOINT = _resolve(
    os.getenv("VGGT_CHECKPOINT", "checkpoints/vggt_omega_1b_512.pt")
)

# Optional YOLO-seg checkpoint used to keep only the segmented object in the
# point cloud. The repo ships yolo26s-seg.pt at its root.
SEG_CHECKPOINT = _resolve(os.getenv("VGGT_SEG_CHECKPOINT", "yolo26s-seg.pt"))

# Where inference artifacts (PLY / depth / mask PNGs) are written. Each request
# gets its own sub-folder named after the job id.
OUTPUT_DIR = _resolve(os.getenv("VGGT_OUTPUT_DIR", "api_outputs"))

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
# The VGGT-Omega forward pass uses torch.autocast(device_type="cuda"), so a
# CUDA device is required in practice. Kept configurable for forward-compat.
DEVICE = os.getenv("VGGT_DEVICE", "cuda")

# ---------------------------------------------------------------------------
# Inference defaults (mirror the CLI flags of inference_vggt.py)
# ---------------------------------------------------------------------------
DEFAULT_RESOLUTION = int(os.getenv("VGGT_RESOLUTION", "512"))
DEFAULT_MODE = os.getenv("VGGT_MODE", "balanced")  # "balanced" | "max_size"
DEFAULT_CONF_THRES = float(os.getenv("VGGT_CONF_THRES", "20.0"))
DEFAULT_SEG_CONF = float(os.getenv("VGGT_SEG_CONF", "0.25"))

# ---------------------------------------------------------------------------
# Upload limits / validation
# ---------------------------------------------------------------------------
# Maximum number of images accepted in a single request.
MAX_IMAGES = int(os.getenv("VGGT_MAX_IMAGES", "32"))
# Maximum size (bytes) of a single uploaded image. Default: 25 MB.
MAX_IMAGE_BYTES = int(os.getenv("VGGT_MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))
# Image extensions the preprocessing pipeline can load (HEIC via pillow-heif).
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
}

# ---------------------------------------------------------------------------
# API metadata
# ---------------------------------------------------------------------------
API_TITLE = "VGGT-Omega Inference API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Run VGGT-Omega multi-view inference over a variable number of images and "
    "export a 3D point cloud (PLY) plus optional depth and object-mask PNGs."
)

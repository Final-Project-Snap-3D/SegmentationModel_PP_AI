#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run VGGT-Omega inference over a set of images.

The model forward pass relies on ``torch.autocast(device_type="cuda", ...)``,
so a CUDA device is required.

Example
-------
    python -m vggt_omega.inference_vggt \
        --checkpoint path/to/vggt_omega_1b_512.pt \
        --images path/to/imageA.png path/to/imageB.png path/to/imageC.png \
        --output predictions.pt
"""

import argparse
import os

import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.segmentation import add_object_masks
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.visualize_predictions import (
    export_depth_pngs,
    export_object_mask_pngs,
    export_point_cloud_ply,
    to_numpy_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT-Omega inference.")
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        help="Path to the VGGT-Omega checkpoint (e.g. vggt_omega_1b_512.pt).",
    )
    parser.add_argument(
        "-i",
        "--images",
        required=True,
        nargs="+",
        help="One or more input image paths.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=512,
        help="Image resolution used for preprocessing (default: 512).",
    )
    parser.add_argument(
        "--mode",
        choices=["balanced", "max_size"],
        default="balanced",
        help="Preprocessing mode (default: balanced).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to run on (default: cuda).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="scene.ply",
        help=(
            "Output path. A .ply extension exports a point cloud; a .pt/.pth "
            "extension saves the raw prediction tensors (default: scene.ply)."
        ),
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=20.0,
        help="Confidence percentile threshold for point filtering in PLY export (default: 20).",
    )
    parser.add_argument(
        "--depth-dir",
        default=None,
        help="If set, also export each depth map as a colorized PNG into this folder.",
    )
    parser.add_argument(
        "--seg-checkpoint",
        default=None,
        help="Path to a YOLO-seg checkpoint (e.g. best.pt). If set, only the "
        "segmented object is kept in the point cloud; the background is dropped.",
    )
    parser.add_argument(
        "--seg-u2net-checkpoint",
        default=None,
        help="Path to a U2Net checkpoint (e.g. u2net.pt). If set, U2Net is used for object segmentation.",
    )
    parser.add_argument(
        "--u2net-thres",
        type=float,
        default=0.5,
        help="Binarisation threshold for the U2Net saliency map (default: 0.5).",
    )
    parser.add_argument(
        "--seg-conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold for the object mask (default: 0.25).",
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="If set, export each object mask as a black/white PNG into this folder.",
    )
    parser.add_argument(
        "--masks-debug",
        action="store_true",
        help="In mixed mode, also save per-frame YOLO, U2Net, and combined masks "
             "(mask_NNN_yolo.png / mask_NNN_u2.png / mask_NNN_mix.png) to --mask-dir.",
    )
    parser.add_argument(
        "--morph-open",
        action="store_true",
        help="Apply morphological opening to the final mask (removes small noise regions).",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=21,
        help="Size of the elliptical structuring element for morphological opening (default: 21). "
             "A large value (e.g. 21–51) removes small spurious regions and thin connections, "
             "keeping only the main compact object.",
    )
    parser.add_argument(
        "--keep-largest",
        action="store_true",
        help="After masking, keep only the largest connected component per frame. "
             "Discards all secondary detections.",
    )
    return parser.parse_args()


@torch.inference_mode()
def run_inference(
    checkpoint_path: str,
    image_names: list[str],
    image_resolution: int = 512,
    mode: str = "balanced",
    device: str = "cuda",
    seg_checkpoint: str | None = None,
    seg_u2net_checkpoint: str | None = None,
    seg_conf: float = 0.25,
    u2net_thres: float = 0.5,
    masks_debug: bool = False,
    morph_open: bool = False,
    morph_kernel: int = 21,
    keep_largest: bool = False,
) -> dict[str, torch.Tensor]:
    """Load the model and images, then return the raw predictions plus
    decoded camera extrinsics/intrinsics."""
    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    images = load_and_preprocess_images(
        image_names,
        mode=mode,
        image_resolution=image_resolution,
    ).to(device)

    predictions = model(images)

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    camera_and_register_tokens = predictions["camera_and_register_tokens"]
    predictions["extrinsics"] = extrinsics
    predictions["intrinsics"] = intrinsics
    predictions["camera_tokens"] = camera_and_register_tokens[:, :, :1]
    predictions["registers"] = camera_and_register_tokens[:, :, 1:]

    if seg_checkpoint is not None or seg_u2net_checkpoint is not None:
        add_object_masks(
            predictions,
            yolo_checkpoint=seg_checkpoint,
            u2net_checkpoint=seg_u2net_checkpoint,
            imgsz=image_resolution,
            conf=seg_conf,
            u2net_thres=u2net_thres,
            device=device,
            masks_debug=masks_debug,
            morph_open=morph_open,
            morph_kernel=morph_kernel,
            keep_largest=keep_largest,
        )

    return predictions


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but VGGT-Omega requires a CUDA device "
            "(the forward pass uses torch.autocast(device_type='cuda'))."
        )

    predictions = run_inference(
        checkpoint_path=args.checkpoint,
        image_names=args.images,
        image_resolution=args.resolution,
        mode=args.mode,
        device=args.device,
        seg_checkpoint=args.seg_checkpoint,
        seg_u2net_checkpoint=args.seg_u2net_checkpoint,
        seg_conf=args.seg_conf,
        u2net_thres=args.u2net_thres,
        masks_debug=args.masks_debug,
        morph_open=args.morph_open,
        morph_kernel=args.morph_kernel,
        keep_largest=args.keep_largest,
    )

    print(f"Processed {len(args.images)} image(s):")
    for key in (
        "pose_enc",
        "depth",
        "depth_conf",
        "extrinsics",
        "intrinsics",
        "camera_tokens",
        "registers",
    ):
        value = predictions.get(key)
        if isinstance(value, torch.Tensor):
            print(f"  {key:<14} {tuple(value.shape)}")

    output_ext = os.path.splitext(args.output)[1].lower()
    if output_ext in (".pt", ".pth"):
        cpu_predictions = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in predictions.items()
        }
        torch.save(cpu_predictions, args.output)
        print(f"Saved predictions to {args.output}")
    elif output_ext == ".ply":
        predictions_np = to_numpy_predictions(predictions)
        export_point_cloud_ply(predictions_np, args.output, conf_thres=args.conf_thres)
        print(f"Saved PLY point cloud to {args.output}")
        if args.depth_dir is not None:
            export_depth_pngs(predictions_np, args.depth_dir)
        if args.mask_dir is not None:
            export_object_mask_pngs(predictions_np, args.mask_dir)
    else:
        raise ValueError(
            f"Unsupported --output extension '{output_ext}'. Use .ply or .pt/.pth."
        )


if __name__ == "__main__":
    main()

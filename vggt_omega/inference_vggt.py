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

import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


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
        default=None,
        help="Optional path to save the predictions as a .pt file.",
    )
    return parser.parse_args()


@torch.inference_mode()
def run_inference(
    checkpoint_path: str,
    image_names: list[str],
    image_resolution: int = 512,
    mode: str = "balanced",
    device: str = "cuda",
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

    if args.output is not None:
        cpu_predictions = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in predictions.items()
        }
        torch.save(cpu_predictions, args.output)
        print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

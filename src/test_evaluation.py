import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import VizWiz
from model import SegmentationModel
from augmentation import DataAugmentation


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    # pred, gt: arrays booleans (H, W) -> retorna dict amb iou, dice, precision, recall
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def run_evaluation(model_path, images_dir, annotations, image_size=512,
                   batch_size=8, threshold=0.5, output_dir=None, device=None):
    # model_path: ruta al checkpoint
    # retorna dict amb mètriques mitjanes (iou, dice, precision, recall) sobre tot el dataset
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Resize + normalize (igual que a val del main.py)
    aug = DataAugmentation(img_size=image_size)
    dataset = VizWiz(images_dir, annotations, transform=aug.val_test())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        enc_channels = checkpoint.get("enc_channels", (3, 16, 32, 64))
        dec_channels = checkpoint.get("dec_channels", (64, 32, 16))
        state_dict   = checkpoint["model_state_dict"]
    else:
        enc_channels = (3, 16, 32, 64)
        dec_channels = (64, 32, 16)
        state_dict   = checkpoint

    model = SegmentationModel(
        encChannels=enc_channels,
        decChannels=dec_channels,
        nbClasses=1,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    all_metrics = {"iou": [], "dice": [], "precision": [], "recall": []}
    sample_idx = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds  = probs > threshold
            gts    = y.numpy().astype(bool)

            for b in range(len(preds)):
                m = compute_metrics(preds[b], gts[b])
                for k, v in m.items():
                    all_metrics[k].append(v)

                if output_dir is not None:
                    mask_img = (preds[b] * 255).astype(np.uint8)
                    Image.fromarray(mask_img).save(
                        os.path.join(output_dir, f"pred_{sample_idx:05d}.png")
                    )
                sample_idx += 1

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


def main():
    # Llegeix els args de la CLI, crida run_evaluation i imprimeix les mètriques per pantalla
    parser = argparse.ArgumentParser(description="Avaluació del model de segmentació")
    parser.add_argument("--model_path",  required=True, help="ruta al checkpoint del model (.pth)")
    parser.add_argument("--images_dir",  required=True, help="directori d'imatges")
    parser.add_argument("--annotations", required=True, help="ruta al JSON d'anotacions")
    parser.add_argument("--image_size",  type=int,   default=512)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="llindar per binaritzar la predicció (default: 0.5)")
    parser.add_argument("--output_dir",  type=str,   default=None,
                        help="directori on guardar màscares predites (opcional)")
    args = parser.parse_args()

    metrics = run_evaluation(
        model_path=args.model_path,
        images_dir=args.images_dir,
        annotations=args.annotations,
        image_size=args.image_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )

    print("\n=== Test Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k.upper():12s}: {v:.4f}")
    print("================================\n")


if __name__ == "__main__":
    main()

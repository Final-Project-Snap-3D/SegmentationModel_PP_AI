import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import VizWiz
from model import SegmentationModel, U2Net
from augmentation import DataAugmentation
from inference import _is_yolo, _load_yolo_model


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


def _eval_yolo(model_path, images_dir, annotations, image_size, conf, output_dir):
    model = _load_yolo_model(model_path)
    dataset = VizWiz(images_dir, annotations, transform=None)
    all_metrics = {"iou": [], "dice": [], "precision": [], "recall": []}

    for idx in range(len(dataset)):
        img_name = dataset.get_name(idx)
        img_path = os.path.join(images_dir, img_name)

        _, gt_tensor = dataset[idx]
        gt = gt_tensor.numpy().astype(bool)
        gt_h, gt_w = gt.shape

        results = model.predict(source=img_path, imgsz=image_size, conf=conf, verbose=False)
        result = results[0]
        orig_h, orig_w = result.orig_shape

        if result.masks is not None and len(result.masks) > 0:
            merged = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for mask_data in result.masks.data:
                mask_np = mask_data.cpu().numpy()
                resized = np.array(
                    Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                        (orig_w, orig_h), Image.NEAREST
                    )
                )
                merged = np.maximum(merged, resized)
            pred = (merged > 127).astype(bool)
        else:
            pred = np.zeros((orig_h, orig_w), dtype=bool)

        # Redimensiona la predicció a les dimensions del GT per a la comparació
        if pred.shape != (gt_h, gt_w):
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8) * 255).resize(
                    (gt_w, gt_h), Image.NEAREST
                )
            ) > 127

        m = compute_metrics(pred, gt)
        for k, v in m.items():
            all_metrics[k].append(v)

        if output_dir is not None:
            Image.fromarray((pred * 255).astype(np.uint8)).save(
                os.path.join(output_dir, f"pred_{idx:05d}.png")
            )

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


def run_evaluation(model_path, images_dir, annotations, image_size=512,
                   batch_size=8, threshold=0.5, conf=0.25, output_dir=None, device=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    is_yolo = os.path.splitext(model_path)[1].lower() == ".pt" and _is_yolo(model_path)
    if is_yolo:
        return _eval_yolo(model_path, images_dir, annotations, image_size, conf, output_dir)

    # ── Torch model (UNet / U2Net) ───────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    aug = DataAugmentation(img_size=image_size)
    dataset = VizWiz(images_dir, annotations, transform=aug.val_test())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        enc_channels = checkpoint.get("enc_channels", (3, 16, 32, 64))
        dec_channels = checkpoint.get("dec_channels", (64, 32, 16))
        model_name   = checkpoint.get("model_name")
        in_channels  = checkpoint.get("in_channels") or 3
        out_channels = checkpoint.get("out_channels") or 1
        state_dict   = checkpoint["model_state_dict"]
    else:
        enc_channels = None
        dec_channels = None
        model_name   = None
        in_channels  = 3
        out_channels = 1
        state_dict   = checkpoint

    if model_name == "U2Net":
        model = U2Net(inChannels=in_channels, outChannels=out_channels).to(device)
    else:
        model = SegmentationModel(
            encChannels=enc_channels or (3, 16, 32, 64),
            decChannels=dec_channels or (64, 32, 16),
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
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
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
    parser = argparse.ArgumentParser(description="Avaluació del model de segmentació")
    parser.add_argument("--model_path",  required=True,
                        help="ruta al checkpoint: .pth per UNet/U2Net, .pt per YOLO")
    parser.add_argument("--images_dir",  required=True, help="directori d'imatges")
    parser.add_argument("--annotations", required=True, help="ruta al JSON d'anotacions")
    parser.add_argument("--image_size",  type=int,   default=512)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="llindar per binaritzar la predicció UNet/U2Net (default: 0.5)")
    parser.add_argument("--conf",        type=float, default=0.25,
                        help="llindar de confiança per YOLO (default: 0.25)")
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
        conf=args.conf,
        output_dir=args.output_dir,
    )

    print("\n=== Test Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k.upper():12s}: {v:.4f}")
    print("================================\n")


if __name__ == "__main__":
    main()

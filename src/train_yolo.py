"""
Train YOLO26 instance segmentation on VizWiz.
Run convert_vizwiz_to_yolo.py once before this to build data_yolo/.

Tuned for GTX 1060 6GB: yolo26s-seg, imgsz=512, batch=4, AMP on.
Lower batch or use yolo26n-seg if OOM. Raise batch to 8 if VRAM holds.
"""
import argparse
from pathlib import Path

import torch
import wandb
from ultralytics import YOLO


class YoloWandbLogger:
    """W&B logger for Ultralytics YOLO training callbacks."""

    def __init__(self, project: str, entity: str | None, run_name: str | None, log_image_every: int):
        wandb.login()
        self.run = wandb.init(project=project, entity=entity)
        if run_name:
            wandb.run.name = run_name
        self.log_image_every = max(int(log_image_every), 0)

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_train_loss(self, trainer) -> dict:
        train_metrics = {}
        tloss = getattr(trainer, "tloss", None)
        if tloss is None:
            return train_metrics

        if isinstance(tloss, torch.Tensor):
            losses = tloss.detach().cpu().tolist()
        else:
            losses = list(tloss)

        loss_names = ["box", "seg", "cls", "dfl", "pose", "kobj"]
        total = 0.0
        for i, loss_value in enumerate(losses):
            loss_float = self._to_float(loss_value)
            if loss_float is None:
                continue
            total += loss_float
            if i < len(loss_names):
                train_metrics[f"train/{loss_names[i]}_loss"] = loss_float
            else:
                train_metrics[f"train/loss_{i}"] = loss_float

        train_metrics["train_loss"] = total
        return train_metrics

    def _extract_val_loss(self, metrics: dict) -> float | None:
        loss_keys = [
            "val/box_loss",
            "val/seg_loss",
            "val/cls_loss",
            "val/dfl_loss",
        ]
        losses = [metrics.get(k) for k in loss_keys if metrics.get(k) is not None]
        if not losses:
            return None
        return float(sum(losses))

    @staticmethod
    def _first_metric(metrics: dict, keys: list[str]) -> float | None:
        for key in keys:
            if key in metrics:
                return metrics[key]
        return None

    def _maybe_build_val_images(self, trainer, epoch: int) -> list:
        if self.log_image_every == 0 or epoch % self.log_image_every != 0:
            return []

        save_dir = Path(getattr(trainer, "save_dir", ""))
        if not save_dir.exists():
            return []

        image_candidates = [
            "val_batch0_pred.jpg",
            "val_batch0_labels.jpg",
            "results.png",
            "confusion_matrix.png",
        ]
        images = []
        for image_name in image_candidates:
            image_path = save_dir / image_name
            if image_path.exists():
                images.append(wandb.Image(str(image_path), caption=image_name))
        return images

    def on_fit_epoch_end(self, trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        raw_metrics = getattr(trainer, "metrics", {}) or {}
        metrics = {}
        for key, value in raw_metrics.items():
            value_float = self._to_float(value)
            if value_float is not None:
                metrics[key] = value_float

        metrics.update(self._extract_train_loss(trainer))

        val_loss = self._extract_val_loss(metrics)
        if val_loss is not None:
            metrics["val_loss"] = val_loss

        # Closest comparable mask-quality metric to IoU/Dice in YOLO logs.
        val_iou_like = self._first_metric(metrics, ["metrics/seg(mAP50)", "metrics/mAP50(M)"])
        if val_iou_like is not None:
            metrics["val_iou_proxy"] = val_iou_like
            metrics["val_dice_proxy"] = (2.0 * val_iou_like) / (1.0 + val_iou_like + 1e-7)

        val_images = self._maybe_build_val_images(trainer, epoch)
        if val_images:
            metrics["val/predictions"] = val_images

        wandb.log(metrics, step=epoch)

    @staticmethod
    def on_train_end(trainer):
        _ = trainer
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_yolo/vizwiz.yaml",
                        help="path to dataset yaml produced by convert_vizwiz_to_yolo.py")
    parser.add_argument("--model", default="yolo26s-seg.pt",
                        help="yolo26n-seg.pt (smallest) | yolo26s-seg.pt | yolo26m-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' to force CPU")
    parser.add_argument("--project", default="runs/yolo_vizwiz")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--wandb_project", default="snap-to-3d")
    parser.add_argument("--wandb_entity", default="pp-snap-to-3d")
    parser.add_argument("--wandb_run_name", default=None,
                        help="Optional explicit W&B run name (defaults to --name).")
    parser.add_argument("--log_image_every", type=int, default=5,
                        help="Log YOLO validation images to W&B every N epochs. Set 0 to disable.")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run convert_vizwiz_to_yolo.py first."
        )

    model = YOLO(args.model)

    wandb_logger = YoloWandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or args.name,
        log_image_every=args.log_image_every,
    )
    model.add_callback("on_fit_epoch_end", wandb_logger.on_fit_epoch_end)
    model.add_callback("on_train_end", wandb_logger.on_train_end)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        amp=True,
        project=args.project,
        name=args.name,
        patience=20,
        save_period=10,
    )


if __name__ == "__main__":
    main()

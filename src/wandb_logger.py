import torch
import torch.nn as nn
import wandb
from datetime import datetime
from data_visualization import visualize
from utils import TaskType
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class WandbLogger():
    """Simple W&B logger for tracking training metrics"""

    def __init__(self, task: TaskType, model: nn.Module):
        """Initialize W&B logger
        
        Args:
            task: TaskType enum for the experiment
            model: PyTorch model being trained
        """
        self.model = model
        
        wandb.login()
        wandb.init(project="snap-to-3d", entity="pp-snap-to-3d")
        wandb.run.name = f'{task.value}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        
        # Log model architecture
        wandb.watch(model, log_freq=100)

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics to Weights & Biases"""
        wandb.log(metrics, step=step)

    def save_checkpoint(self, model_path: str):
        """Save model checkpoint locally"""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'enc_channels': self.model.encChannels,
            'dec_channels': self.model.decChannels,
            'timestamp': datetime.now().isoformat(),
        }, model_path)

    def log_model(self, model_path: str, name: str = None):
        """Log model to W&B artifacts"""
        artifact = wandb.Artifact(name=name or "model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    def log_images(self, images: list, masks: list, step: int = None):
        """Log images and masks to W&B"""
        for img, mask in zip(images, masks):
            # Convert CHW float tensor to HWC uint8 image in [0, 255] for W&B.
            if isinstance(img, torch.Tensor):
                img_np = img.detach().cpu().float()
                if img_np.ndim == 3:
                    img_np = img_np.permute(1, 2, 0)
                img_np = (img_np * 255.0).round().to(torch.uint8).numpy()
            else:
                img_np = np.asarray(img)

            # Convert mask to 2D integer array expected by wandb.Image(..., masks=...).
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu()
                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                    mask_np = mask_np.squeeze(0)
                mask_np = mask_np.long().numpy()
            else:
                mask_np = np.asarray(mask)
                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                    mask_np = np.squeeze(mask_np, axis=0)
                mask_np = mask_np.astype(np.int32)
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_np)
            axes[0].set_title(f"Image")
            axes[0].axis('off')

            plt.tight_layout()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{ts}.png")
            plt.close()
            print(f"Saved {ts}.png")

            wandb.log({
                "image": [
                    wandb.Image(
                        img_np,
                        masks={
                            "predictions": {
                                "mask_data": mask_np,
                                "class_labels": {0: "background", 1: "object"},
                            }
                        },
                    )
                ]
            }, step=step)

    def _to_hwc_uint8_image(self, img):
        """Convert tensor/array image to HWC uint8 in [0, 255]."""
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().float()
            if img_np.ndim == 3:
                img_np = img_np.permute(1, 2, 0)
            return (img_np * 255.0).round().to(torch.uint8).numpy()

        img_np = np.asarray(img)
        if img_np.dtype != np.uint8:
            if np.issubdtype(img_np.dtype, np.floating):
                img_np = np.clip(img_np, 0.0, 1.0)
                img_np = (img_np * 255.0).round().astype(np.uint8)
            else:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return img_np

    def _to_hw_uint8_mask(self, mask):
        """Convert tensor/array mask to HW uint8 (0 or 255)."""
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu()
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np.squeeze(0)
            mask_np = mask_np.long().numpy()
        else:
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = np.squeeze(mask_np, axis=0)

        mask_np = (mask_np > 0).astype(np.uint8) * 255
        return mask_np

    def log_image_mask_artifacts(self, images: list, masks: list, epoch: int, base_dir: str, max_items: int = 3):
        """Store image/mask pairs as independent W&B artifacts."""
        artifacts_root = Path(base_dir) / "wandb_artifacts" / f"epoch_{epoch:04d}"
        artifacts_root.mkdir(parents=True, exist_ok=True)

        for idx, (img, mask) in enumerate(zip(images[:max_items], masks[:max_items]), start=1):
            sample_dir = artifacts_root / f"sample_{idx}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            image_np = self._to_hwc_uint8_image(img)
            mask_np = self._to_hw_uint8_mask(mask)

            image_path = sample_dir / "image.png"
            mask_path = sample_dir / "mask.png"
            Image.fromarray(image_np).save(image_path)
            Image.fromarray(mask_np).save(mask_path)

            artifact = wandb.Artifact(
                name=f"val-sample-{idx}-epoch-{epoch:04d}",
                type="validation-sample",
                metadata={"epoch": epoch, "sample_index": idx},
            )
            artifact.add_file(str(image_path), name="image.png")
            artifact.add_file(str(mask_path), name="mask.png")
            wandb.log_artifact(artifact)

    def finish(self):
        """Finish the W&B run"""
        wandb.finish()
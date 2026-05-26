import torch
import torch.nn as nn
import wandb
from datetime import datetime
from utils import TaskType
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
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
                img_np = img_np.permute(1, 2, 0)  # CHW -> HWC
            mean = torch.tensor([0.485, 0.456, 0.406])
            std  = torch.tensor([0.229, 0.224, 0.225])
            img_np = (img_np * std + mean).clamp(0.0, 1.0)
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

    def build_segmentation_images(self, images: list, masks: list, max_items: int = 3) -> list:
        """Build list of wandb.Image from matplotlib figures (image | predicted mask side by side)."""
        wb_images = []
        for i, (img, mask) in enumerate(zip(images[:max_items], masks[:max_items]), start=1):
            img_np = self._to_hwc_uint8_image(img)
            mask_np = self._to_hw_uint8_mask(mask)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img_np)
            axes[0].set_title(f"Image #{i}")
            axes[0].axis('off')
            axes[1].imshow(mask_np, cmap='gray')
            axes[1].set_title(f"Pred Mask #{i}")
            axes[1].axis('off')
            plt.tight_layout()

            wb_images.append(wandb.Image(fig))
            plt.close(fig)

        return wb_images

    def finish(self):
        """Finish the W&B run"""
        wandb.finish()
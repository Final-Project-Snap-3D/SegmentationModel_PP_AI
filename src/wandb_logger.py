import torch
import torch.nn as nn
import wandb
from logger import Logger
from datetime import datetime
from utils import TaskType
from pathlib import Path


class WandbLogger(Logger):
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
            'timestamp': datetime.now().isoformat(),
        }, model_path)

    def log_model(self, model_path: str, name: str = None):
        """Log model to W&B artifacts"""
        artifact = wandb.Artifact(name=name or "model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    def finish(self):
        """Finish the W&B run"""
        wandb.finish()
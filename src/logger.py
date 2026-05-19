from abc import ABC, abstractmethod
from typing import Dict, Any


class Logger(ABC):
    """Abstract logger interface for training runs."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        pass

    @abstractmethod
    def save_checkpoint(self, model_path: str):
        pass

    @abstractmethod
    def finish(self):
        pass

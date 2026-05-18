from enum import Enum


class TaskType(Enum):
    """Task types for the training pipeline"""
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    CLASSIFICATION = "classification"

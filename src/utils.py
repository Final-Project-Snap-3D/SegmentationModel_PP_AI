from enum import Enum


class TaskType(Enum):
    """Task types for the training pipeline"""
    U2NET_SEGMENTATION = "u2net_segmentation"
    UNET_SEGMENTATION = "unet_segmentation"
    YOLO26_SEGMENTATION = "yolo26_segmentation"

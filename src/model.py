import torch
import torch.nn as nn

# ENCODER (downsamplig): dues conv + BN + ReLU + MaxPool? 
# a SAM2: "apply layer decay (Clark et al., 2020) on the image encoder"
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        pass

    def forward(self, x):
        pass


# DECODER (upsampling):  dues transconv + skip connection + BN + ReLU?
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        pass

    def forward(self, x, skip):
        pass


class SegmentationModel(nn.Module):
    """UNet per segmentació binària de salient object detection"""

    def __init__():
        super().__init__()
        pass

    def forward(self, x):
        pass

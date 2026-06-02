import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Block, self).__init__()
        # TODO: create the convolution and RELU layers
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, padding = 1)
        # Adding batch normalization since we are training from scratch, can be deleted 
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outChannels, outChannels, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        # TODO: apply CONV => RELU => CONV block to the inputs and return it
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
# ENCODER (downsamplig): dues conv + BN + ReLU + MaxPool? 
# a SAM2: "apply layer decay (Clark et al., 2020) on the image encoder"
class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64)):
            super().__init__()
            # TODO: create the encoder blocks and maxpooling layer
            self.encBlocks = nn.ModuleList([
              Block(channels[i], channels[i+1])
                    for i in range(len(channels)-1)
            ])
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # TODO: loop through the encoder blocks and update the blockOutputs list
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)

        return blockOutputs

# DECODER (upsampling):  dues transconv + skip connection + BN + ReLU?
class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)):
          super().__init__()
          # TODO: initialize the number of channels, upsampler blocks, and decoder blocks
          self.upconvs = nn.ModuleList([
              nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2)
              for i in range(len(channels)-1)
          ])
          self.dec_blocks = nn.ModuleList([
              Block(channels[i+1] * 2, channels[i+1])
              for i in range(len(channels)-1)
          ])

    def forward(self, x, encFeatures):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            x = torch.cat([x, encFeatures[-(i+2)]], dim=1)            # cat along channel dim
            x = self.dec_blocks[i](x)                     # index the block
        return x
    
class SegmentationModel(nn.Module):
    """UNet per segmentació binària de salient object detection"""

    def __init__(self, encChannels=(3, 16, 32, 64),
          decChannels=(64, 32, 16),
          nbClasses=1):

        super().__init__()
        self.encChannels = encChannels
        self.decChannels = decChannels
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        # initialize the regression head and store the class variables
        self.head = nn.Conv2d(decChannels[-1], nbClasses, kernel_size=1)

    def forward(self, x):
        # TODO: grab the features from the encoder
        encFeatures = self.encoder(x)

        # TODO: pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[-1], encFeatures)

        # TODO: pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)

        # return the segmentation map
        return map


class REBNCONV(nn.Module):
    def __init__(self, inChannels, outChannels, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            inChannels,
            outChannels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RSU7(nn.Module):
    def __init__(self, inChannels, outChannels, midChannels):
        super().__init__()
        self.rebnconvin = REBNCONV(inChannels, outChannels)

        self.stage1 = REBNCONV(outChannels, midChannels)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = REBNCONV(midChannels, midChannels)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = REBNCONV(midChannels, midChannels)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = REBNCONV(midChannels, midChannels)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = REBNCONV(midChannels, midChannels)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = REBNCONV(midChannels, midChannels)

        self.stage7 = REBNCONV(midChannels, midChannels, dilation=2)

        self.stage6d = REBNCONV(midChannels * 2, midChannels)
        self.stage5d = REBNCONV(midChannels * 2, midChannels)
        self.stage4d = REBNCONV(midChannels * 2, midChannels)
        self.stage3d = REBNCONV(midChannels * 2, midChannels)
        self.stage2d = REBNCONV(midChannels * 2, midChannels)
        self.stage1d = REBNCONV(midChannels * 2, outChannels)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx = hxin

        hx1 = self.stage1(hx)
        hx = self.pool1(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool2(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool3(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool4(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool5(hx5)

        hx6 = self.stage6(hx)

        hx7 = self.stage7(hx6)

        hx6d = self.stage6d(torch.cat((hx7, hx6), dim=1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode="bilinear", align_corners=False)

        hx5d = self.stage5d(torch.cat((hx6dup, hx5), dim=1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=False)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=False)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=False)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=False)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, inChannels, outChannels, midChannels):
        super().__init__()
        self.rebnconvin = REBNCONV(inChannels, outChannels)

        self.stage1 = REBNCONV(outChannels, midChannels)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = REBNCONV(midChannels, midChannels)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = REBNCONV(midChannels, midChannels)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = REBNCONV(midChannels, midChannels)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = REBNCONV(midChannels, midChannels)

        self.stage6 = REBNCONV(midChannels, midChannels, dilation=2)

        self.stage5d = REBNCONV(midChannels * 2, midChannels)
        self.stage4d = REBNCONV(midChannels * 2, midChannels)
        self.stage3d = REBNCONV(midChannels * 2, midChannels)
        self.stage2d = REBNCONV(midChannels * 2, midChannels)
        self.stage1d = REBNCONV(midChannels * 2, outChannels)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx = hxin

        hx1 = self.stage1(hx)
        hx = self.pool1(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool2(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool3(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool4(hx4)

        hx5 = self.stage5(hx)

        hx6 = self.stage6(hx5)

        hx5d = self.stage5d(torch.cat((hx6, hx5), dim=1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=False)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=False)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=False)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=False)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, inChannels, outChannels, midChannels):
        super().__init__()
        self.rebnconvin = REBNCONV(inChannels, outChannels)

        self.stage1 = REBNCONV(outChannels, midChannels)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = REBNCONV(midChannels, midChannels)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = REBNCONV(midChannels, midChannels)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = REBNCONV(midChannels, midChannels)

        self.stage5 = REBNCONV(midChannels, midChannels, dilation=2)

        self.stage4d = REBNCONV(midChannels * 2, midChannels)
        self.stage3d = REBNCONV(midChannels * 2, midChannels)
        self.stage2d = REBNCONV(midChannels * 2, midChannels)
        self.stage1d = REBNCONV(midChannels * 2, outChannels)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx = hxin

        hx1 = self.stage1(hx)
        hx = self.pool1(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool2(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool3(hx3)

        hx4 = self.stage4(hx)

        hx5 = self.stage5(hx4)

        hx4d = self.stage4d(torch.cat((hx5, hx4), dim=1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=False)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=False)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=False)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, inChannels, outChannels, midChannels):
        super().__init__()
        self.rebnconvin = REBNCONV(inChannels, outChannels)

        self.stage1 = REBNCONV(outChannels, midChannels)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = REBNCONV(midChannels, midChannels)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = REBNCONV(midChannels, midChannels)

        self.stage4 = REBNCONV(midChannels, midChannels, dilation=2)

        self.stage3d = REBNCONV(midChannels * 2, midChannels)
        self.stage2d = REBNCONV(midChannels * 2, midChannels)
        self.stage1d = REBNCONV(midChannels * 2, outChannels)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx = hxin

        hx1 = self.stage1(hx)
        hx = self.pool1(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool2(hx2)

        hx3 = self.stage3(hx)

        hx4 = self.stage4(hx3)

        hx3d = self.stage3d(torch.cat((hx4, hx3), dim=1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=False)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=False)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, inChannels, outChannels, midChannels):
        super().__init__()
        self.rebnconvin = REBNCONV(inChannels, outChannels)

        self.stage1 = REBNCONV(outChannels, midChannels, dilation=1)
        self.stage2 = REBNCONV(midChannels, midChannels, dilation=2)
        self.stage3 = REBNCONV(midChannels, midChannels, dilation=4)
        self.stage4 = REBNCONV(midChannels, midChannels, dilation=8)

        self.stage3d = REBNCONV(midChannels * 2, midChannels, dilation=4)
        self.stage2d = REBNCONV(midChannels * 2, midChannels, dilation=2)
        self.stage1d = REBNCONV(midChannels * 2, outChannels, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx = hxin

        hx1 = self.stage1(hx)
        hx2 = self.stage2(hx1)
        hx3 = self.stage3(hx2)
        hx4 = self.stage4(hx3)

        hx3d = self.stage3d(torch.cat((hx4, hx3), dim=1))
        hx2d = self.stage2d(torch.cat((hx3d, hx2), dim=1))
        hx1d = self.stage1d(torch.cat((hx2d, hx1), dim=1))

        return hx1d + hxin


class U2Net(nn.Module):
    def __init__(self, inChannels=3, outChannels=1):
        super().__init__()

        self.in_channels = inChannels
        self.out_channels = outChannels

        self.stage1 = RSU7(inChannels, 64, 32)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 128, 32)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 256, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 512, 128)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 512, 256)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 512, 256)

        self.stage5d = RSU4F(1024, 512, 256)
        self.stage4d = RSU4(1024, 256, 128)
        self.stage3d = RSU5(512, 128, 64)
        self.stage2d = RSU6(256, 64, 32)
        self.stage1d = RSU7(128, 64, 32)

        self.side1 = nn.Conv2d(64, outChannels, 3, padding=1)
        self.side2 = nn.Conv2d(64, outChannels, 3, padding=1)
        self.side3 = nn.Conv2d(128, outChannels, 3, padding=1)
        self.side4 = nn.Conv2d(256, outChannels, 3, padding=1)
        self.side5 = nn.Conv2d(512, outChannels, 3, padding=1)
        self.side6 = nn.Conv2d(512, outChannels, 3, padding=1)

        self.outconv = nn.Conv2d(outChannels * 6, outChannels, 1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode="bilinear", align_corners=False)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), dim=1))

        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=False)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), dim=1))

        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=False)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), dim=1))

        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=False)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), dim=1))

        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=False)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), dim=1))

        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)

        d1 = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)
        d2 = F.interpolate(d2, size=x.shape[2:], mode="bilinear", align_corners=False)
        d3 = F.interpolate(d3, size=x.shape[2:], mode="bilinear", align_corners=False)
        d4 = F.interpolate(d4, size=x.shape[2:], mode="bilinear", align_corners=False)
        d5 = F.interpolate(d5, size=x.shape[2:], mode="bilinear", align_corners=False)
        d6 = F.interpolate(d6, size=x.shape[2:], mode="bilinear", align_corners=False)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))

        return [d0, d1, d2, d3, d4, d5, d6]

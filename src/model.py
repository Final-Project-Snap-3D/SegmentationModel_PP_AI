import torch
import torch.nn as nn

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

        map_binary = torch.sigmoid(map)

        # return the segmentation map
        return map_binary

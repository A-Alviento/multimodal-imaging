# U-Net is a convolutional neural network architecture that's predominantly
# used for biomedical image segmentation. Its name derives from its U-shaped structure.

# The architecture consists of an encoder (contracting path) and a decoder (expanding path).
# During encoding, the spatial dimensions of the input image are reduced while retaining
# crucial information. The decoder then recovers the spatial dimensions from the encoded
# image to match the original input's dimensions, and in the process, classifies each pixel.

from . import config
from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

# The `Block` class is a fundamental building block of our U-Net model.
# It contains two sequential convolutional layers, followed by a ReLU activation function.
class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # Define two convolutional layers. 
        # These layers will extract features and perform spatial transformations.
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()  # Activation function introduces non-linearity.
        self.conv2 = Conv2d(outChannels, outChannels, 3)

    def forward(self, x):
        # Input 'x' is passed through two convolutions with a ReLU activation in between.
        return self.conv2(self.relu(self.conv1(x)))


# Encoder is responsible for the "contracting" part of U-Net. 
# It progressively downsamples the image, reducing its spatial dimensions 
# and thereby compacting its information.
class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        # Define a sequence of blocks (convolutions). Each block extracts features 
        # from the image and reduces its spatial dimensions.
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)  # Downsample the spatial dimensions by a factor of 2.

    def forward(self, x):
        # List to store outputs of each block (useful for the decoder later).
        blockOutputs = []
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)  # Downsample (reduce size).
        return blockOutputs
    

# Decoder represents the "expanding" part of U-Net.
# It aims to recover the spatial dimensions and information lost during encoding.
class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # Define upsampling layers to increase the spatial dimensions of the image.
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
        # Decoder blocks further refine the upsampled image.
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            # Crop and concatenate feature maps from the encoder. This combination 
            # provides localization cues from the encoder to the decoder.
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    # Utility function to crop encoder features to match the size of the current decoder output.
    # Necessary because the encoder and decoder outputs might not have the same spatial dimensions due to 
    # convolutional and pooling operations.
    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        return encFeatures


# The U-Net model combines the encoder and decoder, along with a final convolution
# that maps the output to the desired number of classes.
class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16), nbClasses=1, retainDim=True, outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        # Define the encoder and decoder.
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # Final layer maps the decoder output to the desired number of output classes.
        # In the context of image segmentation, each class represents a type of object/region in the image.
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim  # Whether to resize output to original dimensions.
        self.outSize = outSize
    
    def forward(self, x):
        # Get intermediate feature maps from the encoder.
        encFeatures = self.encoder(x)
        # Decoder uses these features to recover spatial information.
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
        # Produce final segmentation map.
        map = self.head(decFeatures)
        # Resize output if required.
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        return map

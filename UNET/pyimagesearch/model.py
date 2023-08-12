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
# inChannels: Refers to the depth or number of channels in the incoming data. For instance, if you're passing an RGB image, inChannels will be 3. If you're passing the output of another convolutional layer as input, then inChannels would be the number of filters used in that previous layer.
# outChannels: Refers to the number of filters we want to use in the convolutional layer. This will also be the number of channels in the output produced by this layer. For example, if outChannels is 64, this convolutional layer will use 64 filters and produce an output with a depth of 64 channels.
class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # Define two convolutional layers. 
        # These layers will extract features and perform spatial transformations.
        self.conv1 = Conv2d(inChannels, outChannels, 3) # 3 refers to the filter size, so in this case, it's a 3x3 filter
        self.relu = ReLU()  # Activation function introduces non-linearity.
        self.conv2 = Conv2d(outChannels, outChannels, 3)

    def forward(self, x):
        # Input 'x' is passed through two convolutions with a ReLU activation in between.
        return self.conv2(self.relu(self.conv1(x)))


# The Encoder acts like a funnel that progressively "squeezes" the spatial details 
# of an image while capturing its essential features. It's the first half of our U-Net.
class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        
        # We create multiple blocks (pairs of convolutional layers) that will 
        # process the image. Each block increases the depth (number of channels) 
        # of the image while keeping its spatial dimensions the same.
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        
        # After processing the image with a block, we downsample it (reduce its 
        # width and height by half) using Max Pooling. This step compresses the 
        # spatial information but retains important features.
        self.pool = MaxPool2d(2)

    def forward(self, x):
        # We'll keep track of the output after each block because the Decoder 
        # will need these details to "reconstruct" the full details of the original image.
        blockOutputs = []

        for block in self.encBlocks:
            x = block(x)            # Process the image with a block.
            blockOutputs.append(x)  # Save the output.
            x = self.pool(x)        # Downsample the image.
        
        # By the end of the Encoder, we have a series of compressed representations 
        # of the original image, each with increasing depth and decreasing spatial dimensions.
        return blockOutputs

    

# The Decoder takes the compact representation from the Encoder and tries to "blow it up"
# or upscale it to its original dimensions. While doing so, it uses saved features from 
# the Encoder to ensure the upscaled image is detailed and accurate.
class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        
        # We define layers that will upscale or "expand" the compact image.
        # They essentially increase the height and width of the image.
        # arguments: input channel, output channel, kernel size, kernel stride
        # stride of 2 means the spatial dimentsions will double after the operation
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
        
        # After each upsampling, the image might be a bit rough around the edges.
        # These blocks (sets of convolutional layers) will refine the image after each upsampling.
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            # Upscale the image.
            x = self.upconvs[i](x)
            
            # Take corresponding features from the Encoder, trim them to match the size,
            # and add them to our upscaled image. This step ensures that our expanded image 
            # has the detailed information it needs.
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            
            # Refine the combined image.
            x = self.dec_blocks[i](x)
        
        # By the end of the Decoder, we've reconstructed an image that's close in detail 
        # to the original, using the compact representation and saved features.
        return x

    # Given two feature maps, this function trims the first one to match the dimensions
    # of the second. It's a utility function to ensure that we can combine features 
    # from the Encoder and Decoder seamlessly.
    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        return encFeatures



class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16), nbClasses=1, retainDim=True, outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
        super().__init__()

        # Define the encoder.
        # The encoder compresses the input image, capturing its major features but at a reduced size.
        self.encoder = Encoder(encChannels)
        
        # Define the decoder.
        # The decoder expands the compressed representation, trying to restore lost spatial details using cues from the encoder.
        self.decoder = Decoder(decChannels)

        # Final convolutional layer.
        # This layer produces the segmentation map, where the depth corresponds to the number of classes.
        # Each pixel in this map provides a score for each class.
        # decChannels[-1]: This is the number of input channels (or depth) the convolutional layer expects.It's the last element in the decChannels tuple, which means it's the number of channels (or depth) produced by the final layer of the Decoder. Essentially, the input to this Conv2d layer will have this depth.
        # nbClasses: This is the number of output channels (or depth) the convolutional layer will produce. Each output channel corresponds to a class in the segmentation task. The layer will produce a score for each class at every spatial location (pixel) of the image. If you're doing binary segmentation (distinguishing between just two regions, e.g., object vs background), nbClasses would be 1. If you're distinguishing among multiple regions (multi-class segmentation), nbClasses would be the number of regions or classes you're trying to distinguish.
        # 1: This is the size of the kernel (filter) that the convolutional layer will use. A value of 1 means the layer will use 1x1 kernels. Such small kernels are often used in neural networks to reduce the number of channels (depth) or to act as a per-pixel classifier, as is the case here. 1x1 convolutions are a way to apply a dense (or fully connected) layer to every individual pixel location. In this context, it's turning the feature information at every pixel location into class scores.
        self.head = Conv2d(decChannels[-1], nbClasses, 1)

        # A flag to check whether the output segmentation map should match the original image's size.
        self.retainDim = retainDim  

        # Desired output size if 'retainDim' is True.
        self.outSize = outSize
    
    def forward(self, x):
        # Pass the input image through the encoder to get compressed feature maps.
        encFeatures = self.encoder(x)

        # Pass the compressed representation through the decoder.
        # Use the feature maps from the encoder to help restore spatial information.
        # encFeatures[::-1]: This reverses the list of encoder feature maps. The U-Net architecture is symmetric. The deepest (most compressed) feature map from the encoder is directly passed into the decoder. Then, as the decoder upsamples the image, it uses feature maps from the encoder at corresponding levels of detail to help refine its output. So, we reverse the list of feature maps to start with the deepest one.
        # encFeatures[::-1][0]: This accesses the first item of the reversed list, which is the deepest encoder feature map. This serves as the initial input to the decoder, providing the most abstract, compressed representation of the original image.
        # encFeatures[::-1][1:]: This accesses the rest of the items in the reversed list, excluding the first one. These are the feature maps from the encoder that will be successively concatenated with the upsampled feature maps in the decoder. This process aids the decoder in restoring spatial details lost during downsampling by the encoder.
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])

        # Produce the final segmentation map from the decoded features.
        # This map assigns a class score to each pixel.
        map = self.head(decFeatures)

        # If required, resize the output to the original input dimensions.
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
            
        return map

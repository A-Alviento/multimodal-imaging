# USAGE
# python predict_serialise.py
# import the necessary packages
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pyimagesearch import config
from pyimagesearch.model import UNet


# make predictions on images in imagePath and save them to outputDirectory
def make_predictions(model, imagePath, outputDirectory):
    # set model to evaluation mode
    model.eval()
    # if output directory does not exist, create it
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # turn off gradient tracking
    with torch.no_grad():
        image = cv2.imread(imagePath) # load the image from disk
        if image is None:
            print(f"Error loading image at {imagePath}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # swap its color channels
        image = image.astype("float32") / 255.0 # cast it to float data type and scale its pixel values
        image = cv2.resize(image, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)) # resize the image

        filename = os.path.basename(imagePath) # find the filename
        filename_no_ext = os.path.splitext(filename)[0] # find the filename without extension
        output_path = os.path.join(outputDirectory, f"{filename_no_ext}.png") # generate the path to the output mask

        image = np.transpose(image, (2, 0, 1)) # make the channel axis to be the leading one
        image = np.expand_dims(image, 0) # add a batch dimension
        image = torch.from_numpy(image).to(config.DEVICE) # create a PyTorch tensor and flash it to the current device

        predMask = model(image).squeeze() # make the prediction
        predMask = torch.sigmoid(predMask) # pass the results through the sigmoid function
        predMask = predMask.cpu().numpy() # convert the result to a NumPy array
        predMask = (predMask > config.THRESHOLD) * 255 # filter out the weak predictions and convert them to integers
        predMask = predMask.astype(np.uint8) # convert the mask to unsigned 8-bit integers

        cv2.imwrite(output_path, predMask)  # Save the predicted mask directly


# load our model from disk and flash it to the current device
unet = UNet().to(config.DEVICE)
unet.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))

inputDirectory = config.TEST_IMAGE_DATASET_PATH
outputDirectory = config.TEST_PREDICT_DATASET_PATH
if not os.path.exists(outputDirectory):
    raise Exception(f"Output directory {outputDirectory} does not exist")

# iterate over the images in the input directory
for filename in os.listdir(inputDirectory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add any other image formats here
        imagePath = os.path.join(inputDirectory, filename)
        make_predictions(unet, imagePath, outputDirectory)

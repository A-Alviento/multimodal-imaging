import torch
import os

# base path of dataset
DATASET_PATH = "../dataset_mr"

# define the path to the training set
IMAGE_DATASET_PATH = os.path.sep.join([DATASET_PATH, "train", "images"])
MASK_DATASET_PATH = os.path.sep.join([DATASET_PATH, "train", "mask"])

# define the path to the test set
TEST_IMAGE_DATASET_PATH = os.path.sep.join([DATASET_PATH, "test", "images"])
TEST_MASK_DATASET_PATH = os.path.sep.join([DATASET_PATH, "test", "mask"])
TEST_PREDICT_DATASET_PATH = os.path.join(DATASET_PATH, "test", "predict_unet")



# determine the device to use for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "mps" 

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in input, number of classes, and number of levels in U-Net
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 16
PATIENCE = 50

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

# define the threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"


# define the path to the output serialised model, model training plot, and testing image paths
LAST_MODEL_PATH = os.path.join(BASE_OUTPUT, "last.pth")
BEST_MODEL_PATH =os.path.join(BASE_OUTPUT, "best.pth")
# PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
HISTORY_PATH = os.path.sep.join([BASE_OUTPUT, "history.pkl"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
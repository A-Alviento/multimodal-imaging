# USAGE
# The following script is used to train the U-Net model for image segmentation.

# Import necessary libraries and modules.
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.eval import dice_score
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import pickle

# List all image paths for training, validation, and testing.
trainImages = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
trainMasks = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
valImages = sorted(list(paths.list_images(config.VAL_IMAGE_DATASET_PATH)))
valMasks = sorted(list(paths.list_images(config.VAL_MASK_DATASET_PATH)))
testImages = sorted(list(paths.list_images(config.TEST_IMAGE_DATASET_PATH)))
testMasks = sorted(list(paths.list_images(config.TEST_MASK_DATASET_PATH)))

# Save test image paths for future evaluation purposes.
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

# Define data transformations:
# 1. Convert to a PIL Image.
# 2. Resize to the desired input dimensions.
# 3. Convert to a Tensor for PyTorch processing.
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()
])

# Create datasets for training and validation using the paths and transformations.
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
testDS = SegmentationDataset(imagePaths=valImages, maskPaths=valMasks, transforms=transforms)

print(f"[INFO] found {len(trainDS)} examples in training set...")
print(f"[INFO] found {len(testDS)} examples in testing set...")

# Create data loaders for efficient batching during training.
trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=0)
testLoader = DataLoader(testDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=0)

# Initialize the U-Net model and send it to the appropriate computing device (CPU or GPU).
unet = UNet().to(config.DEVICE)

# Define the loss function and optimizer for training.
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# Calculate the number of steps (batches) for each epoch.
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

# Dictionary to keep track of training and validation losses for each epoch.
H = {"train_loss": [], "test_loss": [], "test_dice": []}

# Start the training process.
print ("[INFO] training the network...")
startTime = time.time()

best_dice_score = 0

# variable to keep track of the number of epochs without improvement in Dice score
epochs_without_improvement = 0

# loop over the epochs
for e in tqdm(range(config.NUM_EPOCHS)): # tqdm is a tool to show progress bar in console
    # Model is set to training mode. This affects certain layers like dropout.
    unet.train()

    # Variables to accumulate losses over batches.
    totalTrainLoss = 0
    totalTestLoss = 0

    # Loop through each batch in the training dataset.
    for (i, (x, y)) in enumerate(trainLoader):
        # Move data to the computing device.
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        # Make predictions, compute loss, and backpropagate errors.
        pred = unet(x)
        loss = lossFunc(pred, y)
        # opt.zero_grad(): Before performing backpropagation, you need to zero out any previously computed gradients. Why? Because PyTorch accumulates gradients. That means, whenever you call .backward(), the gradients are added to any previously existing gradients rather than replacing them. If you don't zero them out, gradients from previous iterations will interfere with the current iteration, leading to incorrect updates.
        opt.zero_grad()
        # loss.backward(): This command computes the gradient of the loss with respect to each parameter of the model (i.e., it performs backpropagation). Essentially, it determines how much each parameter (weight/bias in the neural network) contributed to the error in the output.
        loss.backward()
        # opt.step(): This updates the model's parameters (weights and biases) using the computed gradients. The specific way the parameters are updated depends on the optimization algorithm. In the provided code, the optimizer is Adam. So, opt.step() will perform an Adam optimization step, adjusting the weights to reduce the error based on the gradients.
        opt.step()

        # Accumulate batch loss.
        totalTrainLoss += loss

    totalDiceScore = 0
    # Evaluate model performance on validation dataset.
    with torch.no_grad():
        # Model is set to evaluation mode.
        unet.eval()

        # Loop through batches in the validation dataset.
        for (x, y) in testLoader:
            # Move data to the computing device.
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # Make predictions and compute loss.
            pred = unet(x)
            totalTestLoss += lossFunc(pred, y)
            totalDiceScore += dice_score(pred, y)
        
    # Calculate average losses for this epoch.
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps
    avgDiceScore = totalDiceScore / testSteps

    # Store epoch losses for later visualization.
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
    H["test_dice"].append(avgDiceScore)

    # print the model training, validation information and dice score
    print("[INFO] EPOCH: {}/{}".format(e+1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}, Test Dice Score: {:.4f}".format(avgTrainLoss, avgTestLoss, avgDiceScore))

    # Save the latest model weights after every epoch.
    torch.save(unet.state_dict(), config.LAST_MODEL_PATH)

    # if this epoch yields the best dice score, save these model weights
    if avgDiceScore > best_dice_score:
        best_dice_score = avgDiceScore
        torch.save(unet.state_dict(), config.BEST_MODEL_PATH)
        epochs_without_improvement = 0 # reset counter
    else:
        epochs_without_improvement += 1

    print("Num epochs w/o improvement: " + epochs_without_improvement)
    print("Current best dice score: " + best_dice_score)

    # Display the total training time.
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f} seconds".format(endTime - startTime))

    # check for early stopping
    if epochs_without_improvement == config.PATIENCE:
        print("[INFO] Early stopping: No improvement in Dice score for the last {} epochs".format(config.PATIENCE))
        break

# # Plot the training and validation losses.
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(H["train_loss"], label="train_loss")
# plt.plot(H["test_loss"], label="test_loss")
# plt.title("Training Loss on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss")
# plt.legend(loc="lower left")
# plt.savefig(config.PLOT_PATH)

# After the training loop
with open(config.HISTORY_PATH, 'wb') as file:
    pickle.dump(H, file)
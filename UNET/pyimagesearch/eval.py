import torch
from pyimagesearch import config

def dice_score(pred, target, threshold=config.THRESHOLD, eps=1e-6):
    pred = (pred > threshold).float() # Threshold the prediction
    target = target.float() # Ensure that target is a float tensor
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 2.0 * intersection / (union + eps) # Add a small epsilon to avoid division by zero

def dice_loss(pred, target, smooth=1.):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

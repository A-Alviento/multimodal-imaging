import torch
from pyimagesearch import config
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

def dice_score(pred, target, threshold=config.THRESHOLD, eps=1e-6):
    pred = (pred > threshold).float() # Threshold the prediction
    target = target.float() # Ensure that target is a float tensor
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 2.0 * intersection / (union + eps) # Add a small epsilon to avoid division by zero

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        dice_loss = 1 - (2 * torch.sum(pred * target) + 1) / (torch.sum(pred) + torch.sum(target) + 1)
        bce_loss = self.bce(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

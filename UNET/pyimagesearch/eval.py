def dice_score(pred, target, eps=1e-7):
    """
    Compute the Dice coefficient given predicted and target masks.
    """
    # Convert predictions to binary using a threshold of 0.5
    pred = torch.sigmoid(pred) >= 0.5
    intersection = (pred & target).float().sum()
    union = pred.float().sum() + target.float().sum()
    dice = (2 * intersection) / (union + eps)
    return dice.item()

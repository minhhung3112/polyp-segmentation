from loss import register_loss
import torch
import torch.nn.functional as F

@register_loss('dice')
def dice_loss(pred, mask):
    pred_s = torch.sigmoid(pred)
    inter = (pred_s * mask).sum(dim=(2,3))
    union = pred_s.sum(dim=(2,3)) + mask.sum(dim=(2,3))
    dice = 1 - (2*inter + 1) / (union + 1)
    return dice.mean()

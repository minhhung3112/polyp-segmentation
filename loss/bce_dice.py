from loss import register_loss
import torch
import torch.nn.functional as F

@register_loss('bce_dice')
def bce_dice_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask)
    pred_s = torch.sigmoid(pred)
    inter = (pred_s * mask).sum(dim=(2,3))
    union = pred_s.sum(dim=(2,3)) + mask.sum(dim=(2,3))
    dice = 1 - (2*inter + 1) / (union + 1)
    return (bce + dice).mean()

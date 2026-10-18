"""
Focal + Dice Compound Loss.

Widely adopted in medical polyp segmentation papers (e.g., SANet, ColonSegNet):
Combines pixel-level hard example mining (Focal Loss) with global region overlap (Dice Loss):
    L = L_Focal + L_Dice
"""

from loss import register_loss
import torch
import torch.nn.functional as F


@register_loss('focal_dice')
@register_loss('focal_dice_loss')
def focal_dice_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Combined Focal and Dice Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Focal loss alpha factor.
        gamma: Focal loss gamma exponent.
        dice_weight: Weight multiplier for Dice loss term.
        focal_weight: Weight multiplier for Focal loss term.
        smooth: Smoothing constant for Dice.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    # 1. Focal Loss term
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    prob = torch.sigmoid(pred)
    p_t = prob * mask + (1.0 - prob) * (1.0 - mask)
    alpha_t = alpha * mask + (1.0 - alpha) * (1.0 - mask)
    focal = (alpha_t * torch.pow((1.0 - p_t).clamp(min=0.0, max=1.0), gamma) * bce).mean()

    # 2. Dice Loss term
    b = pred.size(0)
    p_flat = prob.view(b, -1)
    g_flat = mask.view(b, -1)
    inter = (p_flat * g_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + g_flat.sum(dim=1)
    dice = (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()

    return focal_weight * focal + dice_weight * dice


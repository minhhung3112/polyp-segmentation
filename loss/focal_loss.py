"""
Focal Loss for Dense Segmentation.

Reference:
    Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection.
    IEEE International Conference on Computer Vision (ICCV 2017), pp. 2980-2988.

Down-weights easy background colon pixels and focuses gradients on hard, ambiguous polyp regions:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""

from loss import register_loss
import torch
import torch.nn.functional as F


@register_loss('focal')
@register_loss('focal_loss')
def focal_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Binary Focal Loss computed directly from logits for numerical stability.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Weighting factor for foreground class (default: 0.25).
        gamma: Focusing parameter for modulating hard vs easy examples (default: 2.0).

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    prob = torch.sigmoid(pred)
    p_t = prob * mask + (1.0 - prob) * (1.0 - mask)
    alpha_t = alpha * mask + (1.0 - alpha) * (1.0 - mask)
    focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=0.0, max=1.0), gamma)

    return (focal_weight * bce).mean()


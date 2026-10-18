"""
IoU (Jaccard) Loss for Binary Segmentation.

Directly optimizes the Intersection-over-Union (Jaccard Index) metric:
    L_IoU = 1 - (Intersection + eps) / (Union + eps)
"""

from loss import register_loss
import torch


@register_loss('iou')
@register_loss('iou_loss')
@register_loss('jaccard')
def iou_loss(pred: torch.Tensor, mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Jaccard / IoU Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        smooth: Smoothing laplace factor to prevent division by zero.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred)
    b = pred.size(0)

    prob_flat = prob.view(b, -1)
    mask_flat = mask.view(b, -1)

    intersection = (prob_flat * mask_flat).sum(dim=1)
    total = prob_flat.sum(dim=1) + mask_flat.sum(dim=1)
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return (1.0 - iou).mean()


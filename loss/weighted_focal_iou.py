"""
Weighted Focal + Weighted IoU Loss (ColonFormer Structure Loss).

Reference:
    Duc, N. T., Oanh, N. T., Thao, N. T. P., Triet, N. T., & Ly, V. S. (2021).
    ColonFormer: An Efficient Transformer Based Method for Polyp Segmentation.
    British Machine Vision Conference (BMVC 2021).
    https://github.com/D-X-Y/ColonFormer

A variant of PraNet's structure loss that replaces the weighted BCE term with a
weighted Focal Loss term to further enhance attention on hard polyp pixels:
    L = wFocal + wIoU
"""

from loss import register_loss
import torch
import torch.nn.functional as F


@register_loss('weighted_focal_iou')
@register_loss('colonformer_loss')
def weighted_focal_iou_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Weighted Focal + Weighted IoU Loss (ColonFormer BMVC 2021).

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Focal loss alpha balance factor (default: 0.25).
        gamma: Focal loss gamma focusing parameter (default: 2.0).

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    # Boundary-aware distance weighting map (PraNet / ColonFormer formulation)
    weit = 1.0 + 5.0 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # Weighted Focal Loss term
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    prob = torch.sigmoid(pred)
    p_t = torch.where(mask == 1.0, prob, 1.0 - prob)
    alpha_t = torch.where(mask == 1.0, alpha, 1.0 - alpha)
    focal = alpha_t * torch.pow((1.0 - p_t).clamp(min=0.0, max=1.0), gamma) * bce
    wfocal = (focal * weit).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)

    # Weighted IoU Loss term
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

    return (wfocal + wiou).mean()

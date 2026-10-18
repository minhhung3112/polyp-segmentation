"""
Focal Tversky Loss for Lesion and Polyp Segmentation.

Reference:
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation.
    IEEE International Symposium on Biomedical Imaging (ISBI 2019), pp. 284-287.

Applies a focal exponent gamma to the Tversky Index to suppress easy examples and
boost gradients on hard, small, and camouflaged polyps:
    FTL = (1 - TI)^gamma
"""

from loss import register_loss
import torch


@register_loss('focal_tversky')
@register_loss('focal_tversky_loss')
def focal_tversky_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Focal Tversky Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Weight of False Positives (default: 0.3).
        beta: Weight of False Negatives (default: 0.7).
        gamma: Focal exponent (default: 0.75; 0.75-1.33 is typical in medical imaging).
        smooth: Smoothing constant.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred)
    b = pred.size(0)

    p_flat = prob.view(b, -1)
    g_flat = mask.view(b, -1)

    true_pos = (p_flat * g_flat).sum(dim=1)
    false_pos = (p_flat * (1.0 - g_flat)).sum(dim=1)
    false_neg = ((1.0 - p_flat) * g_flat).sum(dim=1)

    ti = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    ftl = torch.pow((1.0 - ti).clamp(min=1e-7, max=1.0), gamma)
    return ftl.mean()


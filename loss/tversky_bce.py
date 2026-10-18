"""
Combined Tversky + BCE Loss.

Reference:
    Kim, T., Lee, H., & Kim, D. (2021).
    UACANet: Uncertainty Augmented Context Attention Network for Medical Image Segmentation.
    ACM Multimedia (ACM MM 2021), pp. 2155-2163.
    https://github.com/plemeri/UACANet

Combines standard binary cross-entropy with asymmetric Tversky index penalty to control
both global pixel classification and lesion false negative rates:
    L = L_BCE + L_Tversky
"""

from loss import register_loss
import torch
import torch.nn.functional as F


@register_loss('tversky_bce')
@register_loss('tversky_bce_loss')
def tversky_bce_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 2.0,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Combined Tversky + BCE Loss (UACANet ACM MM 2021).

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: False positive weight in Tversky index (default: 0.5).
        beta: False negative weight in Tversky index (default: 0.5).
        gamma: Exponent on Tversky distance (default: 2.0).
        smooth: Smoothing constant.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    # 1. BCE loss term
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    # 2. Tversky loss term
    prob = torch.sigmoid(pred)
    p_flat = prob.view(-1)
    g_flat = mask.view(-1)

    tp = (p_flat * g_flat).sum()
    fp = ((1.0 - g_flat) * p_flat).sum()
    fn = (g_flat * (1.0 - p_flat)).sum()

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    l_tversky = torch.pow((1.0 - tversky).clamp(min=1e-7, max=1.0), gamma)

    return bce + l_tversky

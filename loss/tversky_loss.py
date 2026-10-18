"""
Tversky Loss for Medical Image Segmentation.

Reference:
    Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017).
    Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks.
    International Workshop on Machine Learning in Medical Imaging (MLMI 2017), pp. 379-387.

Generalization of Dice loss adding asymmetric penalties for False Positives (alpha) and
False Negatives (beta). In clinical polyp segmentation, beta > alpha places higher penalty
on missing polyps (False Negatives).
"""

from loss import register_loss
import torch


@register_loss('tversky')
@register_loss('tversky_loss')
def tversky_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Tversky Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Weight of False Positives (default: 0.3).
        beta: Weight of False Negatives (default: 0.7).
        smooth: Laplace smoothing constant.

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

    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return (1.0 - tversky_index).mean()


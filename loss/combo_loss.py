"""
Combo Loss for Medical Image Segmentation.

Reference:
    Taghanaki, S. A., Zheng, Y., Zhou, S. K., Georgescu, B., Sharma, P., Xu, D., ... & Hamarneh, G. (2019).
    Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation.
    Computerized Medical Imaging and Graphics (CMIG), 75, pp. 24-33.

Combines weighted cross-entropy with Dice loss using a convex balancing parameter alpha and
beta for controlling false positive vs false negative penalties.
"""

from loss import register_loss
import torch


@register_loss('combo')
@register_loss('combo_loss')
def combo_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1.0,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Combo Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Weight balancing BCE vs Dice (default: 0.5).
        beta: Weight balancing FP vs FN in BCE (default: 0.5).
        smooth: Smoothing constant for Dice.
        eps: Epsilon to avoid log(0).

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred).clamp(min=eps, max=1.0 - eps)
    b = pred.size(0)

    p_flat = prob.view(b, -1)
    g_flat = mask.view(b, -1)

    # 1. Weighted modified cross-entropy
    wbce = -(beta * g_flat * torch.log(p_flat) + (1.0 - beta) * (1.0 - g_flat) * torch.log(1.0 - p_flat)).mean()

    # 2. Dice loss
    inter = (p_flat * g_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + g_flat.sum(dim=1)
    dice = (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()

    return alpha * wbce + (1.0 - alpha) * dice


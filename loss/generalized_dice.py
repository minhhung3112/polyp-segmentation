"""
Generalized Dice Loss (GDL).

Reference:
    Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Cardoso, M. J. (2017).
    Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations.
    Deep Learning in Medical Image Analysis (DLMIA 2017), LNCS 10553, pp. 240-248.

Weights class contributions by the inverse of their squared volume, solving the issue
where large background regions dominate the Dice gradient over tiny polyps.
"""

from loss import register_loss
import torch


@register_loss('generalized_dice')
@register_loss('gdl')
def generalized_dice_loss(pred: torch.Tensor, mask: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    Generalized Dice Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred)
    b = pred.size(0)

    # 2 classes: foreground and background
    p_fg = prob.view(b, -1)
    g_fg = mask.view(b, -1)

    p_bg = 1.0 - p_fg
    g_bg = 1.0 - g_fg

    # Inverse squared volume weights
    w_fg = 1.0 / (torch.sum(g_fg, dim=1).pow(2) + smooth)
    w_bg = 1.0 / (torch.sum(g_bg, dim=1).pow(2) + smooth)

    numerator = 2.0 * (w_fg * torch.sum(p_fg * g_fg, dim=1) + w_bg * torch.sum(p_bg * g_bg, dim=1))
    denominator = (
        w_fg * torch.sum(p_fg + g_fg, dim=1) +
        w_bg * torch.sum(p_bg + g_bg, dim=1) +
        smooth
    )

    gdl = 1.0 - (numerator + smooth) / denominator
    return gdl.mean()


"""
Binary Cross-Entropy Loss with Logits.

Standard baseline loss for binary semantic segmentation.
Computes cross-entropy directly on raw model logits for numerical stability.
"""

from loss import register_loss
import torch
import torch.nn.functional as F


@register_loss('bce')
@register_loss('binary_cross_entropy')
def bce_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Standard Binary Cross-Entropy Loss with Logits.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    return F.binary_cross_entropy_with_logits(pred, mask.float())


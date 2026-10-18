"""
Approximate Hausdorff Distance Loss.

Reference:
    Karimi, D., & Salcudean, S. E. (2019).
    Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks.
    IEEE Transactions on Medical Imaging (TMI), 39(2), pp. 499-513.

Provides a differentiable surrogate for the Hausdorff Distance based on distance transforms,
directly penalizing spatial outliers and large contour discrepancies.
"""

from loss import register_loss
import torch
import numpy as np
import scipy.ndimage


def _compute_distance_weight(mask_np: np.ndarray) -> np.ndarray:
    """Compute distance transform from boundaries."""
    pos = mask_np.astype(bool)
    if not pos.any():
        return np.ones_like(mask_np, dtype=np.float32)

    # Boundary contour
    struct = scipy.ndimage.generate_binary_structure(2, 1)
    border = pos ^ scipy.ndimage.binary_erosion(pos, structure=struct)
    dist = scipy.ndimage.distance_transform_edt(~border)

    # Normalize
    max_d = max(dist.max(), 1.0)
    return (dist / max_d).astype(np.float32)


@register_loss('hausdorff')
@register_loss('hausdorff_loss')
@register_loss('hd_loss')
def hausdorff_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Combined Dice + Approximate Hausdorff Distance Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Weight for Hausdorff distance surrogate term (default: 0.5).
        smooth: Smoothing factor for Dice loss.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred)
    b = pred.size(0)

    # 1. Dice loss term
    p_flat = prob.view(b, -1)
    g_flat = mask.view(b, -1)
    inter = (p_flat * g_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + g_flat.sum(dim=1)
    l_dice = (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()

    # 2. Distance-weighted error map (Hausdorff surrogate)
    mask_np = mask.detach().cpu().numpy()
    dist_maps = []
    for i in range(b):
        m_2d = mask_np[i].squeeze()
        dist_maps.append(_compute_distance_weight(m_2d))

    dist_tensor = torch.from_numpy(np.stack(dist_maps)).to(pred.device).view_as(prob)
    error = torch.abs(prob - mask)
    l_hd = (error * (dist_tensor ** 2)).mean()

    return l_dice + alpha * l_hd


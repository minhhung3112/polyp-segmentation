"""
Boundary Loss / Surface Loss for Medical Image Segmentation.

Reference:
    Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., Dolz, J., & Ben Ayed, I. (2019).
    Boundary Loss for Highly Unbalanced Segmentation.
    Medical Imaging with Deep Learning (MIDL 2019), PMLR 102, pp. 285-296.

Computes the signed distance map (SDM) of ground truth contours and evaluates distance-weighted
boundary alignment to guide the prediction towards sharp, precise polyp edges:
    L = L_Dice + alpha * L_Boundary
"""

from loss import register_loss
import torch
import numpy as np
import scipy.ndimage


def _compute_sdf(mask_np: np.ndarray) -> np.ndarray:
    """Compute normalized signed distance map from a binary ground truth mask."""
    posmask = mask_np.astype(bool)
    if not posmask.any():
        return np.ones_like(mask_np, dtype=np.float32)
    negmask = ~posmask
    posdis = scipy.ndimage.distance_transform_edt(posmask)
    negdis = scipy.ndimage.distance_transform_edt(negmask)
    sdf = negdis - posdis
    # Normalize to [-1, 1] range
    max_d = max(np.abs(sdf).max(), 1.0)
    return (sdf / max_d).astype(np.float32)


@register_loss('boundary')
@register_loss('boundary_loss')
@register_loss('surface_loss')
def boundary_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Combined Dice + Boundary Surface Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Weight multiplier for the boundary distance term (default: 0.5).
        smooth: Smoothing factor for Dice loss.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred)
    b = pred.size(0)

    # 1. Dice loss component
    p_flat = prob.view(b, -1)
    g_flat = mask.view(b, -1)
    inter = (p_flat * g_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + g_flat.sum(dim=1)
    l_dice = (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()

    # 2. Boundary distance component
    mask_np = mask.detach().cpu().numpy()
    sdf_maps = []
    for i in range(b):
        m_2d = mask_np[i].squeeze()
        sdf_maps.append(_compute_sdf(m_2d))

    sdf_tensor = torch.from_numpy(np.stack(sdf_maps)).to(pred.device).view_as(prob)
    l_boundary = (prob * sdf_tensor).mean()

    return l_dice + alpha * l_boundary


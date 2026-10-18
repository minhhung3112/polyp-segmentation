"""
Edge-Aware / Contour Loss for Polyp Segmentation.

Reference:
    Fan, D. P., Ji, G. P., Zhou, T., Chen, G., Fu, H., Shen, J., & Shao, L. (2020).
    PraNet: Parallel Reverse Attention Network for Polyp Segmentation.
    MICCAI 2020, pp. 263-273.

Supervises boundary gradients using a Laplacian filter to force sharp, unambiguous polyp contours:
    L = L_BCE + L_Dice + alpha * L_Edge
"""

from loss import register_loss
import torch
import torch.nn.functional as F


def _extract_edges(x: torch.Tensor) -> torch.Tensor:
    """Extract boundary gradients using a normalized Laplacian kernel."""
    laplacian_kernel = torch.tensor(
        [[-1.0, -1.0, -1.0],
         [-1.0,  8.0, -1.0],
         [-1.0, -1.0, -1.0]],
        dtype=torch.float32,
        device=x.device
    ).view(1, 1, 3, 3)

    if x.dim() == 3:
        x = x.unsqueeze(1)

    edges = F.conv2d(x, laplacian_kernel, padding=1)
    return torch.clamp(edges, 0.0, 1.0)


@register_loss('edge_loss')
@register_loss('contour_loss')
@register_loss('boundary_aware')
def edge_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    edge_weight: float = 1.0,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Edge-Aware Contour Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        edge_weight: Weight for edge/contour loss component.
        smooth: Smoothing constant.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred)
    b = pred.size(0)

    # 1. Base BCE + Dice
    bce = F.binary_cross_entropy_with_logits(pred, mask)
    p_flat = prob.view(b, -1)
    g_flat = mask.view(b, -1)
    inter = (p_flat * g_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + g_flat.sum(dim=1)
    dice = (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()

    # 2. Edge supervision term
    pred_edges = _extract_edges(prob)
    gt_edges = _extract_edges(mask)

    pe_flat = pred_edges.view(b, -1)
    ge_flat = gt_edges.view(b, -1)
    edge_inter = (pe_flat * ge_flat).sum(dim=1)
    edge_union = pe_flat.sum(dim=1) + ge_flat.sum(dim=1)
    l_edge = (1.0 - (2.0 * edge_inter + smooth) / (edge_union + smooth)).mean()

    return bce + dice + edge_weight * l_edge


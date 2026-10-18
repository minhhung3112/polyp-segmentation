"""
Area Adaptive Loss (AAL) for Polyp Segmentation.

Reference:
    Du, F., Yu, X., Zhou, S., Lin, Y., Wang, W., Xu, L., Wang, Z., Hu, C., Qian, N., & Wang, Z. (2023).
    DoubleAANet: Enhancing Polyp Segmentation with Auxiliary Attention and Area Adaptive.
    Preprints.org, doi:10.20944/preprints202309.1326.v1 (Section 3.3, Eq. 4-9).

Re-weights the per-sample DiceBCE loss inside a batch so that images with
multiple and diminutive polyps receive more attention than a single large polyp:

    DiceBCE_k = Dice_k + BCE_k                                   (Eq. 6)
    f(P, Q)   = C / ( sum_i P_i - gamma * (Q - 1) )              (Eq. 8)
    w_k       = softmax_k( f(P_k, Q_k) / T )                      (Eq. 9)
    AAL       = sum_k w_k * DiceBCE_k                             (Eq. 7)

where P_i is the pixel area of the i-th polyp (connected component) and Q is
the number of polyps in the image. Smaller total area and more polyps give a
smaller denominator, hence a larger f and a larger weight.

Notes (implementation choices not specified in the paper):
    * Areas are normalized by H*W so that C, gamma and T do not depend on resolution.
    * The denominator is clamped to ``min_denom`` to keep f finite and positive.
    * Images without any polyp (Q = 0) get the lowest possible score f = C,
      i.e. they are treated like an image fully covered by one polyp.
    * The paper does not publish C, gamma, T; the defaults below are starting values.
    * Weights are computed without gradient (they act as constant sample weights).
"""

from loss import register_loss
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage


def _polyp_stats(mask_np: np.ndarray) -> tuple[float, int]:
    """Return (normalized total polyp area, number of polyps) of one binary mask.

    Args:
        mask_np: 2D binary array of shape (H, W).

    Returns:
        Tuple ``(sum_i P_i / (H*W), Q)`` where Q is the number of connected components.
    """
    posmask = mask_np > 0.5
    _, num = scipy.ndimage.label(posmask)
    area = float(posmask.sum()) / posmask.size
    return area, int(num)


def _area_adaptive_weights(
    masks_np: np.ndarray,
    num_polyps: torch.Tensor | None,
    C: float,
    gamma: float,
    T: float,
    min_denom: float,
) -> torch.Tensor:
    """Compute the softmax-normalized area adaptive weights w_k (Eq. 8-9).

    Args:
        masks_np: Binary masks of shape (B, H, W) used to measure polyp areas.
        num_polyps: Optional (B,) tensor of polyp counts Q (e.g. from text labels).
            If None, Q is counted as connected components of ``masks_np``.
        C, gamma, T, min_denom: See ``area_adaptive_loss``.

    Returns:
        Float tensor of shape (B,) summing to 1.
    """
    scores = []
    for k in range(masks_np.shape[0]):
        area, q = _polyp_stats(masks_np[k])
        if num_polyps is not None:
            q = int(num_polyps[k].item())
        if q == 0 or area == 0.0:
            f = C  # background-only image -> lowest score
        else:
            denom = max(area - gamma * (q - 1), min_denom)
            f = C / denom
        scores.append(f)

    scores = torch.tensor(scores, dtype=torch.float32)
    return torch.softmax(scores / T, dim=0)


@register_loss('aal')
@register_loss('area_adaptive')
@register_loss('area_adaptive_loss')
def area_adaptive_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    num_polyps: torch.Tensor | None = None,
    C: float = 1.0,
    gamma: float = 0.01,
    T: float = 5.0,
    min_denom: float = 0.02,
    area_source: str = 'gt',
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Area Adaptive Loss: area/quantity-weighted DiceBCE.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        num_polyps: Optional (B,) tensor with the polyp count Q of each image
            (the "number of polyps" text label of TGANet). Counted from the mask if None.
        C: Constant numerator of the fusion function f (Eq. 8).
        gamma: Scaling factor converting polyp count into (normalized) area (Eq. 8).
        T: Softmax temperature; larger T gives smoother weights (Eq. 9).
        min_denom: Lower clamp of the denominator of f.
        area_source: ``'gt'`` measures areas on the ground truth mask,
            ``'pred'`` on the thresholded prediction (as literally written in the paper).
        smooth: Smoothing factor for Dice loss.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()
    b = pred.size(0)

    # 1. Per-sample DiceBCE (Eq. 4-6)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    bce = bce.view(b, -1).mean(dim=1)

    prob = torch.sigmoid(pred)
    p_flat = prob.view(b, -1)
    g_flat = mask.view(b, -1)
    inter = (p_flat * g_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + g_flat.sum(dim=1)
    dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)

    dice_bce = dice + bce  # (B,)

    # 2. Area adaptive weights (Eq. 8-9)
    with torch.no_grad():
        src = mask if area_source == 'gt' else (prob > 0.5).float()
        src_np = src.detach().cpu().numpy().reshape(b, *src.shape[-2:])
        weights = _area_adaptive_weights(
            src_np, num_polyps, C, gamma, T, min_denom
        ).to(pred.device)

    # 3. AAL = sum_k w_k * DiceBCE_k (Eq. 7)
    return (weights * dice_bce).sum()

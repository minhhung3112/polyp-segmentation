"""
Ambiguity-Tolerant Boundary Band Loss.

Motivation:
    Polyp boundaries are ambiguous: annotations from different experts differ
    by a few pixels. Forcing a pixel-exact match at the edge makes the model
    fit annotation noise, which hurts cross-dataset generalization
    (e.g. train Kvasir + CVC-ClinicDB, test ETIS / ColonDB).

Construction (per image, from the GT signed distance map d, in pixels,
d > 0 outside, d < 0 inside):
    - Band half-width per polyp, proportional to its size:
          delta_k = clip(delta_ratio * sqrt(Area_k), delta_min, delta_max)
      assigned to every pixel through its nearest polyp.
    - Band            : |d| < delta
    - Soft target     : t = sigmoid(-d / tau) inside the band, GT outside
                        (tau = delta / 3 by default -> t ~ 0.05 / 0.95 at band edges)
    - Pixel weight    : band_weight inside the band, 1 outside

    L = weighted BCE(pred, t) + soft Dice(pred, t)

Hard labels are kept away from the boundary, so the polyp body and the
background are still learned sharply; only the ambiguous rim is relaxed.
"""

from loss import register_loss
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage


# ----------------------------------------------------------------------
# Helpers (self-contained, no extra module needed)
# ----------------------------------------------------------------------
_CONN8 = np.ones((3, 3), dtype=bool)


def to_4d(pred: torch.Tensor, mask: torch.Tensor):
    """Bring ``pred`` / ``mask`` to shape (B, 1, H, W), mask as float."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if mask.shape != pred.shape:
        mask = mask.reshape(pred.shape)
    return pred, mask.float()

def label_components(mask_2d: np.ndarray, min_size: int = 0):
    """
    8-connected components of a binary numpy mask.

    Components smaller than ``min_size`` pixels are dropped (useful because
    resized GT masks often contain a few stray pixels).

    Returns:
        (labeled, n): int array with ids 1..n (0 = background), and n.
    """
    labeled, n = scipy.ndimage.label(mask_2d > 0.5, structure=_CONN8)
    if min_size > 0 and n > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < min_size
        small[0] = False
        if small.any():
            labeled[small[labeled]] = 0
            labeled, n = scipy.ndimage.label(labeled > 0, structure=_CONN8)
    return labeled, n

def nearest_component_map(labeled: np.ndarray) -> np.ndarray:
    """Assign every pixel to its nearest GT component id (Voronoi partition)."""
    if labeled.max() == 0:
        return labeled
    _, idx = scipy.ndimage.distance_transform_edt(labeled == 0, return_indices=True)
    return labeled[idx[0], idx[1]]

def signed_distance(mask_2d: np.ndarray):
    """
    Signed distance map in PIXELS (not normalized): > 0 outside, < 0 inside,
    same sign convention as ``boundary_loss.py``.

    Returns ``None`` if the mask is empty or full (no boundary).
    """
    pos = mask_2d > 0.5
    if not pos.any() or pos.all():
        return None
    out = scipy.ndimage.distance_transform_edt(~pos)
    inn = scipy.ndimage.distance_transform_edt(pos)
    return (out - inn).astype(np.float32)


def _band_targets(mask_2d: np.ndarray, delta_ratio: float, delta_min: float,
                  delta_max: float, tau, band_weight: float, min_size: int):
    """Return (soft_target, weight) numpy maps for one image."""
    target = (mask_2d > 0.5).astype(np.float32)
    weight = np.ones_like(target)

    sdf = signed_distance(mask_2d)
    if sdf is None:
        return target, weight

    labeled, n = label_components(mask_2d, min_size)
    if n > 0:
        areas = np.bincount(labeled.ravel()).astype(np.float32)
        deltas = np.clip(delta_ratio * np.sqrt(areas), delta_min, delta_max)
        delta = deltas[nearest_component_map(labeled)]
    else:
        delta = np.full_like(sdf, delta_min)

    tau_map = delta / 3.0 if tau is None else np.full_like(sdf, tau)
    band = np.abs(sdf) < delta
    soft = 1.0 / (1.0 + np.exp(np.clip(sdf / tau_map, -50, 50)))

    target[band] = soft[band]
    weight[band] = band_weight
    return target, weight


@register_loss('ambiguity_band')
@register_loss('ambiguity_band_loss')
@register_loss('band_loss')
def ambiguity_band_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    delta_ratio: float = 0.05,
    delta_min: float = 1.0,
    delta_max: float = 8.0,
    tau: float = None,
    band_weight: float = 0.5,
    min_size: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Weighted BCE + soft Dice against a boundary-relaxed soft target.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        delta_ratio: Band half-width as a fraction of sqrt(polyp area).
            A 100x100 polyp -> 5 px band.
        delta_min, delta_max: Clamp of the band half-width (pixels).
            Tune delta_max with the training resolution (8 px is for ~352x352).
        tau: Soft-target temperature in pixels. ``None`` -> delta / 3.
        band_weight: BCE weight inside the band (1.0 = only soft labels,
            no down-weighting).
        min_size: Ignore GT components smaller than this many pixels when
            estimating band width.
        smooth: Dice smoothing.

    Returns:
        Scalar loss tensor.
    """
    pred, mask = to_4d(pred, mask)
    B = pred.size(0)
    mask_np = mask.detach().cpu().numpy()[:, 0]

    targets, weights = [], []
    for b in range(B):
        t, w = _band_targets(mask_np[b], delta_ratio, delta_min, delta_max,
                             tau, band_weight, min_size)
        targets.append(t)
        weights.append(w)
    target = torch.from_numpy(np.stack(targets)).to(pred.device).view_as(pred)
    weight = torch.from_numpy(np.stack(weights)).to(pred.device).view_as(pred)

    # 1. Weighted BCE with soft target
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    l_bce = ((weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))).mean()

    # 2. Soft Dice with soft target
    prob = torch.sigmoid(pred)
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    l_dice = (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()

    return l_bce + l_dice
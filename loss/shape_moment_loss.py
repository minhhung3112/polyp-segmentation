"""
Moment-Matching Shape Loss (lightweight shape prior).

Motivation:
    Polyps are roughly convex / elliptical blobs, but CNN/Transformer outputs
    are often jagged, holed or fragmented. Matching low-order image moments
    (area, centroid, covariance = an equivalent ellipse) gives a cheap,
    fully differentiable global shape constraint per polyp.

Per polyp k (GT component), on its Voronoi region R_k (pixels nearer to k
than to any other polyp, so predictions of neighbouring polyps don't mix):
    A      = sum w_i                       (0th moment, area)
    mu     = sum w_i x_i / A               (1st moment, centroid)
    Sigma  = sum w_i (x_i-mu)(x_i-mu)^T / A (2nd central moment)
with w = predicted prob (for the prediction) or GT (for the target), and
coordinates x normalized to [0, 1].

    L_area   = |A_hat - A| / A
    L_center = ||mu_hat - mu||_2 / sqrt(A_frac)
    L_cov    = ||Sigma_hat - Sigma||_F / A_frac
(A_frac = A / (H*W); normalizing by size keeps small polyps equally weighted.)

    L = L_structure + lam * mean_k (w_area*L_area + w_center*L_center + w_cov*L_cov)
"""

from loss import register_loss
import numpy as np
import torch
import scipy.ndimage
import torch.nn.functional as F


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

def boundary_weight(mask: torch.Tensor, kernel: int = 31) -> torch.Tensor:
    """PraNet-style boundary emphasis ``|avgpool(mask) - mask|`` in [0, 1]."""
    return torch.abs(
        F.avg_pool2d(mask, kernel, stride=1, padding=kernel // 2) - mask
    )

def structure_loss(pred: torch.Tensor, mask: torch.Tensor,
                   weit: torch.Tensor = None) -> torch.Tensor:
    """
    PraNet structure loss (weighted BCE + weighted IoU).

    Args:
        pred: Logits (B, 1, H, W).
        mask: GT (B, 1, H, W) in [0, 1].
        weit: Optional pixel weight map (B, 1, H, W). Defaults to
            ``1 + 5 * boundary_weight(mask)`` as in PraNet.
    """
    if weit is None:
        weit = 1 + 5 * boundary_weight(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    prob = torch.sigmoid(pred)
    inter = (prob * mask * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

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

def zero_like_graph(pred: torch.Tensor) -> torch.Tensor:
    """A scalar 0 that stays attached to the autograd graph."""
    return pred.sum() * 0.0


def _moments(w: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor, eps: float):
    """Area, centroid (2,), covariance (2, 2) of a weighted point set."""
    A = w.sum()
    my = (w * ys).sum() / (A + eps)
    mx = (w * xs).sum() / (A + eps)
    dy, dx = ys - my, xs - mx
    syy = (w * dy * dy).sum() / (A + eps)
    sxx = (w * dx * dx).sum() / (A + eps)
    sxy = (w * dx * dy).sum() / (A + eps)
    mu = torch.stack([my, mx])
    cov = torch.stack([torch.stack([syy, sxy]), torch.stack([sxy, sxx])])
    return A, mu, cov


@register_loss('shape_moment')
@register_loss('shape_moment_loss')
@register_loss('moment_loss')
def shape_moment_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 0.5,
    w_area: float = 1.0,
    w_center: float = 1.0,
    w_cov: float = 1.0,
    min_size: int = 10,
    clip: float = 5.0,
    use_base: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Structure loss + per-polyp moment matching.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        lam: Weight of the shape term.
        w_area, w_center, w_cov: Weights of the 0th / 1st / 2nd-order terms
            (set one to 0 for ablation).
        min_size: Ignore GT components smaller than this many pixels.
        clip: Upper clamp per term, avoids huge gradients early in training
            when the prediction is nearly empty.
        use_base: Add PraNet structure loss as the pixel-level base.
        eps: Numerical stability.

    Returns:
        Scalar loss tensor.
    """
    pred, mask = to_4d(pred, mask)
    prob = torch.sigmoid(pred)
    B, _, H, W = pred.shape

    yy, xx = torch.meshgrid(
        torch.arange(H, device=pred.device, dtype=prob.dtype) / max(H - 1, 1),
        torch.arange(W, device=pred.device, dtype=prob.dtype) / max(W - 1, 1),
        indexing='ij',
    )
    ys_all, xs_all = yy.reshape(-1), xx.reshape(-1)
    mask_np = mask.detach().cpu().numpy()[:, 0]

    terms = []
    for b in range(B):
        labeled, n = label_components(mask_np[b], min_size)
        if n == 0:
            continue  # empty GT: base loss already penalizes any foreground
        region = nearest_component_map(labeled)
        p_flat = prob[b, 0].reshape(-1)
        lab_flat = torch.from_numpy(labeled.reshape(-1)).to(pred.device)

        for k in range(1, n + 1):
            idx = torch.from_numpy(np.flatnonzero(region == k)).to(pred.device)
            ys, xs = ys_all[idx], xs_all[idx]
            g = (lab_flat[idx] == k).to(prob.dtype)
            p = p_flat[idx]

            A_g, mu_g, cov_g = _moments(g, ys, xs, eps)
            A_p, mu_p, cov_p = _moments(p, ys, xs, eps)
            A_frac = A_g / (H * W)

            l_area = (torch.abs(A_p - A_g) / (A_g + eps)).clamp(max=clip)
            l_center = (torch.norm(mu_p - mu_g) / (A_frac.sqrt() + eps)).clamp(max=clip)
            l_cov = (torch.norm(cov_p - cov_g, p='fro') / (A_frac + eps)).clamp(max=clip)
            terms.append(w_area * l_area + w_center * l_center + w_cov * l_cov)

    l_shape = torch.stack(terms).mean() if terms else zero_like_graph(pred)
    l_base = structure_loss(pred, mask) if use_base else zero_like_graph(pred)
    return l_base + lam * l_shape
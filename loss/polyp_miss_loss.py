"""
Polyp-level Miss Penalty Loss (detection-aware loss).

Motivation:
    Pixel losses (Dice/IoU/BCE) are dominated by large polyps: missing a whole
    200-pixel polyp next to a 20k-pixel one barely changes Dice, yet clinically
    it is a missed lesion. This loss treats every polyp (GT connected component)
    as ONE unit, regardless of its size.

Terms:
    For each GT component C_k, a smooth max of the predicted probability:
        s_k = tau * ( logsumexp(p_i / tau) - log|C_k| ),  i in C_k
    (s_k lies between mean(p) and max(p); tau -> 0 gives the max).

    Miss term  : L_miss = mean_k  relu(margin - s_k)
        -> "the model must fire strongly SOMEWHERE inside each polyp".

    FP term    : on predicted blobs (p > fp_thresh) lying outside a dilated GT,
        L_fp = mean_j relu(s_j - (1 - margin))
        -> "spurious blobs (e.g. specular highlights) must be suppressed".

    Total:
        L = L_structure + lam * (L_miss + fp_weight * L_fp)

Note:
    Component extraction runs on CPU (scipy) on detached tensors; gradients
    flow only through the probabilities, as in ``boundary_loss.py``.
"""

from loss import register_loss
import math
import numpy as np
import scipy.ndimage
import torch
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

def zero_like_graph(pred: torch.Tensor) -> torch.Tensor:
    """A scalar 0 that stays attached to the autograd graph."""
    return pred.sum() * 0.0


def _smooth_max(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Log-mean-exp: between mean(x) and max(x)."""
    return tau * (torch.logsumexp(x / tau, dim=0) - math.log(x.numel()))


def _miss_fp_terms(prob: torch.Tensor, mask: torch.Tensor, margin: float,
                   tau: float, fp_thresh: float, dilate: int, min_size: int,
                   use_fp: bool):
    """Collect per-polyp miss terms and per-blob FP terms over the batch."""
    B = prob.size(0)
    mask_np = mask.detach().cpu().numpy()[:, 0]
    prob_np = prob.detach().cpu().numpy()[:, 0]
    miss_terms, fp_terms = [], []

    for b in range(B):
        p_flat = prob[b, 0].reshape(-1)

        # --- miss: one term per GT polyp ---
        gt_lab, n_gt = label_components(mask_np[b], min_size)
        for k in range(1, n_gt + 1):
            idx = torch.from_numpy(np.flatnonzero(gt_lab == k)).to(prob.device)
            miss_terms.append(F.relu(margin - _smooth_max(p_flat[idx], tau)))

        # --- false positives: one term per spurious predicted blob ---
        if use_fp:
            gt_bin = mask_np[b] > 0.5
            gt_dil = (scipy.ndimage.binary_dilation(gt_bin, iterations=dilate)
                      if dilate > 0 and gt_bin.any() else gt_bin)
            fp_lab, n_fp = label_components((prob_np[b] > fp_thresh) & ~gt_dil,
                                            min_size)
            for j in range(1, n_fp + 1):
                idx = torch.from_numpy(np.flatnonzero(fp_lab == j)).to(prob.device)
                fp_terms.append(F.relu(_smooth_max(p_flat[idx], tau) - (1 - margin)))

    return miss_terms, fp_terms


@register_loss('polyp_miss')
@register_loss('polyp_miss_loss')
@register_loss('miss_loss')
def polyp_miss_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 1.0,
    margin: float = 0.7,
    tau: float = 0.1,
    fp_weight: float = 0.5,
    fp_thresh: float = 0.5,
    dilate: int = 5,
    min_size: int = 10,
    use_base: bool = True,
) -> torch.Tensor:
    """
    Structure loss + polyp-level miss / false-positive penalty.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        lam: Weight of the polyp-level term.
        margin: Required smooth-max confidence inside each polyp. FP blobs are
            pushed below ``1 - margin``.
        tau: Temperature of the smooth max (smaller = closer to hard max).
        fp_weight: Relative weight of the FP term (0 disables it).
        fp_thresh: Probability threshold used to extract predicted blobs.
        dilate: GT dilation (pixels) so boundary spill is not counted as FP.
        min_size: Ignore components (GT or FP) smaller than this many pixels.
        use_base: Add PraNet structure loss as the pixel-level base.

    Returns:
        Scalar loss tensor.
    """
    pred, mask = to_4d(pred, mask)
    prob = torch.sigmoid(pred)

    miss_terms, fp_terms = _miss_fp_terms(
        prob, mask, margin, tau, fp_thresh, dilate, min_size, fp_weight > 0
    )
    l_miss = torch.stack(miss_terms).mean() if miss_terms else zero_like_graph(pred)
    l_fp = torch.stack(fp_terms).mean() if fp_terms else zero_like_graph(pred)

    l_base = structure_loss(pred, mask) if use_base else zero_like_graph(pred)
    return l_base + lam * (l_miss + fp_weight * l_fp)
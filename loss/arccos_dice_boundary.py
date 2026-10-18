"""
Arccosine Dice Loss + Balanced Boundary (Surface) Loss for Polyp Segmentation.

References:
    Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., Dolz, J., & Ben Ayed, I. (2019).
    Boundary Loss for Highly Unbalanced Segmentation.
    Medical Imaging with Deep Learning (MIDL 2019), PMLR 102, pp. 285-296.

Combines two complementary terms:
    L = (2 / pi) * arccos(DSC_w) + alpha_t * L_Boundary

1. Arccos Dice (region term):
    Same [0, 1] range and monotonicity as ``1 - DSC``, but with a steeper
    gradient near DSC = 1. DSC is boundary-weighted (PraNet-style weight
    map) and the arccos backward gradient is clipped to ``max_grad``.

2. Balanced boundary loss (distance term):
    mean(prob * SDF). The original SDF is normalized by ONE global max
    distance, which for a typical polyp gives |SDF| ~0.05 inside vs ~0.5
    outside: the term pushes background down ~10x harder than it pulls the
    polyp up, making the model conservative (under-segmentation, lower Dice
    on large in-domain polyps such as Kvasir / CVC-ClinicDB).
    Here the inside and outside are normalized SEPARATELY, so both lie in
    [-1, 0] and [0, 1] respectively and pull with comparable strength.
    Set ``balanced_sdf=False`` to recover the original behaviour.

3. Warm-up of alpha (Kervadec et al.):
    Pass ``epoch`` from the training loop and the boundary weight is ramped
    linearly from 0 to ``alpha`` over ``warmup_epochs``:
        loss = loss_fn(pred, mask, epoch=epoch)
    Without ``epoch``, the full ``alpha`` is used.
"""

from loss import register_loss
import math
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage


def _boundary_weight_map(mask: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
    """Up to 6x weight for pixels near the mask boundary, 1x deep inside a flat region."""
    return 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) - mask
    )


def _dice_coefficient(prob: torch.Tensor, mask: torch.Tensor,
                      weight: torch.Tensor, smooth: float) -> torch.Tensor:
    """Soft Dice similarity coefficient per image, weighted by ``weight``."""
    inter = (prob * mask * weight).sum(dim=(1, 2, 3))
    union = ((prob + mask) * weight).sum(dim=(1, 2, 3))
    return (2.0 * inter + smooth) / (union + smooth)


class _SafeArccosFn(torch.autograd.Function):
    """arccos(x) with the backward gradient magnitude clipped to ``max_grad``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, max_grad: float):
        ctx.save_for_backward(x)
        ctx.max_grad = max_grad
        return torch.arccos(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        denom = torch.sqrt(torch.clamp(1.0 - x * x, min=1e-8))
        local_grad = torch.clamp(-1.0 / denom, min=-ctx.max_grad, max=ctx.max_grad)
        return grad_output * local_grad, None


def _compute_sdf(mask_np: np.ndarray, balanced: bool = True) -> np.ndarray:
    """
    Signed distance map from a binary GT mask (> 0 outside, < 0 inside).

    balanced=True : inside / max_inside in [-1, 0], outside / max_outside in [0, 1].
    balanced=False: (outside - inside) / max|SDF|  (original single normalization).
    Empty mask -> all ones (any foreground prediction is penalized).
    """
    posmask = mask_np > 0.5
    if not posmask.any():
        return np.ones_like(mask_np, dtype=np.float32)
    negmask = ~posmask
    posdis = scipy.ndimage.distance_transform_edt(posmask)   # inside distances
    negdis = scipy.ndimage.distance_transform_edt(negmask)   # outside distances
    if balanced:
        sdf = negdis / max(negdis.max(), 1.0) - posdis / max(posdis.max(), 1.0)
    else:
        sdf = negdis - posdis
        sdf = sdf / max(np.abs(sdf).max(), 1.0)
    return sdf.astype(np.float32)


@register_loss('arccos_dice_boundary')
@register_loss('arccos_boundary')
def arccos_dice_boundary_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.1,
    smooth: float = 1.0,
    kernel_size: int = 31,
    max_grad: float = 10.0,
    balanced_sdf: bool = True,
    epoch: int = None,
    warmup_epochs: int = 20
) -> torch.Tensor:
    """
    Combined Arccos Dice + Balanced Boundary Surface Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Final weight of the boundary term (default 0.1; try 0.05-0.25).
        smooth: Laplace smoothing added to the Dice ratio.
        kernel_size: Window used to detect boundary pixels for the weight map.
        max_grad: Gradient-magnitude clip for the arccos backward pass.
        balanced_sdf: Normalize inside / outside distances separately
            (False = original Kervadec normalization, for ablation).
        epoch: Current epoch (0-based). If given, alpha is ramped from 0 to
            ``alpha`` over ``warmup_epochs``. None = full alpha.
        warmup_epochs: Length of the linear warm-up.

    Returns:
        Scalar loss tensor.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred)
    b = pred.size(0)

    # 1. Arccos Dice component (DSC is always in (0, 1] when smooth > 0)
    weight = _boundary_weight_map(mask, kernel_size)
    dsc = _dice_coefficient(prob, mask, weight, smooth)
    l_arccos = ((2.0 / math.pi) * _SafeArccosFn.apply(dsc, max_grad)).mean()

    # 2. Boundary weight with optional linear warm-up
    if epoch is not None and warmup_epochs > 0:
        alpha_t = alpha * min(1.0, (epoch + 1) / warmup_epochs)
    else:
        alpha_t = alpha
    if alpha_t <= 0:
        return l_arccos

    # 3. Balanced boundary distance component
    mask_np = mask.detach().cpu().numpy()
    sdf_maps = []
    for i in range(b):
        m_2d = mask_np[i].squeeze()
        sdf_maps.append(_compute_sdf(m_2d, balanced_sdf))

    sdf_tensor = torch.from_numpy(np.stack(sdf_maps)).to(pred.device).view_as(prob)
    l_boundary = (prob * sdf_tensor).mean()

    return l_arccos + alpha_t * l_boundary
"""
Arccosine Dice Loss for Medical Image Segmentation.

Reformulates the Dice loss through arccos instead of a linear
``1 - DSC``:

    L = (2 / pi) * arccos(DSC)

Same [0, 1] range and monotonicity as plain Dice loss (0 for a
perfect match, 1 for no overlap), but with a steeper gradient near
DSC = 1 and a flatter one near DSC = 0, which can push training
harder once a prediction is already close to the target.

Two stability fixes are applied on top of the plain formula:
  1. The arccos gradient, -1 / sqrt(1 - x^2), diverges as DSC -> 1.
     Clamping the *value* of DSC keeps the loss finite but not the
     gradient, so the gradient itself is clipped to ``max_grad``
     (see ``_SafeArccosFn``).
  2. DSC is computed with the same boundary weight map used
     elsewhere in this codebase (see ``_boundary_weight_map``), so
     pixels near the mask boundary count for more in the ratio —
     without this, a model can inflate DSC via easy interior pixels
     alone.
"""

from loss import register_loss
import math
import torch
import torch.nn.functional as F


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


@register_loss('arccos_dice')
@register_loss('dice_arccos')
def arccos_dice_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-4,
    kernel_size: int = 31,
    max_grad: float = 10.0
) -> torch.Tensor:
    """
    Arccosine Dice Loss, boundary-weighted and gradient-safe.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        smooth: Laplace smoothing added to the Dice ratio.
        eps: Margin kept between DSC and {0, 1} before calling arccos.
        kernel_size: Window used to detect boundary pixels for the weight map.
        max_grad: Gradient-magnitude clip for the arccos backward pass;
            lower is more conservative near DSC = 1 (see module docstring).

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred)
    weight = _boundary_weight_map(mask, kernel_size)
    dsc = _dice_coefficient(prob, mask, weight, smooth)
    dsc = torch.clamp(dsc, min=eps, max=1.0 - eps)

    loss = (2.0 / math.pi) * _SafeArccosFn.apply(dsc, max_grad)
    return loss.mean()
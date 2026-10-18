"""
Camouflage-Aware (Image-Conditioned) Structure Loss.

Motivation:
    PraNet's structure loss weights pixels by distance to the GT boundary,
    i.e. it only looks at the MASK. But which boundary is hard depends on the
    IMAGE: where the polyp colour is close to the surrounding mucosa
    (low contrast, "camouflaged"), the model needs more pressure.

Weight map:
    1. Convert the image to CIE-Lab (perceptually uniform colour space).
    2. In a local window around each pixel, compute the mean colour of the
       polyp side (mu_in) and of the background side (mu_out).
    3. Local contrast  Delta = || mu_in - mu_out ||_2   (approx. CIE76 Delta-E).
    4. Hardness        h = exp(-Delta / sigma)   (only where both sides exist)
    5. Weight          w = 1 + 5 * bw * (1 + alpha * h)
       where bw = |avgpool(mask) - mask| is PraNet's boundary weight.

    The weight map is detached; the loss is PraNet's structure loss with w.

IMPORTANT - needs the input image:
    The registry calls ``fn(pred, mask)``, so in the training loop call it as
        loss_fn(pred, mask, image=images)
    Without ``image`` it falls back to the plain structure loss (with a warning).
"""

from loss import register_loss
import warnings
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Helpers (self-contained, no extra module needed)
# ----------------------------------------------------------------------

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

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_warned = False


def _to_unit_rgb(image: torch.Tensor, image_range: str,
                 mean=_IMAGENET_MEAN, std=_IMAGENET_STD) -> torch.Tensor:
    """Bring the image to RGB in [0, 1]."""
    image = image.float()
    if image_range == 'auto':
        if image.min() < 0:
            image_range = 'normalized'
        elif image.max() > 1.5:
            image_range = '255'
        else:
            image_range = 'unit'
    if image_range == 'normalized':
        m = torch.tensor(mean, device=image.device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=image.device).view(1, 3, 1, 1)
        image = image * s + m
    elif image_range == '255':
        image = image / 255.0
    return image.clamp(0, 1)


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB in [0, 1] (B, 3, H, W) -> CIE-Lab (D65), differentiable torch op."""
    lin = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = torch.tensor([[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]],
                     device=rgb.device, dtype=rgb.dtype)
    xyz = torch.einsum('ij,bjhw->bihw', M, lin)
    white = torch.tensor([0.950456, 1.0, 1.088754],
                         device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    xyz = xyz / white
    eps = 0.008856
    f = torch.where(xyz > eps, xyz.clamp(min=eps) ** (1 / 3), 7.787 * xyz + 16 / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], dim=1)


@torch.no_grad()
def camouflage_weight(image: torch.Tensor, mask: torch.Tensor,
                      alpha: float = 1.0, sigma: float = 10.0,
                      kernel: int = 15, bw_kernel: int = 31,
                      image_range: str = 'auto') -> torch.Tensor:
    """
    Compute the image-conditioned weight map (B, 1, H, W). Exposed separately
    so you can visualize it (very useful for the paper figure).
    """
    if image.shape[-2:] != mask.shape[-2:]:
        image = F.interpolate(image, size=mask.shape[-2:], mode='bilinear',
                              align_corners=False)
    lab = _rgb_to_lab(_to_unit_rgb(image, image_range))

    def pool(x):
        return F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2,
                            count_include_pad=False)

    m_in, m_out = pool(mask), pool(1 - mask)
    mu_in = pool(lab * mask) / (m_in + 1e-6)
    mu_out = pool(lab * (1 - mask)) / (m_out + 1e-6)
    delta = torch.norm(mu_in - mu_out, dim=1, keepdim=True)

    both_sides = ((m_in > 0.02) & (m_out > 0.02)).float()
    hard = torch.exp(-delta / sigma) * both_sides

    bw = boundary_weight(mask, bw_kernel)
    return 1 + 5 * bw * (1 + alpha * hard)


@register_loss('camouflage')
@register_loss('camouflage_aware')
@register_loss('camouflage_aware_loss')
def camouflage_aware_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    image: torch.Tensor = None,
    alpha: float = 1.0,
    sigma: float = 10.0,
    kernel: int = 15,
    bw_kernel: int = 31,
    image_range: str = 'auto',
) -> torch.Tensor:
    """
    Structure loss whose boundary weight is amplified in low-contrast regions.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        image: Input RGB batch (B, 3, H', W'); resized to the mask if needed.
        alpha: Extra boundary weight for fully camouflaged regions
            (alpha=0 recovers PraNet exactly -> clean ablation).
        sigma: Lab contrast scale; Delta-E ~ sigma gives h ~ 0.37.
        kernel: Window size for local colour statistics.
        bw_kernel: Window size of PraNet boundary weight (31 in PraNet).
        image_range: 'auto' | 'normalized' (ImageNet mean/std) | 'unit' | '255'.

    Returns:
        Scalar loss tensor.
    """
    global _warned
    pred, mask = to_4d(pred, mask)

    if image is None:
        if not _warned:
            warnings.warn("[camouflage_aware_loss] no `image` given -> falling "
                          "back to plain structure loss. Call loss_fn(pred, "
                          "mask, image=images).")
            _warned = True
        return structure_loss(pred, mask)

    weit = camouflage_weight(image, mask, alpha, sigma, kernel, bw_kernel,
                             image_range)
    return structure_loss(pred, mask, weit)
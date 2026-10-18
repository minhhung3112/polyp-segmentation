"""
Lovász-Hinge Loss for Binary Semantic Segmentation.

Reference:
    Berman, M., Triki, A. R., & Blaschko, M. B. (2018).
    The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure.
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2018), pp. 4413-4421.

Optimizes the Jaccard index using its convex Lovász extension for binary masks.
"""

from loss import register_loss
import torch


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovász hinge loss on 1D flattened tensors."""
    if len(labels) == 0:
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(torch.relu(errors_sorted), grad)
    return loss


@register_loss('lovasz')
@register_loss('lovasz_hinge')
@register_loss('lovasz_loss')
def lovasz_hinge_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Lovász-Hinge Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) with values in {0, 1}.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)

    b = pred.size(0)
    losses = []
    for i in range(b):
        logits_flat = pred[i].view(-1)
        labels_flat = (mask[i].view(-1) > 0.5).long()
        losses.append(_lovasz_hinge_flat(logits_flat, labels_flat))

    return torch.stack(losses).mean()


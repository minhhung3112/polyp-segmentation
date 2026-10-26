"""Drop-in CMUNeXt + Highlight-Aware Target (HAT) training method.

Place this file in ``models/`` and train with the unchanged CLI:

    python train.py --model cmunext_hat --seeds 42 --batchsize 8

HAT changes *training masks*, not network layers.  The model is the exact
CMUNeXt base network, with the same forward pass and state_dict layout.

The framework builds its DataLoaders before calling get_model().  Therefore
the registered model's constructor patches the existing augmentation class's
``__call__`` method; even already-instantiated training augmentations then
use HAT when the first batch is loaded.  Validation/test do not instantiate
PolypAugmentation and are unaffected.  On Colab (Linux), worker processes
start after model construction and inherit this patch.
"""

import random

import torch
from models import register_model
from models.cmunext import CMUNeXt


def highlight_aware_target(image, mask, rng, probability=0.5):
    """Expand the foreground into nearby bright, low-saturation highlights.

    Work on a copy of the mask and only within a thin exterior dilation band.
    The 10-pixel radius at 1008px is scaled to the original image size, before
    dataset resizing and color jitter.  Original GT files are never changed.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("HAT probability must be in [0, 1]")
    if probability == 0 or rng.random() >= probability:
        return mask

    import numpy as np
    from PIL import Image, ImageFilter

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    value = rgb.max(axis=2)
    saturation = (value - rgb.min(axis=2)) / (value + 1e-6)
    specular = (value > 0.90) & (saturation < 0.25)

    original = mask.convert("L")
    pixels = np.asarray(original)
    foreground = pixels > 127
    if not specular.any() or not foreground.any():
        return mask

    height, width = pixels.shape
    radius = max(1, round(10 * min(height, width) / 1008))
    dilated = np.asarray(
        original.filter(ImageFilter.MaxFilter(2 * radius + 1))
    ) > 127
    add_pixels = specular & dilated & ~foreground

    modified = pixels.copy()
    modified[add_pixels] = 255
    return Image.fromarray(modified, mode="L")


def _enable_hat_for_training():
    """Activate HAT on already-created and future training augmentations.

    This is intentionally lazy: importing the module only registers a model,
    so other ``--model`` values keep their original augmentations.
    """
    from utils.augmentation import PolypAugmentation

    if getattr(PolypAugmentation, "_cmunext_hat_active", False):
        return

    original_call = PolypAugmentation.__call__

    def with_hat(self, image, mask):
        image, mask = original_call(self, image, mask)
        rng = getattr(self, "_cmunext_hat_rng", None)
        if rng is None:
            # Dedicated per-worker RNG; baseline flip/rotation draws remain
            # on their original RNG stream for fair comparisons.
            rng = random.Random(torch.initial_seed() ^ 0x4841542026)
            self._cmunext_hat_rng = rng
        return image, highlight_aware_target(image, mask, rng, 0.5)

    PolypAugmentation.__call__ = with_hat
    PolypAugmentation._cmunext_hat_active = True
    print("[HAT] Enabled training-only highlight-aware targets (p=0.5)")


@register_model("cmunext_hat")
class CMUNeXtHAT(CMUNeXt):
    """CMUNeXt base architecture with HAT applied to training targets."""

    def __init__(self):
        super().__init__(
            input_channel=3,
            num_classes=1,
            dims=[16, 32, 128, 160, 256],
            depths=[1, 1, 1, 3, 1],
            kernels=[3, 3, 7, 7, 7],
        )
        _enable_hat_for_training()

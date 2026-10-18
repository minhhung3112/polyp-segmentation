"""
Joint Data Augmentation for Polyp Segmentation.

Applies synchronized spatial and color transformations on paired image and ground truth mask.
"""

import random
from PIL import Image
import torchvision.transforms.functional as TF


class PolypAugmentation:
    """Joint random transformations for image and ground truth mask."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        angles: tuple = (90, 180, 270),
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob
        self.angles = angles

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Random Horizontal Flip
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random Vertical Flip
        if random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random Discrete Rotation
        if random.random() < self.rotate_prob:
            angle = random.choice(self.angles)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        return image, mask

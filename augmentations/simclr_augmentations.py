"""
SimCLR Augmentations
====================
Data augmentation pipeline following the SimCLR paper (Chen et al., 2020).

Key insight: SimCLR requires generating TWO different augmented views
of the same image. This module provides a transform that returns a pair
of augmented views (x_i, x_j) from a single input image.

Augmentation pipeline:
  1. RandomResizedCrop
  2. RandomHorizontalFlip
  3. ColorJitter (with probability)
  4. RandomGrayscale
  5. GaussianBlur (with probability)
  6. Normalize to ImageNet stats
"""

import torch
import torchvision.transforms as T
from PIL import ImageFilter
import random


class GaussianBlur:
    """
    Gaussian blur augmentation as used in SimCLR.

    Applied with a given probability using a random kernel size.
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class SimCLRTransform:
    """
    SimCLR augmentation transform that generates TWO augmented views
    of a single input image.

    Returns:
        Tuple of (view_1, view_2) — two independently augmented crops.

    Args:
        image_size: Target crop size (96 for STL10, 32 for CIFAR10).
        strength: Augmentation strength multiplier for color jitter.
        gaussian_blur_prob: Probability of applying Gaussian blur.
    """

    def __init__(
        self,
        image_size: int = 96,
        strength: float = 1.0,
        gaussian_blur_prob: float = 0.5,
    ):
        # Color jitter strengths (scaled by strength parameter)
        s = strength
        color_jitter = T.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)

        # Build the augmentation pipeline
        self.transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=gaussian_blur_prob),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, x):
        """
        Apply two independent random augmentations to the same image.

        Args:
            x: PIL Image.

        Returns:
            (x_i, x_j): Tuple of two augmented tensor views.
        """
        x_i = self.transform(x)
        x_j = self.transform(x)
        return x_i, x_j


def get_eval_transform(image_size: int = 96):
    """
    Standard evaluation transform (no augmentation, just resize + normalize).

    Used for linear evaluation, kNN evaluation, and t-SNE.
    """
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_finetune_transform(image_size: int = 96):
    """
    Light augmentation for fine-tuning (less aggressive than SimCLR pretrain).
    """
    return T.Compose([
        T.RandomResizedCrop(size=image_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

"""
Medical Image Augmentations
============================
Safe augmentation pipeline for chest X-ray and other medical images.

Design principles:
  - Avoid heavy color distortion (X-ray intensity has diagnostic meaning)
  - Use geometric augmentations that preserve anatomical structures
  - Small brightness/contrast changes are acceptable (exposure variation)
  - Gaussian noise simulates sensor noise in real scanners

Two modes:
  1. MedicalSSLTransform — dual-view augmentation for SSL pretraining
  2. get_medical_eval_transform — standard eval (resize + center crop + normalize)
  3. get_medical_finetune_transform — light augmentations for fine-tuning
"""

import torch
import torchvision.transforms as T
import random
import numpy as np


# Medical image normalization (grayscale X-ray repeated to 3 channels)
MEDICAL_MEAN = [0.485, 0.456, 0.406]
MEDICAL_STD = [0.229, 0.224, 0.225]


class GaussianNoise:
    """Add Gaussian noise to simulate sensor noise in medical imaging."""

    def __init__(self, std_range=(0.01, 0.05)):
        self.std_range = std_range

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = T.functional.to_tensor(tensor)
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class MedicalSSLTransform:
    """
    Dual-view augmentation for SSL pretraining on medical images.

    Returns (view_1, view_2) — two independently augmented crops.

    Augmentation pipeline (safe for X-rays):
      1. RandomResizedCrop (scale 0.5–1.0, no aggressive cropping)
      2. RandomHorizontalFlip
      3. Small rotation (±10 degrees)
      4. Small brightness/contrast change
      5. Gaussian noise
      6. Normalize
    """

    def __init__(self, image_size=224, noise_prob=0.3, rotation_degrees=10):
        self.transform = T.Compose([
            T.Resize(int(image_size * 1.1)),
            T.RandomResizedCrop(size=image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=rotation_degrees),
            T.RandomApply([
                T.ColorJitter(brightness=0.15, contrast=0.15),
            ], p=0.5),
            T.ToTensor(),
            T.RandomApply([GaussianNoise(std_range=(0.01, 0.03))], p=noise_prob),
            T.Normalize(mean=MEDICAL_MEAN, std=MEDICAL_STD),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def get_medical_eval_transform(image_size=224):
    """Standard evaluation transform for medical images."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=MEDICAL_MEAN, std=MEDICAL_STD),
    ])


def get_medical_finetune_transform(image_size=224):
    """Light augmentation for fine-tuning on medical images."""
    return T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(size=image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=MEDICAL_MEAN, std=MEDICAL_STD),
    ])

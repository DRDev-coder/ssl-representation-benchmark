"""
CIFAR-10 Dataset Loaders
========================
CIFAR-10 as an alternative dataset for SimCLR experiments:
  - 50,000 training images (used unlabeled for pretraining)
  - 10,000 test images
  - 10 classes, 32x32 resolution

For pretraining, labels are discarded to simulate self-supervised learning.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision

from augmentations.simclr_augmentations import (
    SimCLRTransform,
    get_eval_transform,
    get_finetune_transform,
)


# CIFAR-10 specific image size
CIFAR10_IMAGE_SIZE = 32


def get_cifar10_pretrain(
    data_dir: str,
    image_size: int = 32,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    augmentation_strength: float = 1.0,
):
    """
    CIFAR-10 training set with SimCLR dual-view augmentation for pretraining.
    Labels are ignored to simulate unlabeled pretraining.

    Returns:
        DataLoader yielding ((view_1, view_2), _) pairs.
    """
    transform = SimCLRTransform(
        image_size=image_size,
        strength=augmentation_strength,
        gaussian_blur_prob=0.0,  # SimCLR paper: no blur for small images
    )

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    return loader


def get_cifar10_train(
    data_dir: str,
    image_size: int = 32,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    label_fraction: float = 1.0,
    augment: bool = True,
    seed: int = 42,
):
    """
    CIFAR-10 labeled training split with optional label subsampling.

    Args:
        label_fraction: Fraction of labels to use (e.g., 0.01, 0.10, 1.0).
        augment: If True, use fine-tuning augmentations; else eval transform.
        seed: Random seed for reproducible label subsampling.

    Returns:
        DataLoader for labeled training data.
    """
    transform = get_finetune_transform(image_size) if augment else get_eval_transform(image_size)

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Subsample labels if needed
    if label_fraction < 1.0:
        dataset = _subsample_dataset(dataset, label_fraction, seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return loader


def get_cifar10_test(
    data_dir: str,
    image_size: int = 32,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    """
    CIFAR-10 test split for evaluation.

    Returns:
        DataLoader for test data with eval transforms.
    """
    transform = get_eval_transform(image_size)

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader


def _subsample_dataset(dataset, fraction: float, seed: int = 42):
    """
    Subsample a dataset while preserving class balance (stratified sampling).
    """
    rng = np.random.RandomState(seed)
    labels = np.array(dataset.targets)
    classes = np.unique(labels)
    indices = []

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        n_keep = max(1, int(len(cls_indices) * fraction))
        selected = rng.choice(cls_indices, size=n_keep, replace=False)
        indices.extend(selected.tolist())

    rng.shuffle(indices)
    print(f"Subsampled {len(indices)} / {len(dataset)} samples ({fraction*100:.1f}%)")
    return Subset(dataset, indices)

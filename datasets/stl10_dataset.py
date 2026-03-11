"""
STL-10 Dataset Loaders
======================
STL-10 is specifically designed for self-supervised learning:
  - 100,000 unlabeled images (for pretraining)
  - 5,000 labeled training images (for evaluation)
  - 8,000 labeled test images
  - 10 classes, 96x96 resolution

This module provides loaders for each split with appropriate transforms.
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


def get_stl10_pretrain(
    data_dir: str,
    image_size: int = 96,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    augmentation_strength: float = 1.0,
):
    """
    STL-10 unlabeled split with SimCLR dual-view augmentation for pretraining.

    Returns:
        DataLoader yielding ((view_1, view_2), _) pairs.
    """
    transform = SimCLRTransform(
        image_size=image_size,
        strength=augmentation_strength,
    )

    dataset = torchvision.datasets.STL10(
        root=data_dir,
        split="unlabeled",
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,           # SimCLR needs consistent batch sizes
        persistent_workers=num_workers > 0,   # Avoid worker restart overhead
    )

    return loader


def get_stl10_train(
    data_dir: str,
    image_size: int = 96,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    label_fraction: float = 1.0,
    augment: bool = True,
    seed: int = 42,
):
    """
    STL-10 labeled training split with optional label subsampling.

    Args:
        label_fraction: Fraction of labels to use (e.g., 0.01, 0.10, 1.0).
        augment: If True, use fine-tuning augmentations; else eval transform.
        seed: Random seed for reproducible label subsampling.

    Returns:
        DataLoader for labeled training data.
    """
    transform = get_finetune_transform(image_size) if augment else get_eval_transform(image_size)

    dataset = torchvision.datasets.STL10(
        root=data_dir,
        split="train",
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


def get_stl10_test(
    data_dir: str,
    image_size: int = 96,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    """
    STL-10 test split for evaluation.

    Returns:
        DataLoader for test data with eval transforms.
    """
    transform = get_eval_transform(image_size)

    dataset = torchvision.datasets.STL10(
        root=data_dir,
        split="test",
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

    Args:
        dataset: PyTorch dataset with .labels attribute.
        fraction: Fraction of data to keep per class.
        seed: Random seed for reproducibility.

    Returns:
        Subset of the dataset.
    """
    rng = np.random.RandomState(seed)
    labels = np.array(dataset.labels)
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

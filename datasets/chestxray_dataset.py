"""
NIH CXR8 (ChestX-ray8) Dataset Loader
=======================================
Dataset loader for the NIH ChestX-ray8 dataset (112,120 frontal-view
chest X-ray images with 14 disease labels).

Binary classification: No Finding (0) vs. Pathology (1).

Data structure expected (CXR8 folder):
    CXR8/
        Data_Entry_2017_v2020.csv   # Main labels file
        train_val_list.txt          # Official train split (86,524 filenames)
        test_list.txt               # Official test split (25,596 filenames)
        images_001/images/          # Image batches 1-12
        images_002/images/
        ...
        images_012/images/

Usage:
    from datasets.chestxray_dataset import get_chestxray_pretrain, get_chestxray_train
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image

from augmentations.simclr_augmentations import SimCLRTransform, get_eval_transform, get_finetune_transform


CHESTXRAY_IMAGE_SIZE = 224
CHESTXRAY_CLASSES = ["no_finding", "pathology"]


def _build_image_index(data_dir):
    """Build a mapping from filename to full path across all image subdirs."""
    index = {}
    for i in range(1, 13):
        subdir = os.path.join(data_dir, f"images_{i:03d}", "images")
        if os.path.isdir(subdir):
            for fname in os.listdir(subdir):
                if fname.endswith(".png"):
                    index[fname] = os.path.join(subdir, fname)
    return index


def _load_metadata(data_dir):
    """Load the CXR8 CSV and return filename-to-binary-label mapping."""
    csv_path = os.path.join(data_dir, "Data_Entry_2017_v2020.csv")
    df = pd.read_csv(csv_path)
    # Binary: 0 = No Finding, 1 = any pathology
    labels = {}
    for _, row in df.iterrows():
        fname = row["Image Index"]
        label = 0 if row["Finding Labels"] == "No Finding" else 1
        labels[fname] = label
    return labels


def _load_split(data_dir, split_file):
    """Load a split file (train_val_list.txt or test_list.txt)."""
    path = os.path.join(data_dir, split_file)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


class CXR8Dataset(Dataset):
    """NIH CXR8 dataset with binary labels (No Finding vs Pathology)."""

    def __init__(self, filenames, labels, image_index, transform=None):
        self.transform = transform
        self.samples = []
        self.targets = []

        for fname in filenames:
            if fname in image_index:
                self.samples.append(image_index[fname])
                self.targets.append(labels.get(fname, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# Alias for backward compatibility
ChestXrayDataset = CXR8Dataset


def get_chestxray_pretrain(
    data_dir="CXR8",
    image_size=224,
    batch_size=64,
    num_workers=0,
    pin_memory=True,
    max_samples=None,
):
    """CXR8 with SimCLR dual-view augmentation for pretraining (uses train split)."""
    transform = SimCLRTransform(image_size=image_size, strength=0.5, gaussian_blur_prob=0.5)

    image_index = _build_image_index(data_dir)
    train_files = _load_split(data_dir, "train_val_list.txt")
    if max_samples and max_samples < len(train_files):
        rng = np.random.RandomState(42)
        rng.shuffle(train_files)
        train_files = train_files[:max_samples]
    labels = _load_metadata(data_dir)
    dataset = CXR8Dataset(train_files, labels, image_index, transform=transform)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=num_workers > 0,
    )
    print(f"CXR8 pretrain: {len(dataset)} images, {len(loader)} batches")
    return loader


def get_chestxray_train(
    data_dir="CXR8",
    image_size=224,
    batch_size=64,
    num_workers=0,
    pin_memory=True,
    label_fraction=1.0,
    augment=True,
    seed=42,
    max_samples=None,
):
    """Labeled CXR8 training split (binary: No Finding vs Pathology)."""
    transform = get_finetune_transform(image_size) if augment else get_eval_transform(image_size)

    image_index = _build_image_index(data_dir)
    train_files = _load_split(data_dir, "train_val_list.txt")
    if max_samples and max_samples < len(train_files):
        rng = np.random.RandomState(seed)
        rng.shuffle(train_files)
        train_files = train_files[:max_samples]
    labels = _load_metadata(data_dir)
    dataset = CXR8Dataset(train_files, labels, image_index, transform=transform)

    if label_fraction < 1.0:
        dataset = _subsample_dataset(dataset, label_fraction, seed)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return loader


def get_chestxray_test(
    data_dir="CXR8",
    image_size=224,
    batch_size=64,
    num_workers=0,
    pin_memory=True,
):
    """CXR8 test split for evaluation (binary: No Finding vs Pathology)."""
    transform = get_eval_transform(image_size)

    image_index = _build_image_index(data_dir)
    test_files = _load_split(data_dir, "test_list.txt")
    labels = _load_metadata(data_dir)
    dataset = CXR8Dataset(test_files, labels, image_index, transform=transform)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return loader


def _subsample_dataset(dataset, fraction, seed=42):
    """Stratified subsampling."""
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
    print(f"Subsampled {len(indices)} / {len(dataset)} ({fraction*100:.1f}%)")
    return Subset(dataset, indices)

"""
Embedding Extractor
===================
Loads a trained encoder and computes L2-normalised embeddings for every
image in a dataset.  Saves the result to disk as a .npz file containing:

    embeddings  – float32 array of shape (N, D)
    labels      – int64 array of shape (N,)
    images      – uint8 array of shape (N, H, W, C)   (original pixels)

Usage:
    python interactive/embedding_extractor.py
    python interactive/embedding_extractor.py --encoder checkpoints/moco_encoder_best.pth
    python interactive/embedding_extractor.py --dataset stl10 --split test
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentations.simclr_augmentations import get_eval_transform
from models.resnet_encoder import ResNetEncoder
from utils.device import get_device

# ── dataset helpers ──────────────────────────────────────────────────────────

IMAGE_SIZES = {"cifar10": 32, "stl10": 96, "chestxray": 224}


def _load_dataset(name, split, data_dir):
    """Return a torchvision dataset in two forms:
    1. *raw* – with ToTensor only (for saving pixel data)
    2. *eval* – with the standard eval transform (for encoding)
    """
    image_size = IMAGE_SIZES.get(name, 32)
    eval_transform = get_eval_transform(image_size)
    raw_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
    ])

    is_train = split == "train"
    if name == "cifar10":
        ds_eval = torchvision.datasets.CIFAR10(
            root=data_dir, train=is_train, download=True, transform=eval_transform)
        ds_raw = torchvision.datasets.CIFAR10(
            root=data_dir, train=is_train, download=True, transform=raw_transform)
    elif name == "stl10":
        stl_split = "train" if is_train else "test"
        ds_eval = torchvision.datasets.STL10(
            root=data_dir, split=stl_split, download=True, transform=eval_transform)
        ds_raw = torchvision.datasets.STL10(
            root=data_dir, split=stl_split, download=True, transform=raw_transform)
    elif name == "chestxray":
        from datasets.chestxray_dataset import CXR8Dataset, _build_image_index, _load_metadata, _load_split
        cx_dir = os.path.join(os.path.dirname(data_dir), "CXR8")
        image_index = _build_image_index(cx_dir)
        labels = _load_metadata(cx_dir)
        split_file = "train_val_list.txt" if is_train else "test_list.txt"
        filenames = _load_split(cx_dir, split_file)
        ds_eval = CXR8Dataset(filenames, labels, image_index, transform=eval_transform)
        ds_raw = CXR8Dataset(filenames, labels, image_index, transform=raw_transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return ds_eval, ds_raw


# ── extraction ───────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder, dataset_eval, dataset_raw, device, batch_size=256):
    """Compute normalised embeddings, labels, and raw pixel arrays."""
    loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    loader_raw = torch.utils.data.DataLoader(
        dataset_raw, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    encoder.eval()
    embeddings, labels, images = [], [], []

    for (imgs_eval, labs), (imgs_raw, _) in tqdm(
            zip(loader_eval, loader_raw), total=len(loader_eval),
            desc="Extracting embeddings"):
        feats = encoder(imgs_eval.to(device, non_blocking=True))
        feats = F.normalize(feats, dim=1)
        embeddings.append(feats.cpu().numpy())
        labels.append(labs.numpy())
        # Store pixels as uint8 (N, H, W, C)
        pixel_arr = (imgs_raw.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        images.append(pixel_arr)

    return (np.concatenate(embeddings),
            np.concatenate(labels),
            np.concatenate(images))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract and save encoder embeddings")
    parser.add_argument("--encoder", default="checkpoints/simclr_encoder_best.pth",
                        help="Path to encoder checkpoint")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "stl10", "chestxray"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default: embeddings/{dataset}_{split}.npz)")
    args = parser.parse_args()

    device = get_device()

    # Load encoder
    encoder = ResNetEncoder(backbone=args.backbone).to(device)
    state = torch.load(args.encoder, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()
    print(f"Loaded encoder from {args.encoder}  ({args.backbone}, dim={encoder.feature_dim})")

    # Load dataset
    data_dir = os.path.join(PROJECT_ROOT, "data")
    ds_eval, ds_raw = _load_dataset(args.dataset, args.split, data_dir)
    print(f"Dataset: {args.dataset} ({args.split}), {len(ds_eval)} images")

    # Extract
    embeddings, labels, images = extract_embeddings(
        encoder, ds_eval, ds_raw, device, args.batch_size)

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, f"{args.dataset}_{args.split}.npz")
    np.savez_compressed(out_path,
                        embeddings=embeddings,
                        labels=labels,
                        images=images)
    print(f"Saved → {out_path}")
    print(f"  embeddings : {embeddings.shape}  (float32)")
    print(f"  labels     : {labels.shape}")
    print(f"  images     : {images.shape}  (uint8)")


if __name__ == "__main__":
    main()

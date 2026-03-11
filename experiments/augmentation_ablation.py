"""
Augmentation Ablation Study
============================
Evaluates the contribution of each SimCLR augmentation component
by training with different augmentation configurations and comparing
downstream linear probe accuracy.

Configurations tested:
  1. Full SimCLR augmentations (baseline)
  2. No ColorJitter
  3. No GaussianBlur
  4. No Grayscale
  5. No RandomResizedCrop (use simple resize)
  6. Crop + Flip only (minimal)

Usage:
    python experiments/augmentation_ablation.py --dataset cifar10 --epochs 50
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentations.simclr_augmentations import GaussianBlur, get_eval_transform
from models.simclr_model import SimCLR
from models.resnet_encoder import ResNetEncoder
from datasets.cifar10_dataset import get_cifar10_train, get_cifar10_test
from utils.losses import NTXentLoss
from utils.device import get_device
from utils.seed import set_seed


class AblationTransform:
    """Configurable SimCLR-style transform for ablation study."""

    def __init__(
        self,
        image_size=32,
        use_crop=True,
        use_flip=True,
        use_color_jitter=True,
        use_grayscale=True,
        use_gaussian_blur=True,
        strength=1.0,
    ):
        transforms = []

        if use_crop:
            transforms.append(T.RandomResizedCrop(size=image_size, scale=(0.08, 1.0)))
        else:
            transforms.append(T.Resize(image_size))
            transforms.append(T.CenterCrop(image_size))

        if use_flip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        if use_color_jitter:
            s = strength
            cj = T.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
            transforms.append(T.RandomApply([cj], p=0.8))

        if use_grayscale:
            transforms.append(T.RandomGrayscale(p=0.2))

        if use_gaussian_blur:
            transforms.append(T.RandomApply([GaussianBlur()], p=0.5))

        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = T.Compose(transforms)

    def __call__(self, x):
        return self.transform(x), self.transform(x)


ABLATION_CONFIGS = {
    "full": dict(use_crop=True, use_flip=True, use_color_jitter=True, use_grayscale=True, use_gaussian_blur=True),
    "no_color_jitter": dict(use_crop=True, use_flip=True, use_color_jitter=False, use_grayscale=True, use_gaussian_blur=True),
    "no_blur": dict(use_crop=True, use_flip=True, use_color_jitter=True, use_grayscale=True, use_gaussian_blur=False),
    "no_grayscale": dict(use_crop=True, use_flip=True, use_color_jitter=True, use_grayscale=False, use_gaussian_blur=True),
    "no_crop": dict(use_crop=False, use_flip=True, use_color_jitter=True, use_grayscale=True, use_gaussian_blur=True),
    "crop_flip_only": dict(use_crop=True, use_flip=True, use_color_jitter=False, use_grayscale=False, use_gaussian_blur=False),
}


def pretrain_simclr(cfg_name, aug_config, dataset_name, image_size, batch_size,
                    epochs, device, data_dir, backbone="resnet18", lr=3e-4, num_workers=0):
    """Pretrain SimCLR with a specific augmentation config."""
    print(f"\n  Pretraining with config: {cfg_name}")

    transform = AblationTransform(image_size=image_size, **aug_config)

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform,
    ) if dataset_name == "cifar10" else torchvision.datasets.STL10(
        root=data_dir, split="unlabeled", download=True, transform=transform,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True,
                        persistent_workers=num_workers > 0)

    model = SimCLR(backbone=backbone).to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler("cuda")

    for epoch in range(epochs):
        model.train()
        total_loss = n = 0
        for (x_i, x_j), _ in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                loss = criterion(z_i, z_j)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n += 1

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f}")

    return model.encoder


def linear_probe(encoder, dataset_name, image_size, batch_size, device, data_dir,
                 epochs=50, num_workers=0):
    """Linear probe evaluation of encoder."""
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    head = nn.Linear(encoder.output_dim, 10).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if dataset_name == "cifar10":
        train_loader = get_cifar10_train(data_dir, image_size, batch_size, augment=False)
        test_loader = get_cifar10_test(data_dir, image_size, batch_size)
    else:
        from datasets.stl10_dataset import get_stl10_train, get_stl10_test
        train_loader = get_stl10_train(data_dir, image_size, batch_size, augment=False)
        test_loader = get_stl10_test(data_dir, image_size, batch_size)

    best_acc = 0.0
    for epoch in range(epochs):
        head.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = encoder(images)
            loss = criterion(head(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            head.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    features = encoder(images)
                    preds = head(features).argmax(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100.0 * correct / total
            best_acc = max(best_acc, acc)

    return best_acc


def plot_ablation_results(results, save_path):
    """Generate bar chart of ablation results."""
    configs = list(results.keys())
    accuracies = [results[c] for c in configs]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ["#2ca02c" if c == "full" else "#1f77b4" for c in configs]
    bars = ax.bar(range(len(configs)), accuracies, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=10)
    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_title("Augmentation Ablation Study", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add baseline reference line
    if "full" in results:
        ax.axhline(y=results["full"], color="green", linestyle="--", alpha=0.5, label="Full augmentation")
        ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Augmentation Ablation Study")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "stl10"])
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to test (default: all)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    image_size = 32 if args.dataset == "cifar10" else 96
    configs_to_test = args.configs or list(ABLATION_CONFIGS.keys())

    results = {}

    for cfg_name in configs_to_test:
        if cfg_name not in ABLATION_CONFIGS:
            print(f"WARNING: Unknown config '{cfg_name}', skipping")
            continue

        print(f"\n{'='*60}")
        print(f"ABLATION: {cfg_name}")
        print(f"{'='*60}")

        encoder = pretrain_simclr(
            cfg_name, ABLATION_CONFIGS[cfg_name],
            dataset_name=args.dataset, image_size=image_size,
            batch_size=args.batch_size, epochs=args.pretrain_epochs,
            device=device, data_dir=data_dir, backbone=args.backbone,
            lr=args.lr, num_workers=args.num_workers,
        )

        acc = linear_probe(
            encoder, args.dataset, image_size, args.batch_size,
            device, data_dir, epochs=args.probe_epochs,
            num_workers=args.num_workers,
        )

        results[cfg_name] = acc
        print(f"  → {cfg_name}: {acc:.2f}%")

    # Save results
    results_file = os.path.join(results_dir, "augmentation_ablation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Plot
    plot_ablation_results(results, os.path.join(results_dir, "augmentation_ablation.png"))

    # Summary
    print(f"\n{'='*50}")
    print("AUGMENTATION ABLATION SUMMARY")
    print(f"{'='*50}")
    for cfg_name, acc in sorted(results.items(), key=lambda x: -x[1]):
        delta = ""
        if "full" in results and cfg_name != "full":
            diff = acc - results["full"]
            delta = f" ({diff:+.1f}%)"
        print(f"  {cfg_name:<20} {acc:.2f}%{delta}")


if __name__ == "__main__":
    main()

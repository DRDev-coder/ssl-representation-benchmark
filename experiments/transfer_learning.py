"""
Transfer Learning Experiment
============================
Evaluates cross-dataset transfer: pretrain on one dataset, evaluate on another.

Default: Pretrain on CIFAR-10, transfer to STL-10 (or vice versa).
Tests whether SSL representations generalize across domains.

Usage:
    python experiments/transfer_learning.py --source cifar10 --target stl10
    python experiments/transfer_learning.py --methods simclr moco byol
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.resnet_encoder import ResNetEncoder
from datasets.cifar10_dataset import get_cifar10_train, get_cifar10_test
from datasets.stl10_dataset import get_stl10_train, get_stl10_test
from utils.device import get_device
from utils.seed import set_seed


class TransferClassifier(nn.Module):
    """Frozen encoder + trainable linear head for transfer evaluation."""

    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.head(features)


def get_loaders(dataset_name, data_dir, image_size, batch_size, label_fraction=1.0, seed=42):
    if dataset_name == "cifar10":
        train_loader = get_cifar10_train(data_dir, image_size, batch_size,
                                         label_fraction=label_fraction, augment=True, seed=seed)
        test_loader = get_cifar10_test(data_dir, image_size, batch_size)
    else:
        train_loader = get_stl10_train(data_dir, image_size, batch_size,
                                       label_fraction=label_fraction, augment=True, seed=seed)
        test_loader = get_stl10_test(data_dir, image_size, batch_size)
    return train_loader, test_loader


def evaluate_transfer(encoder, target_dataset, data_dir, device,
                      epochs=50, batch_size=128, lr=1e-3, num_classes=10,
                      image_size=96, seed=42):
    """Train linear head on target dataset using frozen encoder."""
    model = TransferClassifier(encoder, num_classes=num_classes).to(device)
    train_loader, test_loader = get_loaders(
        target_dataset, data_dir, image_size, batch_size, seed=seed
    )

    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images).argmax(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100.0 * correct / total
            best_acc = max(best_acc, acc)
            print(f"    Epoch {epoch+1}/{epochs} | Target Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

    return best_acc


def plot_transfer_results(results, source, target, save_path):
    """Bar chart of transfer learning results."""
    methods = list(results.keys())
    accs = [results[m] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(range(len(methods)), accs,
                  color=colors[:len(methods)], edgecolor="black", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods], fontsize=12)
    ax.set_ylabel("Target Test Accuracy (%)", fontsize=12)
    ax.set_title(f"Transfer Learning: {source.upper()} → {target.upper()}", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Transfer plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning Experiment")
    parser.add_argument("--methods", nargs="+", default=["simclr", "moco", "byol"])
    parser.add_argument("--source", default="cifar10", choices=["cifar10", "stl10"])
    parser.add_argument("--target", default="stl10", choices=["cifar10", "stl10"])
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    checkpoint_dir = os.path.join(PROJECT_ROOT, args.checkpoint_dir)
    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(results_dir, exist_ok=True)

    target_image_size = 96 if args.target == "stl10" else 32
    num_classes = 10

    results = {}

    for method in args.methods:
        encoder_path = os.path.join(checkpoint_dir, f"{method}_encoder_best.pth")
        if not os.path.exists(encoder_path):
            print(f"WARNING: {encoder_path} not found, skipping {method}")
            continue

        print(f"\n{'='*60}")
        print(f"Transfer: {method.upper()} ({args.source} → {args.target})")
        print(f"{'='*60}")

        encoder = ResNetEncoder(backbone=args.backbone).to(device)
        encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu", weights_only=True)
        )
        encoder.eval()

        acc = evaluate_transfer(
            encoder, args.target, data_dir, device,
            epochs=args.epochs, batch_size=args.batch_size,
            num_classes=num_classes, image_size=target_image_size,
            seed=args.seed,
        )
        results[method] = acc
        print(f"  → {method.upper()}: {acc:.2f}%")

    # Save results
    results_file = os.path.join(results_dir, f"transfer_{args.source}_to_{args.target}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Plot
    if results:
        plot_transfer_results(
            results, args.source, args.target,
            os.path.join(results_dir, f"transfer_{args.source}_to_{args.target}.png"),
        )

    # Summary
    print(f"\n{'='*50}")
    print(f"TRANSFER LEARNING SUMMARY ({args.source} → {args.target})")
    print(f"{'='*50}")
    for method, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {method.upper():<15} {acc:.2f}%")


if __name__ == "__main__":
    main()

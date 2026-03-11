"""
Label Efficiency Experiment
===========================
Systematically evaluates SSL methods at multiple label fractions
and generates accuracy vs. label fraction curves.

Compares: SimCLR, MoCo v2, BYOL, Supervised baseline

Usage:
    python experiments/label_efficiency.py --methods simclr moco byol supervised
    python experiments/label_efficiency.py --dataset cifar10 --fractions 0.01 0.05 0.10 0.50 1.0
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.resnet_encoder import ResNetEncoder
from datasets.cifar10_dataset import get_cifar10_train, get_cifar10_test
from datasets.stl10_dataset import get_stl10_train, get_stl10_test
from utils.device import get_device
from utils.seed import set_seed


class LinearClassifier(nn.Module):
    """Encoder + linear head for label efficiency evaluation."""

    def __init__(self, encoder, num_classes=10, freeze=False):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


def train_and_evaluate(encoder_path, backbone, dataset, fraction, device,
                       epochs=50, batch_size=128, lr=1e-3, num_classes=10,
                       image_size=32, data_dir="data", freeze=True, seed=42):
    """Train a linear classifier on top of frozen/unfrozen encoder."""
    encoder = ResNetEncoder(backbone=backbone)

    if encoder_path is not None:
        encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu", weights_only=True)
        )

    model = LinearClassifier(encoder, num_classes=num_classes, freeze=freeze).to(device)

    if dataset == "cifar10":
        train_loader = get_cifar10_train(data_dir, image_size, batch_size,
                                         label_fraction=fraction, augment=True, seed=seed)
        test_loader = get_cifar10_test(data_dir, image_size, batch_size)
    else:
        train_loader = get_stl10_train(data_dir, image_size, batch_size,
                                       label_fraction=fraction, augment=True, seed=seed)
        test_loader = get_stl10_test(data_dir, image_size, batch_size)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
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
            print(f"    Epoch {epoch+1}/{epochs} | Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

    return best_acc


def plot_label_efficiency(results, save_path):
    """Generate label efficiency curve plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    markers = {"simclr": "o", "moco": "s", "byol": "D", "supervised": "^"}
    colors = {"simclr": "#1f77b4", "moco": "#ff7f0e", "byol": "#2ca02c", "supervised": "#d62728"}

    for method, data in results.items():
        fractions = sorted(data.keys())
        accs = [data[f] for f in fractions]
        pct_labels = [f * 100 for f in fractions]
        ax.plot(pct_labels, accs,
                marker=markers.get(method, "o"),
                color=colors.get(method, "#333333"),
                linewidth=2, markersize=8, label=method.upper())

    ax.set_xlabel("Labeled Data (%)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Label Efficiency: SSL Methods vs Supervised", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Label efficiency plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Label Efficiency Experiment")
    parser.add_argument("--methods", nargs="+", default=["simclr", "moco", "byol", "supervised"])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.50, 1.0])
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "stl10"])
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    checkpoint_dir = os.path.join(PROJECT_ROOT, args.checkpoint_dir)
    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(results_dir, exist_ok=True)

    image_size = 32 if args.dataset == "cifar10" else 96
    num_classes = 10

    all_results = {}

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Label Efficiency — {method.upper()}")
        print(f"{'='*60}")

        if method == "supervised":
            encoder_path = None  # Train from scratch
        else:
            encoder_path = os.path.join(checkpoint_dir, f"{method}_encoder_best.pth")
            if not os.path.exists(encoder_path):
                # Fallback to simclr naming convention
                if method == "simclr":
                    encoder_path = os.path.join(checkpoint_dir, "simclr_encoder_best.pth")
                if not os.path.exists(encoder_path):
                    print(f"WARNING: Encoder not found at {encoder_path}, skipping {method}")
                    continue

        method_results = {}
        freeze = method != "supervised"  # Freeze encoder for SSL, train from scratch for supervised

        for frac in args.fractions:
            print(f"\n  {method.upper()} @ {frac*100:.0f}% labels:")
            acc = train_and_evaluate(
                encoder_path=encoder_path,
                backbone=args.backbone,
                dataset=args.dataset,
                fraction=frac,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_classes=num_classes,
                image_size=image_size,
                data_dir=data_dir,
                freeze=freeze,
                seed=args.seed,
            )
            method_results[frac] = acc
            print(f"  → {method.upper()} @ {frac*100:.0f}%: {acc:.2f}%")

        all_results[method] = method_results

    # Save results
    results_file = os.path.join(results_dir, "label_efficiency_results.json")
    serializable = {m: {str(k): v for k, v in d.items()} for m, d in all_results.items()}
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Plot
    plot_label_efficiency(all_results, os.path.join(results_dir, "label_efficiency_curve.png"))

    # Print summary table
    print(f"\n{'='*70}")
    print("LABEL EFFICIENCY SUMMARY")
    print(f"{'='*70}")
    header = f"{'Method':<15}" + "".join(f"  {f*100:>5.0f}%" for f in args.fractions)
    print(header)
    print("-" * len(header))
    for method, data in all_results.items():
        row = f"{method.upper():<15}"
        for frac in args.fractions:
            row += f"  {data.get(frac, 0):>5.1f}%"
        print(row)


if __name__ == "__main__":
    main()

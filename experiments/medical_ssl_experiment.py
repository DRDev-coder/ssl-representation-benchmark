"""
Medical SSL Experiment
======================
End-to-end experiment: SSL pretraining on chest X-rays,
then fine-tuning with limited labels for disease classification.

Demonstrates the practical value of SSL in medical imaging where
labeled data is scarce and expensive.

Usage:
    python experiments/medical_ssl_experiment.py --data_dir data/chestxray
    python experiments/medical_ssl_experiment.py --method simclr --pretrain_epochs 50
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

from models.ssl_methods import SimCLRMethod, MoCoV2, BYOL
from models.resnet_encoder import ResNetEncoder
from datasets.chestxray_dataset import (
    get_chestxray_pretrain, get_chestxray_train, get_chestxray_test,
)
from utils.device import get_device
from utils.seed import set_seed
from utils.losses import NTXentLoss
from mlops.mlflow_logger import MLflowLogger

SSL_REGISTRY = {"simclr": SimCLRMethod, "moco": MoCoV2, "byol": BYOL}


class MedicalClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2, freeze=True):
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


def pretrain_ssl(method_name, data_dir, device, epochs=50, batch_size=64,
                 lr=3e-4, backbone="resnet18", image_size=224, num_workers=0,
                 max_samples=None, mlflow_logger=None):
    """Pretrain SSL on chest X-rays."""
    print(f"\n  Pretraining {method_name.upper()} on medical data...")

    train_loader = get_chestxray_pretrain(
        data_dir=data_dir, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers,
        max_samples=max_samples,
    )

    cls = SSL_REGISTRY[method_name]
    kwargs = dict(backbone=backbone, projection_hidden_dim=2048, projection_output_dim=128)
    if method_name == "simclr":
        kwargs["temperature"] = 0.5
    elif method_name == "moco":
        kwargs["temperature"] = 0.07
        kwargs["queue_size"] = min(len(train_loader.dataset), 4096)
    model = cls(**kwargs).to(device)

    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler("cuda")

    for epoch in range(epochs):
        model.train()
        total_loss = n = 0
        for (x_i, x_j), _ in train_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                loss = model(x_i, x_j)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / n
        if mlflow_logger:
            mlflow_logger.log_metric("pretrain_loss", avg_loss, step=epoch + 1)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    return model.encoder


def finetune_and_evaluate(encoder, data_dir, device, num_classes=2,
                          label_fraction=1.0, epochs=30, batch_size=64,
                          lr=1e-3, image_size=224, freeze=True, num_workers=0, seed=42,
                          max_samples=None):
    """Fine-tune and evaluate on labeled medical data."""
    model = MedicalClassifier(encoder, num_classes=num_classes, freeze=freeze).to(device)

    train_loader = get_chestxray_train(
        data_dir=data_dir, image_size=image_size, batch_size=batch_size,
        label_fraction=label_fraction, augment=True, seed=seed, num_workers=num_workers,
        max_samples=max_samples,
    )
    test_loader = get_chestxray_test(
        data_dir=data_dir, image_size=image_size, batch_size=batch_size,
        num_workers=num_workers,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )
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

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images).argmax(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)
            if total > 0:
                acc = 100.0 * correct / total
                best_acc = max(best_acc, acc)
                print(f"    Epoch {epoch+1}/{epochs} | Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

    return best_acc


def plot_medical_results(results, save_path):
    """Plot medical experiment results."""
    methods = list(results.keys())
    fractions = sorted(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"simclr": "o", "moco": "s", "byol": "D", "supervised": "^"}
    colors = {"simclr": "#1f77b4", "moco": "#ff7f0e", "byol": "#2ca02c", "supervised": "#d62728"}

    for method in methods:
        pct = [f * 100 for f in fractions]
        accs = [results[method].get(f, 0) for f in fractions]
        ax.plot(pct, accs, marker=markers.get(method, "o"),
                color=colors.get(method, "#333"), linewidth=2, markersize=8,
                label=method.upper())

    ax.set_xlabel("Labeled Data (%)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Medical SSL: Chest X-Ray Classification", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Medical results plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Medical SSL Experiment")
    parser.add_argument("--methods", nargs="+", default=["simclr", "moco", "byol"])
    parser.add_argument("--data_dir", default="CXR8")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=30)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 1.0])
    parser.add_argument("--supervised", action="store_true", default=True,
                        help="Also run supervised baseline")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap training images for faster runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_results = {}

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"MEDICAL SSL EXPERIMENT — {method.upper()}")
        print(f"{'='*60}")

        logger = MLflowLogger(experiment_name="medical_ssl")
        logger.start_run(run_name=f"{method}_pretrain_ep{args.pretrain_epochs}")
        logger.log_params({
            "method": method,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "max_samples": args.max_samples or "all",
            "backbone": args.backbone,
        })

        # Pretrain
        encoder = pretrain_ssl(
            method, data_dir, device, epochs=args.pretrain_epochs,
            batch_size=args.batch_size, backbone=args.backbone,
            image_size=args.image_size, num_workers=args.num_workers,
            max_samples=args.max_samples, mlflow_logger=logger,
        )

        # Save encoder
        encoder_path = os.path.join(checkpoint_dir, f"medical_{method}_encoder.pth")
        torch.save(encoder.state_dict(), encoder_path)

        # Evaluate at different label fractions
        method_results = {}
        for frac in args.fractions:
            print(f"\n  Fine-tuning @ {frac*100:.0f}% labels:")
            acc = finetune_and_evaluate(
                encoder, data_dir, device, num_classes=args.num_classes,
                label_fraction=frac, epochs=args.finetune_epochs,
                batch_size=args.batch_size, image_size=args.image_size,
                num_workers=args.num_workers, seed=args.seed,
                max_samples=args.max_samples,
            )
            method_results[frac] = acc
            logger.log_metric(f"acc_frac_{int(frac*100)}pct", acc)
            print(f"  → {method.upper()} @ {frac*100:.0f}%: {acc:.2f}%")

        logger.end_run()
        all_results[method] = method_results

    # Supervised baseline (train from scratch, no pretraining)
    if args.supervised:
        print(f"\n{'='*60}")
        print("MEDICAL SSL EXPERIMENT — SUPERVISED (from scratch)")
        print(f"{'='*60}")
        logger_sup = MLflowLogger(experiment_name="medical_ssl")
        logger_sup.start_run(run_name="supervised_baseline")
        logger_sup.log_params({
            "method": "supervised",
            "finetune_epochs": args.finetune_epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "max_samples": args.max_samples or "all",
            "backbone": args.backbone,
        })
        encoder_sup = ResNetEncoder(backbone=args.backbone).to(device)
        sup_results = {}
        for frac in args.fractions:
            print(f"\n  Supervised from scratch @ {frac*100:.0f}% labels:")
            acc = finetune_and_evaluate(
                encoder_sup, data_dir, device, num_classes=args.num_classes,
                label_fraction=frac, epochs=args.finetune_epochs,
                batch_size=args.batch_size, image_size=args.image_size,
                freeze=False, num_workers=args.num_workers, seed=args.seed,
                max_samples=args.max_samples,
            )
            sup_results[frac] = acc
            logger_sup.log_metric(f"acc_frac_{int(frac*100)}pct", acc)
            print(f"  → SUPERVISED @ {frac*100:.0f}%: {acc:.2f}%")
        logger_sup.end_run()
        all_results["supervised"] = sup_results

    # Save
    results_file = os.path.join(results_dir, "medical_ssl_results.json")
    serializable = {m: {str(k): v for k, v in d.items()} for m, d in all_results.items()}
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)

    # Plot
    if all_results:
        plot_medical_results(all_results, os.path.join(results_dir, "medical_label_efficiency_curve.png"))

    # Summary
    print(f"\n{'='*60}")
    print("MEDICAL SSL EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for method, data in all_results.items():
        print(f"\n  {method.upper()}:")
        for frac, acc in sorted(data.items()):
            print(f"    {frac*100:>5.0f}% labels → {acc:.2f}%")


if __name__ == "__main__":
    main()

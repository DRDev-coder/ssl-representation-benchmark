"""
Supervised Baseline Training
=============================
Trains a ResNet from scratch with full supervision.
Used as the baseline to compare against SimCLR representations.

Runs at the same label fractions (1%, 10%, 100%) as fine-tuning
to produce the comparative label efficiency curve.

Usage:
    # Default (uses training=simclr config; override to supervised)
    python training/train_supervised.py training=supervised

    # Override dataset/model
    python training/train_supervised.py training=supervised dataset=cifar10

    # Custom fractions
    python training/train_supervised.py training=supervised training.label_fractions=[0.01,0.10,1.0]
"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torch.amp import autocast, GradScaler

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.stl10_dataset import get_stl10_train, get_stl10_test
from datasets.cifar10_dataset import get_cifar10_train, get_cifar10_test
from utils.device import get_device
from utils.seed import set_seed
from mlops.mlflow_logger import MLflowLogger
from mlops.mlflow_tracker import ExperimentTracker


def _resolve_path(path: str) -> str:
    """Resolve a relative path against project root."""
    if not os.path.isabs(path):
        return os.path.join(PROJECT_ROOT, path)
    return path


def build_supervised_model(backbone: str = "resnet18", num_classes: int = 10):
    """Build a supervised ResNet model from scratch."""
    if backbone == "resnet34":
        model = models.resnet34(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_supervised_at_fraction(
    cfg, label_fraction, device, checkpoint_dir, results_dir, logger=None,
):
    """
    Train supervised baseline at a single label fraction.

    Args:
        cfg: Hydra DictConfig.
        label_fraction: Fraction of labels to use (e.g. 0.01, 0.10, 1.0).
        device: torch.device.
        checkpoint_dir: Absolute path for saving checkpoints.
        results_dir: Absolute path for saving results.
        logger: MLflowLogger instance (or None).

    Returns:
        Test accuracy (%).
    """
    print(f"\n--- Supervised training with {label_fraction*100:.1f}% labels ---")

    model = build_supervised_model(
        backbone=cfg.model.backbone,
        num_classes=cfg.dataset.num_classes,
    ).to(device)

    data_dir = _resolve_path(cfg.paths.data_dir)

    # Data
    if cfg.dataset.name == "stl10":
        train_loader = get_stl10_train(
            data_dir, cfg.dataset.image_size, cfg.batch_size,
            label_fraction=label_fraction, augment=True, seed=cfg.seed,
        )
        test_loader = get_stl10_test(
            data_dir, cfg.dataset.image_size, cfg.batch_size,
        )
    else:
        train_loader = get_cifar10_train(
            data_dir, cfg.dataset.image_size, cfg.batch_size,
            label_fraction=label_fraction, augment=True, seed=cfg.seed,
        )
        test_loader = get_cifar10_test(
            data_dir, cfg.dataset.image_size, cfg.batch_size,
        )

    if cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.get("momentum", 0.9),
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    best_acc = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda", enabled=cfg.use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            test_acc = 100.0 * correct / total
            if test_acc > best_acc:
                best_acc = test_acc

            avg_loss = total_loss / len(train_loader)
            print(
                f"  Epoch [{epoch+1}/{cfg.epochs}] | "
                f"Loss: {avg_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)"
            )

    print(f"  => Supervised best with {label_fraction*100:.1f}% labels: {best_acc:.2f}%")

    # Save supervised encoder (for t-SNE comparison)
    if label_fraction == 1.0:
        save_path = os.path.join(checkpoint_dir, "supervised_encoder_best.pth")
        state = {k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}
        torch.save(state, save_path)
        print(f"  Supervised encoder saved to {save_path}")

    if logger is not None:
        logger.log_metric(
            f"supervised/accuracy_{label_fraction*100:.0f}pct", best_acc
        )

    return best_acc


def train_supervised(cfg: DictConfig):
    """
    Run supervised baseline at multiple label fractions.

    Args:
        cfg: Hydra DictConfig.

    Returns:
        Dict mapping label fraction -> accuracy.
    """
    set_seed(cfg.seed)
    device = get_device()

    checkpoint_dir = _resolve_path(cfg.paths.checkpoint_dir)
    results_dir = _resolve_path(cfg.paths.results_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    label_fractions = list(cfg.training.get("label_fractions", [0.01, 0.10, 1.0]))

    print("\n" + "=" * 60)
    print("SUPERVISED BASELINE TRAINING")
    print("=" * 60)
    print(f"Dataset:      {cfg.dataset.name}")
    print(f"Backbone:     {cfg.model.backbone}")
    print(f"Fractions:    {label_fractions}")
    print(f"Epochs:       {cfg.epochs}")
    print(f"Batch size:   {cfg.batch_size}")
    print(f"Optimizer:    {cfg.optimizer.name} (lr={cfg.optimizer.lr})")
    print("=" * 60)

    logger = None
    if cfg.mlflow.enabled:
        logger = MLflowLogger(experiment_name=cfg.mlflow.experiment_name)
        logger.start_run(run_name=f"supervised_{cfg.dataset.name}")
        logger.log_hydra_config(cfg)
        # Log dataset info
        tracker = ExperimentTracker(cfg)
        tracker._run = logger._run
        tracker.active = logger.active
        tracker.log_dataset_info()

    results = {}
    for fraction in label_fractions:
        acc = train_supervised_at_fraction(
            cfg, fraction, device, checkpoint_dir, results_dir, logger
        )
        results[fraction] = acc

    # Print final comparison
    print("\n" + "=" * 60)
    print("LABEL EFFICIENCY RESULTS (Supervised Baseline)")
    print("=" * 60)
    for frac, acc in sorted(results.items()):
        print(f"  {frac*100:6.1f}% labels: {acc:.2f}% accuracy")

    results_path = os.path.join(results_dir, "supervised_results.json")
    with open(results_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if logger is not None:
        logger.log_artifact(results_path)
        # Log supervised encoder checkpoint
        enc_path = os.path.join(checkpoint_dir, "supervised_encoder_best.pth")
        if os.path.isfile(enc_path):
            logger.log_artifact(enc_path, artifact_path="model_checkpoints")
        logger.end_run()

    return results


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Hydra entry point for supervised baseline training."""
    print(OmegaConf.to_yaml(cfg))
    train_supervised(cfg)


if __name__ == "__main__":
    main()

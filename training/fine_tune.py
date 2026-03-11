"""
Semi-Supervised Fine-Tuning
============================
Fine-tunes the SimCLR pretrained encoder with limited labeled data.

Evaluates at multiple label fractions (1%, 10%, 100%) to produce
the label efficiency curve — a key deliverable of the project.

Protocol:
  1. Load pretrained SimCLR encoder
  2. Add a classification head
  3. Fine-tune the full model (encoder + head) with limited labels
  4. Report accuracy at each label fraction
  5. Compare against supervised baseline (from train_supervised.py)

Usage:
    # Default (override training config to finetune)
    python training/fine_tune.py training=finetune

    # With custom fractions
    python training/fine_tune.py training=finetune training.label_fractions=[0.01,0.10,1.0]

    # Freeze encoder
    python training/fine_tune.py training=finetune training.freeze_encoder=true
"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.resnet_encoder import ResNetEncoder
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


class FineTuneClassifier(nn.Module):
    """
    Encoder + classification head for fine-tuning.
    Optionally freeze encoder weights.
    """

    def __init__(self, encoder: ResNetEncoder, num_classes: int = 10, freeze: bool = False):
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


def train_and_evaluate(cfg, label_fraction, device, logger=None):
    """
    Train and evaluate at a single label fraction.

    Args:
        cfg: Hydra DictConfig.
        label_fraction: Fraction of labels to use (e.g. 0.01, 0.10, 1.0).
        device: torch.device.
        logger: MLflowLogger instance (or None).

    Returns:
        Test accuracy (%).
    """
    print(f"\n--- Fine-tuning with {label_fraction*100:.1f}% labels ---")

    # Load pretrained encoder
    encoder_path = _resolve_path(cfg.training.get("encoder_path", "checkpoints/simclr_encoder_best.pth"))
    encoder = ResNetEncoder(backbone=cfg.model.backbone)
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True)
    )

    freeze_encoder = cfg.training.get("freeze_encoder", False)
    model = FineTuneClassifier(
        encoder=encoder,
        num_classes=cfg.dataset.num_classes,
        freeze=freeze_encoder,
    ).to(device)

    data_dir = _resolve_path(cfg.paths.data_dir)

    # Data loaders
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

    # Optimizer — use different learning rates for encoder vs head
    encoder_lr_scale = cfg.training.get("encoder_lr_scale", 0.1)
    if not freeze_encoder:
        param_groups = [
            {"params": model.encoder.parameters(), "lr": cfg.optimizer.lr * encoder_lr_scale},
            {"params": model.head.parameters(), "lr": cfg.optimizer.lr},
        ]
    else:
        param_groups = [
            {"params": model.head.parameters(), "lr": cfg.optimizer.lr},
        ]

    optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    # Training
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

        # Evaluate every 10 epochs or at the end
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

    print(f"  => Best accuracy with {label_fraction*100:.1f}% labels: {best_acc:.2f}%")

    if logger is not None:
        logger.log_metric(
            f"finetune/accuracy_{label_fraction*100:.0f}pct", best_acc
        )

    return best_acc


def train_finetune(cfg: DictConfig):
    """
    Run fine-tuning at multiple label fractions using Hydra config.

    Args:
        cfg: Hydra DictConfig.

    Returns:
        Dict mapping label fraction -> accuracy.
    """
    set_seed(cfg.seed)
    device = get_device()

    results_dir = _resolve_path(cfg.paths.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    label_fractions = list(cfg.training.get("label_fractions", [0.01, 0.10, 1.0]))
    encoder_path = _resolve_path(cfg.training.get("encoder_path", "checkpoints/simclr_encoder_best.pth"))

    print("\n" + "=" * 60)
    print("SEMI-SUPERVISED FINE-TUNING")
    print("=" * 60)
    print(f"Dataset:      {cfg.dataset.name}")
    print(f"Encoder:      {encoder_path}")
    print(f"Fractions:    {label_fractions}")
    print(f"Freeze enc:   {cfg.training.get('freeze_encoder', False)}")
    print(f"Batch size:   {cfg.batch_size}")
    print(f"Epochs:       {cfg.epochs}")
    print(f"Optimizer:    {cfg.optimizer.name} (lr={cfg.optimizer.lr})")
    print("=" * 60)

    logger = None
    if cfg.mlflow.enabled:
        logger = MLflowLogger(experiment_name=cfg.mlflow.experiment_name)
        logger.start_run(run_name=f"finetune_{cfg.dataset.name}")
        logger.log_hydra_config(cfg)
        # Log dataset info
        tracker = ExperimentTracker(cfg)
        tracker._run = logger._run
        tracker.active = logger.active
        tracker.log_dataset_info()

    # Run fine-tuning at each label fraction
    results = {}
    for fraction in label_fractions:
        acc = train_and_evaluate(cfg, fraction, device, logger)
        results[fraction] = acc

    # Save results
    print("\n" + "=" * 60)
    print("LABEL EFFICIENCY RESULTS (SimCLR Fine-Tune)")
    print("=" * 60)
    for frac, acc in sorted(results.items()):
        print(f"  {frac*100:6.1f}% labels: {acc:.2f}% accuracy")

    results_path = os.path.join(results_dir, "finetune_results.json")
    with open(results_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if logger is not None:
        logger.log_artifact(results_path)
        # Log encoder checkpoint used
        enc_path = _resolve_path(cfg.training.get("encoder_path", "checkpoints/simclr_encoder_best.pth"))
        if os.path.isfile(enc_path):
            logger.log_artifact(enc_path, artifact_path="model_checkpoints")
        logger.end_run()

    return results


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Hydra entry point for semi-supervised fine-tuning."""
    print(OmegaConf.to_yaml(cfg))
    train_finetune(cfg)


if __name__ == "__main__":
    main()

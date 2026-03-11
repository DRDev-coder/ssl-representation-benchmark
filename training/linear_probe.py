"""
Linear Probe Evaluation
=======================
Evaluates the quality of SimCLR representations by training a linear
classifier on top of frozen encoder features.

This is the standard evaluation protocol from the SimCLR paper:
  1. Load pretrained encoder
  2. Freeze all encoder weights
  3. Train a single linear layer (FC) for classification
  4. Report test accuracy

Higher linear probe accuracy = better learned representations.

Usage:
    # Default (uses training=simclr config; override to linear_eval)
    python training/linear_probe.py training=linear_eval

    # Specify custom encoder path
    python training/linear_probe.py training=linear_eval training.encoder_path=checkpoints/my_encoder.pth

    # Switch dataset
    python training/linear_probe.py training=linear_eval dataset=cifar10
"""

import os
import sys
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


def _get_train_test_loaders(cfg, data_dir):
    """Build train/test dataloaders for any supported dataset."""
    if cfg.dataset.name == "stl10":
        train = get_stl10_train(data_dir, cfg.dataset.image_size, cfg.batch_size, augment=False)
        test = get_stl10_test(data_dir, cfg.dataset.image_size, cfg.batch_size)
    elif cfg.dataset.name == "chestxray":
        from datasets.chestxray_dataset import get_chestxray_train, get_chestxray_test
        cx_dir = os.path.join(PROJECT_ROOT, "CXR8")
        train = get_chestxray_train(cx_dir, cfg.dataset.image_size, cfg.batch_size, augment=False)
        test = get_chestxray_test(cx_dir, cfg.dataset.image_size, cfg.batch_size)
    else:
        train = get_cifar10_train(data_dir, cfg.dataset.image_size, cfg.batch_size, augment=False)
        test = get_cifar10_test(data_dir, cfg.dataset.image_size, cfg.batch_size)
    return train, test
from utils.seed import set_seed
from utils.metrics import accuracy
from mlops.mlflow_logger import MLflowLogger
from mlops.mlflow_tracker import ExperimentTracker


def _resolve_path(path: str) -> str:
    """Resolve a relative path against project root."""
    if not os.path.isabs(path):
        return os.path.join(PROJECT_ROOT, path)
    return path


class LinearClassifier(nn.Module):
    """
    Linear classifier for evaluating encoder representations.

    Architecture: Frozen encoder -> Linear layer -> Softmax
    """

    def __init__(self, encoder: ResNetEncoder, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, num_classes)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)


def evaluate(model, test_loader, device):
    """Evaluate model on test set. Returns accuracy."""
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

    return 100.0 * correct / total


def train_linear_probe(cfg: DictConfig):
    """
    Main linear evaluation logic using Hydra config.

    Args:
        cfg: Hydra DictConfig.

    Returns:
        Best test accuracy (%).
    """
    set_seed(cfg.seed)
    device = get_device()

    # Resolve paths
    encoder_path = _resolve_path(cfg.training.get("encoder_path", "checkpoints/simclr_encoder_best.pth"))
    data_dir = _resolve_path(cfg.paths.data_dir)

    print("\n" + "=" * 60)
    print("LINEAR PROBE EVALUATION")
    print("=" * 60)
    print(f"Dataset:      {cfg.dataset.name}")
    print(f"Encoder:      {encoder_path}")
    print(f"Epochs:       {cfg.epochs}")
    print(f"Batch size:   {cfg.batch_size}")
    print(f"Optimizer:    {cfg.optimizer.name} (lr={cfg.optimizer.lr})")
    print("=" * 60 + "\n")

    # ── Load pretrained encoder ──────────────────────────────────
    encoder = ResNetEncoder(backbone=cfg.model.backbone)
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True)
    )
    print(f"Loaded encoder from {encoder_path}")

    # ── Build linear classifier ──────────────────────────────────
    model = LinearClassifier(encoder, num_classes=cfg.dataset.num_classes).to(device)

    # ── Data ─────────────────────────────────────────────────────
    train_loader, test_loader = _get_train_test_loaders(cfg, data_dir)

    # ── Optimizer ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),  # Only train the linear layer
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    # ── MLflow ───────────────────────────────────────────────────
    logger = None
    if cfg.mlflow.enabled:
        logger = MLflowLogger(experiment_name=cfg.mlflow.experiment_name)
        logger.start_run(run_name=f"linear_eval_{cfg.dataset.name}")
        logger.log_hydra_config(cfg)
        # Log dataset statistics
        tracker = ExperimentTracker(cfg)
        tracker._run = logger._run
        tracker.active = logger.active
        tracker.log_dataset_info(
            num_samples=len(train_loader.dataset),
            extra={"eval_samples": len(test_loader.dataset)},
        )

    # ── Training ─────────────────────────────────────────────────
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

        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{cfg.epochs}] | "
                f"Loss: {avg_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)"
            )

        if logger is not None:
            logger.log_metric("linear_eval/loss", avg_loss, step=epoch)
            logger.log_metric("linear_eval/test_accuracy", test_acc, step=epoch)

    print(f"\nFinal Linear Probe Accuracy: {best_acc:.2f}%")

    if logger is not None:
        logger.log_metric("linear_eval/best_accuracy", best_acc)
        logger.end_run()

    return best_acc


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Hydra entry point for linear probe evaluation."""
    print(OmegaConf.to_yaml(cfg))
    train_linear_probe(cfg)


if __name__ == "__main__":
    main()

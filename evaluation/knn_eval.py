"""
kNN Evaluation
==============
Evaluates encoder quality using k-Nearest Neighbors classification
on the learned representations — no additional training required.

This is a quick, non-parametric way to assess representation quality:
  1. Extract features from train set using the frozen encoder
  2. For each test sample, find k nearest neighbors in the train features
  3. Classify by weighted voting
  4. Report accuracy

Higher kNN accuracy = more separable, useful features.

Usage:
    python -m evaluation.knn_eval [--encoder_path checkpoints/simclr_encoder_best.pth]
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.simclr_config import EvalConfig
from models.resnet_encoder import ResNetEncoder
from datasets.stl10_dataset import get_stl10_train, get_stl10_test
from datasets.cifar10_dataset import get_cifar10_train, get_cifar10_test
from utils.device import get_device
from utils.seed import set_seed
from mlops.mlflow_logger import MLflowLogger
from mlops.mlflow_tracker import ExperimentTracker


@torch.no_grad()
def extract_features(encoder, loader, device):
    """
    Extract features and labels from a data loader using the encoder.

    Returns:
        features: Tensor of shape (N, D) — L2-normalized feature vectors.
        labels: Tensor of shape (N,) — ground truth labels.
    """
    encoder.eval()
    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        features = encoder(images)
        features = F.normalize(features, dim=1)  # L2 normalize
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


@torch.no_grad()
def knn_evaluate(
    encoder,
    train_loader,
    test_loader,
    device,
    k: int = 200,
    temperature: float = 0.5,
    num_classes: int = 10,
):
    """
    Weighted kNN evaluation on encoder features.

    Uses cosine similarity and temperature-weighted voting:
      weight = exp(similarity / temperature)

    Args:
        encoder: Frozen feature encoder.
        train_loader: DataLoader for training features (memory bank).
        test_loader: DataLoader for test features.
        device: Torch device.
        k: Number of nearest neighbors.
        temperature: Temperature for weighting.
        num_classes: Number of classes.

    Returns:
        kNN classification accuracy (%).
    """
    print("Extracting training features...")
    train_features, train_labels = extract_features(encoder, train_loader, device)

    print("Extracting test features...")
    test_features, test_labels = extract_features(encoder, test_loader, device)

    print(f"Train features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")
    print(f"Running kNN with k={k}...")

    # Move to GPU for faster similarity computation
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    total = 0
    correct = 0
    correct_top5 = 0

    # Process in batches to avoid OOM
    batch_size = 256
    for start_idx in range(0, len(test_features), batch_size):
        end_idx = min(start_idx + batch_size, len(test_features))
        test_batch = test_features[start_idx:end_idx].to(device)
        test_batch_labels = test_labels[start_idx:end_idx].to(device)

        # Cosine similarity: (batch, num_train)
        similarity = torch.mm(test_batch, train_features.T)

        # Top-k most similar training samples
        top_k_sim, top_k_idx = similarity.topk(k, dim=1)

        # Get labels of top-k neighbors
        top_k_labels = train_labels[top_k_idx]  # (batch, k)

        # Temperature-weighted voting
        weights = torch.exp(top_k_sim / temperature)  # (batch, k)

        # Aggregate votes per class
        one_hot = F.one_hot(top_k_labels, num_classes).float()  # (batch, k, C)
        weighted_votes = (weights.unsqueeze(-1) * one_hot).sum(dim=1)  # (batch, C)

        # Predict — Top-1
        predictions = weighted_votes.argmax(dim=1)
        correct += predictions.eq(test_batch_labels).sum().item()

        # Top-5
        _, top5_preds = weighted_votes.topk(min(5, num_classes), dim=1)
        top5_match = top5_preds.eq(test_batch_labels.unsqueeze(1)).any(dim=1)
        correct_top5 += top5_match.sum().item()

        total += test_batch_labels.size(0)

    top1_accuracy = 100.0 * correct / total
    top5_accuracy = 100.0 * correct_top5 / total
    print(f"kNN Top-1 Accuracy (k={k}): {top1_accuracy:.2f}%")
    print(f"kNN Top-5 Accuracy (k={k}): {top5_accuracy:.2f}%")
    return top1_accuracy, top5_accuracy


def main(args=None):
    """Run kNN evaluation."""
    parser = argparse.ArgumentParser(description="kNN Evaluation")
    parser.add_argument("--dataset", type=str, default="stl10", choices=["stl10", "cifar10"])
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--no_mlflow", action="store_true")
    args = parser.parse_args(args)

    config = EvalConfig(dataset=args.dataset)
    if args.dataset == "cifar10":
        config.image_size = 32

    if args.encoder_path is not None:
        config.encoder_path = args.encoder_path

    set_seed(config.seed)
    device = get_device()

    print("\n" + "=" * 60)
    print("kNN EVALUATION")
    print("=" * 60)

    # Load encoder
    encoder = ResNetEncoder(backbone="resnet18").to(device)
    encoder.load_state_dict(
        torch.load(config.encoder_path, map_location="cpu", weights_only=True)
    )
    encoder.eval()

    # Data
    if config.dataset == "stl10":
        train_loader = get_stl10_train(
            config.data_dir, config.image_size, batch_size=256, augment=False,
        )
        test_loader = get_stl10_test(
            config.data_dir, config.image_size, batch_size=256,
        )
    else:
        train_loader = get_cifar10_train(
            config.data_dir, config.image_size, batch_size=256, augment=False,
        )
        test_loader = get_cifar10_test(
            config.data_dir, config.image_size, batch_size=256,
        )

    # Run kNN
    knn_top1, knn_top5 = knn_evaluate(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        k=args.k,
        temperature=config.knn_temperature,
        num_classes=config.num_classes,
    )

    # Log to MLflow
    if not args.no_mlflow:
        logger = MLflowLogger(experiment_name="knn_evaluation")
        logger.start_run(run_name=f"knn_{config.dataset}")
        logger.log_params({
            "encoder_path": config.encoder_path,
            "k": args.k,
            "dataset": config.dataset,
            "image_size": config.image_size,
            "num_classes": config.num_classes,
        })
        logger.log_metric("knn/top1_accuracy", knn_top1)
        logger.log_metric("knn/top5_accuracy", knn_top5)
        logger.end_run()

    return knn_top1, knn_top5


if __name__ == "__main__":
    main()

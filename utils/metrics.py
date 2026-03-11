"""
Metrics Utilities
=================
Common evaluation metrics for classification tasks.
"""

import torch
import numpy as np
from typing import Tuple


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute top-1 classification accuracy.

    Args:
        output: Model predictions (logits), shape (N, C).
        target: Ground truth labels, shape (N,).

    Returns:
        Accuracy as a float in [0, 100].
    """
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        total = target.size(0)
        return 100.0 * correct / total


def top_k_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute top-k classification accuracy.

    Args:
        output: Model predictions (logits), shape (N, C).
        target: Ground truth labels, shape (N,).
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy as a float in [0, 100].
    """
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        target_expanded = target.unsqueeze(1).expand_as(pred)
        correct = pred.eq(target_expanded).any(dim=1).sum().item()
        total = target.size(0)
        return 100.0 * correct / total


def compute_label_efficiency_curve(
    results: dict,
) -> Tuple[list, list, list]:
    """
    Compute label efficiency curve data from fine-tuning results.

    Args:
        results: Dict mapping label_fraction -> {"ssl_acc": float, "supervised_acc": float}.

    Returns:
        (fractions, ssl_accs, supervised_accs) — lists for plotting.
    """
    fractions = sorted(results.keys())
    ssl_accs = [results[f]["ssl_acc"] for f in fractions]
    supervised_accs = [results[f]["supervised_acc"] for f in fractions]
    return fractions, ssl_accs, supervised_accs

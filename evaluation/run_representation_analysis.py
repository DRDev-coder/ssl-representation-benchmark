"""
Representation Analysis Runner
================================
Runs all representation quality metrics (alignment, uniformity, CKA,
collapse detection) across all available SSL encoders and generates
publication-quality plots.

Usage:
    python evaluation/run_representation_analysis.py
    python evaluation/run_representation_analysis.py --dataset cifar10
    python evaluation/run_representation_analysis.py --methods simclr moco byol supervised
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.resnet_encoder import ResNetEncoder
from utils.device import get_device
from utils.seed import set_seed
from mlops.mlflow_tracker import ExperimentTracker
from evaluation.representation_metrics import (
    compute_alignment,
    compute_uniformity,
    compute_cka_matrix,
    compute_embedding_variance,
    compute_covariance_rank,
)


# ── Style palette ────────────────────────────────────────────────────────────

METHOD_STYLE = {
    "simclr": {"color": "#1f77b4", "label": "SimCLR"},
    "moco": {"color": "#ff7f0e", "label": "MoCo v2"},
    "byol": {"color": "#2ca02c", "label": "BYOL"},
    "supervised": {"color": "#d62728", "label": "Supervised"},
}


# ── Feature extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(encoder, dataloader, device, max_samples=5000):
    """Extract L2-normalised encoder features."""
    encoder.eval()
    feats, labs = [], []
    n = 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        h = encoder(images)
        h = F.normalize(h, dim=1)
        feats.append(h.cpu())
        labs.append(labels)
        n += images.shape[0]
        if n >= max_samples:
            break
    feats = torch.cat(feats)[:max_samples]
    labs = torch.cat(labs)[:max_samples]
    return feats, labs


@torch.no_grad()
def extract_dual_view_features(encoder, pretrain_loader, device, max_samples=2000):
    """Extract features from both augmented views for alignment computation."""
    encoder.eval()
    z1_list, z2_list = [], []
    n = 0
    for (x_i, x_j), _ in pretrain_loader:
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)
        h_i = F.normalize(encoder(x_i), dim=1)
        h_j = F.normalize(encoder(x_j), dim=1)
        z1_list.append(h_i.cpu())
        z2_list.append(h_j.cpu())
        n += x_i.shape[0]
        if n >= max_samples:
            break
    return torch.cat(z1_list)[:max_samples], torch.cat(z2_list)[:max_samples]


# ── Data loaders ─────────────────────────────────────────────────────────────

def get_dataloaders(dataset, data_dir, image_size, batch_size=256, num_workers=0):
    """Return (pretrain_loader, test_loader) for the given dataset."""
    if dataset == "cifar10":
        from datasets.cifar10_dataset import get_cifar10_pretrain, get_cifar10_test
        pretrain = get_cifar10_pretrain(data_dir, image_size, batch_size, num_workers)
        test = get_cifar10_test(data_dir, image_size, batch_size, num_workers)
    elif dataset == "stl10":
        from datasets.stl10_dataset import get_stl10_pretrain, get_stl10_test
        pretrain = get_stl10_pretrain(data_dir, image_size, batch_size, num_workers)
        test = get_stl10_test(data_dir, image_size, batch_size, num_workers)
    elif dataset == "chestxray":
        from datasets.chestxray_dataset import get_chestxray_pretrain, get_chestxray_test
        pretrain = get_chestxray_pretrain(data_dir, image_size, batch_size, num_workers)
        test = get_chestxray_test(data_dir, image_size, batch_size, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return pretrain, test


# ── Encoder loading ──────────────────────────────────────────────────────────

def find_encoders(checkpoint_dir, methods):
    """Return {method: path} for available encoder checkpoints."""
    found = {}
    for method in methods:
        path = os.path.join(checkpoint_dir, f"{method}_encoder_best.pth")
        if os.path.exists(path):
            found[method] = path
        else:
            path2 = os.path.join(checkpoint_dir, f"supervised_encoder_best.pth")
            if method == "supervised" and os.path.exists(path2):
                found[method] = path2
    return found


def load_encoder(path, backbone, device):
    encoder = ResNetEncoder(backbone=backbone).to(device)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not any(k.startswith("encoder.") for k in ckpt.keys()):
        ckpt = {"encoder." + k: v for k, v in ckpt.items()}
    encoder.load_state_dict(ckpt, strict=False)
    encoder.eval()
    return encoder


# ── Plot: CKA heatmap ───────────────────────────────────────────────────────

def plot_cka_heatmap(methods, matrix, save_path, dpi=200):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    labels = [METHOD_STYLE.get(m, {}).get("label", m) for m in methods]
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color="white" if matrix[i, j] > 0.6 else "black")
    ax.set_title("CKA Similarity Between Methods", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ── Plot: Embedding variance ────────────────────────────────────────────────

def plot_embedding_variance(variance_data, save_path, dpi=200):
    methods = list(variance_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of mean variance
    colors = [METHOD_STYLE.get(m, {}).get("color", "#333") for m in methods]
    labels = [METHOD_STYLE.get(m, {}).get("label", m) for m in methods]
    mean_vars = [variance_data[m]["mean_variance"] for m in methods]
    axes[0].bar(labels, mean_vars, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_ylabel("Mean Embedding Variance", fontsize=12)
    axes[0].set_title("Average Feature Variance per Method", fontsize=13, fontweight="bold")
    for i, v in enumerate(mean_vars):
        axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    # Right: per-dimension variance distribution
    for m in methods:
        per_dim = variance_data[m]["per_dim_variance"]
        style = METHOD_STYLE.get(m, {})
        axes[1].plot(sorted(per_dim, reverse=True),
                     color=style.get("color", "#333"),
                     label=style.get("label", m), linewidth=1.5)
    axes[1].set_xlabel("Dimension (sorted by variance)", fontsize=12)
    axes[1].set_ylabel("Variance", fontsize=12)
    axes[1].set_title("Per-Dimension Variance Distribution", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ── Plot: Alignment vs Uniformity scatter ────────────────────────────────────

def plot_alignment_uniformity(metrics, save_path, dpi=200):
    fig, ax = plt.subplots(figsize=(7, 6))
    for m, data in metrics.items():
        style = METHOD_STYLE.get(m, {})
        ax.scatter(data["uniformity"], data["alignment"],
                   color=style.get("color", "#333"),
                   label=style.get("label", m),
                   s=150, edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(style.get("label", m),
                    (data["uniformity"], data["alignment"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)

    ax.set_xlabel("Uniformity (lower = more uniform)", fontsize=12)
    ax.set_ylabel("Alignment (lower = better aligned)", fontsize=12)
    ax.set_title("Alignment vs Uniformity", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Representation Quality Analysis")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "stl10", "chestxray"])
    parser.add_argument("--methods", nargs="+", default=["simclr", "moco", "byol", "supervised"])
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    if args.dataset == "chestxray":
        data_dir = os.path.join(PROJECT_ROOT, "CXR8")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    image_size = {"cifar10": 32, "stl10": 96, "chestxray": 224}[args.dataset]

    print("\n" + "=" * 60)
    print("  REPRESENTATION QUALITY ANALYSIS")
    print("=" * 60)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Methods:     {args.methods}")
    print(f"  Max samples: {args.max_samples}")
    print("=" * 60)

    # Find available encoders
    encoders = find_encoders(checkpoint_dir, args.methods)
    if not encoders:
        print("No encoder checkpoints found. Run pretraining first.")
        return

    print(f"\nFound {len(encoders)} encoders: {list(encoders.keys())}")

    # ── MLflow ────────────────────────────────────────────────────────────
    tracker = None
    if not args.no_mlflow:
        tracker = ExperimentTracker(
            experiment_name=f"SSL_Benchmark_{args.dataset.upper()}",
            run_name=f"representation_analysis_{args.dataset}",
        )
        tracker.start_experiment()
        tracker.log_parameters({
            "analysis.dataset": args.dataset,
            "analysis.backbone": args.backbone,
            "analysis.max_samples": args.max_samples,
            "analysis.methods": ",".join(args.methods),
            "analysis.num_encoders_found": len(encoders),
        })
        tracker.log_dataset_info(
            name=args.dataset,
            image_size=image_size,
        )

    # Load data
    pretrain_loader, test_loader = get_dataloaders(
        args.dataset, data_dir, image_size, num_workers=args.num_workers)

    # ── Extract features ─────────────────────────────────────────────────
    all_features = {}
    all_metrics = {}

    for method, enc_path in encoders.items():
        print(f"\n-- {METHOD_STYLE.get(method, {}).get('label', method)} --")
        encoder = load_encoder(enc_path, args.backbone, device)

        # Single-view features (for uniformity, CKA, collapse)
        feats, labels = extract_features(encoder, test_loader, device, args.max_samples)
        all_features[method] = feats.numpy()

        # Dual-view features (for alignment)
        z1, z2 = extract_dual_view_features(encoder, pretrain_loader, device, min(args.max_samples, 2000))

        # Compute metrics
        align = compute_alignment(z1, z2)
        uniform = compute_uniformity(feats)
        var_stats = compute_embedding_variance(feats.numpy())
        rank_stats = compute_covariance_rank(feats.numpy())

        all_metrics[method] = {
            "alignment": align,
            "uniformity": uniform,
            "mean_variance": var_stats["mean_variance"],
            "effective_rank": rank_stats["effective_rank"],
            "rank_ratio": rank_stats["rank_ratio"],
        }

        print(f"  Alignment:      {align:.4f}")
        print(f"  Uniformity:     {uniform:.4f}")
        print(f"  Mean variance:  {var_stats['mean_variance']:.6f}")
        print(f"  Effective rank: {rank_stats['effective_rank']}/{rank_stats['total_dims']}"
              f" ({rank_stats['rank_ratio']:.1%})")

        # Log per-method metrics to MLflow
        if tracker is not None:
            tracker.log_metrics({
                f"{method}/alignment": align,
                f"{method}/uniformity": uniform,
                f"{method}/mean_variance": var_stats["mean_variance"],
                f"{method}/effective_rank": float(rank_stats["effective_rank"]),
                f"{method}/rank_ratio": rank_stats["rank_ratio"],
            })

    # ── CKA similarity ───────────────────────────────────────────────────
    if len(all_features) >= 2:
        print("\n-- CKA Similarity --")
        cka_methods, cka_matrix = compute_cka_matrix(all_features)
        for i, m_i in enumerate(cka_methods):
            for j, m_j in enumerate(cka_methods):
                if j > i:
                    print(f"  {m_i} vs {m_j}: {cka_matrix[i, j]:.4f}")
                    # Log CKA pair to MLflow
                    if tracker is not None:
                        tracker.log_metric(f"cka/{m_i}_vs_{m_j}", float(cka_matrix[i, j]))

        plot_cka_heatmap(
            cka_methods, cka_matrix,
            os.path.join(results_dir, "cka_similarity_heatmap.png"), args.dpi)

    # ── Variance analysis plot ───────────────────────────────────────────
    variance_data = {}
    for method in all_features:
        variance_data[method] = compute_embedding_variance(all_features[method])

    plot_embedding_variance(
        variance_data,
        os.path.join(results_dir, "embedding_variance_analysis.png"), args.dpi)

    # ── Alignment vs Uniformity plot ─────────────────────────────────────
    plot_alignment_uniformity(
        all_metrics,
        os.path.join(results_dir, "alignment_uniformity.png"), args.dpi)

    # ── Save JSON report ─────────────────────────────────────────────────
    report = {m: {k: v for k, v in d.items()} for m, d in all_metrics.items()}
    report_path = os.path.join(results_dir, "representation_metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [OK] {report_path}")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Method':<14} {'Align':>8} {'Uniform':>10} {'Variance':>10} {'Rank':>6} {'Ratio':>8}")
    print(f"{'-'*70}")
    for m, d in all_metrics.items():
        label = METHOD_STYLE.get(m, {}).get("label", m)
        print(f"{label:<14} {d['alignment']:>8.4f} {d['uniformity']:>10.4f} "
              f"{d['mean_variance']:>10.6f} {d['effective_rank']:>6d} {d['rank_ratio']:>8.1%}")
    print(f"{'='*70}")

    # ── Log artifacts to MLflow ──────────────────────────────────────────
    if tracker is not None:
        for fname in [
            "cka_similarity_heatmap.png",
            "embedding_variance_analysis.png",
            "alignment_uniformity.png",
            "representation_metrics.json",
        ]:
            fpath = os.path.join(results_dir, fname)
            if os.path.isfile(fpath):
                tracker.log_artifacts(fpath, artifact_path="representation_analysis")
        tracker.end_experiment()


if __name__ == "__main__":
    main()

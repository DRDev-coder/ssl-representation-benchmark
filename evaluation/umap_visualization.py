"""
UMAP Visualization
==================
Generates UMAP dimensionality reduction plots to visualize learned
representations from SSL and supervised encoders.

UMAP is faster than t-SNE and better at preserving global structure.

Usage:
    python -m evaluation.umap_visualization --dataset cifar10
    python -m evaluation.umap_visualization --methods simclr moco byol supervised
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.resnet_encoder import ResNetEncoder
from datasets.cifar10_dataset import get_cifar10_test
from datasets.stl10_dataset import get_stl10_test
from utils.device import get_device
from utils.seed import set_seed

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@torch.no_grad()
def extract_features_and_labels(encoder, loader, device, max_samples=5000):
    encoder.eval()
    all_features, all_labels = [], []
    count = 0
    for images, labels in loader:
        if count >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        features = encoder(images)
        features = F.normalize(features, dim=1)
        batch_keep = min(images.size(0), max_samples - count)
        all_features.append(features[:batch_keep].cpu().numpy())
        all_labels.append(labels[:batch_keep].numpy())
        count += batch_keep
    return np.concatenate(all_features), np.concatenate(all_labels)


def plot_umap(features, labels, class_names, title, save_path,
              n_neighbors=15, min_dist=0.1, seed=42):
    """Generate and save a UMAP plot."""
    try:
        from umap import UMAP
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        print("Falling back to t-SNE...")
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=seed,
                       max_iter=1000, learning_rate="auto", init="pca")
        embeddings = reducer.fit_transform(features)
        title = title.replace("UMAP", "t-SNE (UMAP fallback)")
        _plot_scatter(embeddings, labels, class_names, title, save_path)
        return

    print(f"Computing UMAP for {title}...")
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                   min_dist=min_dist, random_state=seed, metric="cosine")
    embeddings = reducer.fit_transform(features)
    _plot_scatter(embeddings, labels, class_names, title, save_path)


def _plot_scatter(embeddings, labels, class_names, title, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=COLORS[class_idx % len(COLORS)],
            label=class_names[class_idx],
            alpha=0.6, s=10, edgecolors="none",
        )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_multi_method_umap(all_embeddings, all_labels, class_names, save_path):
    """Side-by-side UMAP for multiple methods."""
    n_methods = len(all_embeddings)
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 7))
    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, embeddings) in zip(axes, all_embeddings.items()):
        labels = all_labels[method_name]
        for class_idx in range(len(class_names)):
            mask = labels == class_idx
            ax.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c=COLORS[class_idx % len(COLORS)],
                label=class_names[class_idx],
                alpha=0.6, s=10, edgecolors="none",
            )
        ax.set_title(method_name.upper(), fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center", bbox_to_anchor=(0.5, -0.02),
               ncol=5, fontsize=11, markerscale=2)

    plt.suptitle("UMAP Visualization — SSL Methods Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Multi-method UMAP saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="UMAP Visualization")
    parser.add_argument("--methods", nargs="+", default=["simclr", "supervised"])
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "stl10"])
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    checkpoint_dir = os.path.join(PROJECT_ROOT, args.checkpoint_dir)
    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(results_dir, exist_ok=True)

    image_size = 32 if args.dataset == "cifar10" else 96
    class_names = CIFAR10_CLASSES if args.dataset == "cifar10" else STL10_CLASSES

    if args.dataset == "cifar10":
        test_loader = get_cifar10_test(data_dir, image_size, batch_size=256)
    else:
        test_loader = get_stl10_test(data_dir, image_size, batch_size=256)

    all_embeddings = {}
    all_labels = {}

    for method in args.methods:
        encoder_path = os.path.join(checkpoint_dir, f"{method}_encoder_best.pth")
        if not os.path.exists(encoder_path):
            print(f"WARNING: {encoder_path} not found, skipping {method}")
            continue

        print(f"\n{'='*40}")
        print(f"UMAP: {method.upper()}")
        print(f"{'='*40}")

        encoder = ResNetEncoder(backbone=args.backbone).to(device)
        encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu", weights_only=True)
        )
        encoder.eval()

        features, labels = extract_features_and_labels(
            encoder, test_loader, device, max_samples=args.max_samples
        )

        # Individual UMAP
        plot_umap(features, labels, class_names,
                  f"UMAP — {method.upper()} Encoder",
                  os.path.join(results_dir, f"umap_{method}.png"))

        # Compute embeddings for multi-panel plot
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                           random_state=args.seed, metric="cosine")
            all_embeddings[method] = reducer.fit_transform(features)
        except ImportError:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=30, random_state=args.seed,
                           max_iter=1000, learning_rate="auto", init="pca")
            all_embeddings[method] = reducer.fit_transform(features)

        all_labels[method] = labels

    # Multi-panel comparison
    if len(all_embeddings) > 1:
        plot_multi_method_umap(all_embeddings, all_labels, class_names,
                               os.path.join(results_dir, "umap_comparison.png"))


if __name__ == "__main__":
    main()

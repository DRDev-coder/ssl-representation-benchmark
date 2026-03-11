"""
t-SNE Visualization
===================
Generates t-SNE plots comparing SimCLR (self-supervised) vs supervised
feature representations.

Produces:
  1. t-SNE of SimCLR encoder representations
  2. t-SNE of supervised encoder representations
  3. Side-by-side comparison plot

Key insight: Good SSL representations should form well-separated class
clusters in t-SNE space — even though the encoder was never trained with labels.

Usage:
    python -m evaluation.tsne_visualization
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.simclr_config import EvalConfig
from models.resnet_encoder import ResNetEncoder
from datasets.stl10_dataset import get_stl10_test
from datasets.cifar10_dataset import get_cifar10_test
from utils.device import get_device
from utils.seed import set_seed

# STL-10 class names
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Colorblind-friendly palette
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@torch.no_grad()
def extract_features_and_labels(encoder, loader, device, max_samples=5000):
    """
    Extract features and labels from a data loader.
    Limits to max_samples for t-SNE efficiency.
    """
    encoder.eval()
    all_features = []
    all_labels = []
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


def plot_tsne(
    features,
    labels,
    class_names,
    title,
    save_path,
    perplexity=30,
    seed=42,
):
    """
    Generate and save a t-SNE plot.

    Args:
        features: numpy array (N, D) of feature vectors.
        labels: numpy array (N,) of integer class labels.
        class_names: List of class name strings.
        title: Plot title.
        save_path: Path to save the figure.
        perplexity: t-SNE perplexity.
        seed: Random state for reproducibility.
    """
    print(f"Computing t-SNE for {title}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=COLORS[class_idx % len(COLORS)],
            label=class_names[class_idx],
            alpha=0.6,
            s=10,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=10,
        markerscale=2,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_side_by_side(
    ssl_features, ssl_labels,
    sup_features, sup_labels,
    class_names, save_path,
    perplexity=30, seed=42,
):
    """
    Generate side-by-side t-SNE comparison: SSL vs Supervised.
    """
    print("Computing t-SNE for side-by-side comparison...")

    # Compute t-SNE for both
    tsne_ssl = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                    max_iter=1000, learning_rate="auto", init="pca")
    emb_ssl = tsne_ssl.fit_transform(ssl_features)

    tsne_sup = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                    max_iter=1000, learning_rate="auto", init="pca")
    emb_sup = tsne_sup.fit_transform(sup_features)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, embeddings, labels, title in [
        (axes[0], emb_ssl, ssl_labels, "SimCLR (Self-Supervised)"),
        (axes[1], emb_sup, sup_labels, "Supervised Baseline"),
    ]:
        for class_idx in range(len(class_names)):
            mask = labels == class_idx
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=COLORS[class_idx % len(COLORS)],
                label=class_names[class_idx],
                alpha=0.6,
                s=10,
                edgecolors="none",
            )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend,
        loc="center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        fontsize=11,
        markerscale=2,
    )

    plt.suptitle(
        "Feature Space Comparison: SSL vs Supervised",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_tsne_comparison(config=None):
    """
    End-to-end t-SNE comparison between SSL and supervised encoders.
    """
    if config is None:
        config = EvalConfig()

    device = get_device()
    class_names = STL10_CLASSES if config.dataset == "stl10" else CIFAR10_CLASSES

    # Load test data
    if config.dataset == "stl10":
        test_loader = get_stl10_test(
            config.data_dir, config.image_size, batch_size=256,
        )
    else:
        test_loader = get_cifar10_test(
            config.data_dir, config.image_size, batch_size=256,
        )

    # ── SimCLR encoder ──────────────────────────────────────────
    ssl_encoder = ResNetEncoder(backbone="resnet18").to(device)
    ssl_encoder.load_state_dict(
        torch.load(config.encoder_path, map_location="cpu", weights_only=True)
    )

    ssl_features, ssl_labels = extract_features_and_labels(
        ssl_encoder, test_loader, device, max_samples=config.tsne_n_samples,
    )

    # Plot SSL t-SNE
    ssl_path = os.path.join(config.results_dir, "tsne_simclr.png")
    plot_tsne(
        ssl_features, ssl_labels, class_names,
        title="SimCLR (Self-Supervised) Feature Space",
        save_path=ssl_path,
        perplexity=config.tsne_perplexity,
        seed=config.tsne_seed,
    )

    # ── Supervised encoder (if available) ────────────────────────
    if os.path.exists(config.supervised_encoder_path):
        sup_encoder = ResNetEncoder(backbone="resnet18").to(device)
        # supervised encoder may have different keys (no fc layer)
        state_dict = torch.load(
            config.supervised_encoder_path, map_location="cpu", weights_only=True
        )
        # Map keys: remove 'encoder.' prefix if present
        mapped = {}
        for k, v in state_dict.items():
            new_key = k.replace("encoder.", "") if k.startswith("encoder.") else k
            mapped[new_key] = v
        sup_encoder.load_state_dict(mapped, strict=False)

        sup_features, sup_labels = extract_features_and_labels(
            sup_encoder, test_loader, device, max_samples=config.tsne_n_samples,
        )

        # Plot supervised t-SNE
        sup_path = os.path.join(config.results_dir, "tsne_supervised.png")
        plot_tsne(
            sup_features, sup_labels, class_names,
            title="Supervised Baseline Feature Space",
            save_path=sup_path,
            perplexity=config.tsne_perplexity,
            seed=config.tsne_seed,
        )

        # Side-by-side comparison
        comparison_path = os.path.join(config.results_dir, "tsne_comparison.png")
        plot_side_by_side(
            ssl_features, ssl_labels,
            sup_features, sup_labels,
            class_names,
            save_path=comparison_path,
            perplexity=config.tsne_perplexity,
            seed=config.tsne_seed,
        )
    else:
        print(
            f"Supervised encoder not found at {config.supervised_encoder_path}. "
            f"Run train_supervised.py first for comparison plots."
        )


def plot_label_efficiency_curve(results_dir=None):
    """
    Plot the label efficiency curve: SSL vs supervised accuracy at different label fractions.
    Reads results from JSON files saved by fine_tune.py and train_supervised.py.
    """
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "results")

    ssl_path = os.path.join(results_dir, "finetune_results.json")
    sup_path = os.path.join(results_dir, "supervised_results.json")

    if not os.path.exists(ssl_path) or not os.path.exists(sup_path):
        print("Results files not found. Run fine_tune.py and train_supervised.py first.")
        return

    with open(ssl_path) as f:
        ssl_results = json.load(f)
    with open(sup_path) as f:
        sup_results = json.load(f)

    # Parse fractions
    fractions = sorted(set(list(ssl_results.keys()) + list(sup_results.keys())))
    frac_values = [float(f) for f in fractions]
    ssl_accs = [ssl_results.get(f, 0) for f in fractions]
    sup_accs = [sup_results.get(f, 0) for f in fractions]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(
        [f * 100 for f in frac_values], ssl_accs,
        "o-", color="#1f77b4", linewidth=2.5, markersize=10,
        label="SimCLR + Fine-tune",
    )
    ax.plot(
        [f * 100 for f in frac_values], sup_accs,
        "s--", color="#d62728", linewidth=2.5, markersize=10,
        label="Supervised (from scratch)",
    )

    ax.set_xlabel("Labeled Data Used (%)", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax.set_title("Label Efficiency Curve: SimCLR vs Supervised", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1%", "10%", "100%"])

    # Annotate data points
    for i, (f, ssl_a, sup_a) in enumerate(zip(frac_values, ssl_accs, sup_accs)):
        ax.annotate(f"{ssl_a:.1f}%", (f * 100, ssl_a), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, color="#1f77b4")
        ax.annotate(f"{sup_a:.1f}%", (f * 100, sup_a), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=10, color="#d62728")

    plt.tight_layout()
    save_path = os.path.join(results_dir, "label_efficiency_curve.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Label efficiency curve saved to {save_path}")


def main(args=None):
    """Generate all visualization outputs."""
    parser = argparse.ArgumentParser(description="t-SNE Visualization")
    parser.add_argument("--dataset", type=str, default="stl10", choices=["stl10", "cifar10"])
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--supervised_path", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--label_curve", action="store_true", help="Also plot label efficiency curve")
    args = parser.parse_args(args)

    config = EvalConfig(dataset=args.dataset)
    if args.dataset == "cifar10":
        config.image_size = 32
    if args.encoder_path is not None:
        config.encoder_path = args.encoder_path
    if args.supervised_path is not None:
        config.supervised_encoder_path = args.supervised_path
    config.tsne_n_samples = args.n_samples

    set_seed(config.seed)

    print("\n" + "=" * 60)
    print("t-SNE VISUALIZATION")
    print("=" * 60)

    generate_tsne_comparison(config)

    if args.label_curve:
        plot_label_efficiency_curve(config.results_dir)

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()

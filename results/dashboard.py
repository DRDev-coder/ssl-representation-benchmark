"""
Results Dashboard
=================
Generates a comprehensive set of publication-quality figures and summary
tables from all experiment outputs.

Reads from:
  results/finetune_results.json          (SimCLR fine-tune)
  results/supervised_results.json        (supervised baseline)
  results/label_efficiency_results.json  (multi-method label sweep)
  results/augmentation_ablation_results.json
  results/transfer_*.json
  results/medical_ssl_results.json
  checkpoints/*_encoder_best.pth         (for t-SNE / UMAP)

Produces (in results/dashboard/):
  01_label_efficiency_curve.png
  02_augmentation_ablation.png
  03_ssl_method_comparison.png
  04_tsne_grid.png
  05_umap_grid.png
  06_knn_bar.png
  07_full_benchmark_table.png
  summary_report.md

Usage:
    python results/dashboard.py
    python results/dashboard.py --no_embeddings          # skip t-SNE/UMAP (fast)
    python results/dashboard.py --dataset cifar10 --dpi 300
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Style ────────────────────────────────────────────────────────────────────

# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
})

METHOD_STYLE = OrderedDict([
    ("simclr",      {"color": "#2176AE", "marker": "o", "label": "SimCLR"}),
    ("moco",        {"color": "#F57C20", "marker": "s", "label": "MoCo v2"}),
    ("byol",        {"color": "#57A773", "marker": "D", "label": "BYOL"}),
    ("supervised",  {"color": "#D64045", "marker": "^", "label": "Supervised"}),
])

CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def style_for(method):
    return METHOD_STYLE.get(method, {"color": "#555555", "marker": "x", "label": method.upper()})


# ── 01  Label Efficiency Curve ───────────────────────────────────────────────

def plot_label_efficiency(results_dir, out_dir, dpi):
    """
    Reads:  label_efficiency_results.json  (multi-method, from experiments/)
            OR falls back to finetune_results.json + supervised_results.json
    """
    multi = load_json(os.path.join(results_dir, "label_efficiency_results.json"))

    if multi is None:
        # Build from original Stage 3+4 results
        ft = load_json(os.path.join(results_dir, "finetune_results.json"))
        sv = load_json(os.path.join(results_dir, "supervised_results.json"))
        if ft is None and sv is None:
            print("  [SKIP] No label-efficiency data found.")
            return
        multi = {}
        if ft:
            multi["simclr"] = ft
        if sv:
            multi["supervised"] = sv

    # Collect all fractions across every method
    all_fracs = set()
    for data in multi.values():
        all_fracs.update(float(k) for k in data.keys())
    fracs = sorted(all_fracs)
    pct = [f * 100 for f in fracs]

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, data in multi.items():
        s = style_for(method)
        accs = [data.get(str(f), data.get(f, None)) for f in fracs]
        # Skip None values
        valid = [(p, a) for p, a in zip(pct, accs) if a is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker=s["marker"], color=s["color"], label=s["label"],
                linewidth=2.2, markersize=8, markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xlabel("Labeled Data (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Label Efficiency: SSL Methods vs. Supervised Baseline")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(pct)
    ax.set_xticklabels([f"{p:g}%" for p in pct])
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc")

    path = os.path.join(out_dir, "01_label_efficiency_curve.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {path}")
    return multi


# ── 02  Augmentation Ablation ────────────────────────────────────────────────

def plot_augmentation_ablation(results_dir, out_dir, dpi):
    data = load_json(os.path.join(results_dir, "augmentation_ablation_results.json"))
    if data is None:
        print("  [SKIP] No augmentation ablation data found.")
        return

    configs = list(data.keys())
    accs = [data[c] for c in configs]
    sorted_pairs = sorted(zip(configs, accs), key=lambda x: -x[1])
    configs, accs = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2176AE" if c == "full" else "#A0C4E8" for c in configs]
    bars = ax.barh(range(len(configs)), accs, color=colors,
                   edgecolor="white", linewidth=0.8, height=0.65)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([c.replace("_", " ").title() for c in configs])
    ax.invert_yaxis()
    ax.set_xlabel("Linear Probe Accuracy (%)")
    ax.set_title("Augmentation Ablation Study")

    # Reference line for full augmentation
    if "full" in data:
        ax.axvline(data["full"], color="#2176AE", linestyle=":", alpha=0.5, linewidth=1.2)

    path = os.path.join(out_dir, "02_augmentation_ablation.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {path}")


# ── 03  SSL Method Comparison (multi-metric bar chart) ───────────────────────

def plot_ssl_comparison(results_dir, out_dir, dpi):
    """
    Combines: linear probe, kNN, fine-tune @1%, @10%, @100% for each method.
    Falls back to whatever data is available.
    """
    ft = load_json(os.path.join(results_dir, "finetune_results.json"))
    sv = load_json(os.path.join(results_dir, "supervised_results.json"))
    le = load_json(os.path.join(results_dir, "label_efficiency_results.json"))

    # Build per-method metric dicts
    methods_data = OrderedDict()

    # From label_efficiency (has all methods)
    if le:
        for method, fracs in le.items():
            methods_data.setdefault(method, {})
            for f, acc in fracs.items():
                methods_data[method][f"FT@{float(f)*100:g}%"] = acc
    else:
        # Fallback to original files
        if ft:
            methods_data["simclr"] = {f"FT@{float(k)*100:g}%": v for k, v in ft.items()}
        if sv:
            methods_data["supervised"] = {f"FT@{float(k)*100:g}%": v for k, v in sv.items()}

    if not methods_data:
        print("  [SKIP] No SSL comparison data found.")
        return

    # Determine common metrics
    all_metrics = []
    for d in methods_data.values():
        for m in d:
            if m not in all_metrics:
                all_metrics.append(m)

    n_methods = len(methods_data)
    n_metrics = len(all_metrics)
    x = np.arange(n_metrics)
    total_width = 0.7
    bar_w = total_width / n_methods

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.8), 5.5))
    for i, (method, data) in enumerate(methods_data.items()):
        s = style_for(method)
        vals = [data.get(m, 0) for m in all_metrics]
        offset = (i - n_methods / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w * 0.9, color=s["color"],
                      label=s["label"], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(all_metrics, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SSL Method Comparison — Multi-Metric Overview")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", loc="upper left")
    ax.set_ylim(0, max(max(d.values()) for d in methods_data.values() if d) + 8)

    path = os.path.join(out_dir, "03_ssl_method_comparison.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {path}")


# ── 04 / 05  Embedding Visualizations (t-SNE + UMAP) ────────────────────────

def _get_encoder_paths(checkpoint_dir):
    """Discover available encoder checkpoints and map to method names."""
    encoders = OrderedDict()
    candidates = [
        ("simclr", "simclr_encoder_best.pth"),
        ("moco", "moco_encoder_best.pth"),
        ("byol", "byol_encoder_best.pth"),
        ("supervised", "supervised_encoder_best.pth"),
    ]
    for method, fname in candidates:
        path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(path):
            encoders[method] = path
    return encoders


def _extract(encoder, loader, device, max_samples=5000):
    """Extract L2-normed features from a data loader."""
    import torch
    import torch.nn.functional as F

    encoder.eval()
    feats, labs = [], []
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            if count >= max_samples:
                break
            images = images.to(device, non_blocking=True)
            f = encoder(images)
            f = F.normalize(f, dim=1)
            keep = min(images.size(0), max_samples - count)
            feats.append(f[:keep].cpu().numpy())
            labs.append(labels[:keep].numpy())
            count += keep
    return np.concatenate(feats), np.concatenate(labs)


def _scatter(ax, embeddings, labels, class_names):
    """Scatter plot with per-class colours."""
    for idx in range(len(class_names)):
        mask = labels == idx
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=CLASS_COLORS[idx % len(CLASS_COLORS)],
                   label=class_names[idx], alpha=0.55, s=6, edgecolors="none")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_embedding_grid(results_dir, checkpoint_dir, out_dir, dpi,
                        dataset="cifar10", method_name="tsne", max_samples=5000, seed=42):
    """
    Grid of t-SNE or UMAP plots — one panel per available encoder.
    """
    import torch
    from models.resnet_encoder import ResNetEncoder
    from datasets.cifar10_dataset import get_cifar10_test
    from datasets.stl10_dataset import get_stl10_test
    from utils.device import get_device

    encoders = _get_encoder_paths(checkpoint_dir)
    if not encoders:
        print(f"  [SKIP] No encoder checkpoints found for {method_name} grid.")
        return

    device = get_device()
    image_size = 32 if dataset == "cifar10" else 96
    class_names = CIFAR10_CLASSES if dataset == "cifar10" else STL10_CLASSES
    data_dir = os.path.join(PROJECT_ROOT, "data")

    if dataset == "cifar10":
        test_loader = get_cifar10_test(data_dir, image_size, batch_size=256)
    else:
        test_loader = get_stl10_test(data_dir, image_size, batch_size=256)

    # Choose reducer
    if method_name == "umap":
        try:
            from umap import UMAP
            def reduce(features):
                return UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=seed, metric="cosine").fit_transform(features)
        except ImportError:
            print("  [INFO] umap-learn not installed, falling back to t-SNE for UMAP grid.")
            from sklearn.manifold import TSNE
            def reduce(features):
                return TSNE(n_components=2, perplexity=30, random_state=seed,
                            max_iter=1000, learning_rate="auto", init="pca").fit_transform(features)
            # method_name stays "umap" so the output is still saved as 05_umap_grid.png
    else:
        from sklearn.manifold import TSNE
        def reduce(features):
            return TSNE(n_components=2, perplexity=30, random_state=seed,
                        max_iter=1000, learning_rate="auto", init="pca").fit_transform(features)

    n = len(encoders)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    method_title = "t-SNE" if "tsne" in method_name else "UMAP"

    for idx, (method, enc_path) in enumerate(encoders.items()):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        print(f"    {method_title} — {method} ...", end=" ", flush=True)
        encoder = ResNetEncoder(backbone="resnet18").to(device)
        ckpt = torch.load(enc_path, map_location="cpu", weights_only=True)
        if not any(k.startswith("encoder.") for k in ckpt.keys()):
            ckpt = {"encoder." + k: v for k, v in ckpt.items()}
        encoder.load_state_dict(ckpt, strict=False)
        features, labels = _extract(encoder, test_loader, device, max_samples)
        emb = reduce(features)
        _scatter(ax, emb, labels, class_names)
        ax.set_title(style_for(method)["label"], fontsize=13, fontweight="bold")
        print("done")

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    # Shared legend below the grid
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center",
               ncol=min(len(class_names), 5), fontsize=9,
               markerscale=2.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"{method_title} Embeddings — {dataset.upper()}", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    tag = "04_tsne_grid" if "tsne" in method_name else "05_umap_grid"
    path = os.path.join(out_dir, f"{tag}.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {path}")


# ── 06  kNN Bar Chart ────────────────────────────────────────────────────────

def plot_knn_bar(results_dir, checkpoint_dir, out_dir, dpi,
                 dataset="cifar10", skip_compute=False):
    """
    Bar chart of kNN Top-1 (and Top-5 if available) per method.
    If skip_compute, only reads existing JSON. Otherwise runs kNN live.
    """
    knn_file = os.path.join(results_dir, "knn_results.json")
    knn_data = load_json(knn_file)

    if knn_data is None and not skip_compute:
        # Run kNN for each available encoder
        import torch
        from models.resnet_encoder import ResNetEncoder
        from datasets.cifar10_dataset import get_cifar10_train, get_cifar10_test
        from datasets.stl10_dataset import get_stl10_train, get_stl10_test
        from evaluation.knn_eval import knn_evaluate, extract_features
        from utils.device import get_device

        encoders = _get_encoder_paths(checkpoint_dir)
        if not encoders:
            print("  [SKIP] No encoders for kNN.")
            return

        device = get_device()
        image_size = 32 if dataset == "cifar10" else 96
        data_dir = os.path.join(PROJECT_ROOT, "data")

        if dataset == "cifar10":
            train_loader = get_cifar10_train(data_dir, image_size, 256, augment=False)
            test_loader = get_cifar10_test(data_dir, image_size, 256)
        else:
            train_loader = get_stl10_train(data_dir, image_size, 256, augment=False)
            test_loader = get_stl10_test(data_dir, image_size, 256)

        knn_data = {}
        for method, enc_path in encoders.items():
            print(f"    kNN — {method} ...", end=" ", flush=True)
            encoder = ResNetEncoder(backbone="resnet18").to(device)
            ckpt = torch.load(enc_path, map_location="cpu", weights_only=True)
            # Supervised checkpoint is saved as a raw ResNet (flat keys).
            # ResNetEncoder wraps it as self.encoder, so remap if needed.
            if not any(k.startswith("encoder.") for k in ckpt.keys()):
                ckpt = {"encoder." + k: v for k, v in ckpt.items()}
            encoder.load_state_dict(ckpt, strict=False)
            encoder.eval()
            top1, top5 = knn_evaluate(encoder, train_loader, test_loader,
                                      device, k=200, temperature=0.5, num_classes=10)
            knn_data[method] = {"top1": top1, "top5": top5}
            print(f"Top-1={top1:.2f}% Top-5={top5:.2f}%")

        with open(knn_file, "w") as f:
            json.dump(knn_data, f, indent=2)

    if knn_data is None:
        print("  [SKIP] No kNN data.")
        return

    methods = list(knn_data.keys())
    top1 = [knn_data[m]["top1"] if isinstance(knn_data[m], dict) else knn_data[m] for m in methods]
    has_top5 = all(isinstance(knn_data[m], dict) and "top5" in knn_data[m] for m in methods)
    top5 = [knn_data[m]["top5"] for m in methods] if has_top5 else None

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 2), 5))

    if has_top5:
        w = 0.35
        bars1 = ax.bar(x - w / 2, top1, w, label="Top-1",
                        color=[style_for(m)["color"] for m in methods],
                        edgecolor="white", linewidth=0.6)
        bars2 = ax.bar(x + w / 2, top5, w, label="Top-5",
                        color=[style_for(m)["color"] for m in methods],
                        alpha=0.55, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars1, top1):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar, v in zip(bars2, top5):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.legend()
    else:
        bars = ax.bar(x, top1, 0.5, color=[style_for(m)["color"] for m in methods],
                      edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, top1):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([style_for(m)["label"] for m in methods])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("kNN Classification Accuracy (k=200)")

    path = os.path.join(out_dir, "06_knn_bar.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {path}")


# ── 07  Benchmark Summary Table (rendered as image) ─────────────────────────

def render_benchmark_table(results_dir, out_dir, dpi):
    """Render a comprehensive benchmark table as a figure."""
    ft = load_json(os.path.join(results_dir, "finetune_results.json"))
    sv = load_json(os.path.join(results_dir, "supervised_results.json"))
    le = load_json(os.path.join(results_dir, "label_efficiency_results.json"))
    knn = load_json(os.path.join(results_dir, "knn_results.json"))
    abl = load_json(os.path.join(results_dir, "augmentation_ablation_results.json"))

    # Build table rows: Method | Linear | kNN | FT@1% | FT@10% | FT@100%
    header = ["Method", "kNN Top-1", "FT @ 1%", "FT @ 10%", "FT @ 100%"]
    rows = []

    sources = le if le else {}
    if not sources:
        if ft:
            sources["simclr"] = ft
        if sv:
            sources["supervised"] = sv

    for method in ["simclr", "moco", "byol", "supervised"]:
        data = sources.get(method)
        if data is None:
            continue
        knn_val = ""
        if knn and method in knn:
            v = knn[method]
            knn_val = f"{v['top1']:.1f}%" if isinstance(v, dict) else f"{v:.1f}%"

        def _get(key):
            val = data.get(key, data.get(str(key), None))
            return f"{val:.1f}%" if val is not None else "—"

        rows.append([
            style_for(method)["label"],
            knn_val or "—",
            _get(0.01), _get(0.1), _get(1.0),
        ])

    if not rows:
        print("  [SKIP] Insufficient data for benchmark table.")
        return

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.45 * len(rows)))
    ax.axis("off")

    colors_header = ["#2176AE"] * len(header)
    cell_colors = [["#f0f4f8"] * len(header) if i % 2 == 0
                   else ["#ffffff"] * len(header) for i in range(len(rows))]

    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellColours=cell_colors,
        colColours=colors_header,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(header)):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_facecolor("#2176AE")

    # Bold best value per column
    for col_idx in range(1, len(header)):
        vals = []
        for row in rows:
            try:
                vals.append(float(row[col_idx].replace("%", "")))
            except (ValueError, AttributeError):
                vals.append(-1)
        if max(vals) > 0:
            best_row = vals.index(max(vals))
            cell = table[best_row + 1, col_idx]
            cell.set_text_props(fontweight="bold")

    ax.set_title("SSL Benchmark — Summary Table", fontsize=14, fontweight="bold", pad=15)

    path = os.path.join(out_dir, "07_full_benchmark_table.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {path}")


# ── Summary Markdown Report ──────────────────────────────────────────────────

def write_summary_report(results_dir, out_dir):
    """Write a Markdown report referencing all generated figures."""
    ft = load_json(os.path.join(results_dir, "finetune_results.json"))
    sv = load_json(os.path.join(results_dir, "supervised_results.json"))

    lines = [
        "# SimCLR Benchmark — Results Dashboard",
        "",
        f"*Auto-generated from experiment outputs in `results/`*",
        "",
        "---",
        "",
        "## 1. Label Efficiency",
        "",
        "![Label Efficiency](01_label_efficiency_curve.png)",
        "",
    ]

    if ft and sv:
        lines += [
            "| Label Fraction | Supervised | SimCLR Fine-Tune | Advantage |",
            "|:-:|:-:|:-:|:-:|",
        ]
        for frac in ["0.01", "0.1", "1.0"]:
            s = sv.get(frac, 0)
            f_val = ft.get(frac, 0)
            diff = f_val - s
            sign = "+" if diff > 0 else ""
            lines.append(f"| {float(frac)*100:g}% | {s:.1f}% | {f_val:.1f}% | {sign}{diff:.1f} pp |")
        lines.append("")

    lines += [
        "## 2. Augmentation Ablation",
        "",
        "![Ablation](02_augmentation_ablation.png)",
        "",
        "## 3. SSL Method Comparison",
        "",
        "![Comparison](03_ssl_method_comparison.png)",
        "",
        "## 4. t-SNE Embeddings",
        "",
        "![t-SNE](04_tsne_grid.png)",
        "",
        "## 5. UMAP Embeddings",
        "",
        "![UMAP](05_umap_grid.png)",
        "",
        "## 6. kNN Evaluation",
        "",
        "![kNN](06_knn_bar.png)",
        "",
        "## 7. Benchmark Table",
        "",
        "![Table](07_full_benchmark_table.png)",
        "",
    ]

    path = os.path.join(out_dir, "summary_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [OK] {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality dashboard from experiment results",
    )
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing experiment JSON outputs")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "stl10"])
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure DPI (300 for publication)")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max samples for t-SNE/UMAP")
    parser.add_argument("--no_embeddings", action="store_true",
                        help="Skip t-SNE / UMAP (much faster)")
    parser.add_argument("--no_knn", action="store_true",
                        help="Skip live kNN computation (use cached JSON only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    checkpoint_dir = os.path.join(PROJECT_ROOT, args.checkpoint_dir)
    out_dir = os.path.join(results_dir, "dashboard")
    ensure_dir(out_dir)

    print("=" * 60)
    print("  RESULTS DASHBOARD GENERATOR")
    print("=" * 60)
    print(f"  Results dir  : {results_dir}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Dataset      : {args.dataset}")
    print(f"  DPI          : {args.dpi}")
    print(f"  Embeddings   : {'yes' if not args.no_embeddings else 'skipped'}")
    print("=" * 60)

    # 01 — Label Efficiency
    print("\n[1/7] Label Efficiency Curve")
    plot_label_efficiency(results_dir, out_dir, args.dpi)

    # 02 — Augmentation Ablation
    print("\n[2/7] Augmentation Ablation")
    plot_augmentation_ablation(results_dir, out_dir, args.dpi)

    # 03 — SSL Method Comparison
    print("\n[3/7] SSL Method Comparison")
    plot_ssl_comparison(results_dir, out_dir, args.dpi)

    # 04 — t-SNE Grid
    if not args.no_embeddings:
        print("\n[4/7] t-SNE Embedding Grid")
        plot_embedding_grid(results_dir, checkpoint_dir, out_dir, args.dpi,
                            dataset=args.dataset, method_name="tsne",
                            max_samples=args.max_samples, seed=args.seed)
    else:
        print("\n[4/7] t-SNE Grid  [SKIPPED]")

    # 05 — UMAP Grid
    if not args.no_embeddings:
        print("\n[5/7] UMAP Embedding Grid")
        plot_embedding_grid(results_dir, checkpoint_dir, out_dir, args.dpi,
                            dataset=args.dataset, method_name="umap",
                            max_samples=args.max_samples, seed=args.seed)
    else:
        print("\n[5/7] UMAP Grid  [SKIPPED]")

    # 06 — kNN Bar Chart
    print("\n[6/7] kNN Bar Chart")
    plot_knn_bar(results_dir, checkpoint_dir, out_dir, args.dpi,
                 dataset=args.dataset, skip_compute=args.no_knn)

    # 07 — Benchmark Table
    print("\n[7/7] Benchmark Table")
    render_benchmark_table(results_dir, out_dir, args.dpi)

    # Summary markdown
    print("\n[+] Summary Report")
    write_summary_report(results_dir, out_dir)

    print("\n" + "=" * 60)
    print(f"  Dashboard complete → {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
MLflow Summary Dashboard
========================
Queries all MLflow experiment runs, extracts metrics, and generates
publication-quality summary tables and plots.

Outputs:
  - results/mlflow_method_comparison.png
  - results/mlflow_label_efficiency.png
  - results/mlflow_representation_comparison.png
  - results/mlflow_summary.csv

Usage:
    python analysis/generate_mlflow_summary.py
    python analysis/generate_mlflow_summary.py --tracking_uri mlruns/
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ── Style ────────────────────────────────────────────────────────────────────
COLORS = {
    "simclr": "#1f77b4",
    "moco": "#ff7f0e",
    "byol": "#2ca02c",
    "supervised": "#d62728",
}
LABELS = {
    "simclr": "SimCLR",
    "moco": "MoCo v2",
    "byol": "BYOL",
    "supervised": "Supervised",
}


def get_all_runs(client: "MlflowClient") -> pd.DataFrame:
    """Fetch all runs from all experiments into a single DataFrame."""
    rows = []
    for exp in client.search_experiments():
        for run in client.search_runs(experiment_ids=[exp.experiment_id]):
            row = {
                "experiment": exp.name,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "",
                "status": run.info.status,
                "start_time": run.info.start_time,
            }
            row.update({f"param.{k}": v for k, v in run.data.params.items()})
            row.update({f"metric.{k}": v for k, v in run.data.metrics.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _safe_float(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert a column to float, coercing errors to NaN."""
    if col not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


# ── Plot 1: Method comparison (best loss / accuracy) ────────────────────────

def plot_method_comparison(df: pd.DataFrame, save_dir: str):
    """Bar chart comparing best loss and linear eval accuracy per method."""
    methods = ["simclr", "moco", "byol", "supervised"]
    losses = {}
    accs = {}

    for method in methods:
        # Best loss from SSL pretraining runs
        loss_col = f"metric.{method}/best_loss"
        if loss_col in df.columns:
            vals = _safe_float(df, loss_col).dropna()
            if len(vals) > 0:
                losses[method] = vals.min()

        # Best loss from main pretrain
        main_loss = _safe_float(df, "metric.train/best_loss").dropna()
        if method == "simclr" and method not in losses and len(main_loss) > 0:
            losses[method] = main_loss.min()

        # Linear eval accuracy
        acc_col = "metric.linear_eval/best_accuracy"
        method_runs = df[df["run_name"].str.contains(method, case=False, na=False)]
        if acc_col in method_runs.columns:
            vals = _safe_float(method_runs, acc_col).dropna()
            if len(vals) > 0:
                accs[method] = vals.max()

    if not losses and not accs:
        print("  No method comparison data found. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss chart
    if losses:
        m_list = [m for m in methods if m in losses]
        ax = axes[0]
        bars = ax.bar(
            [LABELS.get(m, m) for m in m_list],
            [losses[m] for m in m_list],
            color=[COLORS.get(m, "#999") for m in m_list],
            edgecolor="white",
        )
        ax.set_ylabel("Best Pretraining Loss")
        ax.set_title("Pretraining Loss Comparison", fontweight="bold")
        for bar, m in zip(bars, m_list):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{losses[m]:.3f}", ha="center", va="bottom", fontsize=10)

    # Accuracy chart
    if accs:
        m_list = [m for m in methods if m in accs]
        ax = axes[1]
        bars = ax.bar(
            [LABELS.get(m, m) for m in m_list],
            [accs[m] for m in m_list],
            color=[COLORS.get(m, "#999") for m in m_list],
            edgecolor="white",
        )
        ax.set_ylabel("Linear Probe Accuracy (%)")
        ax.set_title("Linear Evaluation Comparison", fontweight="bold")
        for bar, m in zip(bars, m_list):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{accs[m]:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, "mlflow_method_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ── Plot 2: Label efficiency curve from MLflow ──────────────────────────────

def plot_label_efficiency(df: pd.DataFrame, save_dir: str):
    """Reconstruct label efficiency curves from logged finetune/supervised metrics."""
    fractions = [1, 5, 10, 100]
    series = {}

    for prefix in ["finetune", "supervised"]:
        vals = {}
        for pct in fractions:
            col = f"metric.{prefix}/accuracy_{pct}pct"
            if col in df.columns:
                v = _safe_float(df, col).dropna()
                if len(v) > 0:
                    vals[pct] = v.max()
        if vals:
            series[prefix] = vals

    if not series:
        print("  No label efficiency data found. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    style_map = {
        "finetune": ("o-", COLORS["simclr"], "SimCLR + Fine-tune"),
        "supervised": ("s--", COLORS["supervised"], "Supervised (scratch)"),
    }

    for key, vals in series.items():
        marker, color, label = style_map.get(key, ("x-", "#999", key))
        xs = sorted(vals.keys())
        ys = [vals[x] for x in xs]
        ax.plot(xs, ys, marker, color=color, linewidth=2.5, markersize=10, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=10, color=color)

    ax.set_xlabel("Labeled Data Used (%)", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax.set_title("Label Efficiency Curve (from MLflow)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 100])
    ax.set_xticklabels(["1%", "5%", "10%", "100%"])

    plt.tight_layout()
    path = os.path.join(save_dir, "mlflow_label_efficiency.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ── Plot 3: Representation metrics comparison ───────────────────────────────

def plot_representation_comparison(df: pd.DataFrame, save_dir: str):
    """Bar charts of alignment, uniformity, rank ratio from MLflow."""
    methods = ["simclr", "moco", "byol", "supervised"]
    metrics_map = {
        "Alignment": "alignment",
        "Uniformity": "uniformity",
        "Rank Ratio": "rank_ratio",
    }

    data = defaultdict(dict)
    for method in methods:
        for label, suffix in metrics_map.items():
            col = f"metric.{method}/{suffix}"
            if col in df.columns:
                v = _safe_float(df, col).dropna()
                if len(v) > 0:
                    data[label][method] = v.iloc[-1]

    if not data:
        print("  No representation metrics found. Skipping plot.")
        return

    n_metrics = len(data)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, vals) in zip(axes, data.items()):
        m_list = [m for m in methods if m in vals]
        bars = ax.bar(
            [LABELS.get(m, m) for m in m_list],
            [vals[m] for m in m_list],
            color=[COLORS.get(m, "#999") for m in m_list],
            edgecolor="white",
        )
        ax.set_title(metric_name, fontweight="bold", fontsize=13)
        for bar, m in zip(bars, m_list):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{vals[m]:.3f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Representation Quality Metrics (from MLflow)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "mlflow_representation_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ── Summary CSV ──────────────────────────────────────────────────────────────

def generate_summary_csv(df: pd.DataFrame, save_dir: str):
    """Save a condensed CSV of all experiments with key metrics."""
    key_metrics = [c for c in df.columns if c.startswith("metric.")]
    key_params = [c for c in df.columns if c.startswith("param.")]

    cols = ["experiment", "run_name", "status"] + key_params[:20] + key_metrics
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    path = os.path.join(save_dir, "mlflow_summary.csv")
    out.to_csv(path, index=False)
    print(f"  [OK] {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate MLflow Summary Dashboard")
    parser.add_argument(
        "--tracking_uri", default=None,
        help="MLflow tracking URI (default: mlruns/ under project root)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory for output plots/CSV (default: results/)",
    )
    args = parser.parse_args()

    if not MLFLOW_AVAILABLE:
        print("ERROR: mlflow is not installed. Run: pip install mlflow")
        sys.exit(1)

    # Set tracking URI
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    else:
        mlruns_path = Path(PROJECT_ROOT) / "mlruns"
        if mlruns_path.exists():
            mlflow.set_tracking_uri(mlruns_path.as_uri())
        else:
            print(f"No mlruns/ directory found at {mlruns_path}. Run experiments first.")
            sys.exit(1)

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)

    client = MlflowClient()

    print("\n" + "=" * 60)
    print("  MLFLOW EXPERIMENT SUMMARY")
    print("=" * 60)

    # Fetch all runs
    df = get_all_runs(client)
    if df.empty:
        print("No MLflow runs found. Run experiments first.")
        return

    print(f"  Total runs: {len(df)}")
    print(f"  Experiments: {df['experiment'].nunique()}")
    print(f"  Metrics tracked: {sum(1 for c in df.columns if c.startswith('metric.'))}")
    print("=" * 60)

    # Print experiment summary table
    print("\nExperiment Summary:")
    print("-" * 50)
    for exp_name, group in df.groupby("experiment"):
        print(f"  {exp_name}: {len(group)} runs")
    print()

    # Generate outputs
    print("Generating plots...")
    plot_method_comparison(df, output_dir)
    plot_label_efficiency(df, output_dir)
    plot_representation_comparison(df, output_dir)
    generate_summary_csv(df, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")
    print("Launch MLflow UI:  mlflow ui --backend-store-uri mlruns/")
    print("View at:           http://localhost:5000")


if __name__ == "__main__":
    main()

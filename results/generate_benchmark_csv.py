"""
Benchmark Results CSV Generator
================================
Aggregates all experiment results into a single CSV for analysis.

Usage:
    python results/generate_benchmark_csv.py
"""

import os
import sys
import json
import csv
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")

    rows = []

    # ── CIFAR-10 results ─────────────────────────────────────────────────

    # kNN results
    knn = load_json(os.path.join(results_dir, "knn_results.json"))
    knn_map = {}
    if knn:
        for method, data in knn.items():
            top1 = data.get("top1", data) if isinstance(data, dict) else data
            knn_map[method] = top1 if isinstance(top1, (int, float)) else 0

    # Linear probe (from individual experiment results)
    linear_probes = {
        "simclr": 67.56,    # from training_log.txt
        "moco": 65.47,
        "byol": 56.88,
    }

    # Finetune results
    finetune = load_json(os.path.join(results_dir, "finetune_results.json"))
    supervised = load_json(os.path.join(results_dir, "supervised_results.json"))

    # Transfer results
    transfer = load_json(os.path.join(results_dir, "transfer_cifar10_to_stl10.json"))
    stl10_acc = None
    if transfer:
        stl10_acc = transfer.get("best_accuracy", transfer.get("accuracy"))

    # Medical results
    medical = load_json(os.path.join(results_dir, "medical_ssl_results.json"))

    # Representation metrics
    rep_metrics = load_json(os.path.join(results_dir, "representation_metrics.json"))

    # Build rows per method
    for method in ["simclr", "moco", "byol", "supervised"]:
        row = {
            "method": method.upper() if method != "moco" else "MoCo v2",
            "cifar10_knn_top1": knn_map.get(method, ""),
            "cifar10_linear_probe": linear_probes.get(method, ""),
        }

        # STL-10 transfer (only SimCLR was tested)
        if method == "simclr" and stl10_acc:
            row["stl10_transfer"] = stl10_acc
        else:
            row["stl10_transfer"] = ""

        # Medical linear probe
        if medical and method in medical:
            # use the largest fraction available as "linear probe"
            fracs = {float(k): v for k, v in medical[method].items()}
            if fracs:
                max_frac = max(fracs.keys())
                row["chestxray_linear_probe"] = fracs[max_frac]
            else:
                row["chestxray_linear_probe"] = ""
        else:
            row["chestxray_linear_probe"] = ""

        # Representation metrics
        if rep_metrics and method in rep_metrics:
            rm = rep_metrics[method]
            row["alignment"] = rm.get("alignment", "")
            row["uniformity"] = rm.get("uniformity", "")
            row["mean_variance"] = rm.get("mean_variance", "")
            row["effective_rank"] = rm.get("effective_rank", "")
        else:
            row["alignment"] = ""
            row["uniformity"] = ""
            row["mean_variance"] = ""
            row["effective_rank"] = ""

        rows.append(row)

    # Write CSV
    csv_path = os.path.join(results_dir, "benchmark_results_full.csv")
    fieldnames = [
        "method", "cifar10_knn_top1", "cifar10_linear_probe",
        "stl10_transfer", "chestxray_linear_probe",
        "alignment", "uniformity", "mean_variance", "effective_rank",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark CSV saved: {csv_path}")
    print(f"\n{'Method':<12} {'CIFAR kNN':>10} {'Linear':>8} {'STL Trans':>10} {'CXR Probe':>10}")
    print("-" * 56)
    for r in rows:
        knn_val = f"{r['cifar10_knn_top1']:.2f}" if isinstance(r['cifar10_knn_top1'], float) else str(r['cifar10_knn_top1'])
        lp_val = f"{r['cifar10_linear_probe']:.2f}" if isinstance(r['cifar10_linear_probe'], float) else str(r['cifar10_linear_probe'])
        stl_val = f"{r['stl10_transfer']:.2f}" if isinstance(r['stl10_transfer'], float) else str(r['stl10_transfer'])
        cxr_val = f"{r['chestxray_linear_probe']:.2f}" if isinstance(r['chestxray_linear_probe'], float) else str(r['chestxray_linear_probe'])
        print(f"{r['method']:<12} {knn_val:>10} {lp_val:>8} {stl_val:>10} {cxr_val:>10}")


if __name__ == "__main__":
    main()

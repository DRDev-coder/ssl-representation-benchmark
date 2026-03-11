"""
Benchmark Table Generator
=========================
Aggregates results from all experiments and generates a comprehensive
benchmark comparison table in CSV and Markdown format.

Collects results from:
  - Label efficiency experiments
  - kNN evaluation
  - Transfer learning
  - Augmentation ablation
  - Medical SSL experiments

Usage:
    python experiments/generate_benchmark.py
    python experiments/generate_benchmark.py --results_dir results --output_format both
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_json_safe(path):
    """Load JSON file, return empty dict if not found."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def generate_benchmark(results_dir):
    """Collect all experiment results into a unified benchmark."""
    benchmark = {
        "label_efficiency": {},
        "knn": {},
        "transfer": {},
        "augmentation_ablation": {},
        "medical": {},
    }

    # Label efficiency
    le_data = load_json_safe(os.path.join(results_dir, "label_efficiency_results.json"))
    if le_data:
        benchmark["label_efficiency"] = le_data

    # Transfer learning
    for fname in os.listdir(results_dir) if os.path.isdir(results_dir) else []:
        if fname.startswith("transfer_") and fname.endswith(".json"):
            data = load_json_safe(os.path.join(results_dir, fname))
            key = fname.replace(".json", "")
            benchmark["transfer"][key] = data

    # Augmentation ablation
    ablation_data = load_json_safe(os.path.join(results_dir, "augmentation_ablation_results.json"))
    if ablation_data:
        benchmark["augmentation_ablation"] = ablation_data

    # Medical
    medical_data = load_json_safe(os.path.join(results_dir, "medical_ssl_results.json"))
    if medical_data:
        benchmark["medical"] = medical_data

    return benchmark


def benchmark_to_csv(benchmark, output_path):
    """Write benchmark results to CSV."""
    lines = []

    # Label efficiency table
    le = benchmark.get("label_efficiency", {})
    if le:
        lines.append("Label Efficiency Results")
        fractions = set()
        for method_data in le.values():
            fractions.update(method_data.keys())
        fractions = sorted(fractions, key=float)

        header = "Method," + ",".join(f"{float(f)*100:.0f}%" for f in fractions)
        lines.append(header)
        for method, data in le.items():
            row = method.upper() + "," + ",".join(
                f"{data.get(f, 'N/A')}" for f in fractions
            )
            lines.append(row)
        lines.append("")

    # Augmentation ablation
    ablation = benchmark.get("augmentation_ablation", {})
    if ablation:
        lines.append("Augmentation Ablation Results")
        lines.append("Configuration,Linear Probe Accuracy (%)")
        for config, acc in sorted(ablation.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
            lines.append(f"{config},{acc}")
        lines.append("")

    # Transfer learning
    transfer = benchmark.get("transfer", {})
    if transfer:
        lines.append("Transfer Learning Results")
        for exp_name, data in transfer.items():
            lines.append(f"\n{exp_name}")
            lines.append("Method,Accuracy (%)")
            for method, acc in data.items():
                lines.append(f"{method.upper()},{acc}")
        lines.append("")

    # Medical
    medical = benchmark.get("medical", {})
    if medical:
        lines.append("Medical SSL Results")
        fractions = set()
        for method_data in medical.values():
            fractions.update(method_data.keys())
        fractions = sorted(fractions, key=float)

        header = "Method," + ",".join(f"{float(f)*100:.0f}%" for f in fractions)
        lines.append(header)
        for method, data in medical.items():
            row = method.upper() + "," + ",".join(
                f"{data.get(f, 'N/A')}" for f in fractions
            )
            lines.append(row)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"CSV benchmark saved to {output_path}")


def benchmark_to_markdown(benchmark, output_path):
    """Write benchmark results to Markdown."""
    lines = ["# SSL Benchmark Results\n"]

    # Label efficiency
    le = benchmark.get("label_efficiency", {})
    if le:
        lines.append("## Label Efficiency\n")
        fractions = set()
        for method_data in le.values():
            fractions.update(method_data.keys())
        fractions = sorted(fractions, key=float)

        header = "| Method | " + " | ".join(f"{float(f)*100:.0f}%" for f in fractions) + " |"
        sep = "|" + "|".join(["---"] * (len(fractions) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for method, data in le.items():
            row = f"| {method.upper()} | " + " | ".join(
                f"{data.get(f, 'N/A')}" for f in fractions
            ) + " |"
            lines.append(row)
        lines.append("")

    # Augmentation ablation
    ablation = benchmark.get("augmentation_ablation", {})
    if ablation:
        lines.append("## Augmentation Ablation\n")
        lines.append("| Configuration | Linear Probe Acc (%) |")
        lines.append("|---|---|")
        for config, acc in sorted(ablation.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
            lines.append(f"| {config} | {acc} |")
        lines.append("")

    # Transfer learning
    transfer = benchmark.get("transfer", {})
    if transfer:
        lines.append("## Transfer Learning\n")
        for exp_name, data in transfer.items():
            lines.append(f"### {exp_name}\n")
            lines.append("| Method | Accuracy (%) |")
            lines.append("|---|---|")
            for method, acc in data.items():
                lines.append(f"| {method.upper()} | {acc} |")
            lines.append("")

    # Medical
    medical = benchmark.get("medical", {})
    if medical:
        lines.append("## Medical SSL (Chest X-Ray)\n")
        fractions = set()
        for method_data in medical.values():
            fractions.update(method_data.keys())
        fractions = sorted(fractions, key=float)

        header = "| Method | " + " | ".join(f"{float(f)*100:.0f}%" for f in fractions) + " |"
        sep = "|" + "|".join(["---"] * (len(fractions) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for method, data in medical.items():
            row = f"| {method.upper()} | " + " | ".join(
                f"{data.get(f, 'N/A')}" for f in fractions
            ) + " |"
            lines.append(row)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown benchmark saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Benchmark Tables")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_format", default="both", choices=["csv", "markdown", "both"])
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    benchmark = generate_benchmark(results_dir)

    # Save full JSON
    json_path = os.path.join(results_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Full benchmark JSON saved to {json_path}")

    if args.output_format in ("csv", "both"):
        benchmark_to_csv(benchmark, os.path.join(results_dir, "benchmark_results.csv"))

    if args.output_format in ("markdown", "both"):
        benchmark_to_markdown(benchmark, os.path.join(results_dir, "benchmark_results.md"))

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for category, data in benchmark.items():
        if data:
            if isinstance(data, dict):
                n = len(data)
                print(f"  {category}: {n} entries")
            else:
                print(f"  {category}: present")
        else:
            print(f"  {category}: (no data)")


if __name__ == "__main__":
    main()

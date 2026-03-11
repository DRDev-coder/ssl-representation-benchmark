"""
Hydra Unified Entry Point
=========================
Optional single entry point that dispatches to the appropriate
training or evaluation task using Hydra config.

Most users will run training scripts directly (each has its own
@hydra.main decorator):

    python training/train_simclr.py                     # SimCLR pretraining
    python training/train_supervised.py training=supervised  # Supervised baseline
    python training/linear_probe.py training=linear_eval     # Linear probe
    python training/fine_tune.py training=finetune           # Semi-supervised fine-tune

This file provides an alternative unified interface:

    python run.py task=pretrain
    python run.py task=supervised training=supervised
    python run.py task=linear_eval training=linear_eval
    python run.py task=finetune training=finetune
    python run.py task=knn_eval
    python run.py task=visualize

Hydra multirun (hyperparameter sweep):

    python run.py -m task=pretrain temperature=0.3,0.5,0.7 batch_size=128,256
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve_paths(cfg: DictConfig):
    """
    Resolve relative paths in config to absolute paths based on project root.

    Hydra changes the working directory, so we need absolute paths.
    """
    for key in ["data_dir", "checkpoint_dir", "results_dir", "mlflow_uri"]:
        path = cfg.paths.get(key, None)
        if path and not os.path.isabs(path):
            cfg.paths[key] = os.path.join(PROJECT_ROOT, path)

    # Resolve encoder paths in training config
    if hasattr(cfg, "training") and hasattr(cfg.training, "encoder_path"):
        ep = cfg.training.get("encoder_path", None)
        if ep and not os.path.isabs(ep):
            cfg.training.encoder_path = os.path.join(PROJECT_ROOT, ep)

    # Create output directories
    for key in ["data_dir", "checkpoint_dir", "results_dir"]:
        os.makedirs(cfg.paths[key], exist_ok=True)


# ── Task runners ─────────────────────────────────────────────

def run_pretrain(cfg: DictConfig):
    """Run SimCLR pretraining."""
    from training.train_simclr import train_simclr
    return train_simclr(cfg)


def run_linear_eval(cfg: DictConfig):
    """Run linear probe evaluation."""
    from training.linear_probe import train_linear_probe
    return train_linear_probe(cfg)


def run_finetune(cfg: DictConfig):
    """Run semi-supervised fine-tuning."""
    from training.fine_tune import train_finetune
    return train_finetune(cfg)


def run_supervised(cfg: DictConfig):
    """Run supervised baseline training."""
    from training.train_supervised import train_supervised
    return train_supervised(cfg)


def run_knn_eval(cfg: DictConfig):
    """Run kNN evaluation."""
    from evaluation.knn_eval import main as knn_main
    args_list = [
        "--dataset", cfg.dataset.name,
        "--encoder_path", cfg.training.get("encoder_path", "checkpoints/simclr_encoder_best.pth"),
        "--k", str(cfg.get("knn_k", 200)),
    ]
    if not cfg.mlflow.enabled:
        args_list.append("--no_mlflow")
    return knn_main(args_list)


def run_visualize(cfg: DictConfig):
    """Run t-SNE visualization + label efficiency curve."""
    from evaluation.tsne_visualization import main as tsne_main
    args_list = [
        "--dataset", cfg.dataset.name,
        "--encoder_path", cfg.training.get("encoder_path", "checkpoints/simclr_encoder_best.pth"),
        "--label_curve",
    ]
    return tsne_main(args_list)


# ── Task dispatch ────────────────────────────────────────────

TASK_REGISTRY = {
    "pretrain": run_pretrain,
    "linear_eval": run_linear_eval,
    "finetune": run_finetune,
    "supervised": run_supervised,
    "knn_eval": run_knn_eval,
    "visualize": run_visualize,
}


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    SimCLR project — Hydra unified entry point.

    Dispatches to the appropriate task based on the 'task' parameter.
    """
    # Print resolved config
    print("=" * 60)
    print("RESOLVED CONFIGURATION")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Resolve paths to absolute
    _resolve_paths(cfg)

    # Determine task — from 'task' key if present, else from training.name
    task_name = cfg.get("task", cfg.training.get("name", "pretrain"))
    if isinstance(task_name, DictConfig):
        task_name = task_name.get("name", "pretrain")

    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )

    print(f"\nRunning task: {task_name}\n")
    result = TASK_REGISTRY[task_name](cfg)

    print(f"\nTask '{task_name}' completed.")
    return result


if __name__ == "__main__":
    main()

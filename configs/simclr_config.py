"""
SimCLR Project Configuration
============================
Centralized configuration for all training, evaluation, and pipeline settings.
Optimized for NVIDIA RTX 5060 (8GB VRAM) with mixed precision training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MLFLOW_TRACKING_URI = os.path.join(PROJECT_ROOT, "mlruns")

# Create directories if they don't exist
for _dir in [DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(_dir, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# SimCLR Pretraining Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SimCLRConfig:
    """Configuration for SimCLR contrastive pretraining."""

    # Dataset
    dataset: str = "stl10"               # "stl10" or "cifar10"
    data_dir: str = DATA_DIR
    image_size: int = 96                 # STL10=96, CIFAR10=32

    # Architecture
    backbone: str = "resnet18"
    projection_hidden_dim: int = 2048
    projection_output_dim: int = 128

    # Training
    epochs: int = 200
    batch_size: int = 256
    accumulation_steps: int = 1          # Gradient accumulation
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.5
    optimizer: str = "adam"               # "adam" or "lars"

    # LR Scheduler
    scheduler: str = "cosine"            # "cosine" or "step"
    warmup_epochs: int = 10

    # Performance
    use_amp: bool = True                 # Mixed precision training
    num_workers: int = 8
    pin_memory: bool = True
    cudnn_benchmark: bool = True

    # Checkpointing
    checkpoint_dir: str = CHECKPOINTS_DIR
    save_every: int = 50                 # Save checkpoint every N epochs
    resume_from: Optional[str] = None

    # Logging
    log_every: int = 10                  # Log metrics every N steps
    mlflow_experiment: str = "simclr_pretrain"

    # Random seed
    seed: int = 42


# ──────────────────────────────────────────────────────────────────────
# Linear Evaluation Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class LinearEvalConfig:
    """Configuration for linear probe evaluation (frozen encoder)."""

    dataset: str = "stl10"
    data_dir: str = DATA_DIR
    image_size: int = 96

    # Pretrained encoder path
    encoder_path: str = os.path.join(CHECKPOINTS_DIR, "simclr_encoder_best.pth")

    # Training
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_classes: int = 10

    # Performance
    use_amp: bool = True
    num_workers: int = 8
    pin_memory: bool = True

    # Logging
    mlflow_experiment: str = "linear_eval"
    seed: int = 42


# ──────────────────────────────────────────────────────────────────────
# Fine-Tuning Config (Semi-Supervised)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class FineTuneConfig:
    """Configuration for semi-supervised fine-tuning with limited labels."""

    dataset: str = "stl10"
    data_dir: str = DATA_DIR
    image_size: int = 96

    # Pretrained encoder path
    encoder_path: str = os.path.join(CHECKPOINTS_DIR, "simclr_encoder_best.pth")

    # Label fractions to evaluate
    label_fractions: List[float] = field(default_factory=lambda: [0.01, 0.10, 1.0])

    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_classes: int = 10

    # Whether to freeze encoder (False = full fine-tuning)
    freeze_encoder: bool = False

    # Performance
    use_amp: bool = True
    num_workers: int = 8
    pin_memory: bool = True

    # Logging
    mlflow_experiment: str = "finetune_semisupervised"
    seed: int = 42


# ──────────────────────────────────────────────────────────────────────
# Supervised Baseline Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SupervisedConfig:
    """Configuration for fully supervised baseline training."""

    dataset: str = "stl10"
    data_dir: str = DATA_DIR
    image_size: int = 96

    # Architecture
    backbone: str = "resnet18"
    num_classes: int = 10

    # Label fractions (to compare with SSL)
    label_fractions: List[float] = field(default_factory=lambda: [0.01, 0.10, 1.0])

    # Training
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Performance
    use_amp: bool = True
    num_workers: int = 8
    pin_memory: bool = True

    # Logging
    mlflow_experiment: str = "supervised_baseline"
    seed: int = 42


# ──────────────────────────────────────────────────────────────────────
# Evaluation Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    """Configuration for kNN and t-SNE evaluation."""

    dataset: str = "stl10"
    data_dir: str = DATA_DIR
    image_size: int = 96

    encoder_path: str = os.path.join(CHECKPOINTS_DIR, "simclr_encoder_best.pth")
    supervised_encoder_path: str = os.path.join(CHECKPOINTS_DIR, "supervised_encoder_best.pth")

    # kNN
    knn_k: int = 200
    knn_temperature: float = 0.5

    # t-SNE
    tsne_perplexity: int = 30
    tsne_n_samples: int = 5000  # Number of samples for t-SNE
    tsne_seed: int = 42

    # Output
    results_dir: str = RESULTS_DIR
    num_classes: int = 10
    seed: int = 42

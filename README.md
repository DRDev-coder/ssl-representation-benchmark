# SimCLR: Self-Supervised Contrastive Learning Benchmark

A comprehensive **self-supervised learning (SSL) benchmark** implementing **SimCLR**, **MoCo v2**, and **BYOL** for learning high-quality image representations without labels. Includes full evaluation: linear probing, semi-supervised fine-tuning, supervised baselines, KNN evaluation, t-SNE/UMAP visualization, augmentation ablation, cross-dataset transfer learning, and medical imaging experiments.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-red.svg)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-green.svg)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Concepts](#key-concepts)
3. [SSL Methods](#ssl-methods)
4. [Project Pipeline](#project-pipeline)
5. [Repository Structure](#repository-structure)
6. [Installation](#installation)
7. [Dataset Setup](#dataset-setup)
8. [How to Run Experiments](#how-to-run-experiments)
9. [Advanced Experiments](#advanced-experiments)
10. [Interactive Demo](#interactive-demo)
11. [Evaluation Methods](#evaluation-methods)
12. [Research Questions](#research-questions)
12. [Results](#results)
14. [MLOps Integration](#mlops-integration)
15. [Hardware Requirements](#hardware-requirements)
16. [License](#license)
17. [Acknowledgements](#acknowledgements)

---

## Project Overview

This project implements a **multi-method self-supervised learning benchmark** featuring three major SSL frameworks — **SimCLR**, **MoCo v2**, and **BYOL** — to train ResNet encoders on unlabeled images using contrastive and non-contrastive learning. The core idea is to learn visual representations by training the model to recognize that two differently augmented views of the same image are semantically equivalent — without ever using class labels during pretraining.

### Implemented SSL Methods

| Method | Type | Key Idea | Paper |
|--------|------|----------|-------|
| **SimCLR** | Contrastive | NT-Xent loss over in-batch negatives | [Chen et al., 2020](https://arxiv.org/abs/2002.05709) |
| **MoCo v2** | Contrastive | Momentum encoder + dynamic queue of negatives | [He et al., 2020](https://arxiv.org/abs/1911.05722) |
| **BYOL** | Non-contrastive | Online/target networks, no negatives needed | [Grill et al., 2020](https://arxiv.org/abs/2006.07733) |

Once pretrained, each encoder is evaluated under multiple protocols:

- **Linear Probe**: freeze the encoder, train only a linear classifier on top using the full labeled set.
- **Semi-supervised Fine-tuning**: unfreeze the encoder and fine-tune with limited labeled data (1%, 5%, 10%, 50%, 100%).
- **Supervised Baseline**: train an identical ResNet from scratch with labels for direct comparison.
- **KNN Evaluation**: classify using nearest neighbors in the embedding space — no training required (Top-1 and Top-5).
- **t-SNE / UMAP Visualization**: project 512-d embeddings to 2-d to qualitatively assess cluster structure.
- **Label Efficiency Curves**: accuracy vs. label fraction across all SSL methods and supervised baseline.
- **Augmentation Ablation**: measure the impact of each augmentation component on representation quality.
- **Cross-Dataset Transfer**: pretrain on CIFAR-10, evaluate on STL-10 (or vice versa).
- **Medical Imaging SSL**: chest X-ray pretraining and disease classification with limited labels.

**Key result:** SimCLR fine-tuning with only **1% of labels (500 images)** achieves **61.81%** accuracy on CIFAR-10 — compared to **38.54%** for a supervised model trained from scratch on the same 500 images. A **+23.3 percentage point** advantage from self-supervised pretraining alone.

---

## Key Concepts

### Self-Supervised Learning

A paradigm where a model learns representations from unlabeled data by solving a pretext task designed to capture semantic structure. No human annotations are required during pretraining; labels are only used in a downstream fine-tuning or evaluation step. This makes it especially valuable in domains where labeled data is scarce or expensive (e.g., medical imaging, satellite imagery, industrial inspection).

### Contrastive Learning

A self-supervised approach that trains an encoder to map semantically similar inputs (positive pairs) close together in embedding space, while pushing dissimilar inputs (negative pairs) apart. SimCLR uses **augmentation-based positive pairs**: two differently augmented crops of the same image are treated as positives; all other images in the batch serve as negatives.

---

## SSL Methods

### SimCLR — Simple Contrastive Learning

SimCLR consists of two learned components:

| Component | Role | Architecture |
|-----------|------|-------------|
| **Encoder** `f(·)` | Maps image → feature vector `h` | ResNet-18/34/50 (no FC head) |
| **Projection Head** `g(·)` | Maps `h` → contrastive embedding `z` | MLP: `512 → 2048 → 128` + BN + ReLU |

During pretraining, the NT-Xent loss operates on the projection head outputs `z`. For all downstream tasks, the projection head is **discarded** and only the encoder representations `h` are used.

### MoCo v2 — Momentum Contrast

MoCo maintains a **momentum-updated key encoder** and a **dynamic queue** of encoded negatives, decoupling the dictionary size from the batch size:

| Component | Role |
|-----------|------|
| **Query encoder** | Online encoder, updated by gradient descent |
| **Key encoder** | Momentum copy: `θ_k = m·θ_k + (1−m)·θ_q` |
| **Queue** | FIFO buffer of 65,536 encoded keys |

This allows MoCo to use a very large number of negatives (64K) regardless of batch size.

### BYOL — Bootstrap Your Own Latent

BYOL learns without negative pairs at all via an **asymmetric architecture**:

| Component | Role |
|-----------|------|
| **Online network** | Encoder + projector + **predictor** (MLP) |
| **Target network** | Encoder + projector (momentum-updated, no predictor) |

The predictor breaks the symmetry, preventing representation collapse. BYOL minimizes the negative cosine similarity between the online prediction and the target projection.

### NT-Xent Loss

The **Normalized Temperature-scaled Cross Entropy** loss:

$$\ell_{i,j} = -\log \frac{\exp\!\bigl(\text{sim}(z_i, z_j)\,/\,\tau\bigr)}{\displaystyle\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]}\,\exp\!\bigl(\text{sim}(z_i, z_k)\,/\,\tau\bigr)}$$

where `sim(u, v)` is cosine similarity, `τ` is the temperature parameter (default `0.5`), and `N` is the batch size. For a batch of `N` images, `2N` augmented views are generated, giving `2(N−1)` negatives per anchor. The theoretical lower bound for a random encoder with batch size 128 is approximately `log(255) ≈ 5.54`.

### Linear Evaluation Protocol

The gold-standard benchmark for self-supervised representations. The pretrained encoder is **completely frozen** and a single `Linear(feature_dim, num_classes)` layer is trained on top using the full labeled training set. This directly measures representation quality without the confound of fine-tuning. High linear probe accuracy indicates that class-discriminative information is linearly accessible in the embedding space.

### Label Efficiency

The primary practical motivation for self-supervised learning: the model can achieve high accuracy with very few labeled examples by leveraging representations learned from large amounts of unlabeled data. This is evaluated at 1%, 10%, and 100% label fractions and compared to a supervised model trained from scratch on the same label budget.

---

## Project Pipeline

```
Raw Images
    │
    ▼
┌─────────────────────────────────────┐
│  Data Augmentation (SimCLRTransform) │
│  • RandomResizedCrop (scale 0.2–1.0) │
│  • RandomHorizontalFlip              │
│  • ColorJitter    (p=0.8)            │
│  • RandomGrayscale (p=0.2)           │
│  • GaussianBlur   (p=0.5)            │
│  → Produces TWO views (x_i, x_j)    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│  Encoder f(·)   │   ResNet-18/34/50
│  x → h ∈ ℝ^512 │   (trained from scratch, no pretraining)
└─────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Projection Head g(·)   │   MLP: 512 → 2048 → 128
│  h → z ∈ ℝ^128          │   BatchNorm + ReLU at each layer
└─────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│  NT-Xent Contrastive Loss     │
│  (z_i, z_j) over 2N views     │
│  Temperature τ = 0.5          │
└───────────────────────────────┘
    │
    ▼ (projection head discarded after pretraining)
┌──────────────────────────────┐
│  Pretrained Encoder f(·)     │
│  h = f(x) ∈ ℝ^512            │
└──────────────────────────────┘
    │
    ├──► Linear Probe      (freeze encoder → train Linear(512,10) only)
    │
    ├──► Fine-Tuning       (unfreeze encoder → train with 1%/10%/100% labels)
    │
    ├──► KNN Evaluation    (encode all images → k-NN majority vote, no training)
    │
    └──► t-SNE             (512-d → 2-d → scatter plot colored by class)
```

---

## Repository Structure

```
simCLR/
│
├── run.py                        # Top-level unified entry point
│
├── augmentations/
│   ├── simclr_augmentations.py   # SimCLRTransform: dual-view augmentation pipeline
│   └── medical_augmentations.py  # Medical-safe augmentations for X-ray images
│
├── configs/
│   ├── config.yaml               # Hydra root config (composes all sub-configs)
│   ├── simclr_config.py          # Dataclass config schema
│   ├── dataset/
│   │   ├── cifar10.yaml          # CIFAR-10 config (32×32, 10 classes)
│   │   ├── stl10.yaml            # STL-10 config  (96×96, 10 classes)
│   │   └── chestxray.yaml        # Chest X-Ray config (224×224, 2 classes)
│   ├── experiments/              # Pre-built experiment presets
│   │   ├── simclr_cifar10.yaml
│   │   ├── simclr_stl10.yaml
│   │   ├── simclr_resnet34.yaml
│   │   ├── moco_cifar10.yaml     # MoCo v2 on CIFAR-10
│   │   └── byol_cifar10.yaml     # BYOL on CIFAR-10
│   ├── method/                   # SSL method configs
│   │   ├── simclr.yaml
│   │   ├── moco.yaml
│   │   └── byol.yaml
│   ├── model/
│   │   ├── resnet18.yaml
│   │   ├── resnet34.yaml
│   │   └── resnet50.yaml
│   ├── optimizer/
│   │   ├── adam.yaml
│   │   └── sgd.yaml
│   └── training/
│       ├── simclr.yaml
│       ├── ssl_pretrain.yaml     # Unified SSL pretraining config
│       ├── linear_eval.yaml
│       ├── finetune.yaml
│       └── supervised.yaml
│
├── datasets/
│   ├── cifar10_dataset.py        # CIFAR-10 DataLoader factory (auto-downloads)
│   ├── stl10_dataset.py          # STL-10 DataLoader factory  (auto-downloads)
│   └── chestxray_dataset.py      # Chest X-Ray DataLoader (folder/CSV-based)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── evaluation/
│   ├── knn_eval.py               # K-NN accuracy (Top-1 and Top-5)
│   ├── tsne_visualization.py     # t-SNE plots + label efficiency curve
│   ├── umap_visualization.py     # UMAP visualization (single + multi-method)
│   ├── run_representation_analysis.py  # Alignment, uniformity, CKA, collapse runner
│   └── representation_metrics/   # Quantitative representation metrics
│       ├── alignment.py          # Positive-pair embedding distance
│       ├── uniformity.py         # Hypersphere distribution quality
│       ├── cka.py                # Centered Kernel Alignment
│       └── collapse_detection.py # Embedding variance & covariance rank
│
├── experiments/
│   ├── label_efficiency.py       # Multi-method label fraction sweep
│   ├── augmentation_ablation.py  # Augmentation component ablation study
│   ├── transfer_learning.py      # Cross-dataset transfer evaluation
│   ├── medical_ssl_experiment.py # Medical imaging SSL pipeline
│   └── generate_benchmark.py     # Aggregate results into benchmark tables
│
├── mlops/
│   ├── mlflow_logger.py          # MLflow experiment tracking wrapper
│   ├── mlflow_tracker.py         # High-level experiment tracker (dataset info, checkpoints, auto-naming)
│   └── dvc_pipeline.yaml         # DVC pipeline definition
│
├── analysis/
│   └── generate_mlflow_summary.py # Query MLflow runs, generate summary plots + CSV
│
├── models/
│   ├── resnet_encoder.py         # ResNet backbone (avgpool output, no FC)
│   ├── projection_head.py        # 2-layer MLP projection head with BatchNorm
│   ├── simclr_model.py           # SimCLR = encoder + projection head
│   └── ssl_methods/              # Unified SSL method implementations
│       ├── simclr_method.py      # SimCLR wrapper (NT-Xent loss)
│       ├── moco.py               # MoCo v2 (momentum encoder + queue)
│       └── byol.py               # BYOL (online/target networks, no negatives)
│
├── scripts/
│   ├── run_pretrain.sh
│   ├── run_linear_eval.sh
│   ├── run_finetune.sh
│   └── run_all.bat
│
├── training/
│   ├── train_simclr.py           # SimCLR contrastive pretraining
│   ├── train_ssl.py              # Unified SSL training (SimCLR/MoCo/BYOL)
│   ├── linear_probe.py           # Frozen encoder + linear classifier
│   ├── fine_tune.py              # SimCLR encoder fine-tuning
│   └── train_supervised.py       # Supervised ResNet-18 baseline
│
├── utils/
│   ├── losses.py                 # NT-Xent loss implementation
│   ├── metrics.py                # Accuracy and evaluation utilities
│   ├── device.py                 # GPU/CPU setup and memory reporting
│   └── seed.py                   # Global reproducibility seed
│
├── interactive/
│   ├── embedding_extractor.py    # Extract & save embeddings to .npz
│   ├── similarity_search.py      # Cosine-similarity nearest-neighbour index
│   └── streamlit_app.py          # Interactive web app for embedding exploration
│
├── research/
│   ├── hypothesis.md             # Formal hypothesis statements (H1–H7)
│   ├── hypothesis_H7_medical_ssl.md  # Medical SSL hypothesis (detailed)
│   ├── experiment_plan.md         # Experiment designs, commands, pass criteria
│   └── analysis_template.md       # Template + completed write-ups for results
│
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.x (recommended: 8 GB+ VRAM)
- NVIDIA driver ≥ 525

### 1. Clone the repository

```bash
git clone https://github.com/your-username/simclr.git
cd simclr
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv simclr_env
simclr_env\Scripts\Activate.ps1

# Linux / macOS
python3.11 -m venv simclr_env
source simclr_env/bin/activate
```

### 3. Install PyTorch with CUDA support

Install PyTorch **before** the rest of the requirements to ensure the correct CUDA build is selected:

```bash
# CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify GPU availability

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

### Docker (alternative)

```bash
# Build image
docker build -t simclr-project -f docker/Dockerfile .

# Run with GPU
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  simclr-project
```

---

## Dataset Setup

Both supported datasets **download automatically** on first run via TorchVision. No manual download is needed.

```
data/
├── cifar-10-batches-py/    # CIFAR-10  (~170 MB)  — auto-downloaded
└── stl10_binary/           # STL-10    (~2.6 GB)  — auto-downloaded
```

| Dataset | Image Size | Classes | Train | Test | Notes |
|---------|-----------|---------|-------|------|-------|
| CIFAR-10 | 32×32 | 10 | 50,000 | 10,000 | Fast to download, ideal for quick experiments |
| STL-10 | 96×96 | 10 | 5,000 + 100,000 unlabeled | 8,000 | Canonical SimCLR benchmark |

> **Recommendation for first runs:** Use `dataset=cifar10`. STL-10 requires a ~2.6 GB download and significantly longer per-epoch training time.

The data directory can be overridden at runtime:

```bash
python training/train_simclr.py dataset=cifar10 paths.data_dir=/path/to/data
```

---

## How to Run Experiments

All training scripts use [Hydra](https://hydra.cc/) for configuration. Any parameter can be overridden directly on the command line without editing YAML files.

> **Windows users:** Always append `num_workers=0` to avoid multiprocessing spawn errors.

---

### Stage 1 — SimCLR Pretraining

Train the ResNet encoder using self-supervised contrastive learning. **No labels are used.**

```bash
# CIFAR-10 (recommended for quick experiments)
python training/train_simclr.py dataset=cifar10 num_workers=0

# STL-10 (canonical benchmark — requires ~2.6 GB download)
python training/train_simclr.py dataset=stl10 num_workers=4

# Custom hyperparameters
python training/train_simclr.py dataset=cifar10 batch_size=256 epochs=200 temperature=0.5

# Hyperparameter sweep (Hydra multirun)
python training/train_simclr.py -m temperature=0.3,0.5,0.7 batch_size=128,256
```

Checkpoints saved to `checkpoints/`:

| File | Contents |
|------|---------|
| `simclr_encoder_best.pth` | Best encoder weights (used for all downstream tasks) |
| `simclr_full_best.pth` | Best full model (encoder + projection head) |
| `simclr_encoder_final.pth` | Final epoch encoder |
| `simclr_checkpoint_ep{N}.pth` | Full training state (optimizer, scheduler, epoch) |

---

### Stage 2 — Linear Probe Evaluation

Evaluate representation quality by training **only** a linear classifier on a fully **frozen** encoder.

```bash
python training/linear_probe.py training=linear_eval dataset=cifar10 num_workers=0
```

---

### Stage 3 — Semi-Supervised Fine-Tuning

Unfreeze the encoder and fine-tune with limited labels (1%, 10%, and 100% of the training set).

```bash
python training/fine_tune.py training=finetune dataset=cifar10 num_workers=0
```

Results saved to `results/finetune_results.json`.

---

### Stage 4 — Supervised Baseline

Train an identical ResNet-18 **from random initialization** using labels for direct comparison against SimCLR.

```bash
python training/train_supervised.py training=supervised dataset=cifar10 num_workers=0
```

Results saved to `results/supervised_results.json`.

---

### Stage 5 — KNN Evaluation

Evaluate the frozen encoder using K-Nearest Neighbors in embedding space. No classifier training required.

```bash
python -m evaluation.knn_eval \
  --dataset cifar10 \
  --encoder_path checkpoints/simclr_encoder_best.pth \
  --k 200 \
  --no_mlflow
```

---

### Stage 6 — t-SNE Visualization

Generate 2D embedding plots and a label efficiency curve comparing SimCLR vs. supervised.

```bash
python -m evaluation.tsne_visualization \
  --dataset cifar10 \
  --encoder_path checkpoints/simclr_encoder_best.pth \
  --label_curve
```

Output files saved to `results/`:

| File | Description |
|------|-------------|
| `tsne_simclr.png` | SimCLR encoder embedding space (colored by class) |
| `tsne_supervised.png` | Supervised encoder embedding space |
| `tsne_comparison.png` | Side-by-side SimCLR vs. Supervised |
| `label_efficiency_curve.png` | Accuracy vs. label fraction curve |

---

## Advanced Experiments

### Unified SSL Training (SimCLR / MoCo v2 / BYOL)

Train any SSL method using the unified training script:

```bash
# SimCLR (default)
python training/train_ssl.py method=simclr dataset=cifar10 num_workers=0

# MoCo v2
python training/train_ssl.py method=moco dataset=cifar10 num_workers=0

# BYOL (no negatives)
python training/train_ssl.py method=byol dataset=cifar10 num_workers=0

# Using experiment presets
python training/train_ssl.py +experiments=moco_cifar10 num_workers=0
python training/train_ssl.py +experiments=byol_cifar10 num_workers=0
```

Each method saves its encoder to `checkpoints/{method}_encoder_best.pth`.

---

### Label Efficiency Experiment

Systematically compare all methods across multiple label fractions (1%, 5%, 10%, 50%, 100%):

```bash
python experiments/label_efficiency.py \
  --methods simclr moco byol supervised \
  --fractions 0.01 0.05 0.10 0.50 1.0 \
  --dataset cifar10 --epochs 50
```

Outputs: `results/label_efficiency_results.json` and `results/label_efficiency_curve.png`.

---

### Augmentation Ablation Study

Measure the contribution of each augmentation component to representation quality:

```bash
python experiments/augmentation_ablation.py \
  --dataset cifar10 --pretrain_epochs 50 --probe_epochs 50

# Test specific configurations
python experiments/augmentation_ablation.py --configs full no_color_jitter crop_flip_only
```

Tested configurations: `full`, `no_color_jitter`, `no_blur`, `no_grayscale`, `no_crop`, `crop_flip_only`.

Outputs: `results/augmentation_ablation_results.json` and `results/augmentation_ablation.png`.

---

### UMAP Visualization

UMAP preserves global structure better than t-SNE and runs faster:

```bash
python -m evaluation.umap_visualization \
  --methods simclr moco byol supervised \
  --dataset cifar10

# Falls back to t-SNE if umap-learn is not installed
pip install umap-learn  # Optional
```

Outputs: `results/umap_{method}.png` per method and `results/umap_comparison.png` (side-by-side).

---

### Cross-Dataset Transfer Learning

Pretrain on CIFAR-10, evaluate on STL-10:

```bash
python experiments/transfer_learning.py \
  --methods simclr moco byol \
  --source cifar10 --target stl10 --epochs 50
```

Outputs: `results/transfer_cifar10_to_stl10.json` and corresponding plot.

---

### Medical Imaging SSL

SSL pretraining + fine-tuning on chest X-ray data:

```bash
# Prepare data in data/chestxray/train/{normal,pneumonia}/ and data/chestxray/test/{normal,pneumonia}/
python experiments/medical_ssl_experiment.py \
  --methods simclr --data_dir data/chestxray \
  --pretrain_epochs 50 --fractions 0.01 0.10 0.50 1.0
```

Outputs: `results/medical_ssl_results.json` and comparison plot.

---

### Generate Benchmark Tables

Aggregate all experiment results into CSV and Markdown tables:

```bash
python experiments/generate_benchmark.py --output_format both
```

Outputs: `results/benchmark_results.csv`, `results/benchmark_results.md`, `results/benchmark_results.json`.

---

## Interactive Demo

An interactive Streamlit application lets you **explore the learned representation space** in real time. Upload any image (or pick one from the dataset), compute its embedding with the trained encoder, and retrieve the most semantically similar images — demonstrating that the self-supervised encoder has learned meaningful visual similarity without ever seeing labels.

### Quick Start

```bash
# 1. Pre-compute embeddings for the reference dataset (one-time)
python interactive/embedding_extractor.py \
  --encoder checkpoints/simclr_encoder_best.pth \
  --dataset cifar10 --split test

# 2. Launch the web app
streamlit run interactive/streamlit_app.py
```

The app opens at `http://localhost:8501` and provides:

| Feature | Description |
|---------|-------------|
| **Image upload** | Upload any PNG / JPG and embed it with the trained encoder |
| **Dataset browser** | Pick an image from the pre-computed index by index number |
| **Nearest neighbours** | Top-k most similar images retrieved by cosine similarity |
| **Similarity histogram** | Distribution of the query's similarity to all dataset images |

### Components

| File | Purpose |
|------|---------|
| `interactive/embedding_extractor.py` | Encodes all images in a dataset and saves embeddings, labels, and pixel data to a `.npz` file in `embeddings/` |
| `interactive/similarity_search.py` | `SimilarityIndex` class: loads a `.npz` file and provides `query(embedding, k)` → `(indices, scores)` via cosine similarity |
| `interactive/streamlit_app.py` | Streamlit web UI wiring the encoder and index together |

### CLI Usage (without Streamlit)

The extractor and search modules also work standalone:

```bash
# Extract embeddings
python interactive/embedding_extractor.py \
  --encoder checkpoints/simclr_encoder_best.pth \
  --dataset cifar10 --split test

# Find neighbours for image #42
python interactive/similarity_search.py \
  --npz embeddings/cifar10_test.npz --query_idx 42 --k 5
```

---

## Evaluation Methods

### Linear Probe Accuracy

The standard benchmark for self-supervised learning. The entire encoder is frozen and only a single `Linear(feature_dim, num_classes)` layer is trained on top using labeled data. A high score confirms that the encoder has learned **linearly separable, semantically meaningful** features without any label supervision during pretraining.

### KNN Accuracy

All training images are encoded and test images are classified by majority vote among their `k` nearest neighbors (cosine similarity). Reports both **Top-1** and **Top-5** accuracy. Requires **zero training** — it is a direct probe of how well-clustered the embedding space is.

### t-SNE / UMAP Visualization

[t-SNE](https://lvdmaaten.github.io/tsne/) and [UMAP](https://umap-learn.readthedocs.io/) reduce the 512-d encoder output to 2-d for qualitative inspection. UMAP is preferred for its speed and better global structure preservation. Multi-method comparison plots allow visual comparison across SSL methods.

### Label Efficiency Curve

Plots downstream classification accuracy as a function of the fraction of labeled data used for fine-tuning (1%, 10%, 100%). The gap between SimCLR and supervised training at low label fractions quantifies the practical benefit of self-supervised pretraining.

---

## Research Questions

This project is structured as a small research study. The `research/` folder contains formal hypothesis statements, experimental plans, and analysis write-ups.

### Core Questions

| # | Research Question | Hypothesis | Status |
|:-:|-------------------|------------|--------|
| 1 | **How much does SSL pretraining help when labels are scarce?** | SimCLR fine-tuning at 1 % labels beats supervised training by ≥ 15 pp | **Confirmed** (+23.27 pp) |
| 2 | **Does the SSL advantage disappear with abundant labels?** | The gap shrinks monotonically and reverses at 100 % | **Confirmed** |
| 3 | **Is the embedding space well-clustered?** | KNN accuracy falls within 5 pp of linear probe accuracy | **Confirmed** (2.81 pp gap) |
| 4 | **Which augmentation matters most?** | Removing color jitter causes the largest accuracy drop | Pending |
| 5 | **Do non-contrastive methods match contrastive ones?** | BYOL ≈ SimCLR ≈ MoCo within 3 pp | Pending |
| 6 | **Do representations transfer across datasets?** | CIFAR-10 → STL-10 transfer achieves ≥ 50 % | Pending |
| 7 | **Does SSL help in domain-specific settings?** | SSL + 10 % labels beats supervised + 10 % labels by ≥ 5 pp on chest X-rays | Pending |

### Research Documentation

| File | Description |
|------|-------------|
| [`research/hypothesis.md`](research/hypothesis.md) | Seven formal hypotheses with rationale, literature grounding, and observed results |
| [`research/experiment_plan.md`](research/experiment_plan.md) | Detailed designs for experiments E1–E7: procedures, commands, metrics, pass criteria |
| [`research/analysis_template.md`](research/analysis_template.md) | Reusable template for experiment write-ups, plus completed analyses for E1 and E2 |

The central finding confirmed so far: **self-supervised pretraining is most valuable in the low-label regime**. At 1 % labels (500 CIFAR-10 images), SimCLR fine-tuning achieves 61.81 % — a +23.27 pp advantage over a supervised model seeing the same 500 images. This advantage diminishes as labels increase, and supervised training overtakes at 100 % (84.21 % vs. 81.10 %).

---

## Results

All experiments run on **CIFAR-10**, **ResNet-18**, 100 epochs, batch size 128, Adam (lr=3e-4), RTX 5060 Laptop GPU (8 GB).

### Stage 1 — SimCLR Pretraining

| Metric | Value |
|--------|-------|
| Initial NT-Xent loss (epoch 1) | 5.2398 |
| Best NT-Xent loss (epoch 99) | **4.1436** |
| Final NT-Xent loss (epoch 100) | 4.1457 |
| Loss reduction | −21.1% |
| Total training time | ~2.3 hours |

> NT-Xent theoretical lower bound for random encoder (batch=128): `log(255) ≈ 5.54`. Reaching 4.14 indicates the encoder has learned meaningful representations; well-optimised SimCLR with larger batches (256–512) and more epochs (200+) typically reaches 2.5–3.5.

### Stage 2 & 5 — Representation Quality (No Fine-tuning)

| Method | Accuracy | Notes |
|--------|---------|-------|
| **Linear Probe** (100% labels, frozen encoder) | **67.56%** | Trained classifier only |
| **KNN** (k=200, no training at all) | **64.75%** | Pure representation quality |
| Random baseline | ~10.0% | 10 classes, uniform |

The 2.81% gap between KNN and linear probe indicates a clean, well-clustered embedding space.

### Stage 3 vs. Stage 4 — Label Efficiency

| Label Fraction | Supervised (scratch) | SimCLR Fine-tune | SimCLR Advantage |
|:--------------:|:-------------------:|:----------------:|:----------------:|
| **1%** (500 images) | 38.54% | **61.81%** | **+23.27 pp** |
| **10%** (5,000 images) | 63.13% | **71.86%** | **+8.73 pp** |
| **100%** (50,000 images) | **84.21%** | 81.10% | −3.11 pp |

**Key takeaways:**

1. At **1% labels** (only 500 images), SimCLR fine-tuning achieves 61.81% vs. 38.54% supervised — a **+23.3 percentage point advantage**. This is the core value proposition of self-supervised learning: the pretrained encoder provides a dramatically better initialization when labels are scarce.

2. At **10% labels**, SimCLR maintains a strong **+8.7 pp** lead, showing the benefit persists well beyond the extreme low-label regime.

3. At **100% labels**, supervised training edges SimCLR by 3.1 pp. This is expected — with full data, supervised training with an unconstrained learning rate is hard to beat, and our fine-tuning used a conservatively scaled encoder learning rate (`encoder_lr = head_lr × 0.1`). Increasing this scale would likely close the gap.

---

## Medical SSL Experiment

### Motivation

Medical imaging is a prototypical **label-scarce domain** — expert annotations are expensive and time-consuming. SSL pretraining leverages large pools of unlabeled scans to learn structural features (organ edges, texture patterns) before seeing any labels.

### Dataset: Chest X-ray Pneumonia

| Property | Value |
|----------|-------|
| Task | Binary classification: Normal vs. Pneumonia |
| Image type | Grayscale chest X-rays |
| Resolution | Resized to 224×224 |
| Data structure | `data/chestxray/train/{NORMAL,PNEUMONIA}/` |

### Medical-Safe Augmentations

Standard SSL augmentations (strong color jitter) can destroy diagnostic information in X-rays. Our medical augmentation pipeline uses safe transforms:

| Augmentation | Parameters |
|-------------|------------|
| RandomResizedCrop | scale=(0.5, 1.0) |
| HorizontalFlip | p=0.5 |
| Small rotation | ±10° |
| Brightness/contrast | ±15% (p=0.5) |
| Gaussian noise | σ=0.01–0.03 (p=0.3) |
| **No** strong color jitter | ✗ |

### Running the Medical Experiment

```bash
# Full pipeline: pretrain + evaluate at 1%/5%/10%/100% + supervised baseline
python experiments/medical_ssl_experiment.py \
  --methods simclr moco byol \
  --pretrain_epochs 50 --finetune_epochs 30

# Using Hydra for individual SSL pretraining
python training/train_ssl.py dataset=chestxray method=simclr num_workers=0

# Linear probe on medical data
python training/linear_probe.py dataset=chestxray \
  training.encoder_path=checkpoints/medical_simclr_encoder.pth num_workers=0
```

### Output

| File | Description |
|------|-------------|
| `results/medical_label_efficiency_curve.png` | Accuracy vs. label fraction for all methods |
| `results/medical_ssl_results.json` | Raw accuracy numbers per method and fraction |
| `checkpoints/medical_{method}_encoder.pth` | Pretrained medical encoders |

### Hypothesis H7

> SSL pretraining on chest X-rays with 10% labels outperforms supervised from scratch by ≥5 pp.

See [research/hypothesis_H7_medical_ssl.md](research/hypothesis_H7_medical_ssl.md) for full details.

---

## Representation Quality Analysis

Quantitative metrics from modern SSL research for evaluating learned representations.

### Metrics

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| **Alignment** | Similarity between positive-pair embeddings | Lower = better |
| **Uniformity** | Distribution of embeddings on the hypersphere | More negative = better |
| **CKA Similarity** | Similarity between representation spaces of different methods | 0–1 scale |
| **Embedding Variance** | Per-dimension feature variance | Higher = less collapse |
| **Covariance Rank** | Effective dimensionality of the feature space | Higher ratio = better |

### Running the Analysis

```bash
# Run all representation metrics across available encoders
python evaluation/run_representation_analysis.py --dataset cifar10

# On medical data
python evaluation/run_representation_analysis.py --dataset chestxray
```

### Output

| File | Description |
|------|-------------|
| `results/cka_similarity_heatmap.png` | Pairwise CKA matrix heatmap |
| `results/embedding_variance_analysis.png` | Per-method variance bar chart + per-dim distribution |
| `results/alignment_uniformity.png` | Alignment vs. uniformity scatter plot |
| `results/representation_metrics.json` | Raw metric numbers |

### Benchmark CSV

Aggregate all results into a single table:

```bash
python results/generate_benchmark_csv.py
```

Output: `results/benchmark_results_full.csv`

| Method | CIFAR kNN | Linear Probe | STL Transfer | ChestXray Probe | Alignment | Uniformity |
|--------|-----------|-------------|-------------|----------------|-----------|------------|
| SimCLR | X | X | X | X | X | X |
| MoCo v2 | X | X | — | X | X | X |
| BYOL | X | X | — | X | X | X |
| Supervised | X | — | — | X | X | X |

---

## MLOps Integration

### MLflow — Experiment Tracking

Every experiment in this project automatically logs **configuration parameters**, **training metrics**, **representation analysis metrics**, **visualization artifacts**, and **model checkpoints** to MLflow — providing a complete, queryable experiment history similar to a machine learning research lab notebook.

#### What Gets Logged

| Category | Examples |
|----------|----------|
| **Parameters** | method, dataset, backbone, lr, batch_size, temperature, projection_dim, epochs, augmentation settings, label fraction |
| **Training Metrics** | train_loss (per epoch), ssl_loss, learning_rate, epoch_time |
| **Evaluation Metrics** | linear_probe_accuracy, knn_top1/top5, finetune accuracy at each label % |
| **Representation Metrics** | alignment, uniformity, CKA similarity, embedding_variance, effective_rank |
| **Dataset Info** | dataset name, num_samples, image_size, num_classes |
| **Artifacts** | encoder checkpoints (.pth), Hydra config YAML, t-SNE/UMAP/CKA plots, label efficiency curves, benchmark CSVs |

#### Experiment Naming

Experiments are organized under clear names:

| Experiment Name | Contents |
|-----------------|----------|
| `SSL_Benchmark_CIFAR10` | All CIFAR-10 SSL runs |
| `SSL_Benchmark_STL10` | All STL-10 SSL runs |
| `SSL_Benchmark_CHESTXRAY` | Medical imaging experiments |
| `ssl_simclr`, `ssl_moco`, `ssl_byol` | Per-method pretraining |
| `knn_evaluation` | kNN evaluation runs |
| `simclr_project` | General project runs |

Each run is named descriptively, e.g. `simclr_cifar10_resnet18`, `finetune_cifar10`, `representation_analysis_cifar10`.

#### Starting the MLflow Dashboard

```bash
# Launch the MLflow UI
mlflow ui --backend-store-uri mlruns/
```

Then open **http://localhost:5000** in your browser. You can:

- **Compare runs** side-by-side (loss curves, accuracy, representation metrics)
- **Filter by experiment** to focus on a specific dataset or method
- **Download artifacts** (model checkpoints, plots, config files)
- **Sort by metric** to find the best-performing configuration

#### Programmatic API

The project provides two MLflow interfaces:

1. **`MLflowLogger`** (`mlops/mlflow_logger.py`) — Low-level wrapper used in training loops for metric/param/artifact logging.
2. **`ExperimentTracker`** (`mlops/mlflow_tracker.py`) — High-level API with automatic experiment naming, dataset info logging, and model checkpoint tracking.

```python
from mlops.mlflow_tracker import ExperimentTracker

# OOP usage (auto-detects experiment name from Hydra config)
tracker = ExperimentTracker(cfg)
tracker.start_experiment()
tracker.log_dataset_info(num_samples=50000)
tracker.log_metrics({"loss": 0.5, "accuracy": 67.5}, step=10)
tracker.log_model_checkpoint("checkpoints/encoder.pth")
tracker.end_experiment()

# Or use the procedural API
from mlops.mlflow_tracker import start_experiment, log_metrics, end_experiment
start_experiment(cfg)
log_metrics({"loss": 0.5})
end_experiment()
```

#### Generating Summary Reports from MLflow

```bash
# Generate publication plots + CSV from all logged MLflow runs
python analysis/generate_mlflow_summary.py
```

This produces:
- `results/mlflow_method_comparison.png` — Bar chart of pretraining loss and linear eval accuracy
- `results/mlflow_label_efficiency.png` — Label efficiency curve reconstructed from MLflow runs
- `results/mlflow_representation_comparison.png` — Alignment, uniformity, rank ratio comparison
- `results/mlflow_summary.csv` — Full table of all runs with parameters and metrics

#### Disabling MLflow

Set `mlflow.enabled: false` in `configs/config.yaml`, or pass `--no_mlflow` to standalone scripts like `knn_eval.py` and `run_representation_analysis.py`.

### DVC — Data & Pipeline Versioning

The `dvc.yaml` defines a fully reproducible pipeline from data download through pretraining to evaluation. Dataset artifacts and model checkpoints are tracked as DVC outputs, enabling exact reproduction of any experiment.

```bash
# Run the full DVC pipeline
dvc repro

# Check what would be re-run
dvc status
```

### Hydra — Configuration Management

All hyperparameters are managed through a hierarchical YAML config system. Configs are composable: dataset, model, optimizer, and training configs are mixed and matched at runtime without editing any files.

```bash
# Override any parameter on the command line
python training/train_simclr.py dataset=stl10 model=resnet34 optimizer=sgd batch_size=256 epochs=200

# Multirun sweep over a parameter grid
python training/train_simclr.py -m temperature=0.3,0.5,0.7 batch_size=128,256
```

Each run produces a timestamped output directory under `outputs/` containing the resolved config, override list, and training log.

### Docker — Reproducibility

A multi-stage `Dockerfile` based on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` ensures a fully reproducible environment. All dependencies are pinned.

```bash
docker build -t simclr-project -f docker/Dockerfile .
docker run --gpus all -v $(pwd)/data:/app/data simclr-project \
  python training/train_simclr.py dataset=cifar10 num_workers=4
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 6 GB | 8 GB+ |
| GPU | GTX 1660 Ti | RTX 3080 / RTX 4070 / RTX 5060+ |
| CUDA | 11.8+ | 12.4+ |
| System RAM | 8 GB | 16 GB |
| Disk | 5 GB | 10 GB |

Mixed precision training (`use_amp: true`) is enabled by default and significantly reduces VRAM usage and training time on Ampere / Ada Lovelace / Blackwell GPUs.

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).

---

## Acknowledgements

This project implements multiple self-supervised learning frameworks:

> **Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton.**
> *A Simple Framework for Contrastive Learning of Visual Representations.*
> ICML 2020. [[arXiv:2002.05709]](https://arxiv.org/abs/2002.05709)

> **Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick.**
> *Momentum Contrast for Unsupervised Visual Representation Learning.*
> CVPR 2020. [[arXiv:1911.05722]](https://arxiv.org/abs/1911.05722)

> **Jean-Bastien Grill et al.**
> *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.*
> NeurIPS 2020. [[arXiv:2006.07733]](https://arxiv.org/abs/2006.07733)

Additional references:
- [SimCLR v2](https://arxiv.org/abs/2006.10029) — Big Self-Supervised Models Are Strong Semi-Supervised Learners
- [MoCo v2 Improvements](https://arxiv.org/abs/2003.04297) — Improved Baselines with Momentum Contrastive Learning
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Hydra](https://hydra.cc/) — Configuration management framework
- [MLflow](https://mlflow.org/) — Experiment tracking platform
- [DVC](https://dvc.org/) — Data version control


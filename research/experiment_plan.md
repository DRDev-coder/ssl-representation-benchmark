# Experiment Plan

## Overview

This document maps every hypothesis in [hypothesis.md](hypothesis.md) to a concrete, reproducible experiment with defined commands, metrics, baselines, and expected outputs.

---

## E1 — Label Efficiency Sweep

**Tests:** H1 (SSL beats supervised at 1 % labels), H2 (diminishing advantage)

### Design

| Parameter | Value |
|-----------|-------|
| Methods | SimCLR, MoCo v2, BYOL, Supervised |
| Dataset | CIFAR-10 |
| Backbone | ResNet-18 |
| Pretraining epochs | 100 |
| Probe/fine-tune epochs | 100 per fraction |
| Label fractions | 1 %, 10 %, 100 % |
| Encoder LR scale | 0.1 (fine-tune runs only) |
| Batch size | 128 |
| Seed | 42 |

### Procedure

1. **Pretrain** each SSL method (SimCLR / MoCo / BYOL) on the full unlabeled CIFAR-10 training set for 100 epochs.
2. **Fine-tune** each pretrained encoder on {1 %, 10 %, 100 %} of labeled data with an unfrozen encoder and a fresh linear head.
3. **Train supervised baselines** from random initialisation on the same label subsets.
4. Record best test accuracy at each fraction for every method.

### Commands

```bash
# Step 1: Pretrain (already completed for SimCLR)
python training/train_simclr.py dataset=cifar10 num_workers=0

# Step 2–3: Fine-tune + supervised sweep
python training/fine_tune.py training=finetune dataset=cifar10 num_workers=0
python training/train_supervised.py training=supervised dataset=cifar10 num_workers=0

# Multi-method sweep (unified script)
python experiments/label_efficiency.py \
  --methods simclr moco byol supervised \
  --fractions 0.01 0.10 1.0 \
  --dataset cifar10 --epochs 100
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Test accuracy @ fraction** | Top-1 accuracy on the full CIFAR-10 test set after fine-tuning |
| **SimCLR advantage** | `acc_simclr - acc_supervised` at each fraction |

### Expected Outputs

- `results/finetune_results.json`
- `results/supervised_results.json`
- `results/label_efficiency_results.json`
- `results/label_efficiency_curve.png`

### Pass Criteria (per hypothesis)

- **H1:** SimCLR @ 1 % − Supervised @ 1 % ≥ 15 pp
- **H2:** Advantage is monotonically non-increasing across fractions {1 %, 10 %, 100 %}

---

## E2 — Representation Quality Probes

**Tests:** H3 (KNN ≈ linear probe)

### Design

| Parameter | Value |
|-----------|-------|
| Encoder | SimCLR, pretrained 100 epochs |
| Evaluation | Linear probe (100 epochs) + KNN (k = 200) |
| Dataset | CIFAR-10 test set (10 000 images) |

### Procedure

1. **Linear probe:** Freeze the SimCLR encoder, train Linear(512 → 10) for 100 epochs on full labels.
2. **KNN:** Encode all train and test images, predict via cosine-similarity majority vote (k = 200).
3. Compare the two accuracies.

### Commands

```bash
# Linear probe
python training/linear_probe.py training=linear_eval dataset=cifar10 num_workers=0

# KNN
python -m evaluation.knn_eval \
  --dataset cifar10 \
  --encoder_path checkpoints/simclr_encoder_best.pth \
  --k 200 --no_mlflow
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Linear probe acc** | Accuracy of trained linear head on frozen encoder |
| **KNN acc (Top-1)** | k-NN classification accuracy |
| **Gap** | `linear_probe - knn` |

### Pass Criterion

- |Linear probe − KNN| ≤ 5.0 pp

---

## E3 — Augmentation Ablation

**Tests:** H4 (color jitter is the most critical augmentation)

### Design

Run SimCLR pretraining (reduced epochs for efficiency) with six augmentation configurations:

| Config | Augmentations included |
|--------|----------------------|
| `full` | Crop + Flip + Color Jitter + Grayscale + Blur (baseline) |
| `no_color_jitter` | Crop + Flip + Grayscale + Blur |
| `no_blur` | Crop + Flip + Color Jitter + Grayscale |
| `no_grayscale` | Crop + Flip + Color Jitter + Blur |
| `no_crop` | Flip + Color Jitter + Grayscale + Blur |
| `crop_flip_only` | Crop + Flip only |

After each pretraining run, evaluate with linear probe (50 epochs).

### Commands

```bash
python experiments/augmentation_ablation.py \
  --dataset cifar10 --pretrain_epochs 50 --probe_epochs 50
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Linear probe acc per config** | Accuracy after probing each ablated encoder |
| **Δ from full** | `acc_full - acc_config` for each ablated config |

### Expected Outputs

- `results/augmentation_ablation_results.json`
- `results/augmentation_ablation.png`

### Pass Criterion

- `Δ(no_color_jitter)` > `Δ(no_blur)`, `Δ(no_grayscale)`, `Δ(no_crop)`

---

## E4 — Multi-Method Comparison

**Tests:** H5 (BYOL ≈ SimCLR ≈ MoCo within 3 pp)

### Design

| Parameter | Value |
|-----------|-------|
| Methods | SimCLR, MoCo v2, BYOL |
| Dataset | CIFAR-10 |
| Pretraining epochs | 100 |
| Evaluation | Linear probe (100 epochs), KNN (k = 200) |
| Batch size | 128 |

### Procedure

1. Pretrain each method using the unified SSL training script.
2. Evaluate each encoder with linear probe and KNN.
3. Compare accuracies.

### Commands

```bash
# Pretrain each method
python training/train_ssl.py method=simclr dataset=cifar10 num_workers=0
python training/train_ssl.py method=moco dataset=cifar10 num_workers=0
python training/train_ssl.py method=byol dataset=cifar10 num_workers=0

# Evaluate each
for method in simclr moco byol; do
  python training/linear_probe.py \
    training=linear_eval dataset=cifar10 num_workers=0 \
    encoder_path=checkpoints/${method}_encoder_best.pth

  python -m evaluation.knn_eval \
    --dataset cifar10 \
    --encoder_path checkpoints/${method}_encoder_best.pth \
    --k 200 --no_mlflow
done
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Linear probe acc** | Per method |
| **KNN Top-1** | Per method |
| **Max spread** | `max(acc) - min(acc)` across methods |

### Pass Criterion

- Max spread ≤ 3.0 pp for linear probe accuracy

---

## E5 — Cross-Dataset Transfer

**Tests:** H6 (CIFAR-10 → STL-10 transfer ≥ 50 %)

### Design

| Parameter | Value |
|-----------|-------|
| Source dataset | CIFAR-10 (pretrain) |
| Target dataset | STL-10 (evaluate) |
| Methods | SimCLR, MoCo v2, BYOL |
| Evaluation | Linear probe on STL-10 (frozen encoder) |

### Procedure

1. Use encoders pretrained on CIFAR-10 from E4.
2. Resize STL-10 test images to 32×32 (or upsample CIFAR-10 encoder input to 96×96).
3. Train a linear classifier on STL-10 train set using the frozen CIFAR-10 encoder.
4. Evaluate on STL-10 test set.

### Commands

```bash
python experiments/transfer_learning.py \
  --methods simclr moco byol \
  --source cifar10 --target stl10 --epochs 50
```

### Expected Outputs

- `results/transfer_cifar10_to_stl10.json`
- `results/transfer_cifar10_to_stl10.png`

### Pass Criterion

- At least one method achieves ≥ 50 % accuracy on STL-10

---

## E6 — Medical Imaging SSL

**Tests:** H7 (SSL helps medical imaging at low labels)

### Design

| Parameter | Value |
|-----------|-------|
| Dataset | Chest X-Ray (normal vs. pneumonia) |
| Methods | SimCLR (with other methods optional) |
| Pretraining | 50 epochs on unlabeled X-rays |
| Fine-tune fractions | 1 %, 10 %, 50 %, 100 % |
| Baseline | Supervised from scratch on X-rays |

### Data Preparation

```
data/chestxray/
├── train/
│   ├── normal/        # normal X-ray images
│   └── pneumonia/     # pneumonia X-ray images
└── test/
    ├── normal/
    └── pneumonia/
```

### Commands

```bash
python experiments/medical_ssl_experiment.py \
  --methods simclr \
  --data_dir data/chestxray \
  --pretrain_epochs 50 \
  --fractions 0.01 0.10 0.50 1.0
```

### Expected Outputs

- `results/medical_ssl_results.json`
- `results/medical_ssl_comparison.png`

### Pass Criterion

- SSL @ 10 % − Supervised @ 10 % ≥ 5 pp

---

## E7 — t-SNE and UMAP Qualitative Analysis

**Tests:** Qualitative complement to all quantitative hypotheses

### Design

Generate 2-D embedding projections for every available encoder checkpoint.

### Commands

```bash
# t-SNE per encoder
python -m evaluation.tsne_visualization \
  --dataset cifar10 \
  --encoder_path checkpoints/simclr_encoder_best.pth \
  --label_curve

# UMAP (multi-method)
python -m evaluation.umap_visualization \
  --methods simclr moco byol supervised \
  --dataset cifar10

# Full dashboard (all plots at once)
python results/dashboard.py --dpi 300
```

### Expected Outputs

- `results/tsne_simclr.png`, `results/tsne_supervised.png`, `results/tsne_comparison.png`
- `results/umap_*.png`
- `results/dashboard/` — all publication-quality figures

### Evaluation

Visual assessment: tighter, more separated clusters indicate higher-quality representations. Compare cluster purity and inter-class separation between SSL methods and the supervised baseline.

---

## Experiment Schedule

| Order | Experiment | Hypotheses | GPU Hours (est.) | Status |
|:-----:|------------|------------|:-----------------:|--------|
| 1 | E1 — Label Efficiency | H1, H2 | ~6 h | **Done** (SimCLR + Supervised) |
| 2 | E2 — Representation Probes | H3 | ~1 h | **Done** |
| 3 | E7 — t-SNE / UMAP | (qualitative) | ~0.5 h | **Done** |
| 4 | E3 — Augmentation Ablation | H4 | ~8 h | Pending |
| 5 | E4 — Multi-Method Comparison | H5 | ~8 h | Pending |
| 6 | E5 — Cross-Dataset Transfer | H6 | ~2 h | Pending |
| 7 | E6 — Medical SSL | H7 | ~3 h | Pending (needs data) |

---

## Reproducibility Notes

- All experiments use `seed = 42` for PyTorch, NumPy, and Python RNG.
- Hydra captures the full resolved config for every training run in `outputs/`.
- MLflow logs all hyperparameters, epoch-level metrics, and artifact paths.
- Encoder checkpoints follow the naming convention `checkpoints/{method}_encoder_best.pth`.

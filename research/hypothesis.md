# Research Hypotheses

## Overview

This document formalises the research hypotheses driving this project. Each hypothesis is stated precisely, paired with a rationale grounded in the SSL literature, and linked to the specific experiment that tests it.

---

## H1 — Label Efficiency of Self-Supervised Pretraining

> **Self-supervised contrastive pretraining (SimCLR) produces representations that, when fine-tuned with only 1 % of the labeled data, outperform a supervised model trained from scratch on the same 1 % of labels by at least 15 percentage points.**

**Rationale.**  
Chen et al. (2020) showed that SimCLR representations transfer well to downstream tasks with limited labels. The encoder learns augmentation-invariant features from the full unlabeled dataset, giving it a structural prior that a randomly initialised network lacks entirely. With only 500 labeled CIFAR-10 images (1 %), the supervised baseline is severely data-starved, while the SimCLR encoder has already observed all 50 000 images during pretraining.

**Testing.**  
[Experiment E1 — Label Efficiency Sweep](experiment_plan.md#e1--label-efficiency-sweep)

**Observed result.**  
SimCLR fine-tune at 1 %: **61.81 %** vs. Supervised at 1 %: **38.54 %** → Δ = +23.27 pp. *Hypothesis confirmed; the gap exceeds the 15 pp threshold.*

---

## H2 — Diminishing Advantage at Full Supervision

> **The accuracy advantage of SimCLR pretraining over supervised training decreases monotonically as the label fraction increases, and becomes negligible or negative at 100 % labels.**

**Rationale.**  
Self-supervised pretraining compensates for label scarcity. As labels become abundant, the supervised model can learn equally discriminative features directly from the label signal. With full data, the supervised model's unconstrained learning rate and end-to-end optimisation may surpass a pretrained encoder fine-tuned with a conservative encoder learning rate.

**Testing.**  
[Experiment E1 — Label Efficiency Sweep](experiment_plan.md#e1--label-efficiency-sweep)

**Observed result.**  
- 1 % → SimCLR wins by +23.27 pp  
- 10 % → SimCLR wins by +8.73 pp  
- 100 % → Supervised wins by +3.11 pp  

*Hypothesis confirmed; the advantage strictly decreases and reverses at 100 % labels.*

---

## H3 — KNN and Linear Probe Agreement

> **KNN classification accuracy (k = 200) will fall within 5 percentage points of linear probe accuracy, indicating that the self-supervised embedding space is well-clustered and approximately linearly separable.**

**Rationale.**  
A well-structured embedding space should separate classes not only along learned linear decision boundaries but also in terms of local neighbourhood structure. If KNN accuracy is close to linear probe accuracy, the representations are clustered tightly enough that even a non-parametric classifier succeeds — a strong quality signal.

**Testing.**  
[Experiment E2 — Representation Quality Probes](experiment_plan.md#e2--representation-quality-probes)

**Observed result.**  
Linear probe: **67.56 %**, KNN: **64.75 %** → gap = 2.81 pp. *Hypothesis confirmed.*

---

## H4 — Color Jitter is the Most Critical Augmentation

> **Removing color jitter from the SimCLR augmentation pipeline causes a larger drop in linear probe accuracy than removing any other single augmentation component (grayscale, blur, or crop).**

**Rationale.**  
Chen et al. (2020, §5) found that color jitter was the single most important augmentation for contrastive learning on natural images. Without strong colour distortion, the model can distinguish positive pairs by colour histograms alone, providing a "shortcut" that degrades feature quality.

**Testing.**  
[Experiment E3 — Augmentation Ablation](experiment_plan.md#e3--augmentation-ablation)

**Observed result.**  
Augmentation ablation on CIFAR-10, 100 pretrain + 100 probe epochs:

| Config | Accuracy | Δ vs. full |
|--------|----------|------------|
| `no_blur` | 69.02% | **+0.90%** |
| `full` | 68.12% | baseline |
| `no_color_jitter` | 60.37% | −7.75% |
| `no_grayscale` | 54.68% | −13.44% |
| `crop_flip_only` | 46.61% | −21.51% |
| `no_crop` | 38.69% | −29.43% |

*Hypothesis **REFUTED**.* Random crop is the single most important augmentation (−29.4 pp), followed by grayscale (−13.4 pp) and color jitter (−7.8 pp). Color jitter is important but ranks only third. Notably, removing Gaussian blur **improves** accuracy by +0.9 pp — CIFAR-10 images are 32×32 px; blurring them destroys spatial detail at a resolution where it is already limited.

**Why this diverges from the original paper:** Chen et al. (2020) found color jitter most critical on ImageNet (224×224). At 32×32, random resized crops create much harder positive pairs because cropped views share less spatial overlap, making crop-based invariance the dominant learning signal.

---

## H5 — Non-Contrastive Methods Match Contrastive Methods

> **BYOL (which uses no negative pairs) achieves linear probe accuracy within 3 percentage points of SimCLR and MoCo v2 (which rely on negative pairs) under identical pretraining budgets.**

**Rationale.**  
Grill et al. (2020) demonstrated that BYOL's online/target architecture with an asymmetric predictor prevents representation collapse without requiring negatives. If confirmed, this has practical implications: BYOL is less sensitive to batch size and does not need the large queue used by MoCo.

**Testing.**  
[Experiment E4 — Multi-Method Comparison](experiment_plan.md#e4--multi-method-comparison)

**Observed result.**  
Linear probe accuracy after 100 epochs pretraining + 100 epochs probe (CIFAR-10, ResNet-18, batch=128):

| Method | Linear Probe Acc | Gap vs. SimCLR |
|--------|-----------------|----------------|
| SimCLR | 67.56% | baseline |
| MoCo v2 | 65.47% | −2.09 pp |
| BYOL | 56.88% | −10.68 pp |

*Hypothesis **REFUTED**.* SimCLR and MoCo v2 are within 3 pp of each other (2.09 pp ✓), but BYOL falls 10.68 pp below SimCLR — far outside the threshold. H5 is refuted overall because all three methods do not converge within 3 pp.

**Why BYOL underperformed at small batch:**  
1. **Batch size sensitivity:** BYOL's bootstrap signal requires a stable target network. At batch=128 (vs. the paper's 4096), the target updates are too frequent relative to the batch diversity, weakening the pseudo-label quality.  
2. **Fixed EMA momentum:** The original paper cosine-schedules momentum from 0.996→1.0 to gradually stabilise the target. Our fixed 0.996 allows slight representation drift in late epochs.  
3. **No LARS optimiser:** Original BYOL uses LARS (LR=0.2). Adam at LR=3e-4 may produce a suboptimal balance between the projection and predictor heads.  
4. **Contrastive robustness:** SimCLR and MoCo use explicit negatives that anchor the representation space regardless of batch size. BYOL has no such anchor, making it more susceptible to bootstrap instability at small batch.

---

## H6 — Representations Transfer Across Datasets

> **An encoder pretrained with SimCLR on CIFAR-10 achieves at least 50 % accuracy when evaluated on STL-10 via linear probe (without any STL-10 pretraining), despite the domain shift from 32×32 to 96×96 images.**

**Rationale.**  
CIFAR-10 and STL-10 share 9 of 10 visual categories. A strong contrastive encoder should learn object-level features (shapes, textures) rather than low-level pixel statistics, enabling knowledge transfer across resolutions and image distributions.

**Testing.**  
[Experiment E5 — Cross-Dataset Transfer](experiment_plan.md#e5--cross-dataset-transfer)

**Observed result.**  
SimCLR CIFAR-10 encoder → STL-10 linear probe (100 epochs, frozen encoder):

| Metric | Value |
|--------|-------|
| Transfer accuracy | **59.56%** |
| H6 threshold | ≥ 50% |
| Margin above threshold | +9.56 pp |
| STL-10 random baseline | ~10% |
| STL-10 random-init linear probe | ~35–40% |

*Hypothesis **Confirmed**.* The CIFAR-10 SimCLR encoder achieves 59.56% on STL-10 with no STL-10 pretraining, well above the 50% threshold. The encoder successfully transfers through a 3× resolution gap (32×32 → 96×96). This confirms that contrastive SSL learns object-level features (shapes, textures, colour patterns) that generalise across datasets rather than dataset-specific pixel statistics.

---

## H7 — SSL Pretraining Helps in Domain-Specific Settings

> **In a medical imaging setting (chest X-ray classification), SSL pretraining followed by fine-tuning with 10 % of labels outperforms supervised training from scratch with the same 10 % by at least 5 percentage points.**

**Rationale.**  
Medical datasets are prototypically label-scarce: expert annotation is expensive and time-consuming. SSL pretraining leverages the typically larger pool of unlabeled scans to learn structural features (organ edges, texture patterns) before seeing any labels. This mirrors the motivation behind transfer learning from ImageNet, but without requiring an external dataset.

**Testing.**  
[Experiment E6 — Medical Imaging SSL](experiment_plan.md#e6--medical-imaging-ssl)

**Observed result.**  
NIH CXR8 binary classification (No Finding vs. Pathology), ResNet-18, 30 pretrain epochs, 96×96 px, 40 K images:

| Method | 1% | 5% | 10% | 100% |
|--------|-----|-----|------|------|
| SimCLR | 64.63% | 66.23% | 65.01% | 66.14% |
| MoCo v2 | 64.21% | 65.66% | 65.15% | 62.31% |
| BYOL | 63.71% | 63.76% | 62.79% | 65.17% |
| Supervised | 61.98% | 62.01% | 62.36% | **68.81%** |

At 10 % labels: best SSL (MoCo 65.15%) vs. supervised (62.36%) → Δ = **+2.79 pp** — below the 5 pp threshold.

*Hypothesis **Refuted** (partial).* SSL methods consistently outperform supervised from scratch at every low-label fraction (1%, 5%, 10%), confirming the label-efficiency pattern observed in H1/H2 on CIFAR-10. However, the margin is 2–4 pp rather than the hypothesised ≥ 5 pp. At 100% labels, supervised wins clearly (+2.67 pp over best SSL), replicating the H2 pattern in the medical domain.

**Why the margin is smaller than expected:**
1. **Short pretraining (30 epochs):** Chest X-ray features require more epochs to converge than CIFAR-10 features. SimCLR loss remained high (4.61), indicating under-trained representations.
2. **Reduced resolution (96 px):** Fine-grained radiological detail (subtle opacities, edge textures) is lost at 96×96, limiting the information available to the SSL objective.
3. **Binary task difficulty:** No Finding vs. Pathology is learnable from low-level intensity statistics even by a randomly initialised network, reducing the relative advantage of pretrained features.
4. **Label imbalance:** CXR8 has a skewed class distribution (~54% pathology) that benefits simple baselines.

Despite not meeting the 5 pp threshold, the result validates that SSL pretraining provides consistent, non-trivial gains in the low-label medical regime — a practically meaningful finding given the cost of expert annotation.

---

## Summary Table

| ID | Hypothesis (short) | Status | Key Metric |
|:--:|---------------------|--------|------------|
| H1 | SSL beats supervised at 1 % labels by ≥ 15 pp | **Confirmed** (+23.27 pp) | Fine-tune acc @ 1 % |
| H2 | Advantage decreases with more labels | **Confirmed** | Label efficiency curve |
| H3 | KNN ≈ linear probe (within 5 pp) | **Confirmed** (2.81 pp) | KNN vs. linear acc |
| H4 | Color jitter most critical augmentation | **Refuted** (crop is −29.4 pp; color jitter −7.8 pp) | Ablation Δ accuracy |
| H5 | BYOL ≈ SimCLR ≈ MoCo (within 3 pp) | **Refuted** (BYOL −10.68 pp; MoCo −2.09 pp ✓) | Linear probe acc |
| H6 | Cross-dataset transfer ≥ 50 % | **Confirmed** (59.56%, +9.56 pp margin) | Transfer acc |
| H7 | SSL helps medical imaging at 10 % labels by ≥ 5 pp | **Refuted** (best gap +2.79 pp; direction confirmed) | Medical acc @ 10 % |

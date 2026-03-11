# Analysis Template

Use this template to write up the analysis for each experiment. Copy the relevant section, fill in the observed values after running the experiment, and record your interpretation.

---

## Experiment: _[Experiment ID and Title]_

**Date:** _YYYY-MM-DD_  
**Hypothesis tested:** _[H1 / H2 / ...]_  
**Status:** _Completed / In Progress / Blocked_

---

### 1. Setup Summary

| Parameter | Value |
|-----------|-------|
| Dataset | |
| Backbone | |
| SSL method(s) | |
| Pretraining epochs | |
| Evaluation epochs | |
| Batch size | |
| Optimizer | |
| Learning rate | |
| Seed | 42 |
| GPU | |
| Training time | |

**Command used:**

```bash
# Paste the exact command(s) run
```

---

### 2. Raw Results

_Fill in the table with the observed numbers. Add or remove rows as needed._

| Condition | Accuracy (%) | Loss | Notes |
|-----------|:------------:|:----:|-------|
| | | | |
| | | | |
| | | | |

**Result file(s):** `results/_____.json`

---

### 3. Hypothesis Evaluation

**Hypothesis statement:**

> _Paste the full hypothesis from hypothesis.md_

**Pass criterion:** _[e.g., Δ ≥ 15 pp]_

**Observed value:** _[e.g., Δ = 23.27 pp]_

**Verdict:** _Confirmed / Rejected / Inconclusive_

**Confidence notes:**  
_Any caveats — was the result borderline? Could a different seed change the outcome? Is the sample size sufficient?_

---

### 4. Detailed Analysis

#### 4.1 Key Observations

1. _First notable finding._
2. _Second notable finding._
3. _..._

#### 4.2 Comparison to Literature

| Source | Setup | Their Result | Our Result | Notes |
|--------|-------|:------------:|:----------:|-------|
| Chen et al. (2020) | ResNet-50, bs=256, 200ep | ~85 % (linear) | | Different backbone / budget |
| | | | | |

#### 4.3 Error Analysis

_Where did the model struggle? Were certain classes harder? Any failure modes?_

#### 4.4 Visualisations

_Reference the relevant plots from `results/dashboard/` or `results/`._

- Label efficiency curve: `results/dashboard/01_label_efficiency_curve.png`
- t-SNE grid: `results/dashboard/04_tsne_grid.png`

---

### 5. Limitations

- _List anything that may limit the generalisability of the result (small batch, few epochs, single seed, etc.)._

---

### 6. Next Steps

- [ ] _Follow-up experiment or parameter change suggested by the findings._
- [ ] _..._

---
---

# Completed Analyses

Below are filled-in analyses for experiments already completed during this project.

---

## Analysis: E1 — Label Efficiency Sweep (SimCLR + Supervised)

**Date:** 2026-03-08  
**Hypotheses tested:** H1, H2  
**Status:** Completed

### 1. Setup Summary

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (50 000 train / 10 000 test) |
| Backbone | ResNet-18 (512-d output) |
| SSL method | SimCLR (NT-Xent, τ = 0.5) |
| Pretraining epochs | 100 |
| Fine-tune / supervised epochs | 100 per fraction |
| Batch size | 128 |
| Optimizer | Adam (head lr = 3e-4, encoder lr = 3e-5) |
| Seed | 42 |
| GPU | RTX 5060 Laptop (8 GB) |
| Total training time | ~10 hours (pretrain + 6 fraction runs) |

### 2. Raw Results

| Label Fraction | Supervised (scratch) | SimCLR Fine-tune | Δ (SimCLR − Supervised) |
|:-:|:-:|:-:|:-:|
| 1 % (500 images) | 38.54 % | 61.81 % | **+23.27 pp** |
| 10 % (5 000 images) | 63.13 % | 71.86 % | **+8.73 pp** |
| 100 % (50 000 images) | 84.21 % | 81.10 % | −3.11 pp |

**Result files:** `results/finetune_results.json`, `results/supervised_results.json`

### 3. Hypothesis Evaluation

**H1 — SSL beats supervised at 1 % by ≥ 15 pp**

- Pass criterion: Δ ≥ 15 pp
- Observed: Δ = 23.27 pp
- **Verdict: Confirmed**

**H2 — Advantage diminishes with more labels**

- Pass criterion: Δ(1 %) > Δ(10 %) > Δ(100 %) and Δ(100 %) ≤ 0
- Observed: +23.27 > +8.73 > −3.11
- **Verdict: Confirmed**

### 4. Detailed Analysis

#### 4.1 Key Observations

1. **The 1 % result is the strongest signal.** SimCLR more than compensates for having seen zero labels during pretraining: 61.81 % vs. 38.54 % shows that augmentation-based contrastive learning captures class-discriminative structure that a randomly initialised network can not learn from 500 images alone.

2. **Supervised training overtakes at 100 %.** With full labels, the supervised model reaches 84.21 %, beating SimCLR fine-tune (81.10 %) by 3.11 pp. This is partly due to the conservative encoder learning rate scale (0.1×). Increasing it, or using a higher base learning rate for fine-tuning, would likely close this gap.

3. **The crossover point lies between 10 % and 100 %.** A future experiment at 50 % would pinpoint where supervised training catches up.

#### 4.2 Comparison to Literature

| Source | Setup | Result @ 1 % | Our Result @ 1 % |
|--------|-------|:------------:|:----------------:|
| Chen et al. (2020) | ResNet-50, bs=256, 200 ep | ~70 % | 61.81 % |

The gap is expected: we used ResNet-18 (half the capacity), bs=128 (half the negatives), and 100 epochs (half the duration).

#### 4.3 Visualisations

- Label efficiency curve: `results/dashboard/01_label_efficiency_curve.png`
- t-SNE comparison: `results/tsne_comparison.png`

### 5. Limitations

- Single seed (42) — variance not measured.
- Small batch size (128 vs. recommended 256+) limits contrastive performance.
- Only three label fractions tested; additional points (5 %, 25 %, 50 %) would improve the curve.
- Fine-tune encoder LR scale (0.1) was not tuned.

### 6. Next Steps

- [ ] Run with seed ∈ {42, 123, 456} to measure variance.
- [ ] Add 5 % and 50 % fractions.
- [ ] Sweep encoder LR scale {0.05, 0.1, 0.2, 0.5} at 100 % to close the supervised gap.

---

## Analysis: E2 — Representation Quality Probes

**Date:** 2026-03-08  
**Hypothesis tested:** H3  
**Status:** Completed

### 1. Setup Summary

| Parameter | Value |
|-----------|-------|
| Encoder | SimCLR, 100 epochs, best checkpoint |
| Linear probe | 100 epochs, frozen encoder, Adam lr = 3e-4 |
| KNN | k = 200, cosine similarity, full train set |
| Dataset | CIFAR-10 |

### 2. Raw Results

| Method | Top-1 Accuracy |
|--------|:-:|
| Linear probe | 67.56 % |
| KNN (k = 200) | 64.75 % |
| **Gap** | **2.81 pp** |

### 3. Hypothesis Evaluation

- Pass criterion: gap ≤ 5 pp
- Observed: 2.81 pp
- **Verdict: Confirmed**

### 4. Detailed Analysis

The small 2.81 pp gap indicates that class structure in the SimCLR embedding space is already well-separated at the neighbourhood level. The linear probe's marginal improvement comes from learning soft decision boundaries that KNN cannot express with a hard majority vote.

### 5. Limitations

- Only tested at k = 200. A sweep over k ∈ {10, 50, 100, 200, 500} would show sensitivity.
- KNN used cosine similarity; Euclidean distance is another common choice.

### 6. Next Steps

- [ ] Run KNN sweep over k values.
- [ ] Repeat for MoCo and BYOL encoders once pretrained.

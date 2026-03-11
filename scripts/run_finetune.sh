#!/bin/bash
# ============================================================
# Full Fine-Tuning + Supervised Baseline + Visualization
# ============================================================
# Runs the complete evaluation pipeline:
#   1. Semi-supervised fine-tuning (1%, 10%, 100%)
#   2. Supervised baseline (1%, 10%, 100%)
#   3. t-SNE visualization + label efficiency curve
#
# Usage:
#   bash scripts/run_finetune.sh [DATASET]
# ============================================================

set -e

DATASET=${1:-stl10}

echo "============================================"
echo " Full Evaluation Pipeline"
echo " Dataset: $DATASET"
echo "============================================"

cd "$(dirname "$0")/.."

# Step 1: SimCLR Fine-tuning
echo ""
echo "========== STEP 1: SimCLR Fine-Tuning =========="
python -m training.fine_tune \
    --dataset "$DATASET" \
    --fractions 0.01 0.10 1.0 \
    --epochs 100

# Step 2: Supervised Baseline
echo ""
echo "========== STEP 2: Supervised Baseline =========="
python -m training.train_supervised \
    --dataset "$DATASET" \
    --fractions 0.01 0.10 1.0 \
    --epochs 100

# Step 3: Visualization
echo ""
echo "========== STEP 3: Visualization =========="
python -m evaluation.tsne_visualization \
    --dataset "$DATASET" \
    --label_curve

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Results saved in: results/"
echo "   - tsne_simclr.png"
echo "   - tsne_supervised.png"
echo "   - tsne_comparison.png"
echo "   - label_efficiency_curve.png"
echo "   - finetune_results.json"
echo "   - supervised_results.json"
echo "============================================"

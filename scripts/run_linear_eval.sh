#!/bin/bash
# ============================================================
# Linear Evaluation + kNN Evaluation Script
# ============================================================
# Usage:
#   bash scripts/run_linear_eval.sh [DATASET] [ENCODER_PATH]
# ============================================================

set -e

DATASET=${1:-stl10}
ENCODER_PATH=${2:-checkpoints/simclr_encoder_best.pth}

echo "============================================"
echo " Linear Evaluation + kNN"
echo " Dataset:    $DATASET"
echo " Encoder:    $ENCODER_PATH"
echo "============================================"

cd "$(dirname "$0")/.."

# Linear probe
echo ""
echo "--- Running Linear Probe ---"
python -m training.linear_probe \
    --dataset "$DATASET" \
    --encoder_path "$ENCODER_PATH" \
    --epochs 100

# kNN evaluation
echo ""
echo "--- Running kNN Evaluation ---"
python -m evaluation.knn_eval \
    --dataset "$DATASET" \
    --encoder_path "$ENCODER_PATH" \
    --k 200

echo ""
echo "Evaluation complete!"

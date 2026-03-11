#!/bin/bash
# ============================================================
# SimCLR Pretraining Script
# ============================================================
# Usage:
#   bash scripts/run_pretrain.sh [DATASET] [EPOCHS] [BATCH_SIZE]
#
# Examples:
#   bash scripts/run_pretrain.sh stl10 200 256
#   bash scripts/run_pretrain.sh cifar10 200 512
# ============================================================

set -e

DATASET=${1:-stl10}
EPOCHS=${2:-200}
BATCH_SIZE=${3:-256}

echo "============================================"
echo " SimCLR Pretraining"
echo " Dataset:    $DATASET"
echo " Epochs:     $EPOCHS"
echo " Batch Size: $BATCH_SIZE"
echo "============================================"

cd "$(dirname "$0")/.."

python -m training.train_simclr \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "Pretraining complete!"
echo "Encoder saved to: checkpoints/simclr_encoder_best.pth"

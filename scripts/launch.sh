#!/bin/bash
# PROMETHEUS — Multi-GPU launcher (single node)
# Usage: bash scripts/launch.sh [config] [nproc]
# Example:
#   bash scripts/launch.sh configs/qwen_1b.yaml 4
#   bash scripts/launch.sh configs/qwen_7b.yaml 8

CONFIG=${1:-"configs/qwen_1b.yaml"}
NPROC=${2:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

echo "PROMETHEUS launcher"
echo "  config : $CONFIG"
echo "  GPUs   : $NPROC"
echo ""

OMP_NUM_THREADS=8 \
torchrun \
  --standalone \
  --nproc_per_node=$NPROC \
  --master_port=29500 \
  train.py \
  --config "$CONFIG" \
  "$@"

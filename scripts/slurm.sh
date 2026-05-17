#!/bin/bash
#SBATCH --job-name=prometheus
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/prometheus-%j.out
#SBATCH --error=logs/prometheus-%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/12.1
source ~/miniconda3/bin/activate prometheus

# ── Distributed setup ─────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL tuning for multi-node
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

echo "PROMETHEUS SLURM job"
echo "  nodes      : $SLURM_NNODES"
echo "  GPUs/node  : $SLURM_NTASKS_PER_NODE"
echo "  world_size : $WORLD_SIZE"
echo "  master     : $MASTER_ADDR:$MASTER_PORT"

mkdir -p logs checkpoints

# ── Launch ────────────────────────────────────────────────────────────────────
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py \
  --config configs/qwen_7b.yaml \
  --run_name "prometheus-slurm-$SLURM_JOB_ID"

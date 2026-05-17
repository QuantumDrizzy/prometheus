# PROMETHEUS

**Distributed LLM pretraining with PyTorch FSDP.**

PROMETHEUS is a research-grade training framework demonstrating full-stack distributed training: FSDP ZeRO-3, activation checkpointing, Flash Attention 2, MFU tracking, and FSDP-aware checkpointing — designed to scale from a single RTX 5060 Ti up to multi-node GPU clusters via SLURM.

Part of the [iNFAMØUS OS](https://github.com/infamous-os) research stack, alongside [SUBSTRATE](https://github.com/infamous-os/substrate) and [SESHAT](https://arxiv.org/abs/xxxx.xxxxx).

---

## Architecture

```
train.py                    Main training loop
config.py                   Typed dataclass config + YAML loading
model.py                    FSDP wrapping + activation checkpointing
dataset.py                  Streaming tokenization + sequence packing
benchmark.py                Throughput/MFU/VRAM benchmark suite
utils/
  distributed.py            Distributed init, NCCL/GLOO auto-select
  checkpointing.py          Full + sharded FSDP checkpointing
  logging.py                MFU tracking, W&B integration
configs/
  qwen_1b.yaml              Qwen2.5-1.5B — 1–4 GPU (16GB VRAM)
  qwen_7b.yaml              Qwen2.5-7B   — 4–8 GPU (24GB VRAM)
scripts/
  launch.sh                 torchrun single-node launcher
  slurm.sh                  SLURM multi-node launcher (2+ nodes)
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt
# Flash Attention 2 (Linux, significant speedup):
pip install flash-attn --no-build-isolation

# Single GPU (dev / local RTX):
python train.py --config configs/qwen_1b.yaml

# 4x GPU (single node):
bash scripts/launch.sh configs/qwen_1b.yaml 4

# 2-node, 4 GPUs/node (via SLURM):
sbatch scripts/slurm.sh
```

---

## Key Design Decisions

### FSDP Sharding Strategies

| Strategy | Shards | VRAM/GPU | Throughput | Use case |
|----------|--------|----------|------------|----------|
| `NO_SHARD` | none | full model | baseline | DDP reference |
| `SHARD_GRAD_OP` | grads + optim | ~50% | +15% | 2–4 GPU |
| `FULL_SHARD` | params + grads + optim | ~25% | +5% | ≥4 GPU, large models |
| `HYBRID_SHARD` | full within node | ~25% intra | +8% | multi-node |

### Sequence Packing
Documents are concatenated (with EOS tokens as separators) and split into fixed `max_seq_len` chunks. No padding waste — every token in every batch is a real training signal.

### MFU (Model FLOP Utilization)
MFU = actual FLOP/s ÷ theoretical peak FLOP/s.

```
actual FLOP/s = 6 × N_params × tokens_per_second
```
Factor 6 = 2 (forward matmul) × 3 (forward + backward passes).
A100 SXM: peak ~35–45% MFU at 7B scale with Flash Attn 2. H100: ~50–60%.

### Activation Checkpointing
Recomputes activations during the backward pass instead of storing them. Reduces peak VRAM by ~60% at ~33% throughput cost. Applied per FSDP unit (each transformer block).

---

## Benchmark Results

Run: `torchrun --nproc_per_node=4 benchmark.py --model Qwen/Qwen2.5-1.5B`

Results populate in `benchmarks/results.json`.

---

## Roadmap

- [ ] Run full benchmark on 4xA10G (Lambda Cloud) and commit results
- [ ] LoRA / QLoRA fine-tuning mode
- [ ] DPO training loop (alignment research track)
- [ ] Mixture of Experts (MoE) FSDP wrapping
- [ ] Integration with SESHAT sparse attention kernel

---

## Hardware Requirements

| Config | Min GPUs | VRAM/GPU | Notes |
|--------|----------|----------|-------|
| qwen_1b (dev) | 1 | 16 GB | Local RTX 5060 Ti |
| qwen_1b (full) | 4 | 16 GB | Single node |
| qwen_7b | 4 | 24 GB | A10G or better |
| qwen_7b (2-node) | 8 | 24 GB | SLURM |

---

## Citation

```bibtex
@software{prometheus2026,
  author = {Antonio},
  title  = {PROMETHEUS: Distributed LLM Training with PyTorch FSDP},
  year   = {2026},
  url    = {https://github.com/infamous-os/prometheus}
}
```

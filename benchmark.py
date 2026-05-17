"""
PROMETHEUS — Benchmark Suite
Measures and compares:
  - Throughput (tokens/sec) across sharding strategies
  - Memory utilization (peak VRAM per GPU)
  - Model FLOP Utilization (MFU)
  - Gradient checkpointing overhead

Run:
  torchrun --nproc_per_node=N benchmark.py --model Qwen/Qwen2.5-1.5B --seq_len 2048
"""
from __future__ import annotations
import argparse
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal

import torch
import torch.nn as nn

from config import ModelConfig, FSDPConfig, TrainingConfig
from model import build_model
from utils.distributed import setup_distributed, teardown_distributed, get_rank, is_main_process, barrier
from utils.logging import Throughput, get_gpu_peak_flops, get_logger

logger = get_logger("benchmark")


@dataclass
class BenchmarkResult:
    strategy: str
    grad_checkpointing: bool
    flash_attn: bool
    batch_size: int
    seq_len: int
    world_size: int
    # Results
    tokens_per_sec: float
    peak_vram_gb: float
    mfu: float
    step_time_ms: float
    n_params_b: float


def run_benchmark(
    model_name: str,
    strategy: str,
    batch_size: int,
    seq_len: int,
    grad_checkpointing: bool,
    flash_attn: bool,
    warmup_steps: int = 3,
    bench_steps: int = 10,
) -> BenchmarkResult:
    rank, world_size = 0, 1
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig(
        name_or_path=model_name,
        max_seq_len=seq_len,
        flash_attn=flash_attn,
        gradient_checkpointing=grad_checkpointing,
    )
    fsdp_cfg = FSDPConfig(
        sharding_strategy=strategy,
        mixed_precision="bf16",
    )

    model, n_params = build_model(model_cfg, fsdp_cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    torch.cuda.reset_peak_memory_stats()
    gpu_peak = get_gpu_peak_flops()
    throughput = Throughput(n_params, gpu_peak * world_size)

    # Warmup
    for _ in range(warmup_steps):
        input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    throughput.start()
    t_start = time.perf_counter()

    for _ in range(bench_steps):
        input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        throughput.update(batch_size * seq_len * world_size)

    torch.cuda.synchronize()
    barrier()

    elapsed = time.perf_counter() - t_start
    tp = throughput.compute()
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    result = BenchmarkResult(
        strategy=strategy,
        grad_checkpointing=grad_checkpointing,
        flash_attn=flash_attn,
        batch_size=batch_size,
        seq_len=seq_len,
        world_size=world_size,
        tokens_per_sec=tp.get("tok_per_sec", 0),
        peak_vram_gb=peak_vram,
        mfu=tp.get("mfu", 0),
        step_time_ms=elapsed / bench_steps * 1000,
        n_params_b=n_params / 1e9,
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return result


def print_table(results: list[BenchmarkResult]) -> None:
    header = f"{'Strategy':<20} {'GradCkpt':<10} {'FlashAttn':<10} {'tok/s':>12} {'VRAM GB':>10} {'MFU':>8} {'ms/step':>10}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(f"  PROMETHEUS Benchmark | {results[0].n_params_b:.2f}B params | "
          f"bs={results[0].batch_size} seq={results[0].seq_len} gpus={results[0].world_size}")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.strategy:<20} "
            f"{'yes' if r.grad_checkpointing else 'no':<10} "
            f"{'yes' if r.flash_attn else 'no':<10} "
            f"{r.tokens_per_sec:>12,.0f} "
            f"{r.peak_vram_gb:>10.2f} "
            f"{r.mfu:>8.3f} "
            f"{r.step_time_ms:>10.1f}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=str, default="benchmarks/results.json")
    args = parser.parse_args()

    rank, world_size = setup_distributed()

    # NO_SHARD keeps full model on each GPU — needs smaller batch
    strategy_batch = {
        "NO_SHARD":      max(1, args.batch_size // 2),
        "SHARD_GRAD_OP": args.batch_size,
        "FULL_SHARD":    args.batch_size,
        "HYBRID_SHARD":  args.batch_size,
    }

    strategies: list[str] = ["NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD"]
    if world_size > 1:
        strategies.append("HYBRID_SHARD")

    results: list[BenchmarkResult] = []

    for strategy in strategies:
        for grad_ckpt in [False, True]:
            if is_main_process():
                logger.info(f"Benchmarking: {strategy} grad_ckpt={grad_ckpt}")
            try:
                r = run_benchmark(
                    model_name=args.model,
                    strategy=strategy,
                    batch_size=strategy_batch[strategy],
                    seq_len=args.seq_len,
                    grad_checkpointing=grad_ckpt,
                    flash_attn=True,
                    warmup_steps=args.warmup,
                    bench_steps=args.steps,
                )
                results.append(r)
            except torch.cuda.OutOfMemoryError:
                if is_main_process():
                    logger.warning(f"OOM: {strategy} grad_ckpt={grad_ckpt} batch={args.batch_size}")

    if is_main_process():
        print_table(results)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        logger.info(f"Results saved → {args.output}")

    teardown_distributed()


if __name__ == "__main__":
    main()

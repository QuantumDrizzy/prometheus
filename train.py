"""
PROMETHEUS — Main training entry point

Single GPU:   python train.py --config configs/qwen_1b.yaml
Multi-GPU:    torchrun --nproc_per_node=4 train.py --config configs/qwen_1b.yaml
SLURM:        sbatch scripts/slurm.sh
"""
from __future__ import annotations
import argparse
import math
import os
import time
import contextlib
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import TrainingConfig
from model import build_model
from dataset import build_dataloader
from utils import (
    setup_distributed, teardown_distributed,
    is_main_process, get_rank, get_world_size, barrier,
    get_logger, log_metrics, init_wandb,
    save_checkpoint, load_checkpoint,
)
from utils.logging import Throughput, get_gpu_peak_flops

logger = get_logger("prometheus")


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup + cosine decay
# ──────────────────────────────────────────────────────────────────────────────
def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# torch.profiler context (optional, first N steps)
# ──────────────────────────────────────────────────────────────────────────────
def get_profiler(profile_steps: int, output_dir: str):
    if profile_steps <= 0:
        return contextlib.nullcontext()
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=profile_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            Path(output_dir) / "profiler"
        ),
        record_shapes=True,
        with_stack=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train(cfg: TrainingConfig) -> None:
    rank, world_size = setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        logger.info(f"PROMETHEUS starting | world_size={world_size} | device={device}")
        logger.info(f"Effective batch size: {cfg.effective_batch_size * world_size} tokens/step")
        os.makedirs(cfg.output_dir, exist_ok=True)
        init_wandb(cfg.run_name, cfg)

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed + rank)

    # ── Model ────────────────────────────────────────────────────────────────
    model, n_params = build_model(cfg.model, cfg.fsdp)
    if is_main_process():
        logger.info(f"Model: {n_params/1e9:.3f}B params")

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Separate weight-decayed and non-decayed params
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.optimizer.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        fused=torch.cuda.is_available(),   # Fused kernel: faster on GPU
    )

    scheduler = get_cosine_schedule(
        optimizer,
        cfg.scheduler.warmup_steps,
        cfg.scheduler.total_steps,
        cfg.scheduler.min_lr_ratio,
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    if cfg.resume_from:
        start_step = load_checkpoint(model, optimizer, scheduler, cfg.resume_from)

    # ── torch.compile (Linux only) ────────────────────────────────────────────
    if cfg.compile:
        if is_main_process():
            logger.info("torch.compile enabled — first step will be slow.")
        model = torch.compile(model)

    # ── DataLoader ───────────────────────────────────────────────────────────
    loader = build_dataloader(
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        batch_size=cfg.batch_size_per_gpu,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
    )

    # ── Throughput tracker ───────────────────────────────────────────────────
    gpu_peak = get_gpu_peak_flops()
    throughput = Throughput(n_params, gpu_peak * world_size)
    throughput.start()

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step = start_step
    accum_loss = torch.tensor(0.0, device=device)
    accum_step = 0

    data_iter = iter(loader)

    with get_profiler(cfg.profile_steps, cfg.output_dir) as profiler:
        while step < cfg.max_steps:
            optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation
            for micro_step in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                n_tokens = input_ids.numel() * world_size

                # No-sync on all but last micro-step (prevents redundant all-reduce)
                is_last_micro = micro_step == cfg.gradient_accumulation_steps - 1
                ctx = model.no_sync() if not is_last_micro else contextlib.nullcontext()

                with ctx:
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / cfg.gradient_accumulation_steps
                    loss.backward()

                accum_loss += loss.detach()
                throughput.update(n_tokens)

            # Gradient clipping
            if cfg.optimizer.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optimizer.grad_clip
                )
            else:
                grad_norm = torch.tensor(0.0)

            optimizer.step()
            scheduler.step()
            step += 1

            # ── Logging ──────────────────────────────────────────────────────
            if step % cfg.log_every == 0:
                tp = throughput.compute()
                current_lr = scheduler.get_last_lr()[0]
                metrics = {
                    "loss": accum_loss.item() * cfg.gradient_accumulation_steps,
                    "lr": current_lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    **tp,
                }
                log_metrics(metrics, step)
                accum_loss.zero_()
                throughput.start()

            # ── Checkpointing ────────────────────────────────────────────────
            if step % cfg.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, step, cfg.output_dir)

            if profiler is not None:
                profiler.step()

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, step, cfg.output_dir)
    if is_main_process():
        logger.info(f"Training complete at step {step}.")

    teardown_distributed()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PROMETHEUS — Distributed LLM Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)

    # CLI overrides
    if args.run_name:
        cfg.run_name = args.run_name
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.output_dir:
        cfg.output_dir = args.output_dir

    train(cfg)


if __name__ == "__main__":
    main()

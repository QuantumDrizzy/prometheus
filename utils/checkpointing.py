"""
PROMETHEUS — FSDP-aware checkpointing
FSDP requires coordinated saving: every rank holds a shard.
Two modes:
  - FULL state dict: rank 0 reconstructs the full model (simpler, requires enough RAM)
  - SHARDED state dict: each rank saves its own shard (scalable, required for large models)
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Optional, Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
    OptimStateDictConfig,
    FullOptimStateDictConfig,
)

from .distributed import get_rank, is_main_process, barrier

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    output_dir: str,
    mode: str = "full",          # "full" | "sharded"
) -> None:
    """
    Save model + optimizer checkpoint.

    full mode:   rank 0 saves a single model.safetensors (needs full model RAM)
    sharded mode: each rank saves its shard (scalable, load with DCP)
    """
    ckpt_dir = Path(output_dir) / f"step-{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if mode == "full":
        _save_full(model, optimizer, scheduler, step, ckpt_dir)
    elif mode == "sharded":
        _save_sharded(model, optimizer, scheduler, step, ckpt_dir)
    else:
        raise ValueError(f"Unknown checkpoint mode: {mode}")

    # Save training state (step, lr) on rank 0
    if is_main_process():
        torch.save(
            {"step": step, "scheduler_state": scheduler.state_dict()},
            ckpt_dir / "training_state.pt",
        )
        logger.info(f"Checkpoint saved → {ckpt_dir}")

    barrier()


def _save_full(model, optimizer, scheduler, step, ckpt_dir):
    """Gather full state dict to rank 0 and save."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy, optim_policy):
        model_sd = model.state_dict()
        optim_sd = FSDP.optim_state_dict(model, optimizer)

    if is_main_process():
        torch.save(model_sd, ckpt_dir / "model.pt")
        torch.save(optim_sd, ckpt_dir / "optimizer.pt")


def _save_sharded(model, optimizer, scheduler, step, ckpt_dir):
    """Each rank saves its own shard via torch DCP."""
    try:
        import torch.distributed.checkpoint as dcp
        save_policy = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_policy):
            state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        dcp.save(state_dict, checkpoint_id=str(ckpt_dir / "sharded"))
    except Exception as e:
        logger.warning(f"Sharded save failed ({e}), falling back to full mode.")
        _save_full(model, optimizer, scheduler, step, ckpt_dir)


def load_checkpoint(
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Any,
    checkpoint_dir: str,
    mode: str = "full",
) -> int:
    """
    Load checkpoint. Returns the step number to resume from.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    # Load training state
    training_state_path = ckpt_dir / "training_state.pt"
    step = 0
    if training_state_path.exists():
        ts = torch.load(training_state_path, map_location="cpu")
        step = ts["step"]
        scheduler.load_state_dict(ts["scheduler_state"])

    if mode == "full":
        _load_full(model, optimizer, ckpt_dir)
    elif mode == "sharded":
        _load_sharded(model, optimizer, ckpt_dir)

    if is_main_process():
        logger.info(f"Resumed from step {step} ← {ckpt_dir}")

    barrier()
    return step


def _load_full(model, optimizer, ckpt_dir):
    model_path = ckpt_dir / "model.pt"
    optim_path = ckpt_dir / "optimizer.pt"

    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        if is_main_process():
            model_sd = torch.load(model_path, map_location="cpu")
        else:
            model_sd = {}
        model.load_state_dict(model_sd)

        if optimizer is not None and optim_path.exists():
            if is_main_process():
                optim_sd = torch.load(optim_path, map_location="cpu")
            else:
                optim_sd = {}
            optim_sd_loaded = FSDP.optim_state_dict_to_load(model, optimizer, optim_sd)
            optimizer.load_state_dict(optim_sd_loaded)


def _load_sharded(model, optimizer, ckpt_dir):
    try:
        import torch.distributed.checkpoint as dcp
        load_policy = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, load_policy):
            state_dict = {"model": model.state_dict()}
            if optimizer is not None:
                state_dict["optimizer"] = optimizer.state_dict()
            dcp.load(state_dict, checkpoint_id=str(ckpt_dir / "sharded"))
            model.load_state_dict(state_dict["model"])
            if optimizer is not None:
                optimizer.load_state_dict(state_dict["optimizer"])
    except Exception as e:
        logger.warning(f"Sharded load failed ({e}), trying full mode.")
        _load_full(model, optimizer, ckpt_dir)

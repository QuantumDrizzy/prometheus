"""
PROMETHEUS — Distributed utilities
Handles init/teardown, rank queries, collective ops.
Auto-selects NCCL (Linux/multi-GPU) or GLOO (Windows/CPU fallback).
"""
from __future__ import annotations
import os
import sys
import socket
import logging
from contextlib import contextmanager

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int]:
    """
    Initialize the process group.
    Reads LOCAL_RANK, RANK, WORLD_SIZE from env (set by torchrun).
    Returns (rank, world_size).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1 and not dist.is_initialized():
        # Single-GPU / CPU — skip distributed init
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
        return rank, world_size

    # Pick backend: NCCL for CUDA+Linux, GLOO otherwise
    if torch.cuda.is_available() and sys.platform != "win32":
        backend = "nccl"
    else:
        backend = "gloo"
        logger.warning(
            "NCCL unavailable (Windows or no CUDA). Using GLOO — "
            "performance will be lower, single-node only."
        )

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    torch.cuda.set_device(local_rank)

    if rank == 0:
        logger.info(
            f"Distributed init: backend={backend} "
            f"world_size={world_size} "
            f"host={socket.gethostname()}"
        )
    return rank, world_size


def teardown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return get_rank() == 0


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor and return the mean across all ranks."""
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / get_world_size()


@contextmanager
def rank_zero_first():
    """Context manager: rank 0 runs first, then all others."""
    if not is_main_process():
        barrier()
    yield
    if is_main_process():
        barrier()

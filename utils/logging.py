"""
PROMETHEUS — Logging utilities
Rank-0 gated logging + structured metric output.
Optional W&B integration.
"""
from __future__ import annotations
import logging
import os
import sys
import time
from typing import Any, Optional

from .distributed import is_main_process


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


_wandb = None
_wandb_enabled = False


def init_wandb(run_name: str, config: Any) -> None:
    global _wandb, _wandb_enabled
    if not is_main_process():
        return
    try:
        import wandb
        _wandb = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "prometheus"),
            name=run_name,
            config=config,
            resume="allow",
        )
        _wandb_enabled = True
    except ImportError:
        logging.getLogger(__name__).warning("wandb not installed — logging to stdout only.")


def log_metrics(metrics: dict[str, float], step: int) -> None:
    """Log to stdout (always) and W&B (if enabled, rank 0 only)."""
    if not is_main_process():
        return
    parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logging.getLogger("prometheus").info(f"step={step:>6d} | {parts}")
    if _wandb_enabled and _wandb is not None:
        _wandb.log(metrics, step=step)


class Throughput:
    """Tracks tokens/sec and Model FLOP Utilization (MFU)."""

    def __init__(self, model_n_params: int, gpu_flops_peak: float):
        """
        Args:
            model_n_params: Number of parameters (used for MFU calc)
            gpu_flops_peak: GPU peak BF16 FLOP/s (e.g. 330e12 for RTX 5060 Ti)
        """
        self.model_n_params = model_n_params
        self.gpu_flops_peak = gpu_flops_peak
        self._t0: Optional[float] = None
        self._tokens: int = 0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._tokens = 0

    def update(self, n_tokens: int) -> None:
        self._tokens += n_tokens

    def compute(self) -> dict[str, float]:
        if self._t0 is None:
            return {}
        elapsed = time.perf_counter() - self._t0
        tok_per_sec = self._tokens / elapsed

        # MFU: actual FLOP/s / peak FLOP/s
        # Forward pass FLOPs ≈ 6 * N * T (N=params, T=tokens) — factor of 6 = 2 (matmul) * 3 (fwd+bwd)
        actual_flops = 6 * self.model_n_params * self._tokens / elapsed
        mfu = actual_flops / self.gpu_flops_peak

        self._t0 = time.perf_counter()
        self._tokens = 0
        return {"tok_per_sec": tok_per_sec, "mfu": mfu}


# GPU peak BF16 FLOP/s for common cards
GPU_FLOPS = {
    "RTX 5060 Ti":  330e12,
    "RTX 4090": 330e12,
    "RTX 4080": 244e12,
    "A100 SXM 80GB": 312e12,
    "H100 SXM": 989e12,
    "H100 PCIe": 756e12,
    "A10G": 125e12,
}


def get_gpu_peak_flops() -> float:
    """Attempt to identify GPU and return peak BF16 FLOP/s."""
    try:
        import torch
        name = torch.cuda.get_device_name(0)
        for k, v in GPU_FLOPS.items():
            if k.lower() in name.lower():
                return v
    except Exception:
        pass
    # Default: conservative estimate
    return 100e12

"""
PROMETHEUS — Model loading + FSDP wrapping
Supports any HuggingFace CausalLM.
Applies:
  - FSDP with transformer_auto_wrap_policy
  - Activation checkpointing (gradient checkpointing)
  - Flash Attention 2 (if available + requested)
  - BF16 / FP16 mixed precision
"""
from __future__ import annotations
import functools
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch

if TYPE_CHECKING:
    from config import FSDPConfig, ModelConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sharding strategy map
# ──────────────────────────────────────────────────────────────────────────────
_SHARDING_MAP = {
    "FULL_SHARD":    ShardingStrategy.FULL_SHARD,      # ZeRO-3
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,   # ZeRO-2
    "HYBRID_SHARD":  ShardingStrategy.HYBRID_SHARD,    # ZeRO-3 intra-node only
    "NO_SHARD":      ShardingStrategy.NO_SHARD,        # DDP
}

_PRECISION_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


# ──────────────────────────────────────────────────────────────────────────────
# Load model from HuggingFace
# ──────────────────────────────────────────────────────────────────────────────
def load_base_model(cfg: "ModelConfig") -> nn.Module:
    from transformers import AutoModelForCausalLM, AutoConfig

    attn_impl = "flash_attention_2" if cfg.flash_attn else "eager"

    hf_config = AutoConfig.from_pretrained(cfg.name_or_path)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.name_or_path,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,   # Load in BF16 to save RAM
            low_cpu_mem_usage=True,
        )
        if cfg.flash_attn:
            logger.info("Flash Attention 2 enabled.")
    except Exception as e:
        logger.warning(f"Flash Attention 2 unavailable ({e}), falling back to SDPA.")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.name_or_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    # Disable KV cache — required for activation checkpointing
    # (cache changes shape between forward and recompute passes)
    model.config.use_cache = False

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {cfg.name_or_path} | {n_params/1e9:.2f}B params")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Detect transformer decoder layer class for auto-wrap
# ──────────────────────────────────────────────────────────────────────────────
def _get_decoder_layer_class(model: nn.Module) -> type:
    """
    Returns the class of a single transformer decoder block.
    Works for Qwen2, Llama, Mistral, Falcon, etc.
    """
    # Try common attribute names
    for attr in ["model", "transformer"]:
        backbone = getattr(model, attr, None)
        if backbone is None:
            continue
        for layer_attr in ["layers", "h", "blocks"]:
            layers = getattr(backbone, layer_attr, None)
            if layers is not None and len(layers) > 0:
                return type(layers[0])

    # Fallback: inspect children for the most common repeated class
    class_counts: dict[type, int] = {}
    for module in model.modules():
        t = type(module)
        class_counts[t] = class_counts.get(t, 0) + 1
    # Most repeated non-trivial class (heuristic)
    candidates = {k: v for k, v in class_counts.items() if v > 1 and k != type(model)}
    if candidates:
        return max(candidates, key=lambda k: candidates[k])

    raise RuntimeError("Could not determine transformer decoder layer class for auto-wrap.")


# ──────────────────────────────────────────────────────────────────────────────
# Wrap model with FSDP
# ──────────────────────────────────────────────────────────────────────────────
def wrap_with_fsdp(model: nn.Module, cfg: "FSDPConfig") -> FSDP:
    dtype = _PRECISION_MAP[cfg.mixed_precision]

    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    ) if cfg.mixed_precision != "fp32" else None

    # Auto-wrap policy: wrap every transformer decoder layer independently
    decoder_cls = _get_decoder_layer_class(model)
    logger.info(f"FSDP auto-wrap: wrapping {decoder_cls.__name__} layers")
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={decoder_cls},
    )

    cpu_offload = CPUOffload(offload_params=True) if cfg.cpu_offload else None
    sharding = _SHARDING_MAP[cfg.sharding_strategy]

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=sharding,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=cfg.use_orig_params,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,         # Broadcast rank 0 weights to all ranks
        limit_all_gathers=True,          # Memory optimization
    )

    logger.info(
        f"FSDP wrapped | strategy={cfg.sharding_strategy} "
        f"precision={cfg.mixed_precision} "
        f"cpu_offload={cfg.cpu_offload}"
    )
    return fsdp_model


# ──────────────────────────────────────────────────────────────────────────────
# Activation checkpointing
# ──────────────────────────────────────────────────────────────────────────────
def apply_activation_checkpointing(model: FSDP) -> None:
    """
    Apply activation checkpointing to each wrapped FSDP unit.
    Trades ~33% speed for ~60% activation memory reduction.
    Must be called AFTER FSDP wrapping.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing as _apply_ac,
        checkpoint_wrapper,
    )

    check_fn = lambda submodule: isinstance(submodule, FSDP)

    # PyTorch 2.4+ removed checkpoint_impl parameter
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
        _apply_ac(
            model,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
    except TypeError:
        _apply_ac(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
    logger.info("Activation checkpointing applied.")


# ──────────────────────────────────────────────────────────────────────────────
# Full model setup
# ──────────────────────────────────────────────────────────────────────────────
def build_model(model_cfg: "ModelConfig", fsdp_cfg: "FSDPConfig") -> tuple[FSDP, int]:
    """
    Load → FSDP wrap → activation checkpointing.
    Returns (fsdp_model, n_params).
    """
    model = load_base_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())

    model = wrap_with_fsdp(model, fsdp_cfg)

    if model_cfg.gradient_checkpointing:
        apply_activation_checkpointing(model)

    return model, n_params

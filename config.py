"""
PROMETHEUS — Training Configuration
Dataclass-driven config, YAML-loadable, fully typed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-1.5B"
    max_seq_len: int = 2048
    flash_attn: bool = True                  # Flash Attention 2 if available
    gradient_checkpointing: bool = True


@dataclass
class FSDPConfig:
    sharding_strategy: Literal[
        "FULL_SHARD",       # ZeRO-3: params + grads + optimizer states sharded
        "SHARD_GRAD_OP",    # ZeRO-2: grads + optimizer states sharded
        "HYBRID_SHARD",     # ZeRO-3 within node, replicated across nodes
        "NO_SHARD",         # DDP equivalent (baseline)
    ] = "FULL_SHARD"
    cpu_offload: bool = False                # Offload params to CPU (enables > VRAM models)
    mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    # Auto-wrap: wrap every transformer decoder layer independently
    min_num_params: int = 1_000_000          # Minimum params to wrap a submodule
    use_orig_params: bool = True             # Required for torch.compile compatibility


@dataclass
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    text_column: str = "text"
    tokenizer_name: Optional[str] = None     # Defaults to model name
    pack_sequences: bool = True              # Pack short seqs to max_seq_len (no padding waste)
    num_workers: int = 4


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    warmup_steps: int = 100
    total_steps: int = 10_000
    min_lr_ratio: float = 0.1               # Final LR = lr * min_lr_ratio (cosine decay)


@dataclass
class TrainingConfig:
    # Core
    model: ModelConfig = field(default_factory=ModelConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Training loop
    batch_size_per_gpu: int = 4              # Local micro-batch
    gradient_accumulation_steps: int = 8    # Effective batch = batch_size_per_gpu * world_size * grad_accum
    max_steps: int = 10_000
    eval_every: int = 500
    save_every: int = 1_000
    log_every: int = 10

    # I/O
    output_dir: str = "./checkpoints"
    run_name: str = "prometheus-run"
    resume_from: Optional[str] = None

    # Infra
    seed: int = 42
    compile: bool = False                   # torch.compile (Linux only, experimental)
    profile_steps: int = 0                  # Set >0 to enable torch.profiler for N steps

    @property
    def effective_batch_size(self) -> int:
        """Computed at runtime after world_size is known."""
        return self.batch_size_per_gpu * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        if "model" in raw:
            cfg.model = ModelConfig(**raw["model"])
        if "fsdp" in raw:
            cfg.fsdp = FSDPConfig(**raw["fsdp"])
        if "data" in raw:
            cfg.data = DataConfig(**raw["data"])
        if "optimizer" in raw:
            cfg.optimizer = OptimizerConfig(**raw["optimizer"])
        if "scheduler" in raw:
            cfg.scheduler = SchedulerConfig(**raw["scheduler"])
        for k, v in raw.items():
            if k not in {"model", "fsdp", "data", "optimizer", "scheduler"}:
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

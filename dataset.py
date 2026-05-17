"""
PROMETHEUS — Dataset & DataLoader
Streaming tokenization with sequence packing.
No padding waste: short sequences are concatenated up to max_seq_len.
"""
from __future__ import annotations
import logging
from typing import Iterator, Optional, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizer

if TYPE_CHECKING:
    from config import DataConfig, ModelConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sequence packing: concatenate docs, split at max_seq_len
# ──────────────────────────────────────────────────────────────────────────────
class PackedStreamingDataset(IterableDataset):
    """
    Streams from HuggingFace datasets, tokenizes on the fly,
    and packs token sequences into fixed-length chunks.

    Each item is (input_ids, labels) of shape (max_seq_len,).
    Labels are input_ids shifted by 1 (standard CLM).
    EOS token is appended between documents.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        split: str = "train",
        text_column: str = "text",
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.text_column = text_column
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def _load_dataset(self):
        from datasets import load_dataset
        return load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

    def _shard_dataset(self, dataset):
        """Each rank gets a non-overlapping shard."""
        # Account for multiple workers per rank
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        total_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id

        return dataset.shard(num_shards=total_shards, index=shard_id)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        dataset = self._load_dataset()
        dataset = self._shard_dataset(dataset)
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10_000)

        eos_id = self.tokenizer.eos_token_id
        buffer: list[int] = []

        for example in dataset:
            text = example[self.text_column]
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            buffer.extend(ids)
            buffer.append(eos_id)

            # Emit full chunks
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[:self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len:]
                input_ids = torch.tensor(chunk[:self.max_seq_len], dtype=torch.long)
                labels = torch.tensor(chunk[1:self.max_seq_len + 1], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def build_dataloader(
    data_cfg: "DataConfig",
    model_cfg: "ModelConfig",
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int = 42,
    split: str = "train",
) -> DataLoader:
    tok_name = data_cfg.tokenizer_name or model_cfg.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    dataset = PackedStreamingDataset(
        dataset_name=data_cfg.dataset_name,
        dataset_config=data_cfg.dataset_config,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.max_seq_len,
        split=split,
        text_column=data_cfg.text_column,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        prefetch_factor=2 if data_cfg.num_workers > 0 else None,
        persistent_workers=data_cfg.num_workers > 0,
    )

    logger.info(
        f"DataLoader ready | dataset={data_cfg.dataset_name}/{data_cfg.dataset_config} "
        f"batch_size={batch_size} rank={rank}/{world_size}"
    )
    return loader

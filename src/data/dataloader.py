"""Tokenization + sharded dataset utilities for Phase 1."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import datasets
from transformers import AutoTokenizer


@dataclass
class DataConfig:
    dataset: str
    split: str = "train"
    text_column: str = "text"
    dataset_config: Optional[str] = None
    tokenizer_path: str = "/scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2"
    seq_len: int = 1024


@dataclass
class ShardConfig:
    output_dir: Path
    tokens_per_shard: int = 2_097_152  # 1024 seqs of len 2048 by default


class ShardWriter:
    """Accumulates token ids and spills to disk once threshold reached."""

    def __init__(self, shard_cfg: ShardConfig) -> None:
        self.cfg = shard_cfg
        self.output_dir = Path(shard_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokens: List[int] = []
        self.shard_idx = 0

    def add_tokens(self, ids: Iterable[int]) -> None:
        self.tokens.extend(ids)
        if len(self.tokens) >= self.cfg.tokens_per_shard:
            self._flush()

    def close(self) -> None:
        if self.tokens:
            self._flush()

    def _flush(self) -> None:
        target = self.output_dir / f"shard_{self.shard_idx:05d}.pt"
        import torch

        tensor = torch.tensor(self.tokens[: self.cfg.tokens_per_shard], dtype=torch.long)
        torch.save({"input_ids": tensor}, target)
        print(f"[ShardWriter] wrote {target} ({tensor.numel()} tokens)")
        self.tokens = self.tokens[self.cfg.tokens_per_shard :]
        self.shard_idx += 1


def build_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def stream_examples(cfg: DataConfig):
    ds = datasets.load_dataset(cfg.dataset, cfg.dataset_config, split=cfg.split, streaming=True)
    for sample in ds:
        text = sample[cfg.text_column]
        if not isinstance(text, str):
            continue
        yield text


def tokenize_stream(text_iter: Iterable[str], tokenizer, seq_len: int) -> Iterator[List[int]]:
    buffer: List[int] = []
    for text in text_iter:
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(enc + [tokenizer.eos_token_id])
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            buffer = buffer[seq_len:]
    if buffer:
        # pad to seq_len for final shard to simplify training loops
        pad_id = tokenizer.pad_token_id
        buffer.extend([pad_id] * (seq_len - len(buffer)))
        yield buffer[:seq_len]


def preprocess(cfg: DataConfig, shard_cfg: ShardConfig) -> None:
    tokenizer = build_tokenizer(cfg.tokenizer_path)
    writer = ShardWriter(shard_cfg)
    total_tokens = 0
    seqs = 0

    for seq in tokenize_stream(stream_examples(cfg), tokenizer, cfg.seq_len):
        writer.add_tokens(seq)
        total_tokens += len(seq)
        seqs += 1
        if seqs % 100 == 0:
            print(f"[preprocess] sequences={seqs}, tokens={total_tokens}")

    writer.close()
    print(
        f"[preprocess] finished {seqs} sequences, ~{total_tokens/1e6:.2f}M tokens," f" {math.ceil(total_tokens / shard_cfg.tokens_per_shard)} shards"
    )


__all__ = [
    "DataConfig",
    "ShardConfig",
    "ShardWriter",
    "preprocess",
]

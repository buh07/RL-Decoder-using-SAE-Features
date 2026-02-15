#!/usr/bin/env python3
"""Tokenize normalized reasoning datasets into fixed-length shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from transformers import AutoTokenizer


@dataclass
class TokenizationArgs:
    raw_dir: Path
    tokenized_dir: Path
    tokenizer_dir: Path
    datasets: List[str]
    splits: Optional[List[str]]
    seq_len: int
    tokens_per_shard: int
    world_size: int
    rank: int
    force: bool
    resume: bool


def load_manifest(raw_dir: Path) -> Dict[str, Dict]:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run datasets/download_datasets.py before tokenizing."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_records(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_text(record: Dict) -> str:
    parts: List[str] = []
    question = (record.get("question") or "").strip()
    if question:
        parts.append(question)
    cot = record.get("cot")
    if cot:
        cot = cot.strip()
        if cot:
            parts.append(cot)
    answer = (record.get("answer") or "").strip()
    if answer:
        parts.append("Answer:\n" + answer)
    return "\n\n".join(parts).strip()


def sequence_iterator(
    jsonl_path: Path,
    tokenizer,
    seq_len: int,
    pad_token_id: int,
    world_size: int,
    rank: int,
) -> Iterator[List[int]]:
    buffer: List[int] = []
    example_idx = 0
    eos_id = tokenizer.eos_token_id

    for record in iter_records(jsonl_path):
        if example_idx % world_size != rank:
            example_idx += 1
            continue
        text = render_text(record)
        if not text:
            example_idx += 1
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids.append(eos_id)
        buffer.extend(token_ids)
        while len(buffer) >= seq_len:
            seq = buffer[:seq_len]
            buffer = buffer[seq_len:]
            yield seq
        example_idx += 1

    remainder = len(buffer) % seq_len
    if remainder:
        buffer.extend([pad_token_id] * (seq_len - remainder))
    while buffer:
        seq = buffer[:seq_len]
        buffer = buffer[seq_len:]
        yield seq


class ShardWriter:
    def __init__(
        self,
        dataset: str,
        split: str,
        output_dir: Path,
        tokenized_root: Path,
        seq_len: int,
        tokens_per_shard: int,
        start_index: int = 0,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenized_root = tokenized_root
        self.seq_len = seq_len
        self.tokens_per_shard = tokens_per_shard
        if tokens_per_shard % seq_len != 0:
            raise ValueError("tokens_per_shard must be divisible by seq_len")
        self.seqs_per_shard = tokens_per_shard // seq_len
        self.shard_idx = start_index
        self.current_sequences: List[List[int]] = []
        self.total_sequences = 0

    def add_sequence(self, seq: List[int]) -> None:
        if len(seq) != self.seq_len:
            raise ValueError("Sequence length mismatch")
        self.current_sequences.append(seq)
        self.total_sequences += 1
        if len(self.current_sequences) >= self.seqs_per_shard:
            self._flush()

    def close(self) -> None:
        if self.current_sequences:
            self._flush()

    def existing_shards(self) -> List[Path]:
        return sorted(self.output_dir.glob("shard_*.pt"))

    def _flush(self) -> None:
        tensor = torch.tensor(self.current_sequences, dtype=torch.int32)
        target = self.output_dir / f"shard_{self.shard_idx:05d}.pt"
        payload = {"input_ids": tensor}
        torch.save(payload, target)
        sha256 = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        seq_count = tensor.shape[0]
        metadata = {
            "dataset": self.dataset,
            "split": self.split,
            "seq_len": self.seq_len,
            "num_sequences": seq_count,
            "token_count": seq_count * self.seq_len,
            "shard_index": self.shard_idx,
            "path": str(target.relative_to(self.tokenized_root)),
            "sha256": sha256,
        }
        meta_path = target.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            f"[shard] {self.dataset}/{self.split} -> {target.name}"
            f" (sequences={seq_count}, tokens={metadata['token_count']})"
        )
        self.shard_idx += 1
        self.current_sequences = []


def write_global_manifest(tokenized_root: Path) -> None:
    entries: List[Dict] = []
    for meta_path in sorted(tokenized_root.rglob("shard_*.meta.json")):
        with meta_path.open("r", encoding="utf-8") as f:
            entry = json.load(f)
        entries.append(entry)
    manifest_path = tokenized_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {manifest_path} ({len(entries)} shards)")


def process_split(
    spec_name: str,
    split: str,
    args: TokenizationArgs,
    tokenizer,
) -> None:
    jsonl_path = args.raw_dir / spec_name / f"{split}.jsonl"
    if not jsonl_path.exists():
        print(f"[skip] Missing {jsonl_path}; did you download this split?")
        return

    split_out_dir = args.tokenized_dir / spec_name / split
    writer = ShardWriter(
        dataset=spec_name,
        split=split,
        output_dir=split_out_dir,
        tokenized_root=args.tokenized_dir,
        seq_len=args.seq_len,
        tokens_per_shard=args.tokens_per_shard,
        start_index=0,
    )

    existing = writer.existing_shards()
    if existing and not (args.force or args.resume):
        print(f"[skip] {split_out_dir} already has shards. Use --force or --resume.")
        return
    if existing and args.force:
        for path in existing:
            path.unlink()
            meta = path.with_suffix(".meta.json")
            if meta.exists():
                meta.unlink()
        existing = []
    if existing and args.resume:
        writer.shard_idx = len(existing)
        print(
            f"[resume] Starting shard index {writer.shard_idx}"
            f" for {spec_name}/{split}"
        )

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    sequences = 0
    for seq in sequence_iterator(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        pad_token_id=pad_token_id,
        world_size=args.world_size,
        rank=args.rank,
    ):
        writer.add_sequence(seq)
        sequences += 1
    writer.close()
    print(
        f"[done] {spec_name}/{split}: {sequences} sequences"
        f" (~{sequences * args.seq_len / 1e6:.2f}M tokens)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize reasoning datasets into fixed-length shards")
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--tokenized-dir", type=Path, default=Path("datasets/tokenized/gpt2"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("assets/tokenizers/gpt2"))
    parser.add_argument("--dataset", dest="datasets", nargs="*", default=None)
    parser.add_argument("--split", dest="splits", nargs="*", default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--tokens-per-shard", type=int, default=2_097_152)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Delete existing shards before tokenizing")
    parser.add_argument("--resume", action="store_true", help="Append new shards after existing ones")
    args_ns = parser.parse_args()

    manifest = load_manifest(args_ns.raw_dir)
    datasets = args_ns.datasets or sorted(manifest.keys())
    splits_filter = args_ns.splits

    tok = AutoTokenizer.from_pretrained(args_ns.tokenizer_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    token_args = TokenizationArgs(
        raw_dir=args_ns.raw_dir,
        tokenized_dir=args_ns.tokenized_dir,
        tokenizer_dir=args_ns.tokenizer_dir,
        datasets=datasets,
        splits=splits_filter,
        seq_len=args_ns.seq_len,
        tokens_per_shard=args_ns.tokens_per_shard,
        world_size=args_ns.world_size,
        rank=args_ns.rank,
        force=args_ns.force,
        resume=args_ns.resume,
    )

    for dataset_name in datasets:
        spec = manifest.get(dataset_name)
        if spec is None:
            print(f"[warn] {dataset_name} missing from manifest; skipping.")
            continue
        splits = splits_filter or spec.get("splits") or []
        if not splits:
            print(f"[warn] No splits listed for {dataset_name}; skipping.")
            continue
        for split in splits:
            process_split(dataset_name, split, token_args, tok)

    write_global_manifest(token_args.tokenized_dir)


if __name__ == "__main__":
    main()

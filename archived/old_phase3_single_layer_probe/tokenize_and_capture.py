#!/usr/bin/env python3
"""
Tokenize full GSM8K split and capture GPT-2 layer-6 activations.

Steps:
  1. Load full GSM8K split from HuggingFace (train=7473, test=1319)
  2. Tokenize question+answer for each example, pad/chunk to 1024 tokens
  3. Save tokenized shards to --tokenized-dir
  4. Run GPT-2 forward pass with activation hooks, save residual shards to --output-dir

Usage:
    CUDA_VISIBLE_DEVICES=5 python phase3/tokenize_and_capture.py --split train \
        --tokenized-dir datasets/tokenized/gpt2/gsm8k_full \
        --output-dir /scratch2/f004ndc/gpt2_gsm8k_acts_full \
        --device cuda:0

    CUDA_VISIBLE_DEVICES=6 python phase3/tokenize_and_capture.py --split test \
        --tokenized-dir datasets/tokenized/gpt2/gsm8k_full \
        --output-dir /scratch2/f004ndc/gpt2_gsm8k_acts_full \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activation_capture import create_gpt2_capture
from model_registry import get_model

# ── constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1024       # tokens per activation shard row
SEQS_PER_SHARD = 512    # flush a new shard every N sequences
LAYER = 6               # GPT-2 small mid-layer


def tokenize_split(split: str, tokenizer, tokenized_dir: Path) -> list[Path]:
    """
    Tokenize full GSM8K split into fixed-size shards of input_ids.
    Each example is formatted as  "<question>\n<answer>"  then chunked to CHUNK_SIZE.
    Returns sorted list of shard paths.
    """
    out_dir = tokenized_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = sorted(out_dir.glob("shard_*.pt"))
    if existing:
        print(f"[tokenize] Found {len(existing)} existing shards in {out_dir}, skipping.")
        return existing

    print(f"[tokenize] Loading GSM8K {split} split...")
    dataset = load_dataset("gsm8k", "main", split=split)
    print(f"[tokenize] {len(dataset)} examples to tokenize")

    pad_id = tokenizer.eos_token_id or 50256
    shard_idx = 0
    current_rows: list[torch.Tensor] = []

    def flush(rows: list[torch.Tensor], idx: int) -> Path:
        stacked = torch.stack(rows, dim=0)          # [N, CHUNK_SIZE]
        path = out_dir / f"shard_{idx:05d}.pt"
        torch.save({"input_ids": stacked}, path)
        meta = {"shard_index": idx, "num_sequences": len(rows), "seq_len": CHUNK_SIZE}
        (out_dir / f"shard_{idx:05d}.meta.json").write_text(json.dumps(meta))
        print(f"  [shard] {path.name}  seqs={len(rows)}")
        return path

    shard_paths: list[Path] = []

    for idx, item in enumerate(dataset):
        if (idx + 1) % 500 == 0:
            print(f"  tokenized {idx+1}/{len(dataset)}")

        text = f"{item['question']}\n{item['answer']}"
        ids = tokenizer.encode(text, add_special_tokens=False)

        # Chunk into CHUNK_SIZE pieces
        for start in range(0, max(1, len(ids)), CHUNK_SIZE):
            chunk = ids[start : start + CHUNK_SIZE]
            # Pad to CHUNK_SIZE
            if len(chunk) < CHUNK_SIZE:
                chunk = chunk + [pad_id] * (CHUNK_SIZE - len(chunk))
            current_rows.append(torch.tensor(chunk[:CHUNK_SIZE], dtype=torch.long))

            if len(current_rows) >= SEQS_PER_SHARD:
                shard_paths.append(flush(current_rows, shard_idx))
                shard_idx += 1
                current_rows = []

    if current_rows:
        shard_paths.append(flush(current_rows, shard_idx))

    print(f"[tokenize] Done. {len(shard_paths)} shards, total rows ~{shard_idx * SEQS_PER_SHARD + len(current_rows)}")
    return sorted(out_dir.glob("shard_*.pt"))


def capture_activations(
    shard_paths: list[Path],
    model,
    output_dir: Path,
    batch_size: int,
    device: str,
) -> None:
    """Run model forward pass on tokenized shards, saving layer-6 residual activations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[capture] Saving to {output_dir}")
    print(f"[capture] Processing {len(shard_paths)} tokenized shards, batch_size={batch_size}")

    total_seqs = 0
    with create_gpt2_capture(
        model,
        output_dir=output_dir,
        layer_indices=[LAYER],
        capture_residual=True,
        capture_mlp_hidden=False,   # residual only; saves disk & time
    ) as capture:
        for shard_path in shard_paths:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            input_ids = payload["input_ids"]    # [N, CHUNK_SIZE]
            n = input_ids.shape[0]
            print(f"  [shard] {shard_path.name}  seqs={n}")

            for start in range(0, n, batch_size):
                batch = input_ids[start : start + batch_size].to(device)
                with torch.inference_mode():
                    model(batch)
                total_seqs += batch.shape[0]

    print(f"[capture] Done. Total sequences processed: {total_seqs}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize GSM8K and capture GPT-2 activations")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--tokenized-dir",
                        default="/scratch2/f004ndc/RL-Decoder with SAE Features/datasets/tokenized/gpt2/gsm8k_full",
                        type=Path)
    parser.add_argument("--output-dir",
                        default="/scratch2/f004ndc/gpt2_gsm8k_acts_full",
                        type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # ── Load model + tokenizer ───────────────────────────────────────────────
    spec = get_model("gpt2")
    print(f"[main] Loading GPT-2 from {spec.hf_local_path} onto {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(str(spec.hf_local_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(spec.hf_local_path),
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    print(f"[main] Model loaded.")

    # ── Step 1: Tokenize ──────────────────────────────────────────────────────
    shard_paths = tokenize_split(args.split, tokenizer, args.tokenized_dir)

    # ── Step 2: Capture activations ───────────────────────────────────────────
    output_dir = args.output_dir / "gsm8k" / args.split
    capture_activations(shard_paths, model, output_dir, args.batch_size, args.device)

    print(f"\n[main] All done. Activations saved to {output_dir}")
    print(f"[main] Shard files:")
    for p in sorted(output_dir.glob("*.pt")):
        print(f"  {p.name}  ({p.stat().st_size // 1024 // 1024} MB)")


if __name__ == "__main__":
    main()

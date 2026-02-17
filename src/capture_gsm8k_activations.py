#!/usr/bin/env python3
"""
Capture GPT-2 activations on GSM8K tokenized shards.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM

from activation_capture import create_gpt2_capture
from model_registry import get_model


def _iter_token_chunks(input_ids: torch.Tensor, max_seq_len: int, pad_id: int) -> Iterable[torch.Tensor]:
    """Yield fixed-length chunks from a token sequence."""
    seq_len = input_ids.shape[0]
    for start in range(0, seq_len, max_seq_len):
        chunk = input_ids[start : start + max_seq_len]
        if chunk.shape[0] < max_seq_len:
            pad = torch.full((max_seq_len - chunk.shape[0],), pad_id, dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=0)
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture GPT-2 activations for GSM8K")
    parser.add_argument("--model", default="gpt2", help="Model slug (default: gpt2)")
    parser.add_argument("--tokenized-dir", default="datasets/tokenized/gpt2", help="Tokenized dataset root")
    parser.add_argument("--dataset", default="gsm8k", help="Dataset name (default: gsm8k)")
    parser.add_argument("--split", default="train", help="Split to capture (default: train)")
    parser.add_argument("--output-dir", default="/tmp/gpt2_gsm8k_acts", help="Output directory root")
    parser.add_argument("--layers", type=int, nargs="+", default=[6], help="Layer indices to capture")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for capture")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    args = parser.parse_args()

    tokenized_dir = Path(args.tokenized_dir) / args.dataset / args.split
    shard_paths = sorted(tokenized_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {tokenized_dir}")

    spec = get_model(args.model)
    print(f"[capture] Loading {spec.name} from {spec.hf_local_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(spec.hf_local_path),
        local_files_only=True,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map=args.device,
    )
    model.eval()

    max_seq_len = int(getattr(model.config, "n_positions", 1024))
    pad_id = int(getattr(model.config, "eos_token_id", 50256))
    print(f"[capture] Using max_seq_len={max_seq_len}, pad_id={pad_id}")

    output_dir = Path(args.output_dir) / args.dataset / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    batch = []

    with create_gpt2_capture(
        model,
        output_dir=output_dir,
        layer_indices=args.layers,
        capture_residual=True,
        capture_mlp_hidden=True,
    ):
        for shard_path in shard_paths:
            payload = torch.load(shard_path, map_location="cpu")
            input_ids = payload["input_ids"]
            print(f"[capture] shard={shard_path.name} sequences={input_ids.shape[0]}")

            for row in input_ids:
                for chunk in _iter_token_chunks(row, max_seq_len=max_seq_len, pad_id=pad_id):
                    batch.append(chunk)
                    if len(batch) >= args.batch_size:
                        input_batch = torch.stack(batch, dim=0).to(model.device)
                        with torch.inference_mode():
                            _ = model(input_batch)
                        total_chunks += len(batch)
                        batch.clear()

        if batch:
            input_batch = torch.stack(batch, dim=0).to(model.device)
            with torch.inference_mode():
                _ = model(input_batch)
            total_chunks += len(batch)
            batch.clear()

    print(f"[capture] Complete. Total chunks processed: {total_chunks}")


if __name__ == "__main__":
    main()

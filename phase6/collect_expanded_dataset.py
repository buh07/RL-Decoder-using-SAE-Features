#!/usr/bin/env python3
"""
Phase 6 — Expanded Dataset Collector
=====================================
Extends Phase 4's arithmetic data collector to:
  1. Process ALL GSM8K examples (train + test splits)
  2. Store raw hidden states (24, 1024) alongside SAE features (24, 12288)
  3. Compute baseline log-prob of the correct result token
  4. Store result_token_id for the decoder's cross-entropy target
  5. Tag each record with gsm8k_split ("train" or "test")

Output: Two .pt files:
  phase6_results/dataset/gsm8k_expanded_train.pt
  phase6_results/dataset/gsm8k_expanded_test.pt

GPU parallelism (3 shards by example index):
  CUDA_VISIBLE_DEVICES=0 python3 phase6/collect_expanded_dataset.py \\
      --split train --shard 0 3

  CUDA_VISIBLE_DEVICES=1 python3 phase6/collect_expanded_dataset.py \\
      --split train --shard 1 3

  CUDA_VISIBLE_DEVICES=2 python3 phase6/collect_expanded_dataset.py \\
      --split train --shard 2 3

  Then: python3 phase6/collect_expanded_dataset.py --merge
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "phase4"))

from sae_architecture import SparseAutoencoder
from sae_config import SAEConfig

NUM_LAYERS = 24
HIDDEN_DIM = 1024
LATENT_DIM = 12288
MODEL_ID   = "gpt2-medium"
ANN_RE     = re.compile(r"<<([^>]*?)=\s*(-?\d+(?:\.\d+)?)\s*>>")


# ── Reuse Phase 4 utilities ─────────────────────────────────────────────────

def load_saes(saes_dir: Path, device: str) -> Dict[int, SparseAutoencoder]:
    saes: Dict[int, SparseAutoencoder] = {}
    for layer_idx in range(NUM_LAYERS):
        ckpt_path = saes_dir / f"gpt2-medium_layer{layer_idx}_sae.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cd = ckpt["config"]
        cfg = SAEConfig(
            input_dim=cd["input_dim"],
            expansion_factor=cd["expansion_factor"],
            use_relu=cd.get("use_relu", True),
            use_topk=cd.get("use_topk", False),
            topk_k=cd.get("topk_k", 0),
            use_amp=False,
        )
        sae = SparseAutoencoder(cfg)
        sae.load_state_dict(ckpt["model_state_dict"])
        sae = sae.to(device).eval()
        saes[layer_idx] = sae
    print(f"  Loaded {len(saes)} SAEs.")
    return saes


def load_norm_stats(activations_dir: Path, device: str) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in range(NUM_LAYERS):
        path = activations_dir / f"gpt2-medium_layer{layer_idx}_activations.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        acts = payload["activations"] if isinstance(payload, dict) else payload
        if acts.dim() == 3:
            acts = acts.reshape(-1, acts.shape[-1])
        acts = acts.float()
        stats[layer_idx] = (
            acts.mean(dim=0).to(device),
            acts.std(dim=0).clamp_min(1e-6).to(device),
        )
    return stats


def normalize(x, stats):
    if stats is None:
        return x
    mean, std = stats
    return (x - mean) / std


def register_hooks(model):
    store: Dict[int, torch.Tensor] = {}
    handles = []
    def make_hook(L):
        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            store[L] = h.detach().cpu()
        return hook_fn
    for i in range(NUM_LAYERS):
        handles.append(model.transformer.h[i].register_forward_hook(make_hook(i)))
    return store, handles


def find_tokens_for_span(offset_mapping, char_start, char_end):
    result = []
    for i, (s, e) in enumerate(offset_mapping):
        if e > char_start and s < char_end and e > s:
            result.append(i)
    return result


def find_token_for_char(offset_mapping, char_pos):
    for i, (s, e) in enumerate(offset_mapping):
        if s <= char_pos < e:
            return i
    best = None
    for i, (s, e) in enumerate(offset_mapping):
        if s <= char_pos:
            best = i
    return best


def parse_annotations(text):
    results = []
    for m in ANN_RE.finditer(text):
        expr_str = m.group(1)
        C = float(m.group(2))
        ann_text = m.group(0)
        ann_start = m.start()
        ann_end = m.end()
        inner_eq_char = ann_start + ann_text.rfind("=")
        expr_start_char = ann_start + 2
        expr_end_char = inner_eq_char
        result_start_char = inner_eq_char + 1
        result_end_char = ann_end - 2
        outer_eq_char = None
        search_start = max(0, ann_start - 12)
        substr = text[search_start:ann_start]
        idx = substr.rfind("=")
        if idx != -1:
            outer_eq_char = search_start + idx
        results.append({
            "ann_text":      ann_text,
            "expr_str":      expr_str.strip(),
            "C":             C,
            "inner_eq_char": inner_eq_char,
            "expr_chars":    (expr_start_char, expr_end_char),
            "result_chars":  (result_start_char, result_end_char),
            "ann_span":      (ann_start, ann_end),
            "outer_eq_char": outer_eq_char,
        })
    return results


# ── Feature extraction (extended to include raw hidden states) ────────────

@torch.no_grad()
def extract_record(
    ann: dict,
    input_ids: torch.Tensor,
    offset_mapping: List[Tuple[int, int]],
    tokens_str: List[str],
    activation_store: Dict[int, torch.Tensor],
    saes: Dict[int, SparseAutoencoder],
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    logits: torch.Tensor,
    device: str,
    example_idx: int,
    ann_idx: int,
    gsm8k_split: str,
) -> Tuple[Optional[dict], Optional[str]]:
    """Extract one annotation record with raw hidden states, SAE features, and baseline logprob."""

    # Map char positions → token indices
    eq_toks = find_tokens_for_span(offset_mapping, ann["inner_eq_char"], ann["inner_eq_char"] + 1)
    pre_eq_toks = find_tokens_for_span(offset_mapping, ann["expr_chars"][0], ann["expr_chars"][1])
    result_toks = find_tokens_for_span(offset_mapping, ann["result_chars"][0], ann["result_chars"][1])

    if not eq_toks or not pre_eq_toks or not result_toks:
        return None, "token_alignment"

    # Filter: only single-token results (decoder predicts one token)
    if len(result_toks) != 1:
        return None, "multi_token_result"

    eq_tok_idx = eq_toks[0]
    result_tok_idx = result_toks[0]
    result_token_id = input_ids[0, result_tok_idx].item()

    # Raw hidden states at eq_tok position: (24, 1024) float16
    raw_hidden = torch.stack([
        activation_store[L][0, eq_tok_idx, :] for L in range(NUM_LAYERS)
    ]).half()  # (24, 1024)

    # SAE features at eq_tok position: (24, 12288) float16
    sae_features_list = []
    for L in range(NUM_LAYERS):
        h_raw = activation_store[L][0, eq_tok_idx, :].to(device).float()
        h_norm = normalize(h_raw, norm_stats.get(L))
        h_feat = saes[L].encode(h_norm.unsqueeze(0)).squeeze(0)
        sae_features_list.append(h_feat.half().cpu())
    sae_features = torch.stack(sae_features_list)  # (24, 12288)

    # Baseline log-prob: model's own prediction of the result token
    # at position eq_tok_idx (autoregressive: logits[eq_tok] predict token at eq_tok+1)
    # Predict result token from the position just before it
    pred_pos = result_tok_idx - 1
    if pred_pos < 0 or pred_pos >= logits.shape[1]:
        return None, "token_alignment"
    log_probs = F.log_softmax(logits[0, pred_pos, :], dim=-1)
    baseline_logprob = log_probs[result_token_id].item()

    # Top-5 predictions at this position
    top5_vals, top5_ids = log_probs.topk(5)

    C = ann["C"]
    return {
        "schema_version": "phase6_v1",
        "example_idx":     example_idx,
        "ann_idx":         ann_idx,
        "ann_text":        ann["ann_text"],
        "expr_str":        ann["expr_str"],
        "C":               C,
        "log_abs_C":       math.log(abs(C) + 1.0),
        "eq_tok_idx":      eq_tok_idx,
        "result_tok_idx":  result_tok_idx,
        "result_token_id": result_token_id,
        "pre_eq_tok_idxs": pre_eq_toks,
        "token_ids":       input_ids[0].tolist(),
        "tokens":          tokens_str,
        "raw_hidden":      raw_hidden,            # (24, 1024) float16
        "sae_features":    sae_features,           # (24, 12288) float16
        "baseline_logprob": baseline_logprob,
        "baseline_top5":   list(zip(top5_ids.tolist(), top5_vals.tolist())),
        "gsm8k_split":    gsm8k_split,
    }, None


# ── Main collection ──────────────────────────────────────────────────────────

def collect(args):
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model + SAEs
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = model.to(device).eval()

    print("Loading SAEs...")
    saes = load_saes(Path(args.saes_dir), device)
    print("Loading norm stats...")
    norm_stats = load_norm_stats(Path(args.activations_dir), device)

    activation_store, handles = register_hooks(model)

    # Load GSM8K data
    gsm8k_path = Path(args.gsm8k_path)
    gsm8k_split = args.split
    print(f"Loading {gsm8k_split} split from {gsm8k_path}...")
    with open(gsm8k_path) as f:
        raw_examples = [json.loads(line) for line in f]
    print(f"  {len(raw_examples)} examples")

    # Shard if requested
    if args.shard is not None:
        shard_idx, n_shards = args.shard
        shard_size = (len(raw_examples) + n_shards - 1) // n_shards
        start = shard_idx * shard_size
        end = min(start + shard_size, len(raw_examples))
        raw_examples = raw_examples[start:end]
        global_offset = start
        print(f"  Shard {shard_idx}/{n_shards}: examples {start}-{end-1} ({len(raw_examples)} examples)")
    else:
        global_offset = 0

    if args.max_examples is not None:
        raw_examples = raw_examples[: args.max_examples]
        print(f"  Limiting to first {len(raw_examples)} examples via --max-examples")

    records: List[dict] = []
    n_skipped_no_annotations = 0
    n_skipped_alignment = 0
    n_multitoken = 0

    for local_idx, ex in enumerate(raw_examples):
        ex_idx = global_offset + local_idx
        text = ex["question"] + "\n" + ex["answer"]
        annotations = parse_annotations(text)
        if not annotations:
            n_skipped_no_annotations += 1
            continue

        enc = tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            truncation=True, max_length=1024,
        )
        input_ids = enc["input_ids"].to(device)
        offset_mapping = enc["offset_mapping"][0].tolist()
        tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # Forward pass — populates activation_store
        with torch.no_grad():
            logits = model(input_ids).logits

        for ann_idx, ann in enumerate(annotations):
            rec, skip_reason = extract_record(
                ann, input_ids, offset_mapping, tokens_str,
                activation_store, saes, norm_stats, logits, device,
                ex_idx, ann_idx, gsm8k_split,
            )
            if rec is None:
                if skip_reason == "multi_token_result":
                    n_multitoken += 1
                elif skip_reason == "token_alignment":
                    n_skipped_alignment += 1
                else:
                    n_skipped_alignment += 1
                continue
            records.append(rec)

        if (local_idx + 1) % 100 == 0 or (local_idx + 1) == len(raw_examples):
            print(f"  {local_idx+1}/{len(raw_examples)} examples, {len(records)} records...")

    for h in handles:
        h.remove()

    # Save
    shard_suffix = f"_shard{args.shard[0]}" if args.shard else ""
    out_path = output_dir / f"gsm8k_expanded_{gsm8k_split}{shard_suffix}.pt"
    torch.save(records, out_path)

    print(f"\n=== Collection Summary ({gsm8k_split}{shard_suffix}) ===")
    print(f"  Records:          {len(records)}")
    print(f"  Skipped (no ann): {n_skipped_no_annotations}")
    print(f"  Skipped (token alignment): {n_skipped_alignment}")
    print(f"  Skipped (multi-token result): {n_multitoken}")
    if records:
        C_vals = [r["C"] for r in records]
        print(f"  C range: [{min(C_vals):.1f}, {max(C_vals):.1f}]")
        bl = [r["baseline_logprob"] for r in records]
        print(f"  Baseline logprob: mean={np.mean(bl):.3f}, median={np.median(bl):.3f}")
    print(f"  Saved: {out_path}")


def merge(args):
    """Merge shards for a given split."""
    output_dir = Path(args.output_dir)
    for split in ["train", "test"]:
        shards = sorted(output_dir.glob(f"gsm8k_expanded_{split}_shard*.pt"))
        if not shards:
            continue
        print(f"Merging {len(shards)} shards for {split}...")
        all_records = []
        for s in shards:
            records = torch.load(s, weights_only=False)
            all_records.extend(records)
            print(f"  {s.name}: {len(records)} records")
        out_path = output_dir / f"gsm8k_expanded_{split}.pt"
        torch.save(all_records, out_path)
        print(f"  Merged: {len(all_records)} records → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--gsm8k-path", default=None,
                   help="Override GSM8K path (default: auto from split)")
    p.add_argument("--saes-dir", default="phase2_results/saes_gpt2_12x_topk/saes")
    p.add_argument("--activations-dir", default="phase2_results/activations")
    p.add_argument("--output-dir", default="phase6_results/dataset")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Process only the first N examples (after sharding), for smoke tests")
    p.add_argument("--shard", type=int, nargs=2, metavar=("IDX", "N"),
                   help="Shard index and total shards (e.g., --shard 0 3)")
    p.add_argument("--merge", action="store_true", help="Merge shards and exit")
    return p.parse_args()


def main():
    args = parse_args()
    if args.merge:
        merge(args)
        return

    if args.gsm8k_path is None:
        args.gsm8k_path = f"datasets/raw/gsm8k/{args.split}.jsonl"
    collect(args)


if __name__ == "__main__":
    main()

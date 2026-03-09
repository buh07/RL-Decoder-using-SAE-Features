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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from phase7.model_registry import create_adapter, resolve_model_spec
from phase7.sae_assets import load_norm_stats, load_saes, model_key_safe

ANN_RE     = re.compile(r"<<([^>]*?)=\s*(-?\d+(?:\.\d+)?)\s*>>")


def normalize(x, stats):
    if stats is None:
        return x
    mean, std = stats
    return (x - mean) / std


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


@torch.no_grad()
def forward_hidden_states_only(adapter, input_ids: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    if adapter.model is None:
        raise RuntimeError("Adapter model is not loaded")
    ids = adapter.tokenize(input_ids)
    model = adapter.model
    if hasattr(model, "model"):  # Qwen/LLaMA style
        out = model.model(ids, output_hidden_states=True, return_dict=True)
        hs = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
        return hs, out.last_hidden_state
    if hasattr(model, "transformer"):  # GPT-2 style
        out = model.transformer(ids, output_hidden_states=True, return_dict=True)
        hs = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
        return hs, out.last_hidden_state
    # Fallback to adapter forward when base-model path is not recognized.
    logits, hs = adapter.forward(ids)
    return hs, hs[-1]


# ── Feature extraction (extended to include raw hidden states) ────────────

@torch.no_grad()
def extract_record(
    ann: dict,
    input_ids: torch.Tensor,
    offset_mapping: List[Tuple[int, int]],
    tokens_str: List[str],
    hidden_states: Tuple[torch.Tensor, ...],
    saes: Dict[int, torch.nn.Module],
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    log_probs_by_pred_pos: Dict[int, torch.Tensor],
    device: str,
    example_idx: int,
    ann_idx: int,
    gsm8k_split: str,
    model_key: str,
    model_family: str,
    tokenizer_id: str,
    capture_extra_anchors: bool = False,
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
    num_layers = len(hidden_states)
    hidden_dim = int(hidden_states[0].shape[-1])
    model_sae_dim = int(next(iter(saes.values())).decoder.weight.shape[1]) if saes else 0

    # Raw hidden states at eq_tok position: (num_layers, hidden_dim) float16
    raw_hidden = torch.stack([
        hidden_states[L][0, eq_tok_idx, :].detach().cpu() for L in range(num_layers)
    ]).half()  # (24, 1024)
    raw_hidden_result = None
    raw_hidden_pre_eq_last = None
    if capture_extra_anchors:
        pre_eq_last_idx = int(pre_eq_toks[-1]) if pre_eq_toks else int(eq_tok_idx)
        raw_hidden_result = torch.stack(
            [hidden_states[L][0, result_tok_idx, :].detach().cpu() for L in range(num_layers)]
        ).half()
        raw_hidden_pre_eq_last = torch.stack(
            [hidden_states[L][0, pre_eq_last_idx, :].detach().cpu() for L in range(num_layers)]
        ).half()

    # SAE features at eq_tok position: (num_layers, latent_dim) float16
    sae_features_list = []
    for L in range(num_layers):
        h_raw = hidden_states[L][0, eq_tok_idx, :].to(device).float()
        h_norm = normalize(h_raw, norm_stats.get(L))
        sae = saes[L]
        sae_dtype = next(sae.parameters()).dtype
        h_feat = sae.encode(h_norm.to(dtype=sae_dtype).unsqueeze(0)).squeeze(0)
        sae_features_list.append(h_feat.half().cpu())
    sae_features = torch.stack(sae_features_list)  # (num_layers, latent_dim)

    # Baseline log-prob: model's own prediction of the result token
    # at the position just before result token.
    pred_pos = result_tok_idx - 1
    lp_vec = log_probs_by_pred_pos.get(int(pred_pos))
    if pred_pos < 0 or lp_vec is None:
        return None, "token_alignment"
    baseline_logprob = lp_vec[result_token_id].item()

    # Top-5 predictions at this position
    top5_vals, top5_ids = lp_vec.topk(5)

    C = ann["C"]
    out = {
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
        "model_key": model_key,
        "model_family": model_family,
        "num_layers": int(num_layers),
        "hidden_dim": int(hidden_dim),
        "sae_dim": int(model_sae_dim),
        "tokenizer_id": tokenizer_id,
    }
    if capture_extra_anchors:
        out["raw_hidden_result"] = raw_hidden_result
        out["raw_hidden_pre_eq_last"] = raw_hidden_pre_eq_last
    return out, None


# ── Main collection ──────────────────────────────────────────────────────────

def collect(args):
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_model_spec(args.model_key, args.adapter_config)

    # Load model + SAEs
    print(f"Loading model adapter for {spec.model_key}...")
    adapter = create_adapter(model_key=spec.model_key, device=device, adapter_config=args.adapter_config).load(device=device)
    if adapter.tokenizer is None:
        raise RuntimeError("Adapter tokenizer is not loaded")
    tokenizer = adapter.tokenizer

    resolved_saes_dir = args.saes_dir if args.saes_dir is not None else spec.sae_dir
    if not resolved_saes_dir:
        raise ValueError(
            f"No SAE directory configured for model_key={spec.model_key!r}; "
            "pass --saes-dir or set model_registry.sae_dir."
        )
    resolved_activations_dir = args.activations_dir if args.activations_dir is not None else "phase2_results/activations"
    print("Loading SAEs...")
    saes = load_saes(
        saes_dir=Path(resolved_saes_dir),
        model_key=spec.model_key,
        num_layers=int(spec.num_layers),
        device=device,
    )
    print("Loading norm stats...")
    norm_stats = load_norm_stats(
        activations_dir=Path(resolved_activations_dir),
        model_key=spec.model_key,
        num_layers=int(spec.num_layers),
        device=device,
    )

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
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=adapter._tokenize_add_special_tokens(),
        )
        input_ids = enc["input_ids"].to(device)
        offset_mapping = enc["offset_mapping"][0].tolist() if "offset_mapping" in enc else [(0, 0)] * int(input_ids.shape[1])
        tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        hidden_states, last_hidden = forward_hidden_states_only(adapter, input_ids)
        if len(hidden_states) != int(spec.num_layers):
            raise RuntimeError(
                f"Hidden-state depth mismatch for model_key={spec.model_key!r}: "
                f"adapter returned {len(hidden_states)} layers, expected {int(spec.num_layers)}"
            )
        pred_positions = set()
        for ann in annotations:
            result_toks = find_tokens_for_span(offset_mapping, ann["result_chars"][0], ann["result_chars"][1])
            if len(result_toks) == 1 and int(result_toks[0]) > 0:
                pred_positions.add(int(result_toks[0]) - 1)
        log_probs_by_pred_pos: Dict[int, torch.Tensor] = {}
        if pred_positions:
            pred_pos_sorted = sorted(pred_positions)
            idx = torch.tensor(pred_pos_sorted, dtype=torch.long, device=last_hidden.device)
            h_sel = last_hidden[0].index_select(0, idx)
            logits_sel = adapter.model.lm_head(h_sel)
            lp_sel = F.log_softmax(logits_sel.float(), dim=-1).detach().cpu()
            for i, pos in enumerate(pred_pos_sorted):
                log_probs_by_pred_pos[int(pos)] = lp_sel[i]

        for ann_idx, ann in enumerate(annotations):
            rec, skip_reason = extract_record(
                ann, input_ids, offset_mapping, tokens_str,
                hidden_states, saes, norm_stats, log_probs_by_pred_pos, device,
                ex_idx, ann_idx, gsm8k_split,
                model_key=spec.model_key,
                model_family=spec.model_family,
                tokenizer_id=spec.tokenizer_id,
                capture_extra_anchors=bool(args.capture_extra_anchors),
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

    # Save
    prefix = str(args.output_prefix).strip() if args.output_prefix else (
        "gsm8k_expanded" if str(spec.model_key) == "gpt2-medium" else f"{model_key_safe(spec.model_key)}_gsm8k_expanded"
    )
    shard_suffix = f"_shard{args.shard[0]}" if args.shard else ""
    out_path = output_dir / f"{prefix}_{gsm8k_split}{shard_suffix}.pt"
    torch.save(records, out_path)

    print(f"\n=== Collection Summary ({spec.model_key} {gsm8k_split}{shard_suffix}) ===")
    print(f"  Records:          {len(records)}")
    print(f"  Skipped (no ann): {n_skipped_no_annotations}")
    print(f"  Skipped (token alignment): {n_skipped_alignment}")
    print(f"  Skipped (multi-token result): {n_multitoken}")
    print(f"  Model:            {spec.model_key} ({spec.model_family})")
    print(f"  SAE dir:          {resolved_saes_dir}")
    print(f"  Activations dir:  {resolved_activations_dir}")
    if records:
        C_vals = [r["C"] for r in records]
        print(f"  C range: [{min(C_vals):.1f}, {max(C_vals):.1f}]")
        bl = [r["baseline_logprob"] for r in records]
        print(f"  Baseline logprob: mean={np.mean(bl):.3f}, median={np.median(bl):.3f}")
    print(f"  Saved: {out_path}")


def merge(args):
    """Merge shards for a given split."""
    output_dir = Path(args.output_dir)
    spec = resolve_model_spec(args.model_key, args.adapter_config)
    prefix = str(args.output_prefix).strip() if args.output_prefix else (
        "gsm8k_expanded" if str(spec.model_key) == "gpt2-medium" else f"{model_key_safe(spec.model_key)}_gsm8k_expanded"
    )
    for split in ["train", "test"]:
        shards = sorted(output_dir.glob(f"{prefix}_{split}_shard*.pt"))
        if not shards:
            continue
        print(f"Merging {len(shards)} shards for {split}...")
        all_records = []
        for s in shards:
            records = torch.load(s, weights_only=False)
            all_records.extend(records)
            print(f"  {s.name}: {len(records)} records")
        out_path = output_dir / f"{prefix}_{split}.pt"
        torch.save(all_records, out_path)
        print(f"  Merged: {len(all_records)} records → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--gsm8k-path", default=None,
                   help="Override GSM8K path (default: auto from split)")
    p.add_argument("--model-key", default="gpt2-medium")
    p.add_argument("--adapter-config", default=None, help="Optional JSON overrides for model registry entry")
    p.add_argument("--saes-dir", default=None)
    p.add_argument("--activations-dir", default=None)
    p.add_argument("--output-prefix", default=None, help="Optional output filename prefix override.")
    p.add_argument("--output-dir", default="phase6_results/dataset")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Process only the first N examples (after sharding), for smoke tests")
    p.add_argument("--shard", type=int, nargs=2, metavar=("IDX", "N"),
                   help="Shard index and total shards (e.g., --shard 0 3)")
    p.add_argument("--merge", action="store_true", help="Merge shards and exit")
    p.add_argument(
        "--capture-extra-anchors",
        action="store_true",
        help="Store additional raw hidden anchors (result token and pre-eq tail token).",
    )
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

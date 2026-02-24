#!/usr/bin/env python3
"""
Phase 5 — Feature Interpreter

For each of the top-50 arithmetic probe features at layers 14–23:
  1. Rank all 674 records by activation value at the result token
  2. Extract ±6-token context window for the top-20 activating examples
  3. Annotate with expression, result value, and coactivation role
  4. Output per-feature JSON cards + a layer-22 summary table

Usage:
    python3 phase5/feature_interpreter.py \
        --dataset  phase4_results/topk/collection/gsm8k_arithmetic_dataset.pt \
        --probe    phase4_results/topk/probe/top_features_per_layer.json \
        --rates    phase4_results/topk/coactivation/activation_rates.npz \
        --output   phase5_results/feature_interpretations
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",  default="phase4_results/topk/collection/gsm8k_arithmetic_dataset.pt")
    p.add_argument("--probe",    default="phase4_results/topk/probe/top_features_per_layer.json")
    p.add_argument("--rates",    default="phase4_results/topk/coactivation/activation_rates.npz")
    p.add_argument("--output",   default="phase5_results/feature_interpretations")
    p.add_argument("--layers",   type=int, nargs="+", default=list(range(14, 24)),
                   help="Layers to interpret (default: 14-23)")
    p.add_argument("--top-n",    type=int, default=50,  help="Top-N features per layer")
    p.add_argument("--top-k",    type=int, default=20,  help="Top-K records per feature")
    p.add_argument("--context",  type=int, default=6,   help="Context window (tokens each side)")
    return p.parse_args()

# ── Role assignment ───────────────────────────────────────────────────────────

ROLE_NAMES = ["operand", "result", "eq_sign", "background"]

def assign_role(rates_layer: np.ndarray, feature_idx: int, threshold: float = 0.1) -> str:
    """Assign primary role based on per-role activation rate at a given layer."""
    feature_rates = rates_layer[:, feature_idx]  # (4,) for [operand, result, eq_sign, background]
    above = {ROLE_NAMES[i]: feature_rates[i] for i in range(4) if feature_rates[i] >= threshold}
    if not above:
        return "background"
    # Prefer non-background roles
    non_bg = {k: v for k, v in above.items() if k != "background"}
    if len(non_bg) > 1:
        return "computation_bridge"  # active at multiple arithmetic positions
    if non_bg:
        return max(non_bg, key=non_bg.get)
    return "background"


# ── Context extraction ────────────────────────────────────────────────────────

def extract_context(record: dict, center_tok_idx: int, window: int) -> str:
    """Return a string showing tokens around center_tok_idx with the center marked."""
    tokens = record["tokens"]
    lo = max(0, center_tok_idx - window)
    hi = min(len(tokens), center_tok_idx + window + 1)
    parts = []
    for i in range(lo, hi):
        tok = tokens[i]
        if i == center_tok_idx:
            parts.append(f"[{tok}]")  # mark result token
        else:
            parts.append(tok)
    return "".join(parts)


# ── Per-feature card ──────────────────────────────────────────────────────────

def build_feature_card(
    feature_idx: int,
    layer: int,
    records: list,
    role: str,
    rates_at_layer: np.ndarray,
    top_k: int,
    context_window: int,
) -> dict:
    """Build a JSON card for a single feature at a single layer."""
    # Collect (activation_value, record_index) pairs at the result token
    activations = []
    for rec_idx, rec in enumerate(records):
        # result_features: (24, 12288); pick layer L and feature F
        val = rec["result_features"][layer, feature_idx].item()
        activations.append((val, rec_idx))

    activations.sort(reverse=True)
    top_activations = activations[:top_k]

    # Build context windows for top examples
    examples = []
    for act_val, rec_idx in top_activations:
        if act_val <= 0:
            break  # No more active examples
        rec = records[rec_idx]
        result_tok_idx = rec["result_tok_idxs"][0]  # primary result token
        ctx = extract_context(rec, result_tok_idx, context_window)
        examples.append({
            "activation": round(float(act_val), 4),
            "expression": rec["expr_str"],
            "result_value": float(rec["C"]),
            "context": ctx,
        })

    # Rate statistics for this feature
    feature_rates = {ROLE_NAMES[i]: float(rates_at_layer[i, feature_idx]) for i in range(4)}

    return {
        "feature_idx": feature_idx,
        "layer": layer,
        "role": role,
        "activation_rates": feature_rates,
        "mean_activation": float(np.mean([a for a, _ in activations if a > 0]) if any(a > 0 for a, _ in activations) else 0.0),
        "active_count": int(sum(1 for a, _ in activations if a > 0)),
        "top_examples": examples,
    }


# ── Layer-22 summary table ────────────────────────────────────────────────────

def build_layer_summary(cards: list) -> list:
    """Build a summary table of all features at a given layer."""
    rows = []
    for card in cards:
        top_ex = card["top_examples"]
        # Dominant expressions: those that appear most in top activating examples
        exprs = [e["expression"] for e in top_ex[:10]]
        rows.append({
            "feature_idx": card["feature_idx"],
            "role": card["role"],
            "active_count": card["active_count"],
            "mean_activation": round(card["mean_activation"], 4),
            "result_rate": round(card["activation_rates"]["result"], 3),
            "operand_rate": round(card["activation_rates"]["operand"], 3),
            "top_expressions": exprs[:5],
            "sample_context": top_ex[0]["context"] if top_ex else "",
        })
    # Sort by active_count desc
    rows.sort(key=lambda r: r["active_count"], reverse=True)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    records = torch.load(args.dataset, weights_only=False)
    print(f"  {len(records)} records")

    print(f"Loading probe features: {args.probe}")
    with open(args.probe) as f:
        probe_data = json.load(f)
    # probe_data["result"] is a list of 24 lists, each with top-50 feature indices
    result_top_features = probe_data["result"]  # List[List[int]]

    print(f"Loading activation rates: {args.rates}")
    rates_npz = np.load(args.rates)
    rates = rates_npz["rates"]  # (24, 4, 12288)

    print(f"\nInterpreting features at layers: {args.layers}")

    all_cards_by_layer = {}

    for layer in args.layers:
        print(f"\n  Layer {layer}:")
        top_feats = result_top_features[layer][:args.top_n]
        rates_at_layer = rates[layer]  # (4, 12288)

        cards = []
        for feat_idx in top_feats:
            role = assign_role(rates_at_layer, feat_idx)
            card = build_feature_card(
                feature_idx=feat_idx,
                layer=layer,
                records=records,
                role=role,
                rates_at_layer=rates_at_layer,
                top_k=args.top_k,
                context_window=args.context,
            )
            cards.append(card)

        # Count roles
        role_counts = {}
        for c in cards:
            role_counts[c["role"]] = role_counts.get(c["role"], 0) + 1
        print(f"    {len(cards)} features | roles: {role_counts}")

        all_cards_by_layer[layer] = cards

        # Save per-layer JSON
        layer_path = out_dir / f"layer_{layer:02d}_features.json"
        with open(layer_path, "w") as f:
            json.dump(cards, f, indent=2)
        print(f"    Saved: {layer_path}")

    # Layer-22 summary table
    if 22 in all_cards_by_layer:
        summary = build_layer_summary(all_cards_by_layer[22])
        summary_path = out_dir / "layer_22_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nLayer-22 summary: {summary_path}")
        # Print top-10 rows to stdout
        print(f"\n{'Feat':>6}  {'Role':20s}  {'Active':>6}  {'MeanAct':>8}  {'ResRate':>7}  Sample context")
        print("-" * 90)
        for row in summary[:10]:
            print(
                f"{row['feature_idx']:6d}  {row['role']:20s}  {row['active_count']:6d}  "
                f"{row['mean_activation']:8.4f}  {row['result_rate']:7.3f}  {row['sample_context'][:40]}"
            )

    # Cross-layer summary
    cross_summary = {}
    for layer, cards in all_cards_by_layer.items():
        cross_summary[str(layer)] = {
            "total_features": len(cards),
            "role_counts": {},
            "mean_active_count": float(np.mean([c["active_count"] for c in cards])),
        }
        for c in cards:
            r = c["role"]
            cross_summary[str(layer)]["role_counts"][r] = cross_summary[str(layer)]["role_counts"].get(r, 0) + 1
    cross_path = out_dir / "cross_layer_summary.json"
    with open(cross_path, "w") as f:
        json.dump(cross_summary, f, indent=2)
    print(f"\nCross-layer summary: {cross_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

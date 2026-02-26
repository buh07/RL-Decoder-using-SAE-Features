#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from common import CAUSAL_PATCH_SPEC_SCHEMA, load_json, save_json

DEFAULT_VARIABLES = [
    "subresult_value",
    "result_token_id",
    "operator",
    "magnitude_bucket",
    "sign",
]


def _probe_layer_map(probe_json: dict, position: str) -> Dict[int, List[int]]:
    if position not in probe_json or not isinstance(probe_json[position], list):
        raise ValueError(f"Probe JSON missing position '{position}'")
    out = {}
    for layer, items in enumerate(probe_json[position]):
        feats = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, int):
                    feats.append(int(it))
                elif isinstance(it, dict) and "feature_idx" in it:
                    feats.append(int(it["feature_idx"]))
        out[layer] = feats
    return out


def _saliency_layer_map(sal_json: dict) -> Dict[int, List[int]]:
    per = sal_json.get("top_features_per_selected_layer", {})
    out: Dict[int, List[int]] = {}
    for k, rows in per.items():
        feats = []
        for row in rows:
            if isinstance(row, dict) and "feature_index" in row:
                feats.append(int(row["feature_index"]))
            elif isinstance(row, int):
                feats.append(int(row))
        out[int(k)] = feats
    return out


def _combine(a: Sequence[int], b: Sequence[int], policy: str, topk: int) -> List[int]:
    if policy == "probe_only" or not b:
        return list(dict.fromkeys(int(x) for x in a))[:topk]
    if policy == "saliency_only":
        return list(dict.fromkeys(int(x) for x in b))[:topk]
    if policy == "intersection":
        aset = set(int(x) for x in a)
        return [int(x) for x in b if int(x) in aset][:topk]
    # union (default): saliency-ranked first, then probe priors
    seen = set()
    out = []
    for x in list(b) + list(a):
        x = int(x)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= topk:
            break
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-top-features", default="phase4_results/topk/probe/top_features_per_layer.json")
    p.add_argument("--probe-position", default="result", choices=["eq", "pre_eq", "result"])
    p.add_argument("--decoder-saliency", default=None, help="Optional phase7 grad saliency JSON (SAE-only decoder recommended)")
    p.add_argument("--layers", type=int, nargs="*", default=[7, 12, 17, 22])
    p.add_argument("--variables", nargs="*", default=DEFAULT_VARIABLES)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--combine-policy", choices=["union", "intersection", "probe_only", "saliency_only"], default="union")
    p.add_argument("--output", default="phase7_results/interventions/variable_subspaces.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    probe = load_json(args.probe_top_features)
    probe_map = _probe_layer_map(probe, args.probe_position)
    sal_map: Dict[int, List[int]] = {}
    if args.decoder_saliency:
        sal = load_json(args.decoder_saliency)
        sal_map = _saliency_layer_map(sal)

    specs = []
    for var in args.variables:
        for layer in args.layers:
            feats = _combine(probe_map.get(layer, []), sal_map.get(layer, []), args.combine_policy, args.top_k)
            specs.append(
                {
                    "schema_version": CAUSAL_PATCH_SPEC_SCHEMA,
                    "variable": var,
                    "step_idx": None,
                    "layer": int(layer),
                    "token_position_role": "eq_or_result",
                    "subspace_source": args.combine_policy if args.decoder_saliency else "probe_only",
                    "feature_indices": feats,
                    "method": "subspace_patch",
                    "donor_match_rules": ["same_operator", "same_step_type", "magnitude_bucket_close"],
                    "probe_position": args.probe_position,
                    "top_k": int(args.top_k),
                }
            )

    out = {
        "schema_version": CAUSAL_PATCH_SPEC_SCHEMA,
        "probe_top_features": args.probe_top_features,
        "decoder_saliency": args.decoder_saliency,
        "combine_policy": args.combine_policy,
        "layers": list(args.layers),
        "variables": list(args.variables),
        "specs": specs,
    }
    save_json(args.output, out)
    print(f"Saved {len(specs)} variable subspace specs -> {args.output}")


if __name__ == "__main__":
    main()

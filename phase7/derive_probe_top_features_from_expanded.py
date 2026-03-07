#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List

import torch

try:  # pragma: no cover
    from .common import save_json
except ImportError:  # pragma: no cover
    from common import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expanded-dataset", required=True)
    p.add_argument("--model-key", default="qwen2.5-7b")
    p.add_argument("--split", choices=["train", "test", "all"], default="train")
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--max-records", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260307)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _is_finite_number(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _prepare_records(args: argparse.Namespace) -> List[dict]:
    records = list(torch.load(args.expanded_dataset, map_location="cpu", weights_only=False))
    if args.split != "all":
        records = [r for r in records if str(r.get("gsm8k_split")) == str(args.split)]
    has_model_key = any("model_key" in r for r in records)
    if has_model_key:
        records = [r for r in records if str(r.get("model_key")) == str(args.model_key)]
    out: List[dict] = []
    for r in records:
        y = r.get("log_abs_C")
        feats = r.get("sae_features")
        if not _is_finite_number(y):
            continue
        if not isinstance(feats, torch.Tensor) or feats.ndim != 2:
            continue
        out.append(r)
    if int(args.max_records) > 0 and len(out) > int(args.max_records):
        rng = random.Random(int(args.seed))
        idxs = list(range(len(out)))
        rng.shuffle(idxs)
        out = [out[i] for i in idxs[: int(args.max_records)]]
    return out


def _top_features_for_layer(
    rows: List[dict],
    *,
    layer: int,
    y: torch.Tensor,
    top_k: int,
) -> Dict[str, object]:
    x = torch.stack([r["sae_features"][int(layer)].float().cpu() for r in rows], dim=0)
    yv = y.float().cpu()
    if x.shape[0] != yv.shape[0]:
        raise RuntimeError("feature/target row count mismatch")
    if x.shape[0] < 4:
        raise RuntimeError("insufficient rows for correlation")
    xm = x.mean(dim=0)
    xv = x - xm
    ym = yv.mean()
    yvc = yv - ym
    x_std = xv.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
    y_std = yvc.pow(2).mean().sqrt().clamp_min(1e-6)
    corr = (xv * yvc.unsqueeze(1)).mean(dim=0) / (x_std * y_std)
    abs_corr = corr.abs()
    k = int(min(max(1, top_k), abs_corr.numel()))
    vals, idxs = torch.topk(abs_corr, k=k, largest=True, sorted=True)
    return {
        "top_indices": [int(i) for i in idxs.tolist()],
        "top_abs_corr": [float(v) for v in vals.tolist()],
        "mean_abs_corr": float(abs_corr.mean().item()),
        "max_abs_corr": float(abs_corr.max().item()),
    }


def main() -> None:
    args = parse_args()
    rows = _prepare_records(args)
    if not rows:
        raise RuntimeError("No valid rows after filtering")
    first_feats = rows[0]["sae_features"]
    num_layers = int(first_feats.shape[0])
    sae_dim = int(first_feats.shape[1])
    ys = torch.tensor([float(r["log_abs_C"]) for r in rows], dtype=torch.float32)

    eq: List[List[int]] = []
    pre_eq: List[List[int]] = []
    result: List[List[int]] = []
    layer_stats: Dict[str, object] = {}
    for layer in range(num_layers):
        row = _top_features_for_layer(rows, layer=layer, y=ys, top_k=int(args.top_k))
        feats = list(row["top_indices"])
        eq.append(feats)
        pre_eq.append(feats)
        result.append(feats)
        layer_stats[str(layer)] = {
            "feature_count": int(len(feats)),
            "mean_abs_corr": float(row["mean_abs_corr"]),
            "max_abs_corr": float(row["max_abs_corr"]),
            "top_abs_corr_head": [float(x) for x in row["top_abs_corr"][:10]],
        }
        print(
            f"[derive_probe_top_features] layer={layer} feature_count={len(feats)} "
            f"max_abs_corr={float(row['max_abs_corr']):.4f}"
        )

    payload = {
        "schema_version": "phase7_probe_top_features_from_expanded_v1",
        "model_key": str(args.model_key),
        "expanded_dataset": str(args.expanded_dataset),
        "split": str(args.split),
        "top_k": int(args.top_k),
        "max_records": int(args.max_records),
        "seed": int(args.seed),
        "num_rows_used": int(len(rows)),
        "num_layers": int(num_layers),
        "sae_dim": int(sae_dim),
        "eq": eq,
        "pre_eq": pre_eq,
        "result": result,
        "layer_stats": layer_stats,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

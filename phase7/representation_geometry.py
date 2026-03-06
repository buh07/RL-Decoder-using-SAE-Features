#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

try:  # pragma: no cover
    from .causal_intervention_engine import _load_control_records_artifact
    from .common import save_json
except ImportError:  # pragma: no cover
    from causal_intervention_engine import _load_control_records_artifact
    from common import save_json


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na = torch.norm(a).item()
    nb = torch.norm(b).item()
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb))


def _mean(xs: List[float]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _cohen_d(a: List[float], b: List[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = (((len(a) - 1) * va) + ((len(b) - 1) * vb)) / max(1, (len(a) + len(b) - 2))
    if pooled <= 0.0:
        return None
    return float((ma - mb) / math.sqrt(pooled))


def _roc_auc(scores_labels: List[Tuple[float, int]]) -> float | None:
    if not scores_labels:
        return None
    p = sum(int(y) for _, y in scores_labels)
    n = len(scores_labels) - p
    if p == 0 or n == 0:
        return None
    ranked = sorted(scores_labels, key=lambda x: x[0])
    rank_sum = 0.0
    for i, (_, y) in enumerate(ranked, start=1):
        if y == 1:
            rank_sum += i
    auc = (rank_sum - (p * (p + 1) / 2.0)) / (p * n)
    return float(auc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-records", required=True, help="Path to phase7 control-records artifact JSON.")
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows, payload = _load_control_records_artifact(args.control_records)
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    labels: Dict[Tuple[str, str], str] = {}
    for r in rows:
        key = (str(r.get("trace_id", "")), str(r.get("control_variant", "unknown")))
        grouped[key].append(r)
        lbl = str(r.get("gold_label", "unknown"))
        if lbl in {"faithful", "unfaithful"}:
            labels[key] = lbl

    per_trace_variant: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        trace_id, variant = key
        lbl = labels.get(key, "unknown")
        if lbl not in {"faithful", "unfaithful"}:
            continue
        ordered = sorted(items, key=lambda x: int(x.get("step_idx", -1)))
        vecs: List[torch.Tensor] = []
        for r in ordered:
            h = r.get("raw_hidden")
            if not isinstance(h, torch.Tensor) or h.ndim != 2:
                continue
            if int(args.layer) < 0 or int(args.layer) >= int(h.shape[0]):
                continue
            vecs.append(h[int(args.layer)].float().cpu())
        if not vecs:
            continue

        consecutive_cos = []
        for i in range(len(vecs) - 1):
            consecutive_cos.append(_cos(vecs[i], vecs[i + 1]))
        norms = [float(torch.norm(v).item()) for v in vecs]
        centroid = torch.stack(vecs, dim=0).mean(dim=0)
        dist_to_centroid = [float(torch.norm(v - centroid).item()) for v in vecs]
        mat = torch.stack(vecs, dim=0)
        within_var = float(torch.var(mat, dim=0, unbiased=False).mean().item()) if mat.shape[0] > 1 else 0.0
        per_trace_variant.append(
            {
                "trace_id": trace_id,
                "control_variant": variant,
                "gold_label": lbl,
                "mean_consecutive_cosine": _mean(consecutive_cos),
                "norm_mean": _mean(norms),
                "norm_std": float(torch.tensor(norms).std(unbiased=False).item()) if len(norms) > 1 else 0.0,
                "within_trace_variance": within_var,
                "distance_to_centroid_mean": _mean(dist_to_centroid),
                "num_steps": int(len(vecs)),
            }
        )

    features = [
        "mean_consecutive_cosine",
        "norm_mean",
        "norm_std",
        "within_trace_variance",
        "distance_to_centroid_mean",
    ]
    summary: Dict[str, Any] = {}
    for feat in features:
        faithful_vals = [float(r[feat]) for r in per_trace_variant if r["gold_label"] == "faithful" and isinstance(r.get(feat), (int, float))]
        unfaith_vals = [float(r[feat]) for r in per_trace_variant if r["gold_label"] == "unfaithful" and isinstance(r.get(feat), (int, float))]
        scored = [(float(r[feat]), 1 if r["gold_label"] == "unfaithful" else 0) for r in per_trace_variant if isinstance(r.get(feat), (int, float))]
        auc_raw = _roc_auc(scored)
        auc_inverted = (1.0 - auc_raw) if isinstance(auc_raw, (int, float)) else None
        summary[feat] = {
            "faithful_mean": _mean(faithful_vals),
            "unfaithful_mean": _mean(unfaith_vals),
            "cohen_d_faithful_minus_unfaithful": _cohen_d(faithful_vals, unfaith_vals),
            "auroc_raw_unfaithful_positive": auc_raw,
            "auroc_inverted_unfaithful_positive": auc_inverted,
            "auroc_best_orientation": (
                max(float(auc_raw), float(auc_inverted))
                if isinstance(auc_raw, (int, float)) and isinstance(auc_inverted, (int, float))
                else None
            ),
            "n_faithful": int(len(faithful_vals)),
            "n_unfaithful": int(len(unfaith_vals)),
        }

    out = {
        "schema_version": "phase7_representation_geometry_v1",
        "source_control_records": str(args.control_records),
        "source_control_records_stats": payload.get("stats"),
        "layer": int(args.layer),
        "num_trace_variants": int(len(per_trace_variant)),
        "feature_summary": summary,
        "rows": per_trace_variant,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    print(f"Saved geometry diagnostics -> {out_path}")


if __name__ == "__main__":
    main()


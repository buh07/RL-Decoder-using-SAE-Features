#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:  # pragma: no cover
    from .common import load_json, load_pt, save_json, sha256_file
    from .sae_assets import load_norm_stats_for_layer, load_sae_for_layer
except ImportError:  # pragma: no cover
    from common import load_json, load_pt, save_json, sha256_file
    from sae_assets import load_norm_stats_for_layer, load_sae_for_layer


FEATURE_SET_CHOICES = ("eq_top50", "result_top50", "eq_pre_result_150")


def _parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    for tok in str(value or "").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("Expected at least one layer id")
    if len(set(out)) != len(out):
        raise ValueError("Duplicate layer id in --layers")
    return sorted(out)


def _load_rows(payload_path: str | Path) -> Tuple[Dict[str, Any], List[dict]]:
    payload = load_json(payload_path)
    rows_path = payload.get("rows_path")
    if not rows_path:
        raise RuntimeError("paired dataset payload missing rows_path")
    rp = Path(str(rows_path))
    if not rp.is_absolute():
        candidate = (Path(payload_path).parent / rp).resolve()
        rp = candidate if candidate.exists() else rp.resolve()
    rows = list(load_pt(rp))
    return payload, rows


def _feature_indices(top_features_path: str | Path, layer: int, feature_set: str) -> List[int]:
    payload = load_json(top_features_path)
    if str(feature_set) == "eq_top50":
        arr = payload.get("eq")
        if not isinstance(arr, list) or layer >= len(arr):
            raise RuntimeError(f"Missing eq features for layer={layer}")
        return [int(x) for x in list(arr[layer])[:50]]
    if str(feature_set) == "result_top50":
        arr = payload.get("result")
        if not isinstance(arr, list) or layer >= len(arr):
            raise RuntimeError(f"Missing result features for layer={layer}")
        return [int(x) for x in list(arr[layer])[:50]]
    if str(feature_set) == "eq_pre_result_150":
        eq = payload.get("eq")
        pre = payload.get("pre_eq")
        res = payload.get("result")
        if not isinstance(eq, list) or not isinstance(pre, list) or not isinstance(res, list):
            raise RuntimeError("Missing eq/pre_eq/result feature blocks")
        if layer >= len(eq) or layer >= len(pre) or layer >= len(res):
            raise RuntimeError(f"Missing features for layer={layer}")
        out: List[int] = []
        seen = set()
        for x in list(eq[layer])[:50] + list(pre[layer])[:50] + list(res[layer])[:50]:
            i = int(x)
            if i in seen:
                continue
            seen.add(i)
            out.append(i)
        return out
    raise ValueError(f"Unsupported feature_set={feature_set!r}")


def _normalize_hidden(h: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (h - mean) / std


def _encode_rows(
    *,
    rows: Sequence[dict],
    layer: int,
    feature_indices: Sequence[int],
    model_key: str,
    saes_dir: Path,
    activations_dir: Path,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    sae = load_sae_for_layer(saes_dir=saes_dir, model_key=model_key, layer=int(layer), device=device)
    mean, std = load_norm_stats_for_layer(
        activations_dir=activations_dir,
        model_key=model_key,
        layer=int(layer),
        device=device,
    )
    idx = torch.tensor([int(x) for x in feature_indices], dtype=torch.long, device=device)
    sae_dtype = next(sae.parameters()).dtype
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, len(rows), int(batch_size)):
            chunk = rows[s : s + int(batch_size)]
            x = torch.stack([r["raw_hidden"][int(layer)].float() for r in chunk], dim=0).to(device)
            xn = _normalize_hidden(x, mean, std)
            z = sae.encode(xn.to(dtype=sae_dtype))
            zs = z.index_select(1, idx).float().cpu()
            outs.append(zs)
    del sae
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, len(feature_indices)), dtype=torch.float32)


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() <= 0:
        return 0.0
    return float(torch.quantile(x, float(q)).item())


def _member_metrics(feat_seq: torch.Tensor) -> Dict[str, float]:
    if feat_seq.ndim != 2 or feat_seq.shape[0] < 2:
        return {
            "transition_mean_cosine": 0.5,
            "transition_min_cosine": 0.5,
            "transition_mean_delta_l2": 0.0,
            "transition_max_delta_l2": 0.0,
            "transition_p95_delta_l2": 0.0,
        }
    a = feat_seq[:-1]
    b = feat_seq[1:]
    cos = F.cosine_similarity(a, b, dim=1, eps=1e-8)
    d = (b - a).norm(dim=1)
    return {
        "transition_mean_cosine": float(cos.mean().item()),
        "transition_min_cosine": float(cos.min().item()),
        "transition_mean_delta_l2": float(d.mean().item()),
        "transition_max_delta_l2": float(d.max().item()),
        "transition_p95_delta_l2": _safe_quantile(d, 0.95),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-dataset", required=True)
    p.add_argument("--model-key", default="qwen2.5-7b")
    p.add_argument("--layers", required=True, help="CSV list of layers for this shard.")
    p.add_argument("--saes-dir", default="phase2_results/saes_qwen25_7b_12x_topk/saes")
    p.add_argument("--activations-dir", default="phase2_results/activations")
    p.add_argument(
        "--phase4-top-features",
        default="phase7_results/runs/20260308_165109_phase7_qwen_trackc_upgrade/interventions/top_features_per_layer_qwen_phase7_qwen_trackc_upgrade_20260308_165109_phase7_qwen_trackc_upgrade.json",
    )
    p.add_argument("--feature-set", choices=list(FEATURE_SET_CHOICES), default="eq_pre_result_150")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--run-tag", default="")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layers = _parse_int_csv(args.layers)
    payload, rows = _load_rows(args.paired_dataset)
    members_meta = list(payload.get("members", []))
    members_by_id: Dict[str, Dict[str, Any]] = {str(m.get("member_id")): dict(m) for m in members_meta}

    rows_ordered = sorted(
        [r for r in rows if isinstance(r, dict)],
        key=lambda r: (
            str(r.get("member_id", "")),
            int(r.get("step_idx", -1)),
            int(r.get("line_index", -1)),
        ),
    )
    row_member_ids = [str(r.get("member_id", "")) for r in rows_ordered]
    member_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, mid in enumerate(row_member_ids):
        if mid:
            member_to_indices[mid].append(int(i))

    per_member_features: Dict[str, Dict[str, float]] = defaultdict(dict)
    all_feature_names: List[str] = []
    for layer in layers:
        feats = _feature_indices(args.phase4_top_features, int(layer), str(args.feature_set))
        enc = _encode_rows(
            rows=rows_ordered,
            layer=int(layer),
            feature_indices=feats,
            model_key=str(args.model_key),
            saes_dir=Path(args.saes_dir),
            activations_dir=Path(args.activations_dir),
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        for mid, idxs in member_to_indices.items():
            seq = enc[idxs, :]
            m = _member_metrics(seq)
            for k, v in m.items():
                fn = f"layer{int(layer)}:{k}"
                per_member_features[mid][fn] = float(v)
        if not all_feature_names:
            for k in _member_metrics(torch.zeros((1, 2))).keys():
                all_feature_names.append(f"layer{int(layer)}:{k}")
        else:
            for k in _member_metrics(torch.zeros((1, 2))).keys():
                all_feature_names.append(f"layer{int(layer)}:{k}")

    members_out: List[Dict[str, Any]] = []
    for mid in sorted(member_to_indices.keys()):
        mm = members_by_id.get(mid, {})
        members_out.append(
            {
                "member_id": str(mid),
                "pair_id": str(mm.get("pair_id", "")),
                "lexical_control": bool(mm.get("lexical_control", False)),
                "pair_ambiguous": bool(mm.get("pair_ambiguous", False)),
                "gold_label": str(mm.get("gold_label", "unknown")),
                "label_binary": int(mm.get("label_binary", -1)),
                "label_defined": bool(mm.get("label_defined", False)),
                "is_correct": bool(mm.get("is_correct", False)),
                "step_count": int(len(member_to_indices.get(mid, []))),
                "features": dict(per_member_features.get(mid, {})),
            }
        )

    out = {
        "schema_version": "phase7_optionc_internal_consistency_partial_v1",
        "status": "ok",
        "run_tag": str(args.run_tag or ""),
        "paired_dataset": str(args.paired_dataset),
        "paired_dataset_sha256": sha256_file(args.paired_dataset),
        "model_key": str(args.model_key),
        "layers": [int(x) for x in layers],
        "feature_set": str(args.feature_set),
        "feature_count": int(len(all_feature_names)),
        "member_count": int(len(members_out)),
        "members": members_out,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(args.output_json, out)
    print(f"Saved Option C consistency partial -> {args.output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:  # pragma: no cover
    from .common import load_json, load_pt, magnitude_bucket, save_json, sha256_file, sign_label
    from .sae_assets import load_norm_stats_for_layer, load_sae_for_layer
    from .state_decoder_core import decode_latent_pred_states, load_model_from_checkpoint
except ImportError:  # pragma: no cover
    from common import load_json, load_pt, magnitude_bucket, save_json, sha256_file, sign_label
    from sae_assets import load_norm_stats_for_layer, load_sae_for_layer
    from state_decoder_core import decode_latent_pred_states, load_model_from_checkpoint


METRIC_KEYS = ("cosine_smoothness", "feature_variance_coherence", "magnitude_monotonicity_coherence")
ANOMALY_KEYS = (
    "max_delta_l2",
    "top2_mean_delta_l2",
    "argmax_step_idx_norm",
    "p95_delta_l2",
    "max_delta_cosine",
    "p95_delta_cosine",
    "max_delta_l2_norm",
)
HYBRID_KEYS = (
    "hybrid_transition_consistency_mean",
    "hybrid_transition_inconsistency_fraction",
    "hybrid_weakest_link_consistency",
    "hybrid_min_abs_transition_error",
    "hybrid_p95_abs_transition_error",
)
FEATURE_SET_CHOICES = ("eq_top50", "result_top50", "eq_pre_result_150", "divergent_top50")


@dataclass
class PairDescriptor:
    trace_id: str
    unfaithful_variant: str
    common_steps: List[int]
    faithful_row_indices: List[int]
    unfaithful_row_indices: List[int]


def _parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    for tok in str(value or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("Expected at least one integer")
    return out


def _load_control_records_artifact(path: str | Path) -> Tuple[List[dict], Dict[str, Any]]:
    payload = load_json(path)
    if str(payload.get("rows_format", "")).lower() == "pt":
        rows_path = payload.get("rows_path")
        if not rows_path:
            raise RuntimeError(f"control-records artifact missing rows_path: {path}")
        rp = Path(str(rows_path))
        if not rp.is_absolute():
            rel_candidate = (Path(path).parent / rp).resolve()
            if rel_candidate.exists():
                rp = rel_candidate
            else:
                rp = rp.resolve()
        rows = list(load_pt(rp))
        return rows, payload
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("control-records artifact rows are not available as list")
    return list(rows), payload


def _normalize_hidden(h: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    m, s = stats
    return (h - m) / s


def _phase4_top50_eq_for_layer(phase4_top_features_path: str | Path, layer: int) -> List[int]:
    payload = load_json(phase4_top_features_path)
    eq = payload.get("eq") if isinstance(payload, dict) else None
    if not isinstance(eq, list) or layer >= len(eq):
        raise RuntimeError(f"Phase4 top-features payload missing eq layer entry for layer={layer}")
    vals = list(eq[layer])
    return [int(x) for x in vals[:50]]


def _phase4_feature_block(phase4_top_features_path: str | Path, block_key: str, layer: int) -> List[int]:
    payload = load_json(phase4_top_features_path)
    block = payload.get(block_key) if isinstance(payload, dict) else None
    if not isinstance(block, list) or layer >= len(block):
        raise RuntimeError(f"Phase4 top-features payload missing key={block_key!r} layer={layer}")
    vals = list(block[layer])
    return [int(x) for x in vals]


def _divergent_top_features_for_layer(divergent_source_path: str | Path, layer: int, top_k: int = 50) -> List[int]:
    payload = load_json(divergent_source_path)
    by_layer = payload.get("by_layer") if isinstance(payload, dict) else None
    if not isinstance(by_layer, dict):
        raise RuntimeError(f"Divergent source missing by_layer map: {divergent_source_path}")
    row = by_layer.get(str(layer))
    if not isinstance(row, dict):
        raise RuntimeError(f"Divergent source missing layer={layer} entry: {divergent_source_path}")
    div = row.get("feature_divergence")
    if not isinstance(div, dict):
        raise RuntimeError(f"Divergent source layer={layer} missing feature_divergence")
    top = div.get("top_features_abs_d")
    if not isinstance(top, list):
        raise RuntimeError(f"Divergent source layer={layer} missing top_features_abs_d")
    out = []
    for item in top[: int(top_k)]:
        if isinstance(item, dict) and "feature_idx" in item:
            out.append(int(item["feature_idx"]))
        elif isinstance(item, (int, float)):
            out.append(int(item))
    if not out:
        raise RuntimeError(f"Divergent source layer={layer} yielded no feature indices")
    return out


def _select_feature_indices(
    *,
    feature_set: str,
    layer: int,
    phase4_top_features_path: str | Path,
    divergent_source_path: Optional[str | Path],
) -> Tuple[List[int], Dict[str, Any]]:
    mode = str(feature_set)
    if mode not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unsupported feature_set={mode!r}; expected one of {FEATURE_SET_CHOICES}")

    if mode == "eq_top50":
        feats = _phase4_feature_block(phase4_top_features_path, "eq", layer)[:50]
        return feats, {"feature_set": mode, "source": str(phase4_top_features_path), "feature_count": int(len(feats))}
    if mode == "result_top50":
        feats = _phase4_feature_block(phase4_top_features_path, "result", layer)[:50]
        return feats, {"feature_set": mode, "source": str(phase4_top_features_path), "feature_count": int(len(feats))}
    if mode == "eq_pre_result_150":
        eq = _phase4_feature_block(phase4_top_features_path, "eq", layer)[:50]
        pre = _phase4_feature_block(phase4_top_features_path, "pre_eq", layer)[:50]
        res = _phase4_feature_block(phase4_top_features_path, "result", layer)[:50]
        ordered: List[int] = []
        seen = set()
        for f in (eq + pre + res):
            fi = int(f)
            if fi not in seen:
                seen.add(fi)
                ordered.append(fi)
        return ordered, {
            "feature_set": mode,
            "source": str(phase4_top_features_path),
            "feature_count": int(len(ordered)),
            "component_counts": {"eq": 50, "pre_eq": 50, "result": 50},
        }
    if not divergent_source_path:
        raise ValueError("--divergent-source is required when --feature-set divergent_top50")
    feats = _divergent_top_features_for_layer(divergent_source_path, layer, top_k=50)
    return feats, {"feature_set": mode, "source": str(divergent_source_path), "feature_count": int(len(feats))}


def _magnitude_features_for_layer(subspace_specs_path: str | Path, layer: int) -> List[int]:
    payload = load_json(subspace_specs_path)
    specs = payload.get("specs") if isinstance(payload, dict) else None
    if not isinstance(specs, list):
        return []
    for s in specs:
        if str(s.get("variable")) == "magnitude_bucket" and int(s.get("layer", -1)) == int(layer):
            return [int(x) for x in list(s.get("feature_indices", []))]
    return []


def _canonical_row_order(rows: Sequence[dict]) -> List[dict]:
    return sorted(
        [r for r in rows if isinstance(r, dict)],
        key=lambda r: (
            str(r.get("trace_id", "")),
            str(r.get("control_variant", "")),
            int(r.get("step_idx", -1)),
            int(r.get("line_index", -1)),
            int(r.get("example_idx", -1)),
        ),
    )


def _build_pair_descriptors(
    rows: Sequence[dict],
    *,
    min_common_steps: int,
    sample_traces: Optional[int],
    seed: int,
) -> Tuple[List[dict], List[PairDescriptor], Dict[str, Any]]:
    ordered = _canonical_row_order(rows)
    filtered: List[dict] = []
    for r in ordered:
        lbl = str(r.get("gold_label", ""))
        if lbl not in {"faithful", "unfaithful"}:
            continue
        h = r.get("raw_hidden")
        if not isinstance(h, torch.Tensor) or h.ndim != 2:
            continue
        tid = str(r.get("trace_id", ""))
        var = str(r.get("control_variant", ""))
        step = int(r.get("step_idx", -1))
        if not tid or not var or step < 0:
            continue
        filtered.append(r)

    tvs: Dict[Tuple[str, str], Dict[int, int]] = defaultdict(dict)
    for idx, r in enumerate(filtered):
        key = (str(r.get("trace_id")), str(r.get("control_variant")))
        step_idx = int(r.get("step_idx"))
        if step_idx not in tvs[key]:
            tvs[key][step_idx] = idx

    trace_to_variants: Dict[str, List[str]] = defaultdict(list)
    for t, v in tvs:
        trace_to_variants[t].append(v)

    trace_ids_pre = sorted(trace_to_variants.keys())
    pair_desc: List[PairDescriptor] = []
    skipped_common_lt3 = 0
    for trace_id in trace_ids_pre:
        faithful_map = tvs.get((trace_id, "faithful"), {})
        if not faithful_map:
            continue
        for variant in sorted(set(trace_to_variants[trace_id])):
            if variant == "faithful":
                continue
            u_map = tvs.get((trace_id, variant), {})
            if not u_map:
                continue
            common = sorted(set(faithful_map).intersection(set(u_map)))
            if len(common) < int(min_common_steps):
                skipped_common_lt3 += 1
                continue
            pair_desc.append(
                PairDescriptor(
                    trace_id=trace_id,
                    unfaithful_variant=variant,
                    common_steps=common,
                    faithful_row_indices=[int(faithful_map[s]) for s in common],
                    unfaithful_row_indices=[int(u_map[s]) for s in common],
                )
            )

    eligible_trace_ids = sorted(set(p.trace_id for p in pair_desc))
    sampled_trace_ids = eligible_trace_ids
    if sample_traces is not None and int(sample_traces) > 0 and len(eligible_trace_ids) > int(sample_traces):
        rng = random.Random(int(seed))
        sampled_trace_ids = sorted(rng.sample(eligible_trace_ids, int(sample_traces)))
    sampled_set = set(sampled_trace_ids)
    pair_desc = [p for p in pair_desc if p.trace_id in sampled_set]

    used_indices = sorted(
        set(i for p in pair_desc for i in (p.faithful_row_indices + p.unfaithful_row_indices))
    )
    old_to_new = {old: new for new, old in enumerate(used_indices)}
    rows_used = [filtered[i] for i in used_indices]
    mapped_pair_desc: List[PairDescriptor] = []
    for p in pair_desc:
        mapped_pair_desc.append(
            PairDescriptor(
                trace_id=p.trace_id,
                unfaithful_variant=p.unfaithful_variant,
                common_steps=list(p.common_steps),
                faithful_row_indices=[int(old_to_new[i]) for i in p.faithful_row_indices],
                unfaithful_row_indices=[int(old_to_new[i]) for i in p.unfaithful_row_indices],
            )
        )

    diag = {
        "rows_total_input": int(len(rows)),
        "rows_total_valid": int(len(filtered)),
        "trace_count_pre_filter": int(len(trace_ids_pre)),
        "trace_count_eligible_common_ge_min": int(len(eligible_trace_ids)),
        "trace_count_sampled": int(len(sampled_trace_ids)),
        "trace_ids_sampled": sampled_trace_ids,
        "variant_pairs_total_common_ge_min": int(len(pair_desc)),
        "skipped_variant_pairs_common_lt_min": int(skipped_common_lt3),
        "rows_used_for_encoding": int(len(rows_used)),
    }
    return rows_used, mapped_pair_desc, diag


def _encode_rows_layer_feature_subset(
    rows: Sequence[dict],
    *,
    model_key: str,
    layer: int,
    feature_indices: Sequence[int],
    saes_dir: Path,
    activations_dir: Path,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    sae = load_sae_for_layer(
        saes_dir=Path(saes_dir),
        model_key=str(model_key),
        layer=int(layer),
        device=device,
    )
    mean, std = load_norm_stats_for_layer(
        activations_dir=Path(activations_dir),
        model_key=str(model_key),
        layer=int(layer),
        device=device,
    )
    sae_dtype = next(sae.parameters()).dtype
    idx = torch.tensor([int(x) for x in feature_indices], dtype=torch.long, device=device)
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start : start + int(batch_size)]
            x = torch.stack([r["raw_hidden"][int(layer)].float() for r in chunk], dim=0).to(device)
            x_norm = _normalize_hidden(x, (mean, std))
            z = sae.encode(x_norm.to(dtype=sae_dtype))
            z_sub = z.index_select(dim=1, index=idx).detach().float().cpu()
            outs.append(z_sub)
    del sae
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, len(feature_indices)), dtype=torch.float32)


def _cosine_smoothness(traj: torch.Tensor) -> Optional[float]:
    if traj.ndim != 2 or traj.shape[0] < 2:
        return None
    sims = F.cosine_similarity(traj[:-1], traj[1:], dim=1, eps=1e-8)
    return float(sims.mean().item())


def _feature_variance_coherence(traj: torch.Tensor) -> Optional[float]:
    if traj.ndim != 2 or traj.shape[0] < 2:
        return None
    var_mean = float(traj.var(dim=0, unbiased=False).mean().item())
    return float(1.0 / (1.0 + var_mean))


def _magnitude_monotonicity_coherence(traj: torch.Tensor, mag_cols: Sequence[int]) -> Optional[float]:
    if traj.ndim != 2 or traj.shape[0] < 3:
        return None
    cols = [int(c) for c in mag_cols if 0 <= int(c) < int(traj.shape[1])]
    if not cols:
        cols = list(range(int(traj.shape[1])))
    if not cols:
        return None
    t = torch.arange(int(traj.shape[0]), dtype=torch.float32)
    t = (t - t.mean()) / t.std().clamp_min(1e-8)
    vals: List[float] = []
    for c in cols:
        x = traj[:, int(c)].float()
        xstd = x.std(unbiased=False)
        if float(xstd.item()) < 1e-8:
            vals.append(0.0)
            continue
        xn = (x - x.mean()) / xstd.clamp_min(1e-8)
        corr = float((xn * t).mean().item())
        vals.append(abs(corr))
    return float(sum(vals) / max(1, len(vals)))


def _roc_auc(scores_labels: Sequence[Tuple[float, int]]) -> Optional[float]:
    if not scores_labels:
        return None
    pos = sum(int(y) for _, y in scores_labels)
    neg = len(scores_labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranked = sorted(scores_labels, key=lambda x: x[0])
    rank_sum = 0.0
    for i, (_, y) in enumerate(ranked, start=1):
        if int(y) == 1:
            rank_sum += i
    return float((rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg))


def _bootstrap_auc_ci(
    scores_labels: Sequence[Tuple[float, int]],
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    auc = _roc_auc(scores_labels)
    out = {
        "auroc_unfaithful_positive": auc,
        "auroc_ci95_lower": None,
        "auroc_ci95_upper": None,
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_effective_n": 0,
        "defined": bool(auc is not None),
    }
    if auc is None:
        return out
    rng = random.Random(int(seed))
    n = len(scores_labels)
    boots: List[float] = []
    for _ in range(int(max(1, n_bootstrap))):
        sample = [scores_labels[rng.randrange(n)] for __ in range(n)]
        bauc = _roc_auc(sample)
        if bauc is not None:
            boots.append(float(bauc))
    if boots:
        boots_sorted = sorted(boots)
        lo_idx = int(max(0, round(0.025 * (len(boots_sorted) - 1))))
        hi_idx = int(min(len(boots_sorted) - 1, round(0.975 * (len(boots_sorted) - 1))))
        out["auroc_ci95_lower"] = float(boots_sorted[lo_idx])
        out["auroc_ci95_upper"] = float(boots_sorted[hi_idx])
        out["bootstrap_effective_n"] = int(len(boots_sorted))
    return out


def _trajectory_samples(
    feat_rows: torch.Tensor,
    pairs: Sequence[PairDescriptor],
    *,
    mag_cols: Sequence[int],
    decoder_preds_by_row: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> List[dict]:
    samples: List[dict] = []
    for p in pairs:
        ftraj = feat_rows[p.faithful_row_indices, :].float()
        utraj = feat_rows[p.unfaithful_row_indices, :].float()
        anomaly = _step_anomaly_block(ftraj, utraj)
        for lbl, traj, row_indices in (
            ("faithful", ftraj, p.faithful_row_indices),
            ("unfaithful", utraj, p.unfaithful_row_indices),
        ):
            metric_row: Dict[str, Optional[float]] = {
                "cosine_smoothness": _cosine_smoothness(traj),
                "feature_variance_coherence": _feature_variance_coherence(traj),
                "magnitude_monotonicity_coherence": _magnitude_monotonicity_coherence(traj, mag_cols),
            }
            if lbl == "unfaithful":
                metric_row.update({k: float(v) for k, v in anomaly["aggregate"].items()})
            else:
                metric_row.update({k: 0.0 for k in ANOMALY_KEYS})
            if decoder_preds_by_row is not None:
                pred_seq = [decoder_preds_by_row[int(i)] for i in row_indices]
                metric_row.update(_hybrid_consistency_block(pred_seq))
            sample = {
                "trace_id": p.trace_id,
                "variant": p.unfaithful_variant,
                "label": lbl,
                "step_count": int(traj.shape[0]),
                "metrics": metric_row,
            }
            if lbl == "unfaithful":
                sample["step_anomaly_block"] = {
                    "delta_l2_step": [float(x) for x in anomaly["delta_l2_step"]],
                    "delta_cosine_step": [float(x) for x in anomaly["delta_cosine_step"]],
                    "delta_l2_norm_step": [float(x) for x in anomaly["delta_l2_norm_step"]],
                }
            samples.append(sample)
    return samples


def _step_anomaly_block(ftraj: torch.Tensor, utraj: torch.Tensor) -> Dict[str, Any]:
    if ftraj.ndim != 2 or utraj.ndim != 2 or ftraj.shape != utraj.shape or ftraj.shape[0] == 0:
        return {
            "delta_l2_step": [],
            "delta_cosine_step": [],
            "delta_l2_norm_step": [],
            "aggregate": {k: 0.0 for k in ANOMALY_KEYS},
        }
    delta = utraj - ftraj
    d_l2 = delta.norm(dim=1)
    denom = ftraj.norm(dim=1).clamp_min(1e-8)
    d_l2_norm = d_l2 / denom
    d_cos = 1.0 - F.cosine_similarity(utraj, ftraj, dim=1, eps=1e-8)

    top2 = torch.topk(d_l2, k=min(2, int(d_l2.numel())), largest=True).values
    argmax = int(torch.argmax(d_l2).item()) if int(d_l2.numel()) > 0 else 0
    denom_steps = max(1, int(d_l2.numel()) - 1)

    aggregate = {
        "max_delta_l2": float(torch.max(d_l2).item()) if int(d_l2.numel()) > 0 else 0.0,
        "top2_mean_delta_l2": float(top2.mean().item()) if int(top2.numel()) > 0 else 0.0,
        "argmax_step_idx_norm": float(argmax / denom_steps),
        "p95_delta_l2": float(torch.quantile(d_l2, 0.95).item()) if int(d_l2.numel()) > 0 else 0.0,
        "max_delta_cosine": float(torch.max(d_cos).item()) if int(d_cos.numel()) > 0 else 0.0,
        "p95_delta_cosine": float(torch.quantile(d_cos, 0.95).item()) if int(d_cos.numel()) > 0 else 0.0,
        "max_delta_l2_norm": float(torch.max(d_l2_norm).item()) if int(d_l2_norm.numel()) > 0 else 0.0,
    }
    return {
        "delta_l2_step": d_l2.detach().cpu().tolist(),
        "delta_cosine_step": d_cos.detach().cpu().tolist(),
        "delta_l2_norm_step": d_l2_norm.detach().cpu().tolist(),
        "aggregate": aggregate,
    }


def _safe_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        xf = float(x)
        if xf == xf and xf not in (float("inf"), float("-inf")):
            return xf
    return None


def _pred_to_mag_sign(v: Optional[float]) -> Tuple[str, str]:
    if v is None:
        return "unknown", "unknown"
    try:
        return str(magnitude_bucket(float(v))), str(sign_label(float(v)))
    except Exception:
        return "unknown", "unknown"


def _hybrid_consistency_block(pred_seq: Sequence[Optional[Dict[str, Any]]]) -> Dict[str, Optional[float]]:
    if len(pred_seq) < 2:
        return {k: 0.5 for k in HYBRID_KEYS}
    consistency: List[float] = []
    abs_errs: List[float] = []
    for i in range(len(pred_seq) - 1):
        cur = pred_seq[i] or {}
        nxt = pred_seq[i + 1] or {}
        cur_sub = _safe_float(cur.get("subresult_value"))
        nxt_lhs = _safe_float(nxt.get("lhs_value"))
        nxt_rhs = _safe_float(nxt.get("rhs_value"))
        cur_mag = str(cur.get("magnitude_bucket", "unknown"))
        cur_sign = str(cur.get("sign", "unknown"))
        lhs_mag, lhs_sign = _pred_to_mag_sign(nxt_lhs)
        rhs_mag, rhs_sign = _pred_to_mag_sign(nxt_rhs)
        match_lhs = cur_mag == lhs_mag and cur_sign == lhs_sign and cur_mag != "unknown"
        match_rhs = cur_mag == rhs_mag and cur_sign == rhs_sign and cur_mag != "unknown"
        consistency.append(1.0 if (match_lhs or match_rhs) else 0.0)
        if cur_sub is not None and (nxt_lhs is not None or nxt_rhs is not None):
            errs = []
            if nxt_lhs is not None:
                errs.append(abs(cur_sub - nxt_lhs))
            if nxt_rhs is not None:
                errs.append(abs(cur_sub - nxt_rhs))
            if errs:
                abs_errs.append(float(min(errs)))
    if not consistency:
        return {k: 0.5 for k in HYBRID_KEYS}
    err_tensor = torch.tensor(abs_errs, dtype=torch.float32) if abs_errs else torch.tensor([0.0], dtype=torch.float32)
    mean_cons = float(sum(consistency) / max(1, len(consistency)))
    return {
        "hybrid_transition_consistency_mean": mean_cons,
        "hybrid_transition_inconsistency_fraction": float(1.0 - mean_cons),
        "hybrid_weakest_link_consistency": float(min(consistency)),
        "hybrid_min_abs_transition_error": float(torch.min(err_tensor).item()) if abs_errs else 0.0,
        "hybrid_p95_abs_transition_error": float(torch.quantile(err_tensor, 0.95).item()) if abs_errs else 0.0,
    }


def _row_identity_hash(rows: Sequence[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        tid = str(r.get("trace_id", ""))
        var = str(r.get("control_variant", ""))
        step = int(r.get("step_idx", -1))
        ex = int(r.get("example_idx", -1))
        line = int(r.get("line_index", -1))
        h.update(f"{tid}|{var}|{step}|{ex}|{line}\n".encode("utf-8"))
    return h.hexdigest()


def _decode_predictions_for_rows(
    rows: Sequence[dict],
    *,
    decoder_checkpoint: str,
    device: str,
    batch_size: int,
    cache_path: Optional[str],
) -> List[Optional[Dict[str, Any]]]:
    row_hash = _row_identity_hash(rows)
    if cache_path:
        cp = Path(cache_path)
        if cp.exists():
            payload = load_json(cp)
            if (
                str(payload.get("row_identity_hash", "")) == row_hash
                and isinstance(payload.get("predictions"), list)
                and int(payload.get("rows_count", -1)) == int(len(rows))
            ):
                preds = list(payload.get("predictions", []))
                if len(preds) == len(rows):
                    return preds

    _, cfg, numeric_stats, model = load_model_from_checkpoint(decoder_checkpoint, device=device)
    full = decode_latent_pred_states(
        model,
        rows,
        cfg,
        numeric_stats,
        device=device,
        batch_size=int(batch_size),
        cache_inputs="auto",
        cache_max_gb=8.0,
        non_blocking_transfer=True,
    )
    preds: List[Optional[Dict[str, Any]]] = []
    for p in full:
        s = p.get("latent_pred_state", {}) if isinstance(p, dict) else {}
        preds.append(
            {
                "magnitude_bucket": s.get("magnitude_bucket"),
                "sign": s.get("sign"),
                "lhs_value": _safe_float(s.get("lhs_value")),
                "rhs_value": _safe_float(s.get("rhs_value")),
                "subresult_value": _safe_float(s.get("subresult_value")),
            }
        )
    if len(preds) != len(rows):
        raise RuntimeError(
            f"decoder prediction length mismatch: preds={len(preds)} rows={len(rows)} checkpoint={decoder_checkpoint}"
        )
    if cache_path:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        save_json(
            cp,
            {
                "schema_version": "phase7_decoder_preds_cache_v1",
                "decoder_checkpoint": str(decoder_checkpoint),
                "rows_count": int(len(rows)),
                "row_identity_hash": str(row_hash),
                "predictions": preds,
            },
        )
    return preds


def _metric_auc_from_samples(samples: Sequence[dict], metric_key: str, *, seed: int, n_bootstrap: int) -> Dict[str, Any]:
    score_labels: List[Tuple[float, int]] = []
    defined_count = 0
    for s in samples:
        m = s.get("metrics", {}).get(metric_key)
        if isinstance(m, (int, float)):
            defined_count += 1
            # coherence is faithful-high; invert so AUROC positive class=unfaithful
            score_labels.append((1.0 - float(m), 1 if str(s.get("label")) == "unfaithful" else 0))
    out = _bootstrap_auc_ci(score_labels, n_bootstrap=n_bootstrap, seed=seed)
    out["defined_fraction"] = float(defined_count / max(1, len(samples)))
    out["sample_count"] = int(len(score_labels))
    return out


def _variant_metric_table(samples: Sequence[dict], metric_key: str, *, seed: int, n_bootstrap: int) -> Dict[str, Any]:
    by_variant: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        by_variant[str(s.get("variant", ""))].append(s)
    out: Dict[str, Any] = {}
    for variant in sorted(by_variant.keys()):
        rows = by_variant[variant]
        auc = _metric_auc_from_samples(rows, metric_key, seed=seed + hash(variant) % 10007, n_bootstrap=n_bootstrap)
        classes = {"faithful": 0, "unfaithful": 0}
        for r in rows:
            lbl = str(r.get("label", ""))
            if lbl in classes:
                classes[lbl] += 1
        out[variant] = {
            **auc,
            "class_counts": classes,
            "pair_count": int(min(classes["faithful"], classes["unfaithful"])),
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-records", required=True)
    p.add_argument("--model-key", default="gpt2-medium")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--saes-dir", default="phase2_results/saes_gpt2_12x_topk/saes")
    p.add_argument("--activations-dir", default="phase2_results/activations")
    p.add_argument("--phase4-top-features", default="phase4_results/topk/probe/top_features_per_layer.json")
    p.add_argument(
        "--feature-set",
        choices=list(FEATURE_SET_CHOICES),
        default="eq_top50",
        help="Feature index set used to build trajectories.",
    )
    p.add_argument(
        "--divergent-source",
        default="phase7_results/results/phase7_sae_feature_discrimination_phase7_sae_20260306_224419_phase7_sae.json",
        help="Required for --feature-set divergent_top50.",
    )
    p.add_argument(
        "--subspace-specs",
        default="phase7_results/interventions/variable_subspaces_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json",
    )
    p.add_argument("--sample-traces", type=int, default=0, help="0 means use all eligible traces")
    p.add_argument("--min-common-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260306)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--decoder-checkpoint", default="", help="Optional decoder checkpoint for hybrid consistency features.")
    p.add_argument("--decoder-device", default="", help="Device for decoder inference (default: --device).")
    p.add_argument("--decoder-batch-size", type=int, default=128)
    p.add_argument(
        "--decoder-preds-cache",
        default="",
        help="Optional JSON cache path for per-row decoder predictions used by hybrid metrics.",
    )
    p.add_argument("--run-tag", default="")
    p.add_argument(
        "--emit-samples",
        action="store_true",
        help="Include per-sample trajectory metric rows in output (larger artifact).",
    )
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    run_tag = str(args.run_tag).strip() or f"phase7_sae_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    layer = int(args.layer)
    rows, payload = _load_control_records_artifact(args.control_records)

    rows_used, pair_desc, pair_diag = _build_pair_descriptors(
        rows,
        min_common_steps=int(args.min_common_steps),
        sample_traces=(None if int(args.sample_traces) <= 0 else int(args.sample_traces)),
        seed=int(args.seed),
    )
    if not rows_used or not pair_desc:
        out = {
            "schema_version": "phase7_sae_trajectory_coherence_partial_v1",
            "status": "blocked_no_pairable_trajectories",
            "run_tag": run_tag,
            "model_key": str(args.model_key),
            "layer": int(layer),
            "source_control_records": str(args.control_records),
            "source_control_records_sha256": sha256_file(args.control_records),
            "coverage_diagnostics": pair_diag,
            "timestamp": datetime.now().isoformat(),
        }
        save_json(args.output, out)
        print(f"Saved blocked partial -> {args.output}")
        return

    top50, selection_meta = _select_feature_indices(
        feature_set=str(args.feature_set),
        layer=layer,
        phase4_top_features_path=args.phase4_top_features,
        divergent_source_path=args.divergent_source,
    )
    mag_feats = _magnitude_features_for_layer(args.subspace_specs, layer=layer)
    mag_overlap_global = [f for f in top50 if f in set(mag_feats)]
    if mag_overlap_global:
        mag_focus_features = list(mag_overlap_global)
        mag_focus_source = "overlap_with_magnitude_subspace"
    else:
        mag_focus_features = list(top50)
        mag_focus_source = "fallback_eq_top50"
    feature_to_col = {int(f): i for i, f in enumerate(top50)}
    mag_focus_cols = [feature_to_col[f] for f in mag_focus_features if f in feature_to_col]

    feat_rows = _encode_rows_layer_feature_subset(
        rows_used,
        model_key=str(args.model_key),
        layer=layer,
        feature_indices=top50,
        saes_dir=Path(args.saes_dir),
        activations_dir=Path(args.activations_dir),
        device=str(args.device),
        batch_size=int(args.batch_size),
    )

    decoder_preds_by_row: Optional[List[Optional[Dict[str, Any]]]] = None
    decoder_checkpoint = str(args.decoder_checkpoint).strip()
    if decoder_checkpoint:
        decoder_device = str(args.decoder_device).strip() or str(args.device)
        decoder_preds_by_row = _decode_predictions_for_rows(
            rows_used,
            decoder_checkpoint=decoder_checkpoint,
            device=decoder_device,
            batch_size=int(args.decoder_batch_size),
            cache_path=(str(args.decoder_preds_cache).strip() or None),
        )

    samples = _trajectory_samples(
        feat_rows,
        pair_desc,
        mag_cols=mag_focus_cols,
        decoder_preds_by_row=decoder_preds_by_row,
    )
    overall: Dict[str, Any] = {}
    by_variant: Dict[str, Any] = {}
    for mi, metric in enumerate(METRIC_KEYS):
        overall[metric] = _metric_auc_from_samples(
            samples,
            metric,
            seed=int(args.seed) + (mi + 1) * 1009,
            n_bootstrap=int(args.n_bootstrap),
        )
        by_variant[metric] = _variant_metric_table(
            samples,
            metric,
            seed=int(args.seed) + (mi + 1) * 2003,
            n_bootstrap=int(args.n_bootstrap),
        )

    variant_pair_counts: Dict[str, int] = defaultdict(int)
    for p in pair_desc:
        variant_pair_counts[p.unfaithful_variant] += 1

    out = {
        "schema_version": "phase7_sae_trajectory_coherence_partial_v1",
        "status": "ok",
        "run_tag": run_tag,
        "model_key": str(args.model_key),
        "layer": int(layer),
        "source_control_records": str(args.control_records),
        "source_control_records_sha256": sha256_file(args.control_records),
        "source_control_records_stats": payload.get("stats"),
        "sae_source": str(args.saes_dir),
        "activations_source": str(args.activations_dir),
        "phase4_top_features_source": str(args.phase4_top_features),
        "subspace_specs_source": str(args.subspace_specs),
        "analysis_config": {
            "sample_traces": int(args.sample_traces),
            "min_common_steps": int(args.min_common_steps),
            "seed": int(args.seed),
            "n_bootstrap": int(args.n_bootstrap),
            "batch_size": int(args.batch_size),
            "device": str(args.device),
            "alignment_policy": "common_step_idx_sorted",
            "metric_policy": "separate_only_no_blend",
            "feature_set": str(args.feature_set),
            "decoder_checkpoint": (decoder_checkpoint or None),
            "decoder_device": (str(args.decoder_device).strip() or None),
            "hybrid_consistency_enabled": bool(decoder_preds_by_row is not None),
        },
        "feature_selection": {
            "selected_features": [int(x) for x in top50],
            "selection_meta": selection_meta,
            "magnitude_subspace_features": [int(x) for x in mag_feats],
            "magnitude_focus_features": [int(x) for x in mag_focus_features],
            "magnitude_focus_source": str(mag_focus_source),
            "magnitude_focus_overlap_count": int(len(mag_overlap_global)),
        },
        "coverage_diagnostics": {
            **pair_diag,
            "pair_descriptor_count": int(len(pair_desc)),
            "variant_pair_counts": {k: int(v) for k, v in sorted(variant_pair_counts.items())},
            "samples_count": int(len(samples)),
            "samples_faithful": int(sum(1 for s in samples if s["label"] == "faithful")),
            "samples_unfaithful": int(sum(1 for s in samples if s["label"] == "unfaithful")),
        },
        "overall_metrics": overall,
        "variant_stratified_metrics": by_variant,
        "timestamp": datetime.now().isoformat(),
    }
    if bool(args.emit_samples):
        serializable_samples: List[Dict[str, Any]] = []
        for s in samples:
            serializable_samples.append(
                {
                    "trace_id": str(s.get("trace_id", "")),
                    "variant": str(s.get("variant", "")),
                    "label": str(s.get("label", "")),
                    "step_count": int(s.get("step_count", 0)),
                    "metrics": {
                        k: (float(v) if isinstance(v, (int, float)) else None)
                        for k, v in dict(s.get("metrics", {})).items()
                    },
                }
            )
        out["sample_metrics"] = serializable_samples
        out["sample_metrics_count"] = int(len(serializable_samples))
    save_json(args.output, out)
    print(f"Saved trajectory coherence partial -> {args.output}")


if __name__ == "__main__":
    main()

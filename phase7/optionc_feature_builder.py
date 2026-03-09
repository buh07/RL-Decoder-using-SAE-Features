#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover
    from .common import load_json, load_pt
    from .optionc_domain_decoder import decode_optionc_domain_states, load_optionc_domain_decoder_checkpoint
    from .state_decoder_core import decode_latent_pred_states, load_model_from_checkpoint
except ImportError:  # pragma: no cover
    from common import load_json, load_pt
    from optionc_domain_decoder import decode_optionc_domain_states, load_optionc_domain_decoder_checkpoint
    from state_decoder_core import decode_latent_pred_states, load_model_from_checkpoint


def _safe_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        x = float(v)
        if x == x and x not in (float("inf"), float("-inf")):
            return x
    return None


def _infer_dataset_domain(payload: Dict[str, Any]) -> str:
    d = str((payload.get("source") or {}).get("scope", "")).strip().lower()
    if d in {"arithmetic", "prontoqa", "entailmentbank"}:
        return d
    return "arithmetic"


def _feature_allowed_by_layer(feature_name: str, layer_allowlist: Optional[set[int]]) -> bool:
    if not layer_allowlist:
        return True
    if not str(feature_name).startswith("layer") or ":" not in str(feature_name):
        return True
    try:
        layer = int(str(feature_name).split(":", 1)[0].replace("layer", ""))
    except Exception:
        return False
    return layer in layer_allowlist


def _pred_to_mag_sign(v: Optional[float]) -> Tuple[str, str]:
    if v is None:
        return "unknown", "unknown"
    if abs(float(v)) < 1e-12:
        return "zero", "zero"
    sign = "positive" if float(v) > 0 else "negative"
    mag = "small" if abs(float(v)) < 10.0 else ("medium" if abs(float(v)) < 100.0 else "large")
    return mag, sign


def _decoder_transition_features(pred_seq: Sequence[Optional[Dict[str, Any]]]) -> Dict[str, float]:
    if len(pred_seq) < 2:
        return {
            "decoder_transition_consistency_mean": 0.5,
            "decoder_transition_inconsistency_fraction": 0.5,
            "decoder_weakest_link_consistency": 0.5,
            "decoder_min_abs_transition_error": 0.0,
            "decoder_p95_abs_transition_error": 0.0,
        }
    consistency: List[float] = []
    abs_errs: List[float] = []
    for i in range(len(pred_seq) - 1):
        cur = pred_seq[i] or {}
        nxt = pred_seq[i + 1] or {}
        cur_sub = _safe_float(cur.get("subresult_value"))
        nxt_lhs = _safe_float(nxt.get("lhs_value"))
        nxt_rhs = _safe_float(nxt.get("rhs_value"))
        cur_mag, cur_sign = _pred_to_mag_sign(cur_sub)
        lhs_mag, lhs_sign = _pred_to_mag_sign(nxt_lhs)
        rhs_mag, rhs_sign = _pred_to_mag_sign(nxt_rhs)
        match_lhs = cur_mag == lhs_mag and cur_sign == lhs_sign and cur_mag != "unknown"
        match_rhs = cur_mag == rhs_mag and cur_sign == rhs_sign and cur_mag != "unknown"
        consistency.append(1.0 if (match_lhs or match_rhs) else 0.0)
        if cur_sub is not None and (nxt_lhs is not None or nxt_rhs is not None):
            errs = []
            if nxt_lhs is not None:
                errs.append(abs(float(cur_sub) - float(nxt_lhs)))
            if nxt_rhs is not None:
                errs.append(abs(float(cur_sub) - float(nxt_rhs)))
            if errs:
                abs_errs.append(float(min(errs)))
    mean_cons = float(sum(consistency) / max(1, len(consistency)))
    min_cons = float(min(consistency)) if consistency else 0.5
    if abs_errs:
        err_t = torch.tensor(abs_errs, dtype=torch.float32)
        min_err = float(err_t.min().item())
        p95_err = float(torch.quantile(err_t, 0.95).item())
    else:
        min_err = 0.0
        p95_err = 0.0
    return {
        "decoder_transition_consistency_mean": mean_cons,
        "decoder_transition_inconsistency_fraction": float(1.0 - mean_cons),
        "decoder_weakest_link_consistency": min_cons,
        "decoder_min_abs_transition_error": min_err,
        "decoder_p95_abs_transition_error": p95_err,
    }


def _decoder_transition_features_logical(
    pred_seq: Sequence[Optional[Dict[str, Any]]],
    *,
    expected_truth: Optional[bool] = None,
    feature_mode: str = "full",
) -> Dict[str, float]:
    mode = str(feature_mode or "full").strip().lower()
    if mode not in {"full", "truth_inference_only"}:
        mode = "full"
    if len(pred_seq) < 2:
        if mode == "truth_inference_only":
            return {
                "decoder_truth_consistency_mean": 0.5,
                "decoder_inference_consistency_mean": 0.5,
                "decoder_answer_alignment": 0.5,
                "decoder_weakest_link_consistency": 0.5,
                "decoder_p95_transition_confidence_gap": 0.0,
            }
        return {
            "decoder_chain_coherence_mean": 0.5,
            "decoder_truth_consistency_mean": 0.5,
            "decoder_answer_alignment": 0.5,
            "decoder_weakest_link_consistency": 0.5,
            "decoder_p95_transition_confidence_gap": 0.0,
        }
    chain_hits: List[float] = []
    truth_hits: List[float] = []
    inference_hits: List[float] = []
    confidence_gaps: List[float] = []
    for i in range(len(pred_seq) - 1):
        cur = pred_seq[i] or {}
        nxt = pred_seq[i + 1] or {}
        cur_conc = int(cur.get("conclusion_class_id", 0) or 0)
        nxt_prem = int(nxt.get("premise_class_id", 0) or 0)
        cur_truth = int(cur.get("truth_value_id", 0) or 0)
        nxt_truth = int(nxt.get("truth_value_id", 0) or 0)
        cur_inf = int(cur.get("inference_type_id", 0) or 0)
        nxt_inf = int(nxt.get("inference_type_id", 0) or 0)
        if cur_conc > 0 and nxt_prem > 0 and mode == "full":
            chain_hits.append(1.0 if cur_conc == nxt_prem else 0.0)
        if cur_truth > 0 and nxt_truth > 0:
            truth_hits.append(1.0 if cur_truth == nxt_truth else 0.0)
        if cur_inf > 0 and nxt_inf > 0:
            inference_hits.append(1.0 if cur_inf == nxt_inf else 0.0)

        conc_prob = _safe_float(cur.get("conclusion_top1_prob"))
        prem_prob = _safe_float(nxt.get("premise_top1_prob"))
        truth_prob_cur = _safe_float(cur.get("truth_top1_prob"))
        truth_prob_nxt = _safe_float(nxt.get("truth_top1_prob"))
        inf_prob_cur = _safe_float(cur.get("inference_top1_prob"))
        inf_prob_nxt = _safe_float(nxt.get("inference_top1_prob"))
        if mode == "full" and conc_prob is not None and prem_prob is not None:
            confidence_gaps.append(float(abs(conc_prob - prem_prob)))
        if truth_prob_cur is not None and truth_prob_nxt is not None:
            confidence_gaps.append(float(abs(truth_prob_cur - truth_prob_nxt)))
        if inf_prob_cur is not None and inf_prob_nxt is not None:
            confidence_gaps.append(float(abs(inf_prob_cur - inf_prob_nxt)))

    chain_mean = float(sum(chain_hits) / max(1, len(chain_hits))) if chain_hits else 0.5
    truth_mean = float(sum(truth_hits) / max(1, len(truth_hits))) if truth_hits else 0.5
    inference_mean = float(sum(inference_hits) / max(1, len(inference_hits))) if inference_hits else 0.5
    weakest = float(min([truth_mean, inference_mean])) if mode == "truth_inference_only" else float(min([chain_mean, truth_mean]))
    if confidence_gaps:
        gaps = torch.tensor(confidence_gaps, dtype=torch.float32)
        p95_gap = float(torch.quantile(gaps, 0.95).item())
    else:
        p95_gap = 0.0

    answer_alignment = 0.5
    last = pred_seq[-1] or {}
    last_truth = int(last.get("truth_value_id", 0) or 0)
    if expected_truth is not None and last_truth > 0:
        expected_id = 1 if bool(expected_truth) else 2
        answer_alignment = 1.0 if int(last_truth) == int(expected_id) else 0.0

    if mode == "truth_inference_only":
        return {
            "decoder_truth_consistency_mean": truth_mean,
            "decoder_inference_consistency_mean": inference_mean,
            "decoder_answer_alignment": float(answer_alignment),
            "decoder_weakest_link_consistency": weakest,
            "decoder_p95_transition_confidence_gap": p95_gap,
        }
    return {
        "decoder_chain_coherence_mean": chain_mean,
        "decoder_truth_consistency_mean": truth_mean,
        "decoder_answer_alignment": float(answer_alignment),
        "decoder_weakest_link_consistency": weakest,
        "decoder_p95_transition_confidence_gap": p95_gap,
    }


def build_optionc_feature_rows(
    *,
    paired_dataset: str,
    partials: Sequence[str],
    decoder_checkpoint: str = "",
    decoder_domain_requested: str = "auto",
    logical_decoder_feature_mode: str = "full",
    sae_layer_allowlist_values: Optional[Sequence[int]] = None,
    decoder_device: str = "cuda:0",
    decoder_batch_size: int = 128,
    require_decoder_enabled: bool = False,
) -> Dict[str, Any]:
    payload = load_json(paired_dataset)
    dataset_domain = _infer_dataset_domain(payload)
    expected_decoder_domain = dataset_domain if str(decoder_domain_requested) == "auto" else str(decoder_domain_requested).strip().lower()

    rows_path = payload.get("rows_path")
    if not rows_path:
        raise RuntimeError("paired dataset missing rows_path")
    rp = Path(str(rows_path))
    if not rp.is_absolute():
        candidate = (Path(paired_dataset).parent / rp).resolve()
        rp = candidate if candidate.exists() else rp.resolve()
    rows = list(load_pt(rp))
    rows_sorted = sorted(
        [r for r in rows if isinstance(r, dict)],
        key=lambda r: (str(r.get("member_id", "")), int(r.get("step_idx", -1)), int(r.get("line_index", -1))),
    )

    members = list(payload.get("members", []))
    members_by_id: Dict[str, Dict[str, Any]] = {str(m.get("member_id")): dict(m) for m in members}
    pairs = list(payload.get("pairs", []))

    mode = str(logical_decoder_feature_mode or "full").strip().lower()
    if mode not in {"full", "truth_inference_only"}:
        mode = "full"
    allow_vals = sorted({int(x) for x in (sae_layer_allowlist_values or [])})
    allow_set = set(allow_vals) if allow_vals else None

    merged_features: Dict[str, Dict[str, float]] = defaultdict(dict)
    partial_paths = [str(p) for p in partials]
    for pp in partial_paths:
        pj = load_json(pp)
        if str(pj.get("status")) != "ok":
            raise RuntimeError(f"partial not ok: {pp}")
        for m in list(pj.get("members", [])):
            mid = str(m.get("member_id", ""))
            if not mid:
                continue
            for k, v in dict(m.get("features", {})).items():
                if isinstance(v, (int, float)):
                    fk = str(k)
                    if _feature_allowed_by_layer(fk, allow_set):
                        merged_features[mid][fk] = float(v)
    sae_feature_names = sorted({k for d in merged_features.values() for k in d.keys() if str(k).startswith("layer")})

    decoder_added = False
    decoder_domain: Optional[str] = None
    decoder_domain_match = False
    decoder_feature_block_status = "disabled_no_checkpoint"
    decoder_quality: Optional[Dict[str, Any]] = None
    if str(decoder_checkpoint).strip():
        ckpt_path = str(decoder_checkpoint)
        try:
            raw_ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            raw_ckpt = {}
        schema = str((raw_ckpt or {}).get("schema_version", ""))
        if schema == "phase7_optionc_domain_decoder_v1":
            ckpt, cfg, model = load_optionc_domain_decoder_checkpoint(ckpt_path, device=str(decoder_device))
            decoder_domain = str(cfg.decoder_domain).strip().lower()
            decoder_quality = dict(ckpt.get("best_val") or {})
            decoder_domain_match = bool(decoder_domain == expected_decoder_domain)
            if not decoder_domain_match:
                decoder_feature_block_status = "blocked_domain_mismatch"
            else:
                preds = decode_optionc_domain_states(
                    model,
                    cfg,
                    rows_sorted,
                    device=str(decoder_device),
                    batch_size=int(decoder_batch_size),
                )
                member_pred: Dict[str, List[Optional[Dict[str, Any]]]] = defaultdict(list)
                for row, pred in zip(rows_sorted, preds):
                    mid = str(row.get("member_id", ""))
                    s = (pred or {}).get("latent_pred_state", {}) if isinstance(pred, dict) else {}
                    c = (pred or {}).get("latent_pred_confidence", {}) if isinstance(pred, dict) else {}
                    member_pred[mid].append(
                        {
                            "inference_type_id": int(s.get("inference_type_id", 0) or 0),
                            "chain_depth_id": int(s.get("chain_depth_id", 0) or 0),
                            "truth_value_id": int(s.get("truth_value_id", 0) or 0),
                            "conclusion_class_id": int(s.get("conclusion_class_id", 0) or 0),
                            "premise_class_id": int(s.get("premise_class_id", 0) or 0),
                            "target_entity_id": int(s.get("target_entity_id", 0) or 0),
                            "inference_top1_prob": _safe_float(c.get("inference_top1_prob")),
                            "conclusion_top1_prob": _safe_float(c.get("conclusion_top1_prob")),
                            "premise_top1_prob": _safe_float(c.get("premise_top1_prob")),
                            "truth_top1_prob": _safe_float(c.get("truth_top1_prob")),
                        }
                    )
                for mid, seq in member_pred.items():
                    mm = members_by_id.get(mid, {})
                    merged_features[mid].update(
                        _decoder_transition_features_logical(
                            seq,
                            expected_truth=bool(mm.get("is_correct", False)),
                            feature_mode=mode,
                        )
                    )
                decoder_added = True
                decoder_feature_block_status = "enabled_logical"
        else:
            decoder_domain = "arithmetic"
            decoder_domain_match = bool(decoder_domain == expected_decoder_domain)
            if not decoder_domain_match:
                decoder_feature_block_status = "blocked_domain_mismatch"
            else:
                _, cfg, numeric_stats, model = load_model_from_checkpoint(ckpt_path, device=str(decoder_device))
                preds = decode_latent_pred_states(
                    model,
                    rows_sorted,
                    cfg,
                    numeric_stats,
                    device=str(decoder_device),
                    batch_size=int(decoder_batch_size),
                    cache_inputs="auto",
                    cache_max_gb=8.0,
                    non_blocking_transfer=True,
                )
                member_pred: Dict[str, List[Optional[Dict[str, Any]]]] = defaultdict(list)
                for row, pred in zip(rows_sorted, preds):
                    mid = str(row.get("member_id", ""))
                    s = (pred or {}).get("latent_pred_state", {}) if isinstance(pred, dict) else {}
                    member_pred[mid].append(
                        {
                            "subresult_value": _safe_float(s.get("subresult_value")),
                            "lhs_value": _safe_float(s.get("lhs_value")),
                            "rhs_value": _safe_float(s.get("rhs_value")),
                        }
                    )
                for mid, seq in member_pred.items():
                    merged_features[mid].update(_decoder_transition_features(seq))
                decoder_added = True
                decoder_feature_block_status = "enabled_arithmetic"

    if bool(require_decoder_enabled) and str(decoder_checkpoint).strip() and not decoder_added:
        raise RuntimeError(
            f"decoder feature block not enabled (status={decoder_feature_block_status}, "
            f"dataset_domain={dataset_domain}, requested={expected_decoder_domain}, actual={decoder_domain})"
        )

    feature_names = sorted({k for d in merged_features.values() for k in d.keys()})
    dropped_ambiguous = 0
    eval_rows: List[Dict[str, Any]] = []
    for mid, mm in sorted(members_by_id.items()):
        if not bool(mm.get("label_defined", False)):
            continue
        if bool(mm.get("pair_ambiguous", False)):
            dropped_ambiguous += 1
            continue
        if mid not in merged_features:
            continue
        feats = {fn: float(merged_features[mid].get(fn, 0.0)) for fn in feature_names}
        eval_rows.append(
            {
                "member_id": str(mid),
                "pair_id": str(mm.get("pair_id", "")),
                "pair_type": str(mm.get("pair_type", "")),
                "label": int(mm.get("label_binary", 1)),
                "gold_label": str(mm.get("gold_label", "unknown")),
                "lexical_control": bool(mm.get("lexical_control", False)),
                "pair_ambiguous": bool(mm.get("pair_ambiguous", False)),
                "features": feats,
            }
        )
    if len(eval_rows) < 20:
        raise RuntimeError(f"Insufficient evaluation rows: {len(eval_rows)}")

    decoder_feature_names = [fn for fn in feature_names if str(fn).startswith("decoder_")]
    return {
        "payload": payload,
        "dataset_domain": str(dataset_domain),
        "expected_decoder_domain": str(expected_decoder_domain),
        "pairs": pairs,
        "members_by_id": members_by_id,
        "eval_rows": eval_rows,
        "partial_paths": partial_paths,
        "logical_decoder_feature_mode": mode,
        "sae_layer_allowlist_values": allow_vals,
        "sae_feature_names": sae_feature_names,
        "feature_names": feature_names,
        "feature_count_total": int(len(feature_names)),
        "sae_feature_count_after_filter": int(len(sae_feature_names)),
        "decoder_feature_count": int(len(decoder_feature_names)),
        "decoder_domain": decoder_domain,
        "decoder_domain_match": bool(decoder_domain_match),
        "decoder_feature_block_status": str(decoder_feature_block_status),
        "decoder_quality": decoder_quality,
        "decoder_features_enabled": bool(decoder_added),
        "dropped_pair_ambiguous": int(dropped_ambiguous),
    }

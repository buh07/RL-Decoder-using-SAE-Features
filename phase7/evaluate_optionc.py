#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover
    from .common import load_json, load_pt, save_json
    from .optionc_domain_decoder import decode_optionc_domain_states, load_optionc_domain_decoder_checkpoint
    from .state_decoder_core import decode_latent_pred_states, load_model_from_checkpoint
except ImportError:  # pragma: no cover
    from common import load_json, load_pt, save_json
    from optionc_domain_decoder import decode_optionc_domain_states, load_optionc_domain_decoder_checkpoint
    from state_decoder_core import decode_latent_pred_states, load_model_from_checkpoint


def _roc_auc(scores_labels: Sequence[Tuple[float, int]]) -> Optional[float]:
    if not scores_labels:
        return None
    pos = sum(int(y) for _, y in scores_labels)
    neg = int(len(scores_labels) - pos)
    if pos <= 0 or neg <= 0:
        return None
    ranked = sorted(scores_labels, key=lambda x: x[0])
    rank_sum = 0.0
    for i, (_, y) in enumerate(ranked, start=1):
        if int(y) == 1:
            rank_sum += i
    return float((rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg))


def _percentile(vals: Sequence[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(float(x) for x in vals)
    if len(xs) == 1:
        return float(xs[0])
    qq = min(1.0, max(0.0, float(q)))
    idx = qq * (len(xs) - 1)
    lo = int(idx)
    hi = min(len(xs) - 1, lo + 1)
    frac = idx - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _bootstrap_group_auc(
    scored: Sequence[Dict[str, Any]],
    *,
    group_key: str,
    n_bootstrap: int,
    seed: int,
    cpu_workers: int = 1,
) -> Dict[str, Any]:
    by_group: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for r in scored:
        g = str(r.get(group_key, ""))
        by_group[g].append((float(r["score"]), int(r["label"])))
    groups = sorted(g for g in by_group.keys() if g)
    pooled = [xy for g in groups for xy in by_group[g]]
    obs = _roc_auc(pooled)
    out = {
        "defined": bool(obs is not None),
        "observed_auroc": obs,
        "ci95_lower": None,
        "ci95_upper": None,
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_effective_n": 0,
        "group_count": int(len(groups)),
        "row_count": int(len(pooled)),
    }
    if obs is None or not groups:
        return out
    vals: List[float] = []
    total_boot = max(1, int(n_bootstrap))

    n_workers = max(1, int(cpu_workers))
    if n_workers <= 1 or total_boot < 100:
        vals.extend(_bootstrap_worker_task(by_group, groups, total_boot, int(seed)))
    else:
        splits = [total_boot // n_workers] * n_workers
        for i in range(total_boot % n_workers):
            splits[i] += 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = []
            for wi, wb in enumerate(splits):
                if wb <= 0:
                    continue
                futs.append(ex.submit(_bootstrap_worker_task, by_group, groups, int(wb), int(seed) + 1009 * (wi + 1)))
            for fut in concurrent.futures.as_completed(futs):
                vals.extend(fut.result())
    out["bootstrap_effective_n"] = int(len(vals))
    out["ci95_lower"] = _percentile(vals, 0.025)
    out["ci95_upper"] = _percentile(vals, 0.975)
    return out


def _bootstrap_worker_task(
    by_group: Dict[str, List[Tuple[float, int]]],
    groups: Sequence[str],
    worker_boot: int,
    worker_seed: int,
) -> List[float]:
    rng = random.Random(int(worker_seed))
    out_vals: List[float] = []
    grp = [str(x) for x in groups]
    for _ in range(int(worker_boot)):
        sample: List[Tuple[float, int]] = []
        for __ in range(len(grp)):
            g = grp[rng.randrange(len(grp))]
            sample.extend(by_group[g])
        auc = _roc_auc(sample)
        if auc is not None:
            out_vals.append(float(auc))
    return out_vals


def _train_logreg(
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    *,
    feature_names: Sequence[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> Dict[str, Any]:
    if not train_rows or not test_rows:
        raise RuntimeError("empty train/test rows")
    x_train = torch.tensor([[float(r["features"][f]) for f in feature_names] for r in train_rows], dtype=torch.float32)
    y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.float32)
    x_test = torch.tensor([[float(r["features"][f]) for f in feature_names] for r in test_rows], dtype=torch.float32)
    y_test = [int(r["label"]) for r in test_rows]

    mu = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std

    dev = torch.device(str(device))
    x_train = x_train.to(dev)
    y_train = y_train.to(dev)
    x_test = x_test.to(dev)

    w = torch.zeros((x_train.shape[1], 1), dtype=torch.float32, device=dev, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=float(lr), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = (x_train @ w + b).view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs_test = torch.sigmoid((x_test @ w + b).view(-1)).detach().cpu().tolist()
        probs_train = torch.sigmoid((x_train @ w + b).view(-1)).detach().cpu().tolist()
    return {
        "train_auroc": _roc_auc(list(zip(probs_train, [int(y) for y in y_train.detach().cpu().tolist()]))),
        "test_auroc": _roc_auc(list(zip(probs_test, y_test))),
        "test_probs": [float(x) for x in probs_test],
        "coef": [float(x) for x in w.detach().cpu().view(-1).tolist()],
        "intercept": float(b.detach().cpu().item()),
    }


def _fit_fold_task(
    *,
    fold_index: int,
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> Dict[str, Any]:
    if len(train_rows) < 10 or len(test_rows) < 10:
        return {
            "fold_index": int(fold_index),
            "status": "blocked_small_fold",
            "pair_count_test": int(len({str(r.get("pair_id", "")) for r in test_rows})),
            "test_rows": int(len(test_rows)),
            "auroc": None,
            "oof_rows": [],
        }
    fit = _train_logreg(
        train_rows,
        test_rows,
        feature_names=feature_names,
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        device=str(device),
    )
    probs = list(fit.get("test_probs", []))
    oof_rows: List[Dict[str, Any]] = []
    for row, sc in zip(test_rows, probs):
        oof_rows.append(
            {
                "member_id": str(row["member_id"]),
                "pair_id": str(row["pair_id"]),
                "label": int(row["label"]),
                "score": float(sc),
                "lexical_control": bool(row["lexical_control"]),
            }
        )
    return {
        "fold_index": int(fold_index),
        "status": "ok",
        "pair_count_test": int(len({str(r.get("pair_id", "")) for r in test_rows})),
        "test_rows": int(len(test_rows)),
        "auroc": _roc_auc([(float(s), int(r["label"])) for s, r in zip(probs, test_rows)]),
        "oof_rows": oof_rows,
    }


def _build_pair_split(
    rows: Sequence[Dict[str, Any]],
    *,
    pair_test_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    pair_ids = sorted({str(r["pair_id"]) for r in rows})
    n_test = max(1, int(round(len(pair_ids) * float(pair_test_fraction))))
    n_test = min(n_test, max(1, len(pair_ids) - 1))
    rng = random.Random(int(seed))
    ids = list(pair_ids)
    rng.shuffle(ids)
    test_set = set(ids[:n_test])
    train = [r for r in rows if str(r["pair_id"]) not in test_set]
    test = [r for r in rows if str(r["pair_id"]) in test_set]
    return train, test, {
        "pair_count_total": int(len(pair_ids)),
        "pair_count_train": int(len(pair_ids) - len(test_set)),
        "pair_count_test": int(len(test_set)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "trace_overlap_count": 0,
    }


def _parse_csv(value: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in str(value or "").split(","):
        t = tok.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in str(value or "").split(","):
        t = tok.strip()
        if not t:
            continue
        v = int(t)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return sorted(out)


def _feature_allowed_by_layer(
    feature_name: str,
    layer_allowlist: Optional[set[int]],
) -> bool:
    if not layer_allowlist:
        return True
    if not str(feature_name).startswith("layer") or ":" not in str(feature_name):
        return True
    try:
        layer = int(str(feature_name).split(":", 1)[0].replace("layer", ""))
    except Exception:
        return False
    return layer in layer_allowlist


def _apply_train_pair_type_exclusion(
    train_rows: Sequence[Dict[str, Any]],
    *,
    excluded_pair_types: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    excluded = set(str(x) for x in excluded_pair_types if str(x))
    kept: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = defaultdict(int)
    for r in train_rows:
        pair_type = str(r.get("pair_type", ""))
        if pair_type in excluded:
            dropped[pair_type] += 1
            continue
        kept.append(r)
    out_diag = {
        "excluded_pair_types": sorted(excluded),
        "excluded_train_rows_total": int(sum(dropped.values())),
        "excluded_train_rows_by_pair_type": {k: int(v) for k, v in sorted(dropped.items())},
        "train_rows_pre": int(len(train_rows)),
        "train_rows_post": int(len(kept)),
    }
    return kept, out_diag


def _build_pair_folds(pair_ids: Sequence[str], k: int, seed: int) -> List[List[str]]:
    ids = sorted(set(str(x) for x in pair_ids))
    if len(ids) < 2:
        return [ids]
    kk = max(2, min(int(k), len(ids)))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    folds: List[List[str]] = [[] for _ in range(kk)]
    for i, pid in enumerate(ids):
        folds[i % kk].append(pid)
    return [f for f in folds if f]


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


def _pred_to_mag_sign(v: Optional[float]) -> Tuple[str, str]:
    if v is None:
        return "unknown", "unknown"
    # Reuse coarse bins from numeric sign.
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
    if mode == "truth_inference_only":
        weakest = float(min([truth_mean, inference_mean]))
    else:
        weakest = float(min([chain_mean, truth_mean]))
    if confidence_gaps:
        gaps = torch.tensor(confidence_gaps, dtype=torch.float32)
        p95_gap = float(torch.quantile(gaps, 0.95).item())
    else:
        p95_gap = 0.0

    answer_alignment = 0.5
    last = pred_seq[-1] or {}
    last_truth = int(last.get("truth_value_id", 0) or 0)
    if expected_truth is not None and last_truth > 0:
        # vocab convention in optionc domain decoder: true=1, false=2, uncertain=3, unknown=0
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-dataset", required=True)
    p.add_argument("--partials", nargs="+", required=True)
    p.add_argument("--decoder-checkpoint", default="", help="Optional state-decoder checkpoint.")
    p.add_argument("--decoder-domain", choices=["auto", "arithmetic", "prontoqa", "entailmentbank"], default="auto")
    p.add_argument(
        "--logical-decoder-feature-mode",
        choices=["full", "truth_inference_only"],
        default="full",
        help="Logical decoder transition feature block mode.",
    )
    p.add_argument(
        "--sae-layer-allowlist",
        default="",
        help="Optional CSV allowlist of SAE layers (e.g. 6,8,14). Empty keeps all layers.",
    )
    p.add_argument("--decoder-device", default="cuda:0")
    p.add_argument("--decoder-batch-size", type=int, default=128)
    p.add_argument("--train-test-fraction", type=float, default=0.20)
    p.add_argument("--split-seed", type=int, default=20260309)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=20260309)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=20260309)
    p.add_argument("--cpu-workers", type=int, default=0, help="CPU workers for CV/bootstrap. 0=auto.")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--fit-device", default="cpu")
    p.add_argument("--primary-auroc-threshold", type=float, default=0.70)
    p.add_argument("--primary-ci-lower-threshold", type=float, default=0.65)
    p.add_argument("--lexical-auroc-max", type=float, default=0.60)
    p.add_argument("--wrong-minus-lexical-min", type=float, default=0.10)
    p.add_argument(
        "--train-exclude-pair-types",
        default="lexical_control",
        help="CSV pair_type values to exclude from training rows (test rows unchanged).",
    )
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--claim-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cpu_workers = int(args.cpu_workers)
    if cpu_workers <= 0:
        cpu_workers = max(1, min(16, (os.cpu_count() or 4)))
    payload = load_json(args.paired_dataset)
    dataset_domain = _infer_dataset_domain(payload)
    expected_decoder_domain = dataset_domain if str(args.decoder_domain) == "auto" else str(args.decoder_domain).strip().lower()
    rows_path = payload.get("rows_path")
    if not rows_path:
        raise RuntimeError("paired dataset missing rows_path")
    rp = Path(str(rows_path))
    if not rp.is_absolute():
        candidate = (Path(args.paired_dataset).parent / rp).resolve()
        rp = candidate if candidate.exists() else rp.resolve()
    rows = list(load_pt(rp))
    rows_sorted = sorted(
        [r for r in rows if isinstance(r, dict)],
        key=lambda r: (str(r.get("member_id", "")), int(r.get("step_idx", -1)), int(r.get("line_index", -1))),
    )

    members = list(payload.get("members", []))
    members_by_id: Dict[str, Dict[str, Any]] = {str(m.get("member_id")): dict(m) for m in members}
    pairs = list(payload.get("pairs", []))
    logical_decoder_feature_mode = str(args.logical_decoder_feature_mode).strip().lower()
    layer_allowlist_values = _parse_int_csv(str(args.sae_layer_allowlist)) if str(args.sae_layer_allowlist).strip() else []
    layer_allowlist_set = set(int(x) for x in layer_allowlist_values) if layer_allowlist_values else None

    # Merge per-layer partial features.
    merged_features: Dict[str, Dict[str, float]] = defaultdict(dict)
    partial_paths = [str(p) for p in args.partials]
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
                    if _feature_allowed_by_layer(fk, layer_allowlist_set):
                        merged_features[mid][fk] = float(v)
    sae_feature_names = sorted({k for d in merged_features.values() for k in d.keys() if str(k).startswith("layer")})

    # Decoder-based transition consistency block.
    decoder_added = False
    decoder_domain: Optional[str] = None
    decoder_domain_match = False
    decoder_feature_block_status = "disabled_no_checkpoint"
    decoder_quality: Optional[Dict[str, Any]] = None
    if str(args.decoder_checkpoint).strip():
        ckpt_path = str(args.decoder_checkpoint)
        try:
            raw_ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            raw_ckpt = {}
        schema = str((raw_ckpt or {}).get("schema_version", ""))
        if schema == "phase7_optionc_domain_decoder_v1":
            ckpt, cfg, model = load_optionc_domain_decoder_checkpoint(ckpt_path, device=str(args.decoder_device))
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
                    device=str(args.decoder_device),
                    batch_size=int(args.decoder_batch_size),
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
                            feature_mode=logical_decoder_feature_mode,
                        )
                    )
                decoder_added = True
                decoder_feature_block_status = "enabled_logical"
        else:
            decoder_domain = "arithmetic"
            decoder_quality = None
            decoder_domain_match = bool(decoder_domain == expected_decoder_domain)
            if not decoder_domain_match:
                decoder_feature_block_status = "blocked_domain_mismatch"
            else:
                _, cfg, numeric_stats, model = load_model_from_checkpoint(ckpt_path, device=str(args.decoder_device))
                preds = decode_latent_pred_states(
                    model,
                    rows_sorted,
                    cfg,
                    numeric_stats,
                    device=str(args.decoder_device),
                    batch_size=int(args.decoder_batch_size),
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

    # Build member rows for scoring.
    eval_rows: List[Dict[str, Any]] = []
    dropped_ambiguous = 0
    feature_names = sorted({k for d in merged_features.values() for k in d.keys()})
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

    # Single split eval.
    train_rows_raw, test_rows, split_diag = _build_pair_split(
        eval_rows,
        pair_test_fraction=float(args.train_test_fraction),
        seed=int(args.split_seed),
    )
    excluded_pair_types = _parse_csv(args.train_exclude_pair_types)
    train_rows, train_excl_diag = _apply_train_pair_type_exclusion(
        train_rows_raw,
        excluded_pair_types=excluded_pair_types,
    )
    if len({int(r["label"]) for r in train_rows}) < 2:
        raise RuntimeError("Training rows became single-class after train pair-type exclusion")
    fit = _train_logreg(
        train_rows,
        test_rows,
        feature_names=feature_names,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=str(args.fit_device),
    )
    test_scored: List[Dict[str, Any]] = []
    for row, score in zip(test_rows, list(fit.get("test_probs", []))):
        test_scored.append(
            {
                "member_id": str(row["member_id"]),
                "pair_id": str(row["pair_id"]),
                "label": int(row["label"]),
                "score": float(score),
                "lexical_control": bool(row["lexical_control"]),
            }
        )
    primary_auc = _roc_auc([(float(r["score"]), int(r["label"])) for r in test_scored])
    primary_boot = _bootstrap_group_auc(
        test_scored,
        group_key="pair_id",
        n_bootstrap=int(args.bootstrap_n),
        seed=int(args.bootstrap_seed),
        cpu_workers=cpu_workers,
    )

    # Lexical confound AUROC: lexical vs non-lexical among faithful members only.
    lexical_rows = [r for r in test_scored if int(r["label"]) == 0]
    lexical_auc = _roc_auc([(float(r["score"]), 1 if bool(r["lexical_control"]) else 0) for r in lexical_rows])

    # Pair-level behavioral baseline.
    pair_labels: List[Tuple[float, int]] = []
    pair_scored_for_ci: List[Dict[str, Any]] = []
    pair_lookup = {str(p.get("pair_id", "")): dict(p) for p in pairs}
    test_pair_ids = sorted({str(r["pair_id"]) for r in test_scored})
    for pid in test_pair_ids:
        p = pair_lookup.get(pid, {})
        score = p.get("behavioral_contradiction_score")
        defined = bool(p.get("behavioral_contradiction_defined", False))
        label = 1 if bool(p.get("pair_any_unfaithful", False)) else 0
        if not defined or not isinstance(score, (int, float)):
            continue
        pair_labels.append((float(score), int(label)))
        pair_scored_for_ci.append({"pair_id": str(pid), "score": float(score), "label": int(label)})
    baseline_pair_auc = _roc_auc(pair_labels)
    baseline_pair_boot = _bootstrap_group_auc(
        pair_scored_for_ci,
        group_key="pair_id",
        n_bootstrap=int(args.bootstrap_n),
        seed=int(args.bootstrap_seed) + 31,
        cpu_workers=cpu_workers,
    )

    # CV pooled member-level.
    all_pair_ids = [str(r["pair_id"]) for r in eval_rows]
    folds = _build_pair_folds(all_pair_ids, int(args.cv_folds), int(args.cv_seed))
    oof: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    overlap_total = 0
    fold_jobs: List[Dict[str, Any]] = []
    for fi, fold in enumerate(folds):
        test_set = set(str(x) for x in fold)
        tr_raw = [r for r in eval_rows if str(r["pair_id"]) not in test_set]
        te = [r for r in eval_rows if str(r["pair_id"]) in test_set]
        tr, fold_excl_diag = _apply_train_pair_type_exclusion(
            tr_raw,
            excluded_pair_types=excluded_pair_types,
        )
        overlap = len(set(str(r["pair_id"]) for r in tr).intersection(test_set))
        overlap_total += int(overlap)
        fold_jobs.append(
            {
                "fold_index": int(fi),
                "train_rows": tr,
                "test_rows": te,
                "train_exclusion_diagnostics": fold_excl_diag,
                "pair_count_test": int(len(test_set)),
                "overlap": int(overlap),
            }
        )

    if cpu_workers <= 1 or len(fold_jobs) <= 1:
        fold_out = [
            _fit_fold_task(
                fold_index=int(job["fold_index"]),
                train_rows=job["train_rows"],
                test_rows=job["test_rows"],
                feature_names=feature_names,
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                device=str(args.fit_device),
            )
            for job in fold_jobs
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(cpu_workers, len(fold_jobs))) as ex:
            futs = [
                ex.submit(
                    _fit_fold_task,
                    fold_index=int(job["fold_index"]),
                    train_rows=job["train_rows"],
                    test_rows=job["test_rows"],
                    feature_names=feature_names,
                    epochs=int(args.epochs),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    device=str(args.fit_device),
                )
                for job in fold_jobs
            ]
            fold_out = [f.result() for f in concurrent.futures.as_completed(futs)]
    by_fold_index = {int(r["fold_index"]): r for r in fold_out}
    for job in sorted(fold_jobs, key=lambda j: int(j["fold_index"])):
        fr = by_fold_index[int(job["fold_index"])]
        fold_rows.append(
            {
                "fold_index": int(fr["fold_index"]),
                "status": str(fr["status"]),
                "pair_count_test": int(job["pair_count_test"]),
                "test_rows": int(fr.get("test_rows", 0)),
                "auroc": fr.get("auroc"),
                "overlap": int(job["overlap"]),
                "train_exclusion_diagnostics": job["train_exclusion_diagnostics"],
            }
        )
        oof.extend(list(fr.get("oof_rows", [])))
    cv_pooled_auc = _roc_auc([(float(r["score"]), int(r["label"])) for r in oof])
    cv_boot = _bootstrap_group_auc(
        oof,
        group_key="pair_id",
        n_bootstrap=int(args.bootstrap_n),
        seed=int(args.bootstrap_seed) + 97,
        cpu_workers=cpu_workers,
    )

    wrong_minus_lex = (
        float(cv_pooled_auc) - float(lexical_auc)
        if isinstance(cv_pooled_auc, (int, float)) and isinstance(lexical_auc, (int, float))
        else None
    )
    primary_gate = bool(
        isinstance(cv_pooled_auc, (int, float))
        and float(cv_pooled_auc) > float(args.primary_auroc_threshold)
        and isinstance(cv_boot.get("ci95_lower"), (int, float))
        and float(cv_boot["ci95_lower"]) >= float(args.primary_ci_lower_threshold)
        and int(overlap_total) == 0
    )
    lexical_gate = bool(isinstance(lexical_auc, (int, float)) and float(lexical_auc) <= float(args.lexical_auroc_max))
    delta_gate = bool(isinstance(wrong_minus_lex, (int, float)) and float(wrong_minus_lex) >= float(args.wrong_minus_lexical_min))
    strict_gate = bool(primary_gate and lexical_gate and delta_gate)

    result = {
        "schema_version": "phase7_optionc_eval_v1",
        "status": "ok",
        "paired_dataset": str(args.paired_dataset),
        "partials": partial_paths,
        "decoder_checkpoint": (str(args.decoder_checkpoint) if str(args.decoder_checkpoint).strip() else None),
        "dataset_domain": str(dataset_domain),
        "decoder_domain_requested": str(expected_decoder_domain),
        "logical_decoder_feature_mode": str(logical_decoder_feature_mode),
        "sae_layer_allowlist": [int(x) for x in layer_allowlist_values],
        "sae_feature_count_after_filter": int(len(sae_feature_names)),
        "decoder_domain": decoder_domain,
        "decoder_domain_match": bool(decoder_domain_match),
        "decoder_feature_block_status": str(decoder_feature_block_status),
        "decoder_quality": decoder_quality,
        "decoder_features_enabled": bool(decoder_added),
        "feature_count": int(len(feature_names)),
        "split_diagnostics": split_diag,
        "train_exclusion_diagnostics": train_excl_diag,
        "row_filtering": {
            "dropped_pair_ambiguous_members": int(dropped_ambiguous),
        },
        "single_split": {
            "primary_member_auroc": primary_auc,
            "wrong_intermediate_probe_auroc": primary_auc,  # compatibility alias
            "primary_member_ci95": {
                "lower": primary_boot.get("ci95_lower"),
                "upper": primary_boot.get("ci95_upper"),
                "defined": bool(primary_boot.get("defined")),
            },
            "lexical_probe_auroc": lexical_auc,
            "behavioral_pair_auroc": baseline_pair_auc,
            "behavioral_pair_ci95": {
                "lower": baseline_pair_boot.get("ci95_lower"),
                "upper": baseline_pair_boot.get("ci95_upper"),
                "defined": bool(baseline_pair_boot.get("defined")),
            },
            "wrong_minus_lexical_delta": wrong_minus_lex,
        },
        "cv_diagnostics": {
            "cv_folds": int(len(folds)),
            "cv_seed": int(args.cv_seed),
            "fold_rows": fold_rows,
            "cv_primary_pooled_auroc": cv_pooled_auc,
            "cv_wrong_intermediate_pooled_auroc": cv_pooled_auc,  # compatibility alias
            "cv_primary_pooled_ci95": {
                "lower": cv_boot.get("ci95_lower"),
                "upper": cv_boot.get("ci95_upper"),
                "defined": bool(cv_boot.get("defined")),
            },
            "cv_wrong_intermediate_pooled_ci95": {
                "lower": cv_boot.get("ci95_lower"),
                "upper": cv_boot.get("ci95_upper"),
                "defined": bool(cv_boot.get("defined")),
            },
            "cv_pair_overlap_count": int(overlap_total),
            "cv_trace_overlap_count": int(overlap_total),
        },
        "claim_gate": {
            "primary_gate_pass": bool(primary_gate),
            "lexical_confound_control_pass": bool(lexical_gate),
            "wrong_minus_lexical_delta_pass": bool(delta_gate),
            "strict_gate_pass": bool(strict_gate),
        },
        "thresholds": {
            "primary_auroc_threshold": float(args.primary_auroc_threshold),
            "primary_ci_lower_threshold": float(args.primary_ci_lower_threshold),
            "lexical_auroc_max": float(args.lexical_auroc_max),
            "wrong_minus_lexical_min": float(args.wrong_minus_lexical_min),
        },
        "runtime": {
            "cpu_workers": int(cpu_workers),
        },
        "test_scored_rows": test_scored,
        "cv_oof_rows": oof,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(args.output_json, result)

    claim = {
        "schema_version": "phase7_optionc_claim_boundary_v1",
        "model_generated_condition_pass": True,
        "lexical_confound_control_pass": bool(lexical_gate),
        "wrong_minus_lexical_delta": wrong_minus_lex,
        "faithfulness_claim_enabled": bool(strict_gate),
        "coherence_only_fallback": bool(not strict_gate),
        "timestamp": datetime.now().isoformat(),
    }
    save_json(args.claim_json, claim)

    lines = [
        "# Option C Evaluation Summary",
        "",
        f"- Dataset domain: {dataset_domain}",
        f"- Decoder feature status: {decoder_feature_block_status}",
        f"- Single-split primary member AUROC: {primary_auc}",
        f"- CV pooled primary member AUROC: {cv_pooled_auc}",
        f"- CV pooled CI95: {result['cv_diagnostics']['cv_primary_pooled_ci95']}",
        f"- Lexical-control AUROC: {lexical_auc}",
        f"- Wrong-minus-lexical delta: {wrong_minus_lex}",
        f"- Behavioral contradiction pair AUROC: {baseline_pair_auc}",
        f"- Strict faithfulness gate: {strict_gate}",
    ]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(f"Saved Option C evaluation -> {args.output_json}")


if __name__ == "__main__":
    main()

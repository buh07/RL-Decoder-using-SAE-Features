#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover
    from .common import load_json, save_json
except ImportError:  # pragma: no cover
    from common import load_json, save_json

METRIC_KEYS = ("cosine_smoothness", "feature_variance_coherence", "magnitude_monotonicity_coherence")
MODEL_LADDER_CHOICES = ("sae_only", "hybrid_only", "mixed")


@dataclass
class Row:
    trace_id: str
    variant: str
    label: int
    features: List[float]


@dataclass
class ScoredRow:
    trace_id: str
    variant: str
    label: int
    score: float


def _parse_variant_csv(value: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in str(value or "").split(","):
        v = tok.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _parse_model_ladder_csv(value: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in str(value or "").split(","):
        m = tok.strip()
        if not m or m in seen:
            continue
        if m not in MODEL_LADDER_CHOICES:
            raise ValueError(f"unsupported model ladder entry: {m!r}; expected one of {MODEL_LADDER_CHOICES}")
        seen.add(m)
        out.append(m)
    if not out:
        raise ValueError("model ladder is empty")
    return out


def _summarize_rows(rows: Sequence[Row]) -> Dict[str, Any]:
    by_variant: Dict[str, Dict[str, int]] = {}
    pos = 0
    for r in rows:
        key = str(r.variant)
        rec = by_variant.setdefault(key, {"rows": 0, "pos": 0, "neg": 0})
        rec["rows"] += 1
        if int(r.label) == 1:
            rec["pos"] += 1
            pos += 1
        else:
            rec["neg"] += 1
    return {
        "rows": int(len(rows)),
        "pos": int(pos),
        "neg": int(len(rows) - pos),
        "by_variant": {k: dict(v) for k, v in sorted(by_variant.items())},
    }


def _apply_train_variant_exclusion(
    train_rows: Sequence[Row], excluded_variants: Sequence[str]
) -> Tuple[List[Row], Dict[str, int]]:
    excluded = set(str(v) for v in excluded_variants)
    kept: List[Row] = []
    dropped: Dict[str, int] = {}
    for r in train_rows:
        if int(r.label) == 1 and str(r.variant) in excluded:
            key = str(r.variant)
            dropped[key] = int(dropped.get(key, 0)) + 1
            continue
        kept.append(r)
    return kept, {k: int(v) for k, v in sorted(dropped.items())}


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


def _percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    qq = min(1.0, max(0.0, float(q)))
    idx = qq * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(len(sorted_values) - 1, lo + 1)
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _bootstrap_trace_pair_auc(
    scored_rows: Sequence[ScoredRow],
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    by_trace: Dict[str, List[Tuple[float, int]]] = {}
    for r in scored_rows:
        by_trace.setdefault(str(r.trace_id), []).append((float(r.score), int(r.label)))
    valid_trace_ids: List[str] = []
    for tid, rows in by_trace.items():
        pos = sum(y for _, y in rows)
        neg = len(rows) - pos
        if pos > 0 and neg > 0:
            valid_trace_ids.append(tid)
    valid_trace_ids = sorted(valid_trace_ids)

    pooled: List[Tuple[float, int]] = []
    for tid in valid_trace_ids:
        pooled.extend(by_trace.get(tid, []))
    observed = _roc_auc(pooled)
    out: Dict[str, Any] = {
        "defined": bool(observed is not None),
        "observed_auroc": observed,
        "ci95_lower": None,
        "ci95_upper": None,
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_effective_n": 0,
        "valid_trace_pair_count": int(len(valid_trace_ids)),
        "row_count": int(len(pooled)),
    }
    if observed is None or len(valid_trace_ids) == 0:
        return out

    rng = random.Random(int(seed))
    values: List[float] = []
    draws = max(1, int(n_bootstrap))
    for _ in range(draws):
        sample_pairs: List[Tuple[float, int]] = []
        for __ in range(len(valid_trace_ids)):
            tid = valid_trace_ids[rng.randrange(len(valid_trace_ids))]
            sample_pairs.extend(by_trace.get(tid, []))
        auc = _roc_auc(sample_pairs)
        if auc is not None:
            values.append(float(auc))
    values = sorted(values)
    out["bootstrap_effective_n"] = int(len(values))
    out["ci95_lower"] = _percentile(values, 0.025)
    out["ci95_upper"] = _percentile(values, 0.975)
    return out


def _bootstrap_delta_auc(
    scored_a: Sequence[ScoredRow],
    scored_b: Sequence[ScoredRow],
    *,
    variant: str,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    map_a = {(r.trace_id, r.variant, r.label): float(r.score) for r in scored_a if str(r.variant) == str(variant)}
    map_b = {(r.trace_id, r.variant, r.label): float(r.score) for r in scored_b if str(r.variant) == str(variant)}
    keys = sorted(set(map_a.keys()).intersection(set(map_b.keys())))
    if not keys:
        return {
            "defined": False,
            "delta_observed": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "bootstrap_n": int(n_bootstrap),
            "bootstrap_effective_n": 0,
        }

    pooled_a = [(map_a[k], int(k[2])) for k in keys]
    pooled_b = [(map_b[k], int(k[2])) for k in keys]
    auc_a = _roc_auc(pooled_a)
    auc_b = _roc_auc(pooled_b)
    if auc_a is None or auc_b is None:
        return {
            "defined": False,
            "delta_observed": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "bootstrap_n": int(n_bootstrap),
            "bootstrap_effective_n": 0,
        }
    delta_obs = float(auc_a - auc_b)

    by_trace: Dict[str, List[Tuple[float, float, int]]] = {}
    for k in keys:
        by_trace.setdefault(str(k[0]), []).append((map_a[k], map_b[k], int(k[2])))
    valid_tids = sorted(t for t, rows in by_trace.items() if sum(y for _, _, y in rows) > 0 and sum(1 - y for _, _, y in rows) > 0)
    if not valid_tids:
        return {
            "defined": True,
            "delta_observed": delta_obs,
            "ci95_lower": None,
            "ci95_upper": None,
            "bootstrap_n": int(n_bootstrap),
            "bootstrap_effective_n": 0,
        }

    rng = random.Random(int(seed))
    vals: List[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        sample: List[Tuple[float, float, int]] = []
        for __ in range(len(valid_tids)):
            tid = valid_tids[rng.randrange(len(valid_tids))]
            sample.extend(by_trace[tid])
        auc_sa = _roc_auc([(a, y) for a, _, y in sample])
        auc_sb = _roc_auc([(b, y) for _, b, y in sample])
        if auc_sa is not None and auc_sb is not None:
            vals.append(float(auc_sa - auc_sb))
    vals = sorted(vals)
    return {
        "defined": True,
        "delta_observed": delta_obs,
        "ci95_lower": _percentile(vals, 0.025),
        "ci95_upper": _percentile(vals, 0.975),
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_effective_n": int(len(vals)),
    }


def _build_trace_folds(trace_ids: Sequence[str], *, k: int, seed: int) -> List[List[str]]:
    if int(k) <= 1:
        raise ValueError("k must be >= 2")
    tids = list(sorted(set(str(x) for x in trace_ids)))
    rng = random.Random(int(seed))
    rng.shuffle(tids)
    kk = min(int(k), len(tids))
    folds: List[List[str]] = [[] for _ in range(kk)]
    for i, tid in enumerate(tids):
        folds[i % kk].append(tid)
    return [f for f in folds if f]


def _split_by_trace(
    rows: Sequence[Row], *, test_fraction: float, seed: int, max_tries: int = 200
) -> Tuple[List[Row], List[Row], Dict[str, Any]]:
    trace_ids = sorted({r.trace_id for r in rows})
    if len(trace_ids) < 2:
        raise RuntimeError("Need at least 2 traces for train/test split")
    n_test = max(1, int(round(len(trace_ids) * float(test_fraction))))
    n_test = min(n_test, len(trace_ids) - 1)

    for attempt in range(max_tries):
        rng = random.Random(int(seed) + attempt)
        t = list(trace_ids)
        rng.shuffle(t)
        test_set = set(t[:n_test])
        train_rows = [r for r in rows if r.trace_id not in test_set]
        test_rows = [r for r in rows if r.trace_id in test_set]
        train_pos = sum(r.label for r in train_rows)
        train_neg = len(train_rows) - train_pos
        test_pos = sum(r.label for r in test_rows)
        test_neg = len(test_rows) - test_pos
        if train_pos > 0 and train_neg > 0 and test_pos > 0 and test_neg > 0:
            return train_rows, test_rows, {
                "trace_split_seed_used": int(seed) + attempt,
                "trace_count_total": int(len(trace_ids)),
                "trace_count_train": int(len(trace_ids) - len(test_set)),
                "trace_count_test": int(len(test_set)),
                "train_rows": int(len(train_rows)),
                "test_rows": int(len(test_rows)),
                "train_pos": int(train_pos),
                "train_neg": int(train_neg),
                "test_pos": int(test_pos),
                "test_neg": int(test_neg),
                "trace_overlap_count": 0,
            }
    raise RuntimeError("Could not produce class-balanced trace-disjoint split")


def _train_logreg(
    train_rows: Sequence[Row],
    test_rows: Sequence[Row],
    *,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, Any]:
    if not train_rows or not test_rows:
        raise RuntimeError("Empty train/test rows")

    d = len(train_rows[0].features)
    x_train = torch.tensor([r.features for r in train_rows], dtype=torch.float32)
    y_train = torch.tensor([r.label for r in train_rows], dtype=torch.float32)
    x_test = torch.tensor([r.features for r in test_rows], dtype=torch.float32)

    mu = x_train.mean(dim=0)
    sigma = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma

    dev = torch.device(device)
    x_train = x_train.to(dev)
    y_train = y_train.to(dev)
    x_test = x_test.to(dev)

    w = torch.zeros((d, 1), dtype=torch.float32, device=dev, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=float(lr), weight_decay=float(weight_decay))

    losses: List[float] = []
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = x_train @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), y_train)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        test_logits = (x_test @ w + b).view(-1)
        test_probs = torch.sigmoid(test_logits).detach().cpu().tolist()
        train_probs = torch.sigmoid((x_train @ w + b).view(-1)).detach().cpu().tolist()

    y_test_cpu = [int(r.label) for r in test_rows]
    y_train_cpu = [int(r.label) for r in train_rows]
    test_auc = _roc_auc(list(zip(test_probs, y_test_cpu)))
    train_auc = _roc_auc(list(zip(train_probs, y_train_cpu)))

    coef = w.detach().cpu().view(-1).tolist()
    return {
        "train_auroc": train_auc,
        "test_auroc": test_auc,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "coefficients": [float(c) for c in coef],
        "intercept": float(b.detach().cpu().item()),
        "test_probabilities": [float(x) for x in test_probs],
        "train_probabilities": [float(x) for x in train_probs],
    }


def _rows_with_feature_subset(rows: Sequence[Row], feature_indices: Sequence[int]) -> List[Row]:
    idx = [int(i) for i in feature_indices]
    out: List[Row] = []
    for r in rows:
        out.append(
            Row(
                trace_id=str(r.trace_id),
                variant=str(r.variant),
                label=int(r.label),
                features=[float(r.features[i]) for i in idx],
            )
        )
    return out


def _evaluate_model(
    model_name: str,
    rows_master: Sequence[Row],
    feature_names_master: Sequence[str],
    model_feature_names: Sequence[str],
    *,
    train_trace_set: set[str],
    test_trace_set: set[str],
    excluded_variants: Sequence[str],
    args: argparse.Namespace,
    cv_folds: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    feature_to_idx = {str(n): i for i, n in enumerate(feature_names_master)}
    missing = [fn for fn in model_feature_names if fn not in feature_to_idx]
    if missing:
        return {
            "status": "blocked_missing_features",
            "missing_features": missing[:20],
            "feature_count": int(len(model_feature_names)),
        }

    model_idx = [feature_to_idx[fn] for fn in model_feature_names]
    rows = _rows_with_feature_subset(rows_master, model_idx)

    train_rows = [r for r in rows if str(r.trace_id) in train_trace_set]
    test_rows = [r for r in rows if str(r.trace_id) in test_trace_set]
    split_diag = {
        "trace_count_total": int(len(train_trace_set) + len(test_trace_set)),
        "trace_count_train": int(len(train_trace_set)),
        "trace_count_test": int(len(test_trace_set)),
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "train_pos": int(sum(r.label for r in train_rows)),
        "train_neg": int(len(train_rows) - sum(r.label for r in train_rows)),
        "test_pos": int(sum(r.label for r in test_rows)),
        "test_neg": int(len(test_rows) - sum(r.label for r in test_rows)),
        "trace_overlap_count": int(len(train_trace_set.intersection(test_trace_set))),
    }

    train_pre = _summarize_rows(train_rows)
    train_rows_effective, dropped_by_variant = _apply_train_variant_exclusion(train_rows, excluded_variants)
    train_post = _summarize_rows(train_rows_effective)
    test_cov = _summarize_rows(test_rows)
    exclusion_diag = {
        "train_counts_pre": train_pre,
        "train_counts_post": train_post,
        "excluded_unfaithful_rows_by_variant": dropped_by_variant,
        "excluded_unfaithful_rows_total": int(sum(dropped_by_variant.values())),
        "test_variant_coverage": test_cov.get("by_variant", {}),
    }

    if int(train_post["pos"]) <= 0 or int(train_post["neg"]) <= 0:
        return {
            "status": "blocked_invalid_train_after_exclusion",
            "split_diagnostics": split_diag,
            "train_exclusion_diagnostics": exclusion_diag,
            "wrong_intermediate_probe_auroc": None,
            "robust_wrong_intermediate_gate_pass": False,
            "feature_count": int(len(model_feature_names)),
        }

    train_result = _train_logreg(
        train_rows_effective,
        test_rows,
        device=str(args.device),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    # Legacy baseline only uses coherence metrics when available.
    legacy_idx = [i for i, fn in enumerate(model_feature_names) if fn.split(":", 1)[-1] in METRIC_KEYS]
    base_scores = []
    if legacy_idx:
        for r in test_rows:
            legacy_vals = [r.features[i] for i in legacy_idx]
            inv_mean = 1.0 - float(sum(legacy_vals) / max(1, len(legacy_vals)))
            base_scores.append((inv_mean, int(r.label), r.variant))
    baseline_auc = _roc_auc([(s, y) for s, y, _ in base_scores]) if base_scores else None

    by_variant_baseline: Dict[str, Any] = {}
    by_variant_probe: Dict[str, Any] = {}
    test_probs = list(train_result.get("test_probabilities", []))
    probe_scored_rows = [
        ScoredRow(
            trace_id=str(r.trace_id),
            variant=str(r.variant),
            label=int(r.label),
            score=float(p),
        )
        for p, r in zip(test_probs, test_rows)
    ]
    probe_scores = [(r.score, r.label, r.variant) for r in probe_scored_rows]
    wrong_single_rows = [r for r in probe_scored_rows if str(r.variant) == "wrong_intermediate"]
    wrong_single_boot = _bootstrap_trace_pair_auc(
        wrong_single_rows,
        n_bootstrap=int(args.wrong_intermediate_bootstrap_n),
        seed=int(args.wrong_intermediate_bootstrap_seed),
    )
    wrong_single_ci_support = bool(
        isinstance(wrong_single_boot.get("ci95_upper"), (int, float))
        and float(wrong_single_boot.get("ci95_upper")) >= float(args.require_wrong_intermediate_auroc)
    )
    variants = sorted({r.variant for r in test_rows})
    for v in variants:
        sr = [(s, y) for s, y, vv in base_scores if vv == v]
        pr = [(s, y) for s, y, vv in probe_scores if vv == v]
        pos = sum(y for _, y in pr)
        neg = len(pr) - pos
        by_variant_baseline[v] = {
            "baseline_mean_inverted_auroc": _roc_auc(sr) if sr else None,
            "count": int(len(sr)),
            "pos": int(sum(y for _, y in sr)),
            "neg": int(len(sr) - sum(y for _, y in sr)),
        }
        by_variant_probe[v] = {
            "probe_auroc": _roc_auc(pr),
            "count": int(len(pr)),
            "pos": int(pos),
            "neg": int(neg),
        }

    coef_pairs = sorted(
        zip(model_feature_names, train_result.get("coefficients", [])),
        key=lambda x: abs(float(x[1])),
        reverse=True,
    )

    wrong_intermediate_probe_auroc = (by_variant_probe.get("wrong_intermediate", {}) or {}).get("probe_auroc")
    robust_pass = bool(
        isinstance(wrong_intermediate_probe_auroc, (int, float))
        and float(wrong_intermediate_probe_auroc) > float(args.require_wrong_intermediate_auroc)
    )

    cv_fold_rows: List[Dict[str, Any]] = []
    cv_wrong_oof: List[ScoredRow] = []
    cv_fold_aurocs: List[float] = []
    cv_trace_overlap_count = 0
    cv_valid_fold_count = 0

    for fi, fold_test_ids in enumerate(cv_folds):
        fold_test_set = set(str(x) for x in fold_test_ids)
        fold_train = [r for r in rows if str(r.trace_id) not in fold_test_set]
        fold_test = [r for r in rows if str(r.trace_id) in fold_test_set]
        overlap = len(set(str(r.trace_id) for r in fold_train).intersection(fold_test_set))
        cv_trace_overlap_count += int(overlap)
        fold_train_eff, fold_dropped = _apply_train_variant_exclusion(fold_train, excluded_variants)
        fold_train_sum = _summarize_rows(fold_train_eff)
        fold_test_sum = _summarize_rows(fold_test)
        fold_entry: Dict[str, Any] = {
            "fold_index": int(fi),
            "test_trace_count": int(len(fold_test_set)),
            "trace_overlap_count": int(overlap),
            "train_counts_post": fold_train_sum,
            "test_counts": fold_test_sum,
            "excluded_unfaithful_rows_by_variant": fold_dropped,
            "wrong_intermediate_auroc": None,
            "wrong_intermediate_count": 0,
            "wrong_intermediate_pos": 0,
            "wrong_intermediate_neg": 0,
            "status": "ok",
        }
        if int(fold_train_sum["pos"]) <= 0 or int(fold_train_sum["neg"]) <= 0:
            fold_entry["status"] = "blocked_invalid_train_after_exclusion"
            cv_fold_rows.append(fold_entry)
            continue
        fold_result = _train_logreg(
            fold_train_eff,
            fold_test,
            device=str(args.device),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        fold_probs = list(fold_result.get("test_probabilities", []))
        fold_scored = [
            ScoredRow(
                trace_id=str(r.trace_id),
                variant=str(r.variant),
                label=int(r.label),
                score=float(p),
            )
            for p, r in zip(fold_probs, fold_test)
        ]
        fold_wrong = [r for r in fold_scored if str(r.variant) == "wrong_intermediate"]
        fold_entry["wrong_intermediate_count"] = int(len(fold_wrong))
        fold_entry["wrong_intermediate_pos"] = int(sum(int(r.label) for r in fold_wrong))
        fold_entry["wrong_intermediate_neg"] = int(len(fold_wrong) - sum(int(r.label) for r in fold_wrong))
        fold_auc = _roc_auc([(float(r.score), int(r.label)) for r in fold_wrong])
        fold_entry["wrong_intermediate_auroc"] = fold_auc
        if isinstance(fold_auc, (int, float)):
            cv_valid_fold_count += 1
            cv_fold_aurocs.append(float(fold_auc))
            cv_wrong_oof.extend(fold_wrong)
        else:
            fold_entry["status"] = "blocked_wrong_intermediate_undefined"
        cv_fold_rows.append(fold_entry)

    cv_pooled_auc = _roc_auc([(float(r.score), int(r.label)) for r in cv_wrong_oof])
    cv_pooled_boot = _bootstrap_trace_pair_auc(
        cv_wrong_oof,
        n_bootstrap=int(args.wrong_intermediate_bootstrap_n),
        seed=int(args.wrong_intermediate_bootstrap_seed) + 101,
    )
    cv_mean = (sum(cv_fold_aurocs) / len(cv_fold_aurocs)) if cv_fold_aurocs else None
    cv_std = None
    if cv_fold_aurocs and len(cv_fold_aurocs) > 1:
        mean_val = float(cv_mean)
        cv_std = float(
            (sum((x - mean_val) ** 2 for x in cv_fold_aurocs) / (len(cv_fold_aurocs) - 1)) ** 0.5
        )
    cv_valid_enough = bool(cv_valid_fold_count >= int(args.cv_min_valid_folds))
    cv_gate_pass_pooled = bool(
        cv_valid_enough
        and isinstance(cv_pooled_auc, (int, float))
        and float(cv_pooled_auc) > float(args.require_wrong_intermediate_auroc)
    )
    cv_ci_upper = cv_pooled_boot.get("ci95_upper")
    cv_evidence_strength = (
        "supports_threshold"
        if cv_valid_enough
        and isinstance(cv_ci_upper, (int, float))
        and float(cv_ci_upper) >= float(args.require_wrong_intermediate_auroc)
        else "not_supported"
    )
    single_split_gate_within_noise = bool((not robust_pass) and wrong_single_ci_support)

    return {
        "status": "ok",
        "feature_count": int(len(model_feature_names)),
        "feature_names": [str(x) for x in model_feature_names],
        "split_diagnostics": split_diag,
        "train_exclusion_diagnostics": exclusion_diag,
        "probe": {
            "method": "logistic_regression_linear",
            "device": str(args.device),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            **train_result,
        },
        "baseline": {
            "method": "mean_inverted_coherence" if base_scores else "undefined_no_legacy_metrics",
            "test_auroc": baseline_auc,
            "by_variant": by_variant_baseline,
        },
        "probe_by_variant": by_variant_probe,
        "confound_check": {
            "probe_wrong_intermediate_auroc": (by_variant_probe.get("wrong_intermediate", {}) or {}).get("probe_auroc"),
            "probe_order_flip_only_auroc": (by_variant_probe.get("order_flip_only", {}) or {}).get("probe_auroc"),
            "baseline_wrong_intermediate_auroc": (by_variant_baseline.get("wrong_intermediate", {}) or {}).get(
                "baseline_mean_inverted_auroc"
            ),
            "baseline_order_flip_only_auroc": (by_variant_baseline.get("order_flip_only", {}) or {}).get(
                "baseline_mean_inverted_auroc"
            ),
        },
        "wrong_intermediate_probe_auroc": wrong_intermediate_probe_auroc,
        "wrong_intermediate_probe_auroc_ci95": {
            "lower": wrong_single_boot.get("ci95_lower"),
            "upper": wrong_single_boot.get("ci95_upper"),
            "bootstrap_n": int(args.wrong_intermediate_bootstrap_n),
            "bootstrap_effective_n": int(wrong_single_boot.get("bootstrap_effective_n", 0)),
            "trace_pair_count": int(wrong_single_boot.get("valid_trace_pair_count", 0)),
            "defined": bool(wrong_single_boot.get("defined", False)),
        },
        "wrong_intermediate_ci_supports_threshold": wrong_single_ci_support,
        "robust_wrong_intermediate_gate_pass": robust_pass,
        "single_split_gate_within_noise": single_split_gate_within_noise,
        "cv_diagnostics": {
            "cv_folds": int(len(cv_folds)),
            "cv_seed": int(args.cv_seed),
            "cv_min_valid_folds": int(args.cv_min_valid_folds),
            "cv_valid_fold_count": int(cv_valid_fold_count),
            "cv_fold_rows": cv_fold_rows,
            "cv_wrong_intermediate_fold_aurocs": [float(x) for x in cv_fold_aurocs],
            "cv_wrong_intermediate_mean_auroc": cv_mean,
            "cv_wrong_intermediate_std_auroc": cv_std,
            "cv_wrong_intermediate_pooled_auroc": cv_pooled_auc,
            "cv_wrong_intermediate_pooled_ci95": {
                "lower": cv_pooled_boot.get("ci95_lower"),
                "upper": cv_pooled_boot.get("ci95_upper"),
                "bootstrap_n": int(args.wrong_intermediate_bootstrap_n),
                "bootstrap_effective_n": int(cv_pooled_boot.get("bootstrap_effective_n", 0)),
                "trace_pair_count": int(cv_pooled_boot.get("valid_trace_pair_count", 0)),
                "defined": bool(cv_pooled_boot.get("defined", False)),
            },
            "cv_trace_overlap_count": int(cv_trace_overlap_count),
            "cv_valid_enough": cv_valid_enough,
        },
        "cv_wrong_intermediate_gate_pass_pooled": cv_gate_pass_pooled,
        "cv_evidence_strength": cv_evidence_strength,
        "top_abs_coefficients": [
            {"feature": str(name), "coef": float(val)} for name, val in coef_pairs[:15]
        ],
        "test_scored_rows": [
            {
                "trace_id": str(r.trace_id),
                "variant": str(r.variant),
                "label": int(r.label),
                "score": float(r.score),
            }
            for r in probe_scored_rows
        ],
    }


def _build_sample_maps(partials: Sequence[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], List[int], List[str], List[str]]:
    layer_to_partial: Dict[int, Dict[str, Any]] = {}
    source_records = str(partials[0].get("source_control_records", ""))
    sampled_ref = list(partials[0].get("coverage_diagnostics", {}).get("trace_ids_sampled", []))

    for p in partials:
        if str(p.get("status")) != "ok":
            raise RuntimeError(f"Partial not ok: {p.get('status')!r}")
        if str(p.get("source_control_records", "")) != source_records:
            raise RuntimeError("Source control records mismatch")
        if list(p.get("coverage_diagnostics", {}).get("trace_ids_sampled", [])) != sampled_ref:
            raise RuntimeError("Sampled trace IDs mismatch across partials")
        if not isinstance(p.get("sample_metrics"), list):
            raise RuntimeError("Partial missing sample_metrics; rerun with --emit-samples")
        layer_to_partial[int(p.get("layer"))] = p

    layers = sorted(layer_to_partial.keys())
    sample_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    metric_names_by_layer: Dict[int, set[str]] = {int(layer): set() for layer in layers}

    for layer in layers:
        p = layer_to_partial[layer]
        for s in p.get("sample_metrics", []):
            trace_id = str(s.get("trace_id", ""))
            variant = str(s.get("variant", ""))
            label_str = str(s.get("label", ""))
            if label_str not in {"faithful", "unfaithful"}:
                continue
            key = (trace_id, variant, label_str)
            rec = sample_map.setdefault(key, {})
            met = s.get("metrics", {}) if isinstance(s, dict) else {}
            if not isinstance(met, dict):
                continue
            for mk, mv in met.items():
                metric_names_by_layer[layer].add(str(mk))
                rec[f"layer{layer}:{mk}"] = mv

    feature_names_all: List[str] = []
    for layer in layers:
        for mk in sorted(metric_names_by_layer[layer]):
            feature_names_all.append(f"layer{layer}:{mk}")

    return sample_map, layers, feature_names_all, sampled_ref


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partials", nargs="+", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--run-tag", default="")
    p.add_argument("--trace-test-fraction", type=float, default=0.20)
    p.add_argument("--trace-split-seed", type=int, default=20260306)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--train-exclude-variants",
        default="",
        help="CSV of unfaithful variants to exclude from TRAINING positives only.",
    )
    p.add_argument(
        "--require-wrong-intermediate-auroc",
        type=float,
        default=0.70,
        help="Gate threshold for wrong_intermediate probe AUROC.",
    )
    p.add_argument("--wrong-intermediate-bootstrap-n", type=int, default=1000)
    p.add_argument("--wrong-intermediate-bootstrap-seed", type=int, default=20260307)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=20260307)
    p.add_argument("--cv-min-valid-folds", type=int, default=3)
    p.add_argument(
        "--model-ladder",
        default="sae_only,hybrid_only,mixed",
        help="CSV subset of sae_only,hybrid_only,mixed.",
    )
    p.add_argument(
        "--mixed-delta-effect-floor",
        type=float,
        default=0.03,
        help="Practical effect floor for mixed-vs-sae gain.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    partials = [load_json(p) for p in args.partials]
    if not partials:
        raise RuntimeError("No partials provided")

    run_tag = str(args.run_tag).strip() or str(partials[0].get("run_tag", "phase7_sae_trajectory_pathc"))
    source_records = str(partials[0].get("source_control_records", ""))

    sample_map, layers, feature_names_all, sampled_ref = _build_sample_maps(partials)

    feature_names_sae = [fn for fn in feature_names_all if ":hybrid_" not in fn]
    feature_names_hybrid = [fn for fn in feature_names_all if ":hybrid_" in fn]
    feature_names_mixed = sorted(set(feature_names_sae + feature_names_hybrid), key=lambda x: feature_names_all.index(x))

    requested_models = _parse_model_ladder_csv(args.model_ladder)
    model_features: Dict[str, List[str]] = {
        "sae_only": feature_names_sae,
        "hybrid_only": feature_names_hybrid,
        "mixed": feature_names_mixed,
    }
    # Master rows use union of requested model features to keep split identical across models.
    master_feature_names: List[str] = []
    seen_master = set()
    for m in requested_models:
        for fn in model_features.get(m, []):
            if fn in seen_master:
                continue
            seen_master.add(fn)
            master_feature_names.append(fn)

    rows_master: List[Row] = []
    dropped_missing = 0
    for (trace_id, variant, label_str), rec in sample_map.items():
        vals: List[float] = []
        ok = True
        for fn in master_feature_names:
            v = rec.get(fn)
            if not isinstance(v, (int, float)):
                ok = False
                break
            vals.append(float(v))
        if not ok:
            dropped_missing += 1
            continue
        rows_master.append(
            Row(
                trace_id=trace_id,
                variant=variant,
                label=(1 if label_str == "unfaithful" else 0),
                features=vals,
            )
        )

    if len(rows_master) < 20:
        out = {
            "schema_version": "phase7_sae_trajectory_pathc_v3",
            "status": "blocked_insufficient_rows",
            "run_tag": run_tag,
            "rows_after_join": int(len(rows_master)),
            "rows_dropped_missing": int(dropped_missing),
            "layers": layers,
            "feature_names_available": feature_names_all,
            "feature_names_master": master_feature_names,
            "timestamp": datetime.now().isoformat(),
        }
        save_json(args.output_json, out)
        Path(args.output_md).write_text("# Path C blocked\n\nInsufficient joined rows.\n")
        print(f"Saved blocked output -> {args.output_json}")
        return

    train_rows_master, test_rows_master, split_diag = _split_by_trace(
        rows_master,
        test_fraction=float(args.trace_test_fraction),
        seed=int(args.trace_split_seed),
    )
    train_trace_set = {str(r.trace_id) for r in train_rows_master}
    test_trace_set = {str(r.trace_id) for r in test_rows_master}

    all_trace_ids = sorted({str(r.trace_id) for r in rows_master})
    cv_folds = _build_trace_folds(all_trace_ids, k=int(args.cv_folds), seed=int(args.cv_seed))

    excluded_variants = _parse_variant_csv(args.train_exclude_variants)
    model_results: Dict[str, Any] = {}
    available_model_order: List[str] = []

    for model_name in requested_models:
        fns = model_features.get(model_name, [])
        if not fns:
            model_results[model_name] = {
                "status": "blocked_missing_features",
                "missing_features": [f"{model_name}:no_features_detected"],
                "feature_count": 0,
            }
            continue
        res = _evaluate_model(
            model_name,
            rows_master,
            master_feature_names,
            fns,
            train_trace_set=train_trace_set,
            test_trace_set=test_trace_set,
            excluded_variants=excluded_variants,
            args=args,
            cv_folds=cv_folds,
        )
        model_results[model_name] = res
        if str(res.get("status")) == "ok":
            available_model_order.append(model_name)

    if not available_model_order:
        out = {
            "schema_version": "phase7_sae_trajectory_pathc_v3",
            "status": "blocked_all_models_unavailable",
            "run_tag": run_tag,
            "source_control_records": source_records,
            "layers": layers,
            "requested_models": requested_models,
            "model_results": model_results,
            "rows_after_join": int(len(rows_master)),
            "rows_dropped_missing": int(dropped_missing),
            "split_diagnostics": split_diag,
            "timestamp": datetime.now().isoformat(),
        }
        save_json(args.output_json, out)
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("# Phase 7-SAE Path C\n\n- Status: `blocked_all_models_unavailable`\n")
        print(f"Saved blocked Path C JSON -> {args.output_json}")
        print(f"Saved blocked Path C MD   -> {args.output_md}")
        return

    primary_model = "mixed" if "mixed" in available_model_order else available_model_order[0]
    primary = model_results[primary_model]

    wrong_intermediate_auroc_by_model = {
        m: (model_results.get(m, {}) or {}).get("wrong_intermediate_probe_auroc")
        for m in requested_models
    }

    delta_m3_vs_m1 = {
        "defined": False,
        "delta_observed": None,
        "ci95_lower": None,
        "ci95_upper": None,
        "bootstrap_n": int(args.wrong_intermediate_bootstrap_n),
        "bootstrap_effective_n": 0,
    }
    if str(model_results.get("mixed", {}).get("status")) == "ok" and str(model_results.get("sae_only", {}).get("status")) == "ok":
        m3_rows = [ScoredRow(**r) for r in model_results["mixed"].get("test_scored_rows", [])]
        m1_rows = [ScoredRow(**r) for r in model_results["sae_only"].get("test_scored_rows", [])]
        delta_m3_vs_m1 = _bootstrap_delta_auc(
            m3_rows,
            m1_rows,
            variant="wrong_intermediate",
            n_bootstrap=int(args.wrong_intermediate_bootstrap_n),
            seed=int(args.wrong_intermediate_bootstrap_seed) + 503,
        )

    mixed_cv = ((model_results.get("mixed", {}) or {}).get("cv_diagnostics", {}) or {}).get("cv_wrong_intermediate_pooled_auroc")
    pass_publishable_threshold = bool(isinstance(mixed_cv, (int, float)) and float(mixed_cv) > float(args.require_wrong_intermediate_auroc))

    delta_obs = delta_m3_vs_m1.get("delta_observed")
    delta_lo = delta_m3_vs_m1.get("ci95_lower")
    mixed_adds_independent_signal = bool(
        isinstance(delta_obs, (int, float))
        and isinstance(delta_lo, (int, float))
        and float(delta_obs) >= float(args.mixed_delta_effect_floor)
        and float(delta_lo) > 0.0
    )

    blocked_reason = None
    if not pass_publishable_threshold:
        blocked_reason = "mixed_wrong_intermediate_below_threshold"
    elif not mixed_adds_independent_signal:
        blocked_reason = "mixed_gain_vs_sae_not_stable_or_too_small"

    fold_wise = {}
    for m in requested_models:
        cvd = (model_results.get(m, {}) or {}).get("cv_diagnostics", {}) or {}
        fold_wise[m] = {
            "valid_fold_count": cvd.get("cv_valid_fold_count"),
            "fold_aurocs": cvd.get("cv_wrong_intermediate_fold_aurocs"),
            "mean": cvd.get("cv_wrong_intermediate_mean_auroc"),
            "std": cvd.get("cv_wrong_intermediate_std_auroc"),
            "pooled": cvd.get("cv_wrong_intermediate_pooled_auroc"),
        }

    result = {
        "schema_version": "phase7_sae_trajectory_pathc_v3",
        "status": "ok",
        "run_tag": run_tag,
        "source_control_records": source_records,
        "layers": layers,
        "feature_names_available": feature_names_all,
        "feature_names_master": master_feature_names,
        "feature_name_blocks": {
            "sae_only": feature_names_sae,
            "hybrid_only": feature_names_hybrid,
            "mixed": feature_names_mixed,
        },
        "rows_after_join": int(len(rows_master)),
        "rows_dropped_missing": int(dropped_missing),
        "sampled_trace_ids": sampled_ref,
        "split_diagnostics": split_diag,
        "train_exclusion_policy": {
            "excluded_train_positive_variants": excluded_variants,
            "require_wrong_intermediate_auroc": float(args.require_wrong_intermediate_auroc),
            "model_ladder": requested_models,
            "mixed_delta_effect_floor": float(args.mixed_delta_effect_floor),
        },
        "model_ladder": model_results,
        "primary_model": primary_model,
        "wrong_intermediate_auroc_by_model": wrong_intermediate_auroc_by_model,
        "delta_m3_vs_m1": delta_m3_vs_m1,
        "fold_wise_stability": fold_wise,
        "pass_publishable_threshold": pass_publishable_threshold,
        "mixed_adds_independent_signal": mixed_adds_independent_signal,
        "blocked_reason": blocked_reason,
        # Legacy compatibility aliases (primary model view)
        "train_exclusion_diagnostics": primary.get("train_exclusion_diagnostics"),
        "probe": primary.get("probe"),
        "baseline": primary.get("baseline"),
        "probe_by_variant": primary.get("probe_by_variant"),
        "confound_check": primary.get("confound_check"),
        "wrong_intermediate_probe_auroc": primary.get("wrong_intermediate_probe_auroc"),
        "wrong_intermediate_probe_auroc_ci95": primary.get("wrong_intermediate_probe_auroc_ci95"),
        "wrong_intermediate_ci_supports_threshold": primary.get("wrong_intermediate_ci_supports_threshold"),
        "robust_wrong_intermediate_gate_pass": primary.get("robust_wrong_intermediate_gate_pass"),
        "single_split_gate_within_noise": primary.get("single_split_gate_within_noise"),
        "cv_diagnostics": primary.get("cv_diagnostics"),
        "cv_wrong_intermediate_gate_pass_pooled": primary.get("cv_wrong_intermediate_gate_pass_pooled"),
        "cv_evidence_strength": primary.get("cv_evidence_strength"),
        "top_abs_coefficients": primary.get("top_abs_coefficients"),
        "timestamp": datetime.now().isoformat(),
    }

    md_lines = [
        "# Phase 7-SAE Path C (Model Ladder)",
        "",
        f"- Run tag: `{run_tag}`",
        f"- Layers: `{','.join(str(x) for x in layers)}`",
        f"- Rows after join: `{result['rows_after_join']}`",
        f"- Primary model: `{primary_model}`",
        f"- Publishable threshold pass (mixed CV > {args.require_wrong_intermediate_auroc}): `{pass_publishable_threshold}`",
        f"- Mixed adds independent signal: `{mixed_adds_independent_signal}`",
        f"- Blocked reason: `{blocked_reason}`",
        "",
        "## Wrong-Intermediate AUROC by Model",
        "",
    ]
    for m in requested_models:
        md_lines.append(f"- `{m}`: `{wrong_intermediate_auroc_by_model.get(m)}`")
    md_lines.extend(
        [
            "",
            "## Mixed vs SAE Delta",
            "",
            f"- Delta observed: `{delta_m3_vs_m1.get('delta_observed')}`",
            f"- CI95: `{delta_m3_vs_m1.get('ci95_lower')}` .. `{delta_m3_vs_m1.get('ci95_upper')}`",
            "",
            "## Top Coefficients (Primary)",
            "",
        ]
    )
    for row in (result.get("top_abs_coefficients") or [])[:15]:
        md_lines.append(f"- `{row['feature']}`: `{row['coef']}`")
    md_lines.append("")

    save_json(args.output_json, result)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Saved Path C JSON -> {args.output_json}")
    print(f"Saved Path C MD   -> {args.output_md}")


if __name__ == "__main__":
    main()

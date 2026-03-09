#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover
    from phase7.common import load_json, save_json
    from phase7 import aggregate_sae_trajectory_pathc as apc
except ImportError:  # pragma: no cover
    from common import load_json, save_json
    import aggregate_sae_trajectory_pathc as apc

TRAJECTORY_METRICS = {
    "cosine_smoothness",
    "feature_variance_coherence",
    "magnitude_monotonicity_coherence",
}
STEP_ANOMALY_METRICS = {
    "max_delta_l2",
    "max_delta_l2_norm",
    "max_delta_cosine",
    "p95_delta_l2",
    "p95_delta_cosine",
    "top2_mean_delta_l2",
    "argmax_step_idx_norm",
}

FEATURE_MODE_CHOICES = ("full_sae", "trajectory_only", "step_anomaly_only", "single_best_layer")


def _parse_csv(value: str) -> List[str]:
    out: List[str] = []
    for tok in str(value or "").split(","):
        t = tok.strip()
        if t and t not in out:
            out.append(t)
    return out


def _parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    for tok in str(value or "").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _summ(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    vals_sorted = sorted(vals)
    mean = sum(vals_sorted) / len(vals_sorted)
    if len(vals_sorted) > 1:
        var = sum((v - mean) ** 2 for v in vals_sorted) / (len(vals_sorted) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    p95 = apc._percentile(vals_sorted, 0.95)
    return {
        "count": int(len(vals_sorted)),
        "mean": float(mean),
        "std": float(std),
        "p95": float(p95) if isinstance(p95, (int, float)) else None,
        "min": float(vals_sorted[0]),
        "max": float(vals_sorted[-1]),
    }


def _load_rows_and_features(partials: Sequence[str]) -> Tuple[List[apc.Row], List[str], List[int], Dict[str, Any]]:
    payloads = [load_json(p) for p in partials]
    sample_map, layers, feature_names_all, sampled_ref = apc._build_sample_maps(payloads)

    feature_names_sae = [fn for fn in feature_names_all if ":hybrid_" not in fn]
    rows_master: List[apc.Row] = []
    dropped_missing = 0
    for (trace_id, variant, label_str), rec in sample_map.items():
        vals: List[float] = []
        ok = True
        for fn in feature_names_sae:
            v = rec.get(fn)
            if not isinstance(v, (int, float)):
                ok = False
                break
            vals.append(float(v))
        if not ok:
            dropped_missing += 1
            continue
        rows_master.append(
            apc.Row(
                trace_id=str(trace_id),
                variant=str(variant),
                label=(1 if str(label_str) == "unfaithful" else 0),
                features=vals,
            )
        )
    meta = {
        "layers": [int(x) for x in layers],
        "sampled_trace_ids": list(sampled_ref),
        "rows_after_join": int(len(rows_master)),
        "rows_dropped_missing": int(dropped_missing),
    }
    return rows_master, feature_names_sae, layers, meta


def _feature_indices(feature_names: Sequence[str], keep_names: Sequence[str]) -> List[int]:
    idx_map = {str(n): i for i, n in enumerate(feature_names)}
    return [int(idx_map[n]) for n in keep_names if n in idx_map]


def _subset_rows(rows: Sequence[apc.Row], idx: Sequence[int]) -> List[apc.Row]:
    return apc._rows_with_feature_subset(rows, idx)


def _resolve_feature_names_for_mode(
    feature_names: Sequence[str],
    *,
    partial_paths: Sequence[str],
    mode: str,
    single_layer_source: str,
    single_layer_id: Optional[int],
) -> Tuple[List[str], Dict[str, Any]]:
    mode_norm = str(mode or "full_sae").strip()
    if mode_norm not in FEATURE_MODE_CHOICES:
        raise RuntimeError(f"Unsupported --feature-mode {mode_norm!r}; expected one of {FEATURE_MODE_CHOICES}")

    traj_names = [n for n in feature_names if n.split(":", 1)[-1] in TRAJECTORY_METRICS]
    step_names = [n for n in feature_names if n.split(":", 1)[-1] in STEP_ANOMALY_METRICS]
    full_names = list(feature_names)

    chosen_layer: Optional[int] = None
    if mode_norm == "single_best_layer":
        src = str(single_layer_source or "auto_best").strip()
        if src == "explicit":
            if single_layer_id is None:
                raise RuntimeError("--single-layer-id is required when --single-layer-source=explicit")
            chosen_layer = int(single_layer_id)
        else:
            chosen_layer = _single_best_layer_from_partials(partial_paths)
        layer_prefix = f"layer{chosen_layer}:"
        selected = [n for n in feature_names if n.startswith(layer_prefix)]
    elif mode_norm == "trajectory_only":
        selected = traj_names
    elif mode_norm == "step_anomaly_only":
        selected = step_names
    else:
        selected = full_names

    if not selected:
        raise RuntimeError(f"No features selected for feature mode: {mode_norm}")

    meta = {
        "feature_mode": mode_norm,
        "selected_layer": (int(chosen_layer) if chosen_layer is not None else None),
        "feature_count": int(len(selected)),
        "feature_names": [str(x) for x in selected],
        "feature_block_counts": {
            "full_sae": int(len(full_names)),
            "trajectory_only": int(len(traj_names)),
            "step_anomaly_only": int(len(step_names)),
        },
    }
    return selected, meta


def _train_eval_wrong(
    train_rows: Sequence[apc.Row],
    test_rows: Sequence[apc.Row],
    *,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, Any]:
    result = apc._train_logreg(
        train_rows,
        test_rows,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    probs = list(result.get("test_probabilities", []))
    scored = [
        apc.ScoredRow(
            trace_id=str(r.trace_id),
            variant=str(r.variant),
            label=int(r.label),
            score=float(p),
        )
        for p, r in zip(probs, test_rows)
    ]
    wrong = [r for r in scored if str(r.variant) == "wrong_intermediate"]
    wrong_auc = apc._roc_auc([(float(r.score), int(r.label)) for r in wrong])
    return {
        "train_auroc": result.get("train_auroc"),
        "test_auroc": result.get("test_auroc"),
        "loss_start": result.get("loss_start"),
        "loss_end": result.get("loss_end"),
        "wrong_intermediate_auroc": wrong_auc,
        "wrong_intermediate_count": int(len(wrong)),
        "wrong_intermediate_pos": int(sum(int(r.label) for r in wrong)),
        "wrong_intermediate_neg": int(len(wrong) - sum(int(r.label) for r in wrong)),
        "scored_rows": [
            {
                "trace_id": str(r.trace_id),
                "variant": str(r.variant),
                "label": int(r.label),
                "score": float(r.score),
            }
            for r in scored
        ],
    }


def _split_with_exclusion(
    rows: Sequence[apc.Row],
    *,
    seed: int,
    test_fraction: float,
    excluded_variants: Sequence[str],
) -> Dict[str, Any]:
    train_rows, test_rows, split_diag = apc._split_by_trace(rows, test_fraction=test_fraction, seed=seed)
    train_pre = apc._summarize_rows(train_rows)
    train_eff, dropped = apc._apply_train_variant_exclusion(train_rows, excluded_variants)
    train_post = apc._summarize_rows(train_eff)
    test_sum = apc._summarize_rows(test_rows)
    return {
        "train_rows": train_eff,
        "test_rows": test_rows,
        "split_diagnostics": split_diag,
        "train_exclusion_diagnostics": {
            "train_counts_pre": train_pre,
            "train_counts_post": train_post,
            "excluded_unfaithful_rows_by_variant": dropped,
            "excluded_unfaithful_rows_total": int(sum(dropped.values())),
            "test_variant_coverage": test_sum.get("by_variant", {}),
        },
    }


def _collect_wrong_scored(scored_rows: Sequence[Dict[str, Any]], seed_tag: str) -> List[apc.ScoredRow]:
    out: List[apc.ScoredRow] = []
    for r in scored_rows:
        if str(r.get("variant")) != "wrong_intermediate":
            continue
        out.append(
            apc.ScoredRow(
                trace_id=f"{seed_tag}:{r.get('trace_id')}",
                variant=str(r.get("variant")),
                label=int(r.get("label", 0)),
                score=float(r.get("score", 0.0)),
            )
        )
    return out


def _single_best_layer_from_partials(partials: Sequence[str]) -> int:
    best_layer = None
    best_auc = -1.0
    for p in partials:
        payload = load_json(p)
        layer = int(payload.get("layer"))
        vs = payload.get("variant_stratified_metrics", {}) if isinstance(payload, dict) else {}
        if not isinstance(vs, dict):
            continue
        for metric_payload in vs.values():
            if not isinstance(metric_payload, dict):
                continue
            wrong = metric_payload.get("wrong_intermediate", {})
            if not isinstance(wrong, dict):
                continue
            auc = wrong.get("auroc_unfaithful_positive")
            if isinstance(auc, (int, float)) and float(auc) > best_auc:
                best_auc = float(auc)
                best_layer = layer
    if best_layer is None:
        raise RuntimeError("Could not resolve single best layer from partials")
    return int(best_layer)


def run_permutation(args: argparse.Namespace) -> Dict[str, Any]:
    rows_master, feature_names, layers, meta = _load_rows_and_features(args.partials)
    selected_feature_names, feature_meta = _resolve_feature_names_for_mode(
        feature_names,
        partial_paths=args.partials,
        mode=str(args.feature_mode),
        single_layer_source=str(args.single_layer_source),
        single_layer_id=args.single_layer_id,
    )
    selected_idx = _feature_indices(feature_names, selected_feature_names)
    excluded = _parse_csv(args.train_exclude_variants)
    split = _split_with_exclusion(
        rows_master,
        seed=int(args.trace_split_seed),
        test_fraction=float(args.trace_test_fraction),
        excluded_variants=excluded,
    )
    train_rows = _subset_rows(split["train_rows"], selected_idx)
    test_rows = _subset_rows(split["test_rows"], selected_idx)

    if not train_rows or not test_rows:
        raise RuntimeError("Empty split rows for permutation")

    observed = _train_eval_wrong(
        train_rows,
        test_rows,
        device=str(args.device),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay_base),
    )
    observed_auc = observed.get("wrong_intermediate_auroc")

    rng = random.Random(int(args.permutation_seed))
    aucs: List[float] = []
    blocked = 0
    for i in range(int(args.permutation_runs)):
        labels = [int(r.label) for r in train_rows]
        rng.shuffle(labels)
        perm_rows: List[apc.Row] = []
        for r, y in zip(train_rows, labels):
            perm_rows.append(
                apc.Row(trace_id=str(r.trace_id), variant=str(r.variant), label=int(y), features=list(r.features))
            )
        # Ensure both classes exist after permutation (should generally hold)
        if sum(x.label for x in perm_rows) in (0, len(perm_rows)):
            blocked += 1
            continue
        out = _train_eval_wrong(
            perm_rows,
            test_rows,
            device=str(args.device),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay_base),
        )
        auc = out.get("wrong_intermediate_auroc")
        if isinstance(auc, (int, float)):
            aucs.append(float(auc))

    stats = _summ(aucs)
    legacy_pass = bool(
        isinstance(stats.get("mean"), (int, float))
        and isinstance(stats.get("p95"), (int, float))
        and isinstance(stats.get("max"), (int, float))
        and 0.45 <= float(stats["mean"]) <= 0.55
        and float(stats["p95"]) < 0.60
        and float(stats["max"]) < 0.70
    )
    empirical_p_value = None
    p_value_significant = None
    p_value_primary_pass = None
    if isinstance(observed_auc, (int, float)) and len(aucs) > 0:
        ge = sum(1 for x in aucs if float(x) >= float(observed_auc))
        empirical_p_value = float((ge + 1) / (len(aucs) + 1))
        p_value_significant = bool(empirical_p_value < 0.01)
        p_value_primary_pass = bool(p_value_significant)

    return {
        "schema_version": "qwen_pathc_stress_permutation_v1",
        "status": "ok",
        "meta": meta,
        "feature_selection": feature_meta,
        "layers": [int(x) for x in layers],
        "split_diagnostics": split["split_diagnostics"],
        "train_exclusion_diagnostics": split["train_exclusion_diagnostics"],
        "observed_wrong_intermediate_auroc": observed_auc,
        "observed_test_auroc": observed.get("test_auroc"),
        "runs_requested": int(args.permutation_runs),
        "runs_effective": int(stats.get("count", 0)),
        "runs_blocked": int(blocked),
        "wrong_intermediate_auroc_distribution": stats,
        "empirical_p_value": empirical_p_value,
        "p_value_significant": p_value_significant,
        "p_value_primary_pass": p_value_primary_pass,
        "legacy_strict_pass": bool(legacy_pass),
        "criteria": {
            "mean_range": [0.45, 0.55],
            "p95_lt": 0.60,
            "max_lt": 0.70,
            "p_value_lt": 0.01,
        },
        "pass": bool(legacy_pass),
        "timestamp": datetime.now().isoformat(),
    }


def run_ablation_reg(args: argparse.Namespace) -> Dict[str, Any]:
    rows_master, feature_names, layers, meta = _load_rows_and_features(args.partials)
    selected_feature_names, feature_meta = _resolve_feature_names_for_mode(
        feature_names,
        partial_paths=args.partials,
        mode=str(args.feature_mode),
        single_layer_source=str(args.single_layer_source),
        single_layer_id=args.single_layer_id,
    )
    excluded = _parse_csv(args.train_exclude_variants)
    split = _split_with_exclusion(
        rows_master,
        seed=int(args.trace_split_seed),
        test_fraction=float(args.trace_test_fraction),
        excluded_variants=excluded,
    )
    idx_selected = _feature_indices(feature_names, selected_feature_names)
    train_rows = _subset_rows(split["train_rows"], idx_selected)
    test_rows = _subset_rows(split["test_rows"], idx_selected)

    selected_name_to_idx = {str(n): i for i, n in enumerate(selected_feature_names)}
    traj_names = [n for n in selected_feature_names if n.split(":", 1)[-1] in TRAJECTORY_METRICS]
    step_names = [n for n in selected_feature_names if n.split(":", 1)[-1] in STEP_ANOMALY_METRICS]
    full_names = list(selected_feature_names)

    if str(args.single_layer_mode) == "auto_best":
        best_layer = _single_best_layer_from_partials(args.partials)
    else:
        best_layer = int(args.single_layer_mode)
    layer_prefix = f"layer{best_layer}:"
    single_names = [n for n in selected_feature_names if n.startswith(layer_prefix)]
    if not single_names:
        single_names = list(selected_feature_names)

    feature_sets = {
        "trajectory_only": traj_names,
        "step_anomaly_only": step_names,
        "single_best_layer": single_names,
        "full_sae": full_names,
    }

    ablations: Dict[str, Any] = {}
    for name, fns in feature_sets.items():
        idx = [selected_name_to_idx[x] for x in fns if x in selected_name_to_idx]
        if not idx:
            ablations[name] = {
                "feature_count": int(len(fns)),
                "wrong_intermediate_auroc": None,
                "test_auroc": None,
                "train_auroc": None,
                "loss_end": None,
                "status": "blocked_no_features",
            }
            continue
        tr = _subset_rows(train_rows, idx)
        te = _subset_rows(test_rows, idx)
        res = _train_eval_wrong(
            tr,
            te,
            device=str(args.device),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay_base),
        )
        ablations[name] = {
            "feature_count": int(len(fns)),
            "wrong_intermediate_auroc": res.get("wrong_intermediate_auroc"),
            "test_auroc": res.get("test_auroc"),
            "train_auroc": res.get("train_auroc"),
            "loss_end": res.get("loss_end"),
        }

    wd_values = [float(x) for x in _parse_csv(args.weight_decay_values)]
    reg: Dict[str, Any] = {}
    for wd in wd_values:
        res = _train_eval_wrong(
            train_rows,
            test_rows,
            device=str(args.device),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(wd),
        )
        reg[str(wd)] = {
            "wrong_intermediate_auroc": res.get("wrong_intermediate_auroc"),
            "test_auroc": res.get("test_auroc"),
            "train_auroc": res.get("train_auroc"),
            "loss_end": res.get("loss_end"),
        }

    wd_pass = True
    for wd, min_req in [("0.01", 0.70), ("0.1", 0.70), ("1.0", 0.65)]:
        auc = (reg.get(wd) or {}).get("wrong_intermediate_auroc")
        if not isinstance(auc, (int, float)) or float(auc) < min_req:
            wd_pass = False

    full_auc = (ablations.get("full_sae") or {}).get("wrong_intermediate_auroc")
    traj_auc = (ablations.get("trajectory_only") or {}).get("wrong_intermediate_auroc")
    step_auc = (ablations.get("step_anomaly_only") or {}).get("wrong_intermediate_auroc")
    full_dep = None
    if all(isinstance(x, (int, float)) for x in [full_auc, traj_auc, step_auc]):
        full_dep = bool((float(full_auc) - float(traj_auc) >= 0.02) and (float(full_auc) - float(step_auc) >= 0.02))

    return {
        "schema_version": "qwen_pathc_stress_ablation_reg_v1",
        "status": "ok",
        "meta": meta,
        "feature_selection": feature_meta,
        "layers": [int(x) for x in layers],
        "split_diagnostics": split["split_diagnostics"],
        "train_exclusion_diagnostics": split["train_exclusion_diagnostics"],
        "single_best_layer": int(best_layer),
        "ablations": ablations,
        "regularization": reg,
        "regularization_pass": bool(wd_pass),
        "criteria": {
            "wd_0.01_min": 0.70,
            "wd_0.1_min": 0.70,
            "wd_1.0_min": 0.65,
            "ablation_material_delta_min": 0.02,
        },
        "full_sae_depends_on_full_feature_design": full_dep,
        "timestamp": datetime.now().isoformat(),
    }


def run_multiseed(args: argparse.Namespace) -> Dict[str, Any]:
    rows_master, feature_names, layers, meta = _load_rows_and_features(args.partials)
    selected_feature_names, feature_meta = _resolve_feature_names_for_mode(
        feature_names,
        partial_paths=args.partials,
        mode=str(args.feature_mode),
        single_layer_source=str(args.single_layer_source),
        single_layer_id=args.single_layer_id,
    )
    idx_selected = _feature_indices(feature_names, selected_feature_names)
    excluded = _parse_csv(args.train_exclude_variants)
    seeds = _parse_int_csv(args.multi_seed_values)
    if not seeds:
        raise RuntimeError("No multi-seed values provided")

    per_seed: List[Dict[str, Any]] = []
    wrong_aucs: List[float] = []
    pooled_wrong_rows: List[apc.ScoredRow] = []
    overlap_total = 0

    for seed in seeds:
        split = _split_with_exclusion(
            rows_master,
            seed=int(seed),
            test_fraction=float(args.trace_test_fraction),
            excluded_variants=excluded,
        )
        train_rows = _subset_rows(split["train_rows"], idx_selected)
        test_rows = _subset_rows(split["test_rows"], idx_selected)
        overlap_total += int((split["split_diagnostics"] or {}).get("trace_overlap_count", 0))
        out = _train_eval_wrong(
            train_rows,
            test_rows,
            device=str(args.device),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay_base),
        )
        auc = out.get("wrong_intermediate_auroc")
        if isinstance(auc, (int, float)):
            wrong_aucs.append(float(auc))
        pooled_wrong_rows.extend(_collect_wrong_scored(out.get("scored_rows", []), seed_tag=str(seed)))
        per_seed.append(
            {
                "seed": int(seed),
                "split_diagnostics": split["split_diagnostics"],
                "train_exclusion_diagnostics": split["train_exclusion_diagnostics"],
                "wrong_intermediate_auroc": auc,
                "wrong_intermediate_count": out.get("wrong_intermediate_count"),
                "wrong_intermediate_pos": out.get("wrong_intermediate_pos"),
                "wrong_intermediate_neg": out.get("wrong_intermediate_neg"),
                "loss_end": out.get("loss_end"),
            }
        )

    pooled_auc = apc._roc_auc([(float(r.score), int(r.label)) for r in pooled_wrong_rows])
    pooled_boot = apc._bootstrap_trace_pair_auc(
        pooled_wrong_rows,
        n_bootstrap=int(args.wrong_intermediate_bootstrap_n),
        seed=int(args.wrong_intermediate_bootstrap_seed),
    )
    stats = _summ(wrong_aucs)

    pass_flag = bool(
        isinstance(stats.get("mean"), (int, float))
        and isinstance(stats.get("std"), (int, float))
        and isinstance(pooled_boot.get("ci95_lower"), (int, float))
        and float(stats["mean"]) >= 0.70
        and float(stats["std"]) <= 0.08
        and float(pooled_boot["ci95_lower"]) >= 0.65
        and int(overlap_total) == 0
    )

    return {
        "schema_version": "qwen_pathc_stress_multiseed_v1",
        "status": "ok",
        "meta": meta,
        "feature_selection": feature_meta,
        "layers": [int(x) for x in layers],
        "seeds": [int(s) for s in seeds],
        "per_seed": per_seed,
        "wrong_intermediate_auroc_distribution": stats,
        "pooled_wrong_intermediate_auroc": pooled_auc,
        "pooled_wrong_intermediate_ci95": {
            "lower": pooled_boot.get("ci95_lower"),
            "upper": pooled_boot.get("ci95_upper"),
            "bootstrap_n": int(args.wrong_intermediate_bootstrap_n),
            "bootstrap_effective_n": int(pooled_boot.get("bootstrap_effective_n", 0)),
            "trace_pair_count": int(pooled_boot.get("valid_trace_pair_count", 0)),
            "defined": bool(pooled_boot.get("defined", False)),
        },
        "trace_overlap_count_total": int(overlap_total),
        "criteria": {
            "mean_min": 0.70,
            "std_max": 0.08,
            "pooled_ci95_lower_min": 0.65,
            "trace_overlap_must_equal": 0,
        },
        "pass": pass_flag,
        "timestamp": datetime.now().isoformat(),
    }


def run_final(args: argparse.Namespace) -> Dict[str, Any]:
    perm = load_json(args.permutation_json)
    abr = load_json(args.ablation_reg_json)
    ms = load_json(args.multiseed_json)

    perm_legacy_pass = bool(perm.get("legacy_strict_pass") is True or perm.get("pass") is True)
    perm_primary_pass = bool(perm.get("p_value_primary_pass") is True)
    reg_pass = bool(abr.get("regularization_pass") is True)
    ms_pass = bool(ms.get("pass") is True)

    inconclusive_reasons: List[str] = []
    for name, obj in [("permutation", perm), ("ablation_reg", abr), ("multiseed", ms)]:
        if str(obj.get("status")) != "ok":
            inconclusive_reasons.append(f"{name}_status_{obj.get('status')}")

    if inconclusive_reasons:
        verdict_legacy = "inconclusive"
        reason_legacy = ",".join(inconclusive_reasons)
        verdict_primary = "inconclusive"
        reason_primary = ",".join(inconclusive_reasons)
    else:
        if perm_legacy_pass and reg_pass and ms_pass:
            verdict_legacy = "pass"
            reason_legacy = "all_legacy_checks_passed"
        else:
            verdict_legacy = "fail"
            fails_legacy: List[str] = []
            if not perm_legacy_pass:
                fails_legacy.append("permutation_legacy")
            if not reg_pass:
                fails_legacy.append("regularization")
            if not ms_pass:
                fails_legacy.append("multiseed")
            reason_legacy = "failed:" + ",".join(fails_legacy)

        if perm_primary_pass and reg_pass and ms_pass:
            verdict_primary = "pass"
            reason_primary = "all_primary_checks_passed"
        else:
            verdict_primary = "fail"
            fails_primary: List[str] = []
            if not perm_primary_pass:
                fails_primary.append("permutation_pvalue")
            if not reg_pass:
                fails_primary.append("regularization")
            if not ms_pass:
                fails_primary.append("multiseed")
            reason_primary = "failed:" + ",".join(fails_primary)

    out = {
        "schema_version": "qwen_pathc_stress_v1",
        "status": "ok",
        "run_id": str(args.run_id),
        "run_tag": str(args.run_tag),
        "source_partials": [str(x) for x in args.partials],
        "strict_policy": {
            "permutation": {
                "mean_range": [0.45, 0.55],
                "p95_lt": 0.60,
                "max_lt": 0.70,
                "p_value_lt": 0.01,
            },
            "regularization": {
                "wd_0.01_min": 0.70,
                "wd_0.1_min": 0.70,
                "wd_1.0_min": 0.65,
            },
            "multiseed": {
                "mean_min": 0.70,
                "std_max": 0.08,
                "pooled_ci95_lower_min": 0.65,
            },
            "ablation_interpretation": {
                "material_delta_min": 0.02,
                "note": "Interpretive only; not hard fail",
            },
        },
        "tests": {
            "permutation": perm,
            "ablation_reg": abr,
            "multiseed": ms,
        },
        "checks": {
            "permutation_legacy_pass": perm_legacy_pass,
            "permutation_pvalue_pass": perm_primary_pass,
            "regularization_pass": reg_pass,
            "multiseed_pass": ms_pass,
            "ablation_full_feature_dependency": abr.get("full_sae_depends_on_full_feature_design"),
        },
        "legacy_strict_pass": bool(perm_legacy_pass and reg_pass and ms_pass),
        "p_value_primary_pass": bool(perm_primary_pass and reg_pass and ms_pass),
        "final_verdict_legacy": verdict_legacy,
        "final_reason_legacy": reason_legacy,
        "final_verdict_primary": verdict_primary,
        "final_reason_primary": reason_primary,
        "final_verdict": verdict_primary,
        "final_reason": reason_primary,
        "timestamp": datetime.now().isoformat(),
    }

    lines = [
        "# Qwen Path C Stress Test Summary",
        "",
        f"- Run id: `{args.run_id}`",
        f"- Verdict (primary): `{verdict_primary}`",
        f"- Reason (primary): `{reason_primary}`",
        f"- Verdict (legacy): `{verdict_legacy}`",
        f"- Reason (legacy): `{reason_legacy}`",
        "",
        "## Strict Checks",
        "",
        f"- Permutation legacy pass: `{perm_legacy_pass}`",
        f"- Permutation p-value pass: `{perm_primary_pass}`",
        f"- Regularization pass: `{reg_pass}`",
        f"- Multi-seed pass: `{ms_pass}`",
        f"- Full-feature dependency (interpretive): `{abr.get('full_sae_depends_on_full_feature_design')}`",
        "",
        "## Key Numbers",
        "",
        f"- Perm mean/p95/max AUROC: `{(perm.get('wrong_intermediate_auroc_distribution') or {}).get('mean')}` / `{(perm.get('wrong_intermediate_auroc_distribution') or {}).get('p95')}` / `{(perm.get('wrong_intermediate_auroc_distribution') or {}).get('max')}`",
        f"- Perm observed AUROC: `{perm.get('observed_wrong_intermediate_auroc')}`",
        f"- Perm empirical p-value: `{perm.get('empirical_p_value')}`",
        f"- Multi-seed mean±std: `{(ms.get('wrong_intermediate_auroc_distribution') or {}).get('mean')}` ± `{(ms.get('wrong_intermediate_auroc_distribution') or {}).get('std')}`",
        f"- Multi-seed pooled CI95 lower: `{((ms.get('pooled_wrong_intermediate_ci95') or {}).get('lower'))}`",
        f"- Regularization @0.01/@0.1/@1.0: `{((abr.get('regularization') or {}).get('0.01') or {}).get('wrong_intermediate_auroc')}` / `{((abr.get('regularization') or {}).get('0.1') or {}).get('wrong_intermediate_auroc')}` / `{((abr.get('regularization') or {}).get('1.0') or {}).get('wrong_intermediate_auroc')}`",
    ]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["permutation", "ablation_reg", "multiseed", "final"], required=True)
    p.add_argument("--partials", nargs="+", default=[])
    p.add_argument("--run-id", default="")
    p.add_argument("--run-tag", default="")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=False, default="")

    p.add_argument("--train-exclude-variants", default="order_flip_only,answer_first_order_flip,reordered_steps")
    p.add_argument("--trace-test-fraction", type=float, default=0.20)
    p.add_argument("--trace-split-seed", type=int, default=20260306)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight-decay-base", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")

    p.add_argument("--permutation-runs", type=int, default=100)
    p.add_argument("--permutation-seed", type=int, default=20260308)

    p.add_argument("--weight-decay-values", default="0.0001,0.01,0.1,1.0")
    p.add_argument("--single-layer-mode", default="auto_best", help="auto_best or explicit layer id")
    p.add_argument("--feature-mode", default="full_sae", choices=FEATURE_MODE_CHOICES)
    p.add_argument("--single-layer-source", default="auto_best", choices=("auto_best", "explicit"))
    p.add_argument("--single-layer-id", type=int, default=None)

    p.add_argument("--multi-seed-values", default="20260307,20260308,20260309,20260310,20260311,20260312,20260313,20260314,20260315,20260316")
    p.add_argument("--wrong-intermediate-bootstrap-n", type=int, default=1000)
    p.add_argument("--wrong-intermediate-bootstrap-seed", type=int, default=20260307)

    p.add_argument("--permutation-json", default="")
    p.add_argument("--ablation-reg-json", default="")
    p.add_argument("--multiseed-json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    task = str(args.task)

    if task != "final":
        if not args.partials:
            raise RuntimeError("--partials is required for non-final tasks")

    if task == "permutation":
        out = run_permutation(args)
    elif task == "ablation_reg":
        out = run_ablation_reg(args)
    elif task == "multiseed":
        out = run_multiseed(args)
    else:
        if not args.permutation_json or not args.ablation_reg_json or not args.multiseed_json:
            raise RuntimeError("final task requires --permutation-json --ablation-reg-json --multiseed-json")
        if not args.output_md:
            raise RuntimeError("final task requires --output-md")
        out = run_final(args)

    save_json(args.output_json, out)
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()

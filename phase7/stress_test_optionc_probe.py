#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover
    from phase7.common import load_json, save_json
    from phase7.optionc_feature_builder import build_optionc_feature_rows
except ImportError:  # pragma: no cover
    from common import load_json, save_json
    from optionc_feature_builder import build_optionc_feature_rows

TASK_CHOICES = ("permutation", "ablation_reg", "multiseed", "final")

TRAJECTORY_SUFFIXES = ("transition_mean_cosine", "transition_min_cosine")
STEP_ANOMALY_SUFFIXES = (
    "transition_mean_delta_l2",
    "transition_max_delta_l2",
    "transition_p95_delta_l2",
)


@dataclass
class Row:
    member_id: str
    pair_id: str
    trace_id: str
    variant: str
    label: int
    lexical_control: bool
    features: List[float]


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


def _roc_auc(scores_labels: Sequence[Tuple[float, int]]) -> Optional[float]:
    if not scores_labels:
        return None
    pos = sum(int(y) for _, y in scores_labels)
    neg = len(scores_labels) - pos
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


def _summ(vals: Sequence[float]) -> Dict[str, Any]:
    xs = [float(v) for v in vals if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not xs:
        return {"count": 0, "mean": None, "std": None, "p95": None, "min": None, "max": None}
    mean = sum(xs) / len(xs)
    if len(xs) > 1:
        var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {
        "count": int(len(xs)),
        "mean": float(mean),
        "std": float(std),
        "p95": _percentile(xs, 0.95),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _bootstrap_group_worker(
    by_group: Dict[str, List[Tuple[float, int]]],
    groups: List[str],
    worker_boot: int,
    worker_seed: int,
) -> List[float]:
    rng = random.Random(int(worker_seed))
    out_vals: List[float] = []
    gg = list(groups)
    for _ in range(int(worker_boot)):
        sample: List[Tuple[float, int]] = []
        for __ in range(len(gg)):
            g = gg[rng.randrange(len(gg))]
            sample.extend(by_group[g])
        auc = _roc_auc(sample)
        if auc is not None:
            out_vals.append(float(auc))
    return out_vals


def _bootstrap_group_auc(
    scored: Sequence[Dict[str, Any]],
    *,
    group_key: str,
    n_bootstrap: int,
    seed: int,
    cpu_workers: int,
) -> Dict[str, Any]:
    by_group: Dict[str, List[Tuple[float, int]]] = {}
    for r in scored:
        g = str(r.get(group_key, ""))
        by_group.setdefault(g, []).append((float(r["score"]), int(r["label"])))
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

    total_boot = max(1, int(n_bootstrap))
    workers = max(1, int(cpu_workers))
    vals: List[float] = []

    if workers <= 1 or total_boot < 100:
        vals.extend(_bootstrap_group_worker(by_group, groups, total_boot, int(seed)))
    else:
        splits = [total_boot // workers] * workers
        for i in range(total_boot % workers):
            splits[i] += 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = []
            for wi, wb in enumerate(splits):
                if wb <= 0:
                    continue
                futs.append(
                    ex.submit(
                        _bootstrap_group_worker,
                        by_group,
                        groups,
                        int(wb),
                        int(seed) + 1009 * (wi + 1),
                    )
                )
            for fut in concurrent.futures.as_completed(futs):
                vals.extend(fut.result())

    out["bootstrap_effective_n"] = int(len(vals))
    out["ci95_lower"] = _percentile(vals, 0.025)
    out["ci95_upper"] = _percentile(vals, 0.975)
    return out


def _load_rows(
    paired_dataset: str,
    partials: Sequence[str],
    *,
    layer_allowlist: Optional[set[int]] = None,
    layer_allowlist_values: Optional[Sequence[int]] = None,
    decoder_checkpoint: str = "",
    decoder_domain: str = "auto",
    logical_decoder_feature_mode: str = "full",
    decoder_device: str = "cuda:0",
    decoder_batch_size: int = 128,
    require_decoder_enabled: bool = False,
) -> Tuple[List[Row], List[str], Dict[str, Any]]:
    assembled = build_optionc_feature_rows(
        paired_dataset=str(paired_dataset),
        partials=[str(p) for p in partials],
        decoder_checkpoint=str(decoder_checkpoint),
        decoder_domain_requested=str(decoder_domain),
        logical_decoder_feature_mode=str(logical_decoder_feature_mode),
        sae_layer_allowlist_values=[int(x) for x in list(layer_allowlist_values or [])],
        decoder_device=str(decoder_device),
        decoder_batch_size=int(decoder_batch_size),
        require_decoder_enabled=bool(require_decoder_enabled),
    )
    ds = assembled["payload"]
    feature_names = list(assembled["feature_names"])
    rows: List[Row] = []
    dropped = 0
    for rec in list(assembled["eval_rows"]):
        mid = str(rec.get("member_id", ""))
        feat_map = rec.get("features", {})
        if not isinstance(feat_map, dict):
            dropped += 1
            continue
        vals: List[float] = []
        ok = True
        for fn in feature_names:
            v = feat_map.get(fn)
            if not isinstance(v, (int, float)):
                ok = False
                break
            vals.append(float(v))
        if not ok:
            dropped += 1
            continue
        lexical = bool(rec.get("lexical_control", False))
        pair_ambiguous = bool(rec.get("pair_ambiguous", False))
        label = int(rec.get("label", 0))
        raw_variant = str(rec.get("pair_type", "")).strip()
        if lexical:
            variant = "lexical_consistent_swap"
        elif pair_ambiguous:
            variant = "pair_ambiguous"
        elif raw_variant:
            variant = raw_variant
        elif label == 1:
            variant = "unfaithful"
        else:
            variant = "faithful"
        rows.append(
            Row(
                member_id=str(mid),
                pair_id=str(rec.get("pair_id", "")),
                trace_id=str(mid),
                variant=str(variant),
                label=int(label),
                lexical_control=lexical,
                features=vals,
            )
        )

    meta = {
        "paired_dataset": str(paired_dataset),
        "paired_dataset_sha256": ds.get("rows_sha256"),
        "rows_after_join": int(len(rows)),
        "rows_dropped": int(dropped),
        "feature_count": int(len(feature_names)),
        "feature_count_total": int(assembled.get("feature_count_total", len(feature_names))),
        "sae_feature_count": int(assembled.get("sae_feature_count_after_filter", 0)),
        "decoder_feature_count": int(assembled.get("decoder_feature_count", 0)),
        "decoder_features_enabled": bool(assembled.get("decoder_features_enabled", False)),
        "decoder_feature_block_status": str(assembled.get("decoder_feature_block_status", "unknown")),
        "logical_decoder_feature_mode": str(assembled.get("logical_decoder_feature_mode", logical_decoder_feature_mode)),
        "sae_layer_allowlist": [int(x) for x in (layer_allowlist_values or [])],
        "member_count_dataset": int(len(list(ds.get("members", [])))),
        "partial_count": int(len(partials)),
    }
    return rows, feature_names, meta


def _resolve_layer_allowlist(args: argparse.Namespace) -> Tuple[List[int], Optional[set[int]]]:
    vals = _parse_int_csv(str(args.sae_layer_allowlist)) if str(args.sae_layer_allowlist).strip() else []
    allow = set(int(x) for x in vals) if vals else None
    return vals, allow


def _subset_rows(rows: Sequence[Row], idx: Sequence[int]) -> List[Row]:
    out: List[Row] = []
    ii = [int(i) for i in idx]
    for r in rows:
        out.append(
            Row(
                member_id=str(r.member_id),
                pair_id=str(r.pair_id),
                trace_id=str(r.trace_id),
                variant=str(r.variant),
                label=int(r.label),
                lexical_control=bool(r.lexical_control),
                features=[float(r.features[i]) for i in ii],
            )
        )
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


def _apply_train_variant_exclusion(train_rows: Sequence[Row], excluded_variants: Sequence[str]) -> Tuple[List[Row], Dict[str, int]]:
    excluded = set(str(v) for v in excluded_variants)
    kept: List[Row] = []
    dropped: Dict[str, int] = {}
    for r in train_rows:
        if str(r.variant) in excluded:
            key = str(r.variant)
            dropped[key] = int(dropped.get(key, 0)) + 1
            continue
        kept.append(r)
    return kept, {k: int(v) for k, v in sorted(dropped.items())}


def _split_by_pair(rows: Sequence[Row], *, test_fraction: float, seed: int, max_tries: int = 200) -> Tuple[List[Row], List[Row], Dict[str, Any]]:
    pair_ids = sorted({str(r.pair_id) for r in rows if str(r.pair_id)})
    if len(pair_ids) < 2:
        raise RuntimeError("Need at least 2 pair_ids")
    n_test = max(1, int(round(len(pair_ids) * float(test_fraction))))
    n_test = min(n_test, len(pair_ids) - 1)

    for attempt in range(max_tries):
        rng = random.Random(int(seed) + attempt)
        ids = list(pair_ids)
        rng.shuffle(ids)
        test_set = set(ids[:n_test])
        train_rows = [r for r in rows if str(r.pair_id) not in test_set]
        test_rows = [r for r in rows if str(r.pair_id) in test_set]
        train_pos = sum(int(r.label) for r in train_rows)
        train_neg = len(train_rows) - train_pos
        test_pos = sum(int(r.label) for r in test_rows)
        test_neg = len(test_rows) - test_pos
        if train_pos > 0 and train_neg > 0 and test_pos > 0 and test_neg > 0:
            return train_rows, test_rows, {
                "pair_split_seed_used": int(seed) + attempt,
                "pair_count_total": int(len(pair_ids)),
                "pair_count_train": int(len(pair_ids) - len(test_set)),
                "pair_count_test": int(len(test_set)),
                "train_rows": int(len(train_rows)),
                "test_rows": int(len(test_rows)),
                "train_pos": int(train_pos),
                "train_neg": int(train_neg),
                "test_pos": int(test_pos),
                "test_neg": int(test_neg),
                "trace_overlap_count": 0,
                "pair_overlap_count": 0,
            }
    raise RuntimeError("Could not produce class-balanced pair-disjoint split")


def _train_logreg(
    train_rows: Sequence[Row],
    test_rows: Sequence[Row],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> Dict[str, Any]:
    if not train_rows or not test_rows:
        raise RuntimeError("empty train/test rows")
    d = len(train_rows[0].features)
    x_train = torch.tensor([r.features for r in train_rows], dtype=torch.float32)
    y_train = torch.tensor([int(r.label) for r in train_rows], dtype=torch.float32)
    x_test = torch.tensor([r.features for r in test_rows], dtype=torch.float32)

    mu = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std

    dev = torch.device(str(device))
    x_train = x_train.to(dev)
    y_train = y_train.to(dev)
    x_test = x_test.to(dev)

    w = torch.zeros((d, 1), dtype=torch.float32, device=dev, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=float(lr), weight_decay=float(weight_decay))
    losses: List[float] = []
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = (x_train @ w + b).view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        train_probs = torch.sigmoid((x_train @ w + b).view(-1)).detach().cpu().tolist()
        test_probs = torch.sigmoid((x_test @ w + b).view(-1)).detach().cpu().tolist()

    y_train_cpu = [int(r.label) for r in train_rows]
    y_test_cpu = [int(r.label) for r in test_rows]
    return {
        "train_auroc": _roc_auc(list(zip(train_probs, y_train_cpu))),
        "test_auroc": _roc_auc(list(zip(test_probs, y_test_cpu))),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "test_probs": [float(x) for x in test_probs],
    }


def _eval_wrong_lexical(
    train_rows: Sequence[Row],
    test_rows: Sequence[Row],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> Dict[str, Any]:
    fit = _train_logreg(train_rows, test_rows, epochs=epochs, lr=lr, weight_decay=weight_decay, device=device)
    probs = list(fit.get("test_probs", []))
    scored = []
    for p, r in zip(probs, test_rows):
        scored.append(
            {
                "member_id": str(r.member_id),
                "pair_id": str(r.pair_id),
                "trace_id": str(r.trace_id),
                "variant": str(r.variant),
                "label": int(r.label),
                "lexical_control": bool(r.lexical_control),
                "score": float(p),
            }
        )
    primary_rows = [
        (float(r["score"]), int(r["label"]))
        for r in scored
        if str(r["variant"]) not in {"lexical_consistent_swap", "pair_ambiguous"}
    ]
    primary_member_auroc = _roc_auc(primary_rows)
    wrong = [(float(r["score"]), int(r["label"])) for r in scored if str(r["variant"]) == "wrong_intermediate"]
    lexical = [(float(r["score"]), int(r["label"])) for r in scored if str(r["variant"]) == "lexical_consistent_swap"]
    lexical_control_rows = [
        (float(r["score"]), 1 if bool(r["lexical_control"]) else 0)
        for r in scored
        if int(r["label"]) == 0
    ]
    wrong_auc = _roc_auc(wrong)
    return {
        "train_auroc": fit.get("train_auroc"),
        "test_auroc": fit.get("test_auroc"),
        "loss_end": fit.get("loss_end"),
        "primary_member_auroc": primary_member_auroc,
        "primary_member_count": int(len(primary_rows)),
        "primary_member_pos": int(sum(y for _, y in primary_rows)),
        "primary_member_neg": int(len(primary_rows) - sum(y for _, y in primary_rows)),
        "wrong_intermediate_auroc": wrong_auc,
        "wrong_intermediate_count": int(len(wrong)),
        "wrong_intermediate_pos": int(sum(y for _, y in wrong)),
        "wrong_intermediate_neg": int(len(wrong) - sum(y for _, y in wrong)),
        # Legacy lexical metric: AUROC within lexical-consistent variant using original labels.
        "lexical_variant_auroc": _roc_auc(lexical),
        "lexical_variant_count": int(len(lexical)),
        "lexical_variant_pos": int(sum(y for _, y in lexical)),
        "lexical_variant_neg": int(len(lexical) - sum(y for _, y in lexical)),
        # Eval-aligned lexical confound metric: among faithful rows, detect lexical-control rows.
        "lexical_control_probe_auroc": _roc_auc(lexical_control_rows),
        "lexical_control_probe_count": int(len(lexical_control_rows)),
        "lexical_control_probe_pos": int(sum(y for _, y in lexical_control_rows)),
        "lexical_control_probe_neg": int(len(lexical_control_rows) - sum(y for _, y in lexical_control_rows)),
        "scored_rows": scored,
    }


def _feature_blocks(feature_names: Sequence[str]) -> Dict[str, List[str]]:
    traj = [fn for fn in feature_names if fn.split(":", 1)[-1] in TRAJECTORY_SUFFIXES]
    step = [fn for fn in feature_names if fn.split(":", 1)[-1] in STEP_ANOMALY_SUFFIXES]
    full = list(feature_names)
    return {
        "trajectory_only": traj,
        "step_anomaly_only": step,
        "full_sae": full,
    }


def _single_best_layer(feature_names: Sequence[str], rows_train: Sequence[Row], rows_test: Sequence[Row], args: argparse.Namespace) -> Tuple[int, float]:
    layer_to_names: Dict[int, List[str]] = {}
    for fn in feature_names:
        if not fn.startswith("layer") or ":" not in fn:
            continue
        layer_s = fn.split(":", 1)[0].replace("layer", "")
        try:
            layer = int(layer_s)
        except Exception:
            continue
        layer_to_names.setdefault(layer, []).append(fn)
    best_layer = None
    best_auc = -1.0
    idx_map = {str(n): i for i, n in enumerate(feature_names)}
    for layer in sorted(layer_to_names.keys()):
        idx = [idx_map[n] for n in layer_to_names[layer] if n in idx_map]
        if not idx:
            continue
        tr = _subset_rows(rows_train, idx)
        te = _subset_rows(rows_test, idx)
        out = _eval_wrong_lexical(
            tr,
            te,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay_base),
            device=str(args.device),
        )
        auc = out.get("primary_member_auroc")
        if not isinstance(auc, (int, float)):
            auc = out.get("wrong_intermediate_auroc")
        if isinstance(auc, (int, float)) and float(auc) > best_auc:
            best_auc = float(auc)
            best_layer = int(layer)
    if best_layer is None:
        raise RuntimeError("Could not resolve single best layer")
    return int(best_layer), float(best_auc)


def _make_split(rows_master: Sequence[Row], args: argparse.Namespace) -> Dict[str, Any]:
    train_rows, test_rows, split_diag = _split_by_pair(
        rows_master,
        test_fraction=float(args.trace_test_fraction),
        seed=int(args.trace_split_seed),
    )
    train_pre = _summarize_rows(train_rows)
    excluded = _parse_csv(args.train_exclude_variants)
    train_eff, dropped = _apply_train_variant_exclusion(train_rows, excluded)
    train_post = _summarize_rows(train_eff)
    test_sum = _summarize_rows(test_rows)
    return {
        "train_rows": train_eff,
        "test_rows": test_rows,
        "split_diagnostics": split_diag,
        "train_exclusion_diagnostics": {
            "train_counts_pre": train_pre,
            "train_counts_post": train_post,
            "excluded_rows_by_variant": dropped,
            "excluded_rows_total": int(sum(dropped.values())),
            "excluded_unfaithful_rows_by_variant": dropped,  # compatibility alias
            "excluded_unfaithful_rows_total": int(sum(dropped.values())),  # compatibility alias
            "test_variant_coverage": test_sum.get("by_variant", {}),
        },
    }


def run_permutation(args: argparse.Namespace) -> Dict[str, Any]:
    layer_vals, layer_allow = _resolve_layer_allowlist(args)
    rows_master, feature_names, meta = _load_rows(
        args.paired_dataset,
        args.partials,
        layer_allowlist=layer_allow,
        layer_allowlist_values=layer_vals,
        decoder_checkpoint=str(args.decoder_checkpoint),
        decoder_domain=str(args.decoder_domain),
        logical_decoder_feature_mode=str(args.logical_decoder_feature_mode),
        decoder_device=str(args.decoder_device),
        decoder_batch_size=int(args.decoder_batch_size),
        require_decoder_enabled=bool(int(args.require_decoder_enabled)),
    )
    split = _make_split(rows_master, args)
    rows_train = split["train_rows"]
    rows_test = split["test_rows"]

    out_obs = _eval_wrong_lexical(
        rows_train,
        rows_test,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay_base),
        device=str(args.device),
    )
    observed = out_obs.get("primary_member_auroc")
    if not isinstance(observed, (int, float)):
        observed = out_obs.get("wrong_intermediate_auroc")

    rng = random.Random(int(args.permutation_seed))
    aucs: List[float] = []
    blocked = 0
    for _ in range(int(args.permutation_runs)):
        ys = [int(r.label) for r in rows_train]
        rng.shuffle(ys)
        pr: List[Row] = []
        for r, y in zip(rows_train, ys):
            pr.append(
                Row(
                    member_id=str(r.member_id),
                    pair_id=str(r.pair_id),
                    trace_id=str(r.trace_id),
                    variant=str(r.variant),
                    label=int(y),
                    lexical_control=bool(r.lexical_control),
                    features=list(r.features),
                )
            )
        if sum(x.label for x in pr) in (0, len(pr)):
            blocked += 1
            continue
        out = _eval_wrong_lexical(
            pr,
            rows_test,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay_base),
            device=str(args.device),
        )
        auc = out.get("primary_member_auroc")
        if not isinstance(auc, (int, float)):
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
    if isinstance(observed, (int, float)) and len(aucs) > 0:
        ge = sum(1 for x in aucs if float(x) >= float(observed))
        empirical_p_value = float((ge + 1) / (len(aucs) + 1))
        p_value_significant = bool(empirical_p_value < 0.01)
        p_value_primary_pass = bool(p_value_significant)

    return {
        "schema_version": "phase7_optionc_stress_permutation_v1",
        "status": "ok",
        "meta": meta,
        "decoder_features_enabled": bool(meta.get("decoder_features_enabled", False)),
        "decoder_feature_block_status": str(meta.get("decoder_feature_block_status", "unknown")),
        "logical_decoder_feature_mode": str(meta.get("logical_decoder_feature_mode", "")),
        "feature_count_total": int(meta.get("feature_count_total", meta.get("feature_count", 0))),
        "sae_feature_count": int(meta.get("sae_feature_count", 0)),
        "decoder_feature_count": int(meta.get("decoder_feature_count", 0)),
        "sae_layer_allowlist": list(meta.get("sae_layer_allowlist", [])),
        "split_diagnostics": split["split_diagnostics"],
        "train_exclusion_diagnostics": split["train_exclusion_diagnostics"],
        "observed_primary_member_auroc": observed,
        "observed_wrong_intermediate_auroc": out_obs.get("wrong_intermediate_auroc"),
        "observed_lexical_variant_auroc": out_obs.get("lexical_variant_auroc"),
        "observed_lexical_control_probe_auroc": out_obs.get("lexical_control_probe_auroc"),
        "observed_test_auroc": out_obs.get("test_auroc"),
        "runs_requested": int(args.permutation_runs),
        "runs_effective": int(stats.get("count", 0)),
        "runs_blocked": int(blocked),
        "primary_member_auroc_distribution": stats,
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
    layer_vals, layer_allow = _resolve_layer_allowlist(args)
    rows_master, feature_names, meta = _load_rows(
        args.paired_dataset,
        args.partials,
        layer_allowlist=layer_allow,
        layer_allowlist_values=layer_vals,
        decoder_checkpoint=str(args.decoder_checkpoint),
        decoder_domain=str(args.decoder_domain),
        logical_decoder_feature_mode=str(args.logical_decoder_feature_mode),
        decoder_device=str(args.decoder_device),
        decoder_batch_size=int(args.decoder_batch_size),
        require_decoder_enabled=bool(int(args.require_decoder_enabled)),
    )
    split = _make_split(rows_master, args)
    rows_train = split["train_rows"]
    rows_test = split["test_rows"]

    blocks = _feature_blocks(feature_names)
    idx_map = {str(n): i for i, n in enumerate(feature_names)}

    best_layer, best_layer_auc = _single_best_layer(feature_names, rows_train, rows_test, args)
    layer_prefix = f"layer{best_layer}:"
    single_names = [n for n in feature_names if n.startswith(layer_prefix)]

    blocks_eval = {
        "trajectory_only": blocks["trajectory_only"],
        "step_anomaly_only": blocks["step_anomaly_only"],
        "single_best_layer": single_names,
        "full_sae": blocks["full_sae"],
    }

    ablations: Dict[str, Any] = {}
    for name, fns in blocks_eval.items():
        idx = [idx_map[n] for n in fns if n in idx_map]
        if not idx:
            ablations[name] = {"status": "blocked_no_features", "feature_count": int(len(fns))}
            continue
        tr = _subset_rows(rows_train, idx)
        te = _subset_rows(rows_test, idx)
        out = _eval_wrong_lexical(
            tr,
            te,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay_base),
            device=str(args.device),
        )
        ablations[name] = {
            "status": "ok",
            "feature_count": int(len(fns)),
            "primary_member_auroc": out.get("primary_member_auroc"),
            "wrong_intermediate_auroc": out.get("wrong_intermediate_auroc"),
            "lexical_variant_auroc": out.get("lexical_variant_auroc"),
            "lexical_control_probe_auroc": out.get("lexical_control_probe_auroc"),
            "test_auroc": out.get("test_auroc"),
            "train_auroc": out.get("train_auroc"),
            "loss_end": out.get("loss_end"),
        }

    wd_values = [float(x) for x in _parse_csv(args.weight_decay_values)]
    reg: Dict[str, Any] = {}
    for wd in wd_values:
        out = _eval_wrong_lexical(
            rows_train,
            rows_test,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(wd),
            device=str(args.device),
        )
        reg[str(wd)] = {
            "primary_member_auroc": out.get("primary_member_auroc"),
            "wrong_intermediate_auroc": out.get("wrong_intermediate_auroc"),
            "lexical_variant_auroc": out.get("lexical_variant_auroc"),
            "lexical_control_probe_auroc": out.get("lexical_control_probe_auroc"),
            "test_auroc": out.get("test_auroc"),
            "train_auroc": out.get("train_auroc"),
            "loss_end": out.get("loss_end"),
        }

    reg_pass = True
    for wd, min_req in (("0.01", 0.70), ("0.1", 0.70), ("1.0", 0.65)):
        auc = (reg.get(wd) or {}).get("primary_member_auroc")
        if not isinstance(auc, (int, float)):
            auc = (reg.get(wd) or {}).get("wrong_intermediate_auroc")
        if not isinstance(auc, (int, float)) or float(auc) < min_req:
            reg_pass = False

    full_auc = (ablations.get("full_sae") or {}).get("primary_member_auroc")
    traj_auc = (ablations.get("trajectory_only") or {}).get("primary_member_auroc")
    step_auc = (ablations.get("step_anomaly_only") or {}).get("primary_member_auroc")
    if not isinstance(full_auc, (int, float)):
        full_auc = (ablations.get("full_sae") or {}).get("wrong_intermediate_auroc")
    if not isinstance(traj_auc, (int, float)):
        traj_auc = (ablations.get("trajectory_only") or {}).get("wrong_intermediate_auroc")
    if not isinstance(step_auc, (int, float)):
        step_auc = (ablations.get("step_anomaly_only") or {}).get("wrong_intermediate_auroc")
    full_dep = None
    if all(isinstance(x, (int, float)) for x in [full_auc, traj_auc, step_auc]):
        full_dep = bool((float(full_auc) - float(traj_auc) >= 0.02) and (float(full_auc) - float(step_auc) >= 0.02))

    return {
        "schema_version": "phase7_optionc_stress_ablation_reg_v1",
        "status": "ok",
        "meta": meta,
        "decoder_features_enabled": bool(meta.get("decoder_features_enabled", False)),
        "decoder_feature_block_status": str(meta.get("decoder_feature_block_status", "unknown")),
        "logical_decoder_feature_mode": str(meta.get("logical_decoder_feature_mode", "")),
        "feature_count_total": int(meta.get("feature_count_total", meta.get("feature_count", 0))),
        "sae_feature_count": int(meta.get("sae_feature_count", 0)),
        "decoder_feature_count": int(meta.get("decoder_feature_count", 0)),
        "sae_layer_allowlist": list(meta.get("sae_layer_allowlist", [])),
        "split_diagnostics": split["split_diagnostics"],
        "train_exclusion_diagnostics": split["train_exclusion_diagnostics"],
        "single_best_layer": int(best_layer),
        "single_best_layer_auroc": float(best_layer_auc),
        "ablations": ablations,
        "regularization": reg,
        "regularization_pass": bool(reg_pass),
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
    layer_vals, layer_allow = _resolve_layer_allowlist(args)
    rows_master, feature_names, meta = _load_rows(
        args.paired_dataset,
        args.partials,
        layer_allowlist=layer_allow,
        layer_allowlist_values=layer_vals,
        decoder_checkpoint=str(args.decoder_checkpoint),
        decoder_domain=str(args.decoder_domain),
        logical_decoder_feature_mode=str(args.logical_decoder_feature_mode),
        decoder_device=str(args.decoder_device),
        decoder_batch_size=int(args.decoder_batch_size),
        require_decoder_enabled=bool(int(args.require_decoder_enabled)),
    )
    seeds = _parse_int_csv(args.multi_seed_values)
    if not seeds:
        raise RuntimeError("No multi-seed values")

    per_seed: List[Dict[str, Any]] = []
    primary_aucs: List[float] = []
    lexical_variant_aucs: List[float] = []
    lexical_control_probe_aucs: List[float] = []
    pooled_primary: List[Dict[str, Any]] = []
    overlap_total = 0

    for seed in seeds:
        split = _make_split(rows_master, argparse.Namespace(**{**vars(args), "trace_split_seed": int(seed)}))
        rows_train = split["train_rows"]
        rows_test = split["test_rows"]
        overlap_total += int((split["split_diagnostics"] or {}).get("trace_overlap_count", 0))
        out = _eval_wrong_lexical(
            rows_train,
            rows_test,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay_base),
            device=str(args.device),
        )
        wauc = out.get("primary_member_auroc")
        if not isinstance(wauc, (int, float)):
            wauc = out.get("wrong_intermediate_auroc")
        lauc = out.get("lexical_variant_auroc")
        lcp_auc = out.get("lexical_control_probe_auroc")
        if isinstance(wauc, (int, float)):
            primary_aucs.append(float(wauc))
        if isinstance(lauc, (int, float)):
            lexical_variant_aucs.append(float(lauc))
        if isinstance(lcp_auc, (int, float)):
            lexical_control_probe_aucs.append(float(lcp_auc))

        scored = list(out.get("scored_rows", []))
        for r in scored:
            if str(r.get("variant")) in {"lexical_consistent_swap", "pair_ambiguous"}:
                continue
            pooled_primary.append(
                {
                    "trace_id": f"{seed}:{r.get('trace_id')}",
                    "pair_id": f"{seed}:{r.get('pair_id')}",
                    "label": int(r.get("label", 0)),
                    "score": float(r.get("score", 0.0)),
                }
            )

        per_seed.append(
            {
                "seed": int(seed),
                "split_diagnostics": split["split_diagnostics"],
                "train_exclusion_diagnostics": split["train_exclusion_diagnostics"],
                "primary_member_auroc": wauc,
                "wrong_intermediate_auroc": out.get("wrong_intermediate_auroc"),
                "lexical_variant_auroc": lauc,
                "lexical_control_probe_auroc": lcp_auc,
                "primary_member_count": out.get("primary_member_count"),
                "wrong_intermediate_count": out.get("wrong_intermediate_count"),
                "lexical_variant_count": out.get("lexical_variant_count"),
                "lexical_control_probe_count": out.get("lexical_control_probe_count"),
                "loss_end": out.get("loss_end"),
            }
        )

    pooled_auc = _roc_auc([(float(r["score"]), int(r["label"])) for r in pooled_primary])
    pooled_boot = _bootstrap_group_auc(
        pooled_primary,
        group_key="pair_id",
        n_bootstrap=int(args.wrong_intermediate_bootstrap_n),
        seed=int(args.wrong_intermediate_bootstrap_seed),
        cpu_workers=max(1, int(args.cpu_workers)),
    )

    mean = _summ(primary_aucs).get("mean")
    std = _summ(primary_aucs).get("std")
    ci_low = pooled_boot.get("ci95_lower")
    multiseed_pass = bool(
        isinstance(mean, (int, float))
        and isinstance(std, (int, float))
        and isinstance(ci_low, (int, float))
        and float(mean) >= 0.70
        and float(std) <= 0.08
        and float(ci_low) >= 0.65
        and int(overlap_total) == 0
    )

    return {
        "schema_version": "phase7_optionc_stress_multiseed_v1",
        "status": "ok",
        "meta": meta,
        "decoder_features_enabled": bool(meta.get("decoder_features_enabled", False)),
        "decoder_feature_block_status": str(meta.get("decoder_feature_block_status", "unknown")),
        "logical_decoder_feature_mode": str(meta.get("logical_decoder_feature_mode", "")),
        "feature_count_total": int(meta.get("feature_count_total", meta.get("feature_count", 0))),
        "sae_feature_count": int(meta.get("sae_feature_count", 0)),
        "decoder_feature_count": int(meta.get("decoder_feature_count", 0)),
        "sae_layer_allowlist": list(meta.get("sae_layer_allowlist", [])),
        "seeds": [int(x) for x in seeds],
        "per_seed": per_seed,
        "primary_member_summary": _summ(primary_aucs),
        "wrong_intermediate_summary": _summ(primary_aucs),
        "lexical_summary": _summ(lexical_variant_aucs),
        "lexical_control_probe_summary": _summ(lexical_control_probe_aucs),
        "pooled_primary_member": {
            "auroc": pooled_auc,
            "ci95": {
                "lower": pooled_boot.get("ci95_lower"),
                "upper": pooled_boot.get("ci95_upper"),
                "defined": bool(pooled_boot.get("defined")),
                "bootstrap_n": int(args.wrong_intermediate_bootstrap_n),
            },
            "row_count": int(len(pooled_primary)),
        },
        "pooled_wrong_intermediate": {
            "auroc": pooled_auc,
            "ci95": {
                "lower": pooled_boot.get("ci95_lower"),
                "upper": pooled_boot.get("ci95_upper"),
                "defined": bool(pooled_boot.get("defined")),
                "bootstrap_n": int(args.wrong_intermediate_bootstrap_n),
            },
            "row_count": int(len(pooled_primary)),
        },
        "cv_trace_overlap_count": int(overlap_total),
        "criteria": {
            "mean_min": 0.70,
            "std_max": 0.08,
            "pooled_ci_lower_min": 0.65,
        },
        "multiseed_pass": bool(multiseed_pass),
        "timestamp": datetime.now().isoformat(),
    }


def run_final(args: argparse.Namespace) -> Dict[str, Any]:
    perm = load_json(args.permutation_json)
    abr = load_json(args.ablation_reg_json)
    ms = load_json(args.multiseed_json)

    p_primary = bool(perm.get("p_value_primary_pass") is True)
    p_legacy = bool(perm.get("legacy_strict_pass") is True)
    reg_pass = bool(abr.get("regularization_pass") is True)
    ms_pass = bool(ms.get("multiseed_pass") is True)

    final_primary = "pass" if (p_primary and reg_pass and ms_pass) else "fail"
    final_legacy = "pass" if (p_legacy and reg_pass and ms_pass) else "fail"

    out = {
        "schema_version": "phase7_optionc_stress_final_v1",
        "status": "ok",
        "inputs": {
            "permutation_json": str(args.permutation_json),
            "ablation_reg_json": str(args.ablation_reg_json),
            "multiseed_json": str(args.multiseed_json),
        },
        "feature_config": {
            "decoder_features_enabled": bool(perm.get("decoder_features_enabled", False)),
            "decoder_feature_block_status": str(perm.get("decoder_feature_block_status", "unknown")),
            "logical_decoder_feature_mode": str(perm.get("logical_decoder_feature_mode", "")),
            "feature_count_total": perm.get("feature_count_total"),
            "sae_feature_count": perm.get("sae_feature_count"),
            "decoder_feature_count": perm.get("decoder_feature_count"),
            "sae_layer_allowlist": perm.get("sae_layer_allowlist"),
        },
        "permutation": {
            "observed_primary_member_auroc": perm.get("observed_primary_member_auroc"),
            "observed_wrong_intermediate_auroc": perm.get("observed_wrong_intermediate_auroc"),
            "observed_lexical_control_probe_auroc": perm.get("observed_lexical_control_probe_auroc"),
            "observed_lexical_variant_auroc_legacy": perm.get("observed_lexical_variant_auroc"),
            "distribution": perm.get("wrong_intermediate_auroc_distribution"),
            "primary_distribution": perm.get("primary_member_auroc_distribution"),
            "empirical_p_value": perm.get("empirical_p_value"),
            "p_value_primary_pass": bool(perm.get("p_value_primary_pass") is True),
            "legacy_strict_pass": bool(perm.get("legacy_strict_pass") is True),
        },
        "regularization": {
            "regularization_pass": bool(reg_pass),
            "details": abr.get("regularization"),
        },
        "multiseed": {
            "multiseed_pass": bool(ms_pass),
            "primary_member_summary": ms.get("primary_member_summary"),
            "wrong_intermediate_summary": ms.get("wrong_intermediate_summary"),
            "lexical_control_probe_summary": ms.get("lexical_control_probe_summary"),
            "lexical_variant_summary_legacy": ms.get("lexical_summary"),
            "pooled_wrong_intermediate": ms.get("pooled_wrong_intermediate"),
            "pooled_primary_member": ms.get("pooled_primary_member"),
        },
        "lexical_metric_definition": {
            "primary": "lexical_control_probe_auroc (eval-aligned): AUROC among label==0 rows for lexical_control vs non-lexical_control",
            "legacy": "lexical_variant_auroc: AUROC within lexical_consistent_swap rows using original labels",
        },
        "final_verdict_primary": str(final_primary),
        "final_verdict_legacy": str(final_legacy),
        "timestamp": datetime.now().isoformat(),
    }

    lines = [
        "# Option C Stress Summary",
        "",
        f"- Final verdict (primary/p-value): `{final_primary}`",
        f"- Final verdict (legacy strict): `{final_legacy}`",
        f"- Observed primary-member AUROC: `{perm.get('observed_primary_member_auroc')}`",
        f"- Observed lexical-control probe AUROC (eval-aligned): `{perm.get('observed_lexical_control_probe_auroc')}`",
        f"- Observed lexical-variant AUROC (legacy): `{perm.get('observed_lexical_variant_auroc')}`",
        f"- Empirical permutation p-value: `{perm.get('empirical_p_value')}`",
        f"- Regularization pass: `{reg_pass}`",
        f"- Multi-seed pass: `{ms_pass}`",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=list(TASK_CHOICES), required=True)
    p.add_argument("--run-id", default="")
    p.add_argument("--run-tag", default="")

    p.add_argument("--paired-dataset", default="")
    p.add_argument("--partials", nargs="*", default=[])
    p.add_argument("--sae-layer-allowlist", default="", help="Optional CSV SAE layer allowlist.")
    p.add_argument("--decoder-checkpoint", default="", help="Optional decoder checkpoint to inject decoder transition features.")
    p.add_argument("--decoder-domain", choices=["auto", "arithmetic", "prontoqa", "entailmentbank"], default="auto")
    p.add_argument("--logical-decoder-feature-mode", choices=["full", "truth_inference_only"], default="full")
    p.add_argument("--decoder-device", default="cuda:0")
    p.add_argument("--decoder-batch-size", type=int, default=128)
    p.add_argument(
        "--require-decoder-enabled",
        type=int,
        default=0,
        help="If 1 and decoder checkpoint is set, fail if decoder feature block is not enabled.",
    )

    p.add_argument("--train-exclude-variants", default="order_flip_only,answer_first_order_flip,reordered_steps")
    p.add_argument("--trace-test-fraction", type=float, default=0.20)
    p.add_argument("--trace-split-seed", type=int, default=20260306)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight-decay-base", type=float, default=0.0001)
    p.add_argument("--weight-decay-values", default="0.0001,0.01,0.1,1.0")
    p.add_argument("--device", default="cpu")

    p.add_argument("--permutation-runs", type=int, default=1000)
    p.add_argument("--permutation-seed", type=int, default=20260308)

    p.add_argument("--multi-seed-values", default="20260307,20260308,20260309,20260310,20260311,20260312,20260313,20260314,20260315,20260316")
    p.add_argument("--wrong-intermediate-bootstrap-n", type=int, default=1000)
    p.add_argument("--wrong-intermediate-bootstrap-seed", type=int, default=20260307)
    p.add_argument("--cpu-workers", type=int, default=0)

    p.add_argument("--permutation-json", default="")
    p.add_argument("--ablation-reg-json", default="")
    p.add_argument("--multiseed-json", default="")

    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    task = str(args.task)
    if int(args.cpu_workers) <= 0:
        args.cpu_workers = max(1, min(16, (os.cpu_count() or 8) // 2 if (os.cpu_count() or 8) > 1 else 1))

    if task in {"permutation", "ablation_reg", "multiseed"}:
        if not str(args.paired_dataset).strip():
            raise RuntimeError("--paired-dataset is required")
        if not args.partials:
            raise RuntimeError("--partials is required")

    if task == "permutation":
        out = run_permutation(args)
    elif task == "ablation_reg":
        out = run_ablation_reg(args)
    elif task == "multiseed":
        out = run_multiseed(args)
    else:
        if not (args.permutation_json and args.ablation_reg_json and args.multiseed_json and args.output_md):
            raise RuntimeError("final task requires --permutation-json --ablation-reg-json --multiseed-json --output-md")
        out = run_final(args)

    save_json(args.output_json, out)
    print(f"Saved stress output -> {args.output_json}")


if __name__ == "__main__":
    main()

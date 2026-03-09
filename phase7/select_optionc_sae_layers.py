#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover
    from .common import load_json, save_json
except ImportError:  # pragma: no cover
    from common import load_json, save_json


class Row:
    __slots__ = ("member_id", "pair_id", "pair_type", "label", "features")

    def __init__(self, member_id: str, pair_id: str, pair_type: str, label: int, features: List[float]):
        self.member_id = str(member_id)
        self.pair_id = str(pair_id)
        self.pair_type = str(pair_type)
        self.label = int(label)
        self.features = list(float(x) for x in features)


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


def _mean_std(vals: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(x) for x in vals if isinstance(x, (int, float)) and not math.isnan(float(x))]
    if not xs:
        return None, None
    mean = float(sum(xs) / len(xs))
    if len(xs) <= 1:
        return mean, 0.0
    var = float(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
    return mean, float(math.sqrt(max(0.0, var)))


def _load_rows(paired_dataset: str, partials: Sequence[str]) -> Tuple[List[Row], List[str], Dict[str, Any]]:
    ds = load_json(paired_dataset)
    members = {
        str(m.get("member_id", "")): dict(m)
        for m in list(ds.get("members", []))
        if isinstance(m, dict) and str(m.get("member_id", ""))
    }

    feats_by_member: Dict[str, Dict[str, float]] = {}
    feature_name_set: set[str] = set()
    for p in partials:
        payload = load_json(p)
        for m in list(payload.get("members", [])):
            if not isinstance(m, dict):
                continue
            mid = str(m.get("member_id", ""))
            if not mid:
                continue
            f = m.get("features", {})
            if not isinstance(f, dict):
                continue
            rec = feats_by_member.setdefault(mid, {})
            for k, v in f.items():
                if not isinstance(v, (int, float)):
                    continue
                fn = str(k)
                if not fn.startswith("layer") or ":" not in fn:
                    continue
                rec[fn] = float(v)
                feature_name_set.add(fn)

    feature_names = sorted(feature_name_set)
    rows: List[Row] = []
    dropped = 0
    for mid, fmap in feats_by_member.items():
        meta = members.get(mid)
        if not meta:
            dropped += 1
            continue
        if not bool(meta.get("label_defined", False)):
            dropped += 1
            continue
        if bool(meta.get("pair_ambiguous", False)):
            dropped += 1
            continue
        vals: List[float] = []
        ok = True
        for fn in feature_names:
            v = fmap.get(fn)
            if not isinstance(v, (int, float)):
                ok = False
                break
            vals.append(float(v))
        if not ok:
            dropped += 1
            continue
        rows.append(
            Row(
                member_id=str(mid),
                pair_id=str(meta.get("pair_id", "")),
                pair_type=str(meta.get("pair_type", "")),
                label=int(meta.get("label_binary", 0)),
                features=vals,
            )
        )

    meta = {
        "paired_dataset": str(paired_dataset),
        "paired_dataset_sha256": ds.get("rows_sha256"),
        "member_count_dataset": int(len(members)),
        "rows_after_join": int(len(rows)),
        "rows_dropped": int(dropped),
        "feature_count": int(len(feature_names)),
        "partial_count": int(len(partials)),
    }
    return rows, feature_names, meta


def _split_by_pair(rows: Sequence[Row], test_fraction: float, seed: int) -> Tuple[List[Row], List[Row], Dict[str, Any]]:
    pair_ids = sorted({str(r.pair_id) for r in rows if str(r.pair_id)})
    if len(pair_ids) < 2:
        raise RuntimeError("Need at least 2 pair IDs")
    n_test = max(1, int(round(len(pair_ids) * float(test_fraction))))
    n_test = min(n_test, max(1, len(pair_ids) - 1))
    rng = random.Random(int(seed))
    ids = list(pair_ids)
    rng.shuffle(ids)
    test_set = set(ids[:n_test])
    train_rows = [r for r in rows if str(r.pair_id) not in test_set]
    test_rows = [r for r in rows if str(r.pair_id) in test_set]
    return train_rows, test_rows, {
        "pair_count_total": int(len(pair_ids)),
        "pair_count_train": int(len(pair_ids) - len(test_set)),
        "pair_count_test": int(len(test_set)),
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "pair_overlap_count": 0,
        "trace_overlap_count": 0,
    }


def _apply_train_exclusion(rows: Sequence[Row], excluded_pair_types: Sequence[str]) -> Tuple[List[Row], Dict[str, int]]:
    excluded = set(str(x) for x in excluded_pair_types if str(x))
    kept: List[Row] = []
    dropped: Dict[str, int] = {}
    for r in rows:
        pt = str(r.pair_type)
        if pt in excluded:
            dropped[pt] = int(dropped.get(pt, 0)) + 1
            continue
        kept.append(r)
    return kept, {k: int(v) for k, v in sorted(dropped.items())}


def _build_folds(pair_ids: Sequence[str], k: int, seed: int) -> List[List[str]]:
    ids = sorted(set(str(x) for x in pair_ids if str(x)))
    if len(ids) < 2:
        return [ids]
    kk = max(2, min(int(k), len(ids)))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    folds: List[List[str]] = [[] for _ in range(kk)]
    for i, pid in enumerate(ids):
        folds[i % kk].append(pid)
    return [f for f in folds if f]


def _subset_rows(rows: Sequence[Row], idx: Sequence[int]) -> List[Row]:
    ii = [int(i) for i in idx]
    out: List[Row] = []
    for r in rows:
        out.append(Row(r.member_id, r.pair_id, r.pair_type, r.label, [float(r.features[i]) for i in ii]))
    return out


def _train_logreg(
    train_rows: Sequence[Row],
    test_rows: Sequence[Row],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> Optional[float]:
    if not train_rows or not test_rows:
        return None
    d = len(train_rows[0].features)
    x_train = torch.tensor([r.features for r in train_rows], dtype=torch.float32)
    y_train = torch.tensor([int(r.label) for r in train_rows], dtype=torch.float32)
    x_test = torch.tensor([r.features for r in test_rows], dtype=torch.float32)
    y_test = [int(r.label) for r in test_rows]
    if sum(y_test) <= 0 or sum(y_test) >= len(y_test):
        return None
    if float(y_train.sum().item()) <= 0 or float(y_train.sum().item()) >= float(y_train.numel()):
        return None

    mu = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std

    dev = torch.device(str(device))
    x_train = x_train.to(dev)
    y_train = y_train.to(dev)
    x_test = x_test.to(dev)

    torch.manual_seed(int(seed))
    w = torch.zeros((d, 1), dtype=torch.float32, device=dev, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=float(lr), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = (x_train @ w + b).view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid((x_test @ w + b).view(-1)).detach().cpu().tolist()
    return _roc_auc([(float(s), int(y)) for s, y in zip(probs, y_test)])


def _rank_layers(
    rows_train_pool: Sequence[Row],
    feature_names: Sequence[str],
    *,
    excluded_pair_types: Sequence[str],
    cv_folds: int,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> List[Dict[str, Any]]:
    layer_to_names: Dict[int, List[str]] = {}
    for fn in feature_names:
        if not fn.startswith("layer") or ":" not in fn:
            continue
        try:
            layer = int(fn.split(":", 1)[0].replace("layer", ""))
        except Exception:
            continue
        layer_to_names.setdefault(layer, []).append(fn)
    idx_map = {str(n): i for i, n in enumerate(feature_names)}

    pair_ids = sorted({str(r.pair_id) for r in rows_train_pool})
    folds = _build_folds(pair_ids, int(cv_folds), int(seed))
    out_rows: List[Dict[str, Any]] = []
    for layer in sorted(layer_to_names.keys()):
        idx = [idx_map[n] for n in layer_to_names[layer] if n in idx_map]
        if not idx:
            continue
        layer_rows = _subset_rows(rows_train_pool, idx)
        fold_aucs: List[float] = []
        for fi, fold in enumerate(folds):
            test_set = set(str(x) for x in fold)
            tr_raw = [r for r in layer_rows if str(r.pair_id) not in test_set]
            te = [r for r in layer_rows if str(r.pair_id) in test_set]
            tr, _ = _apply_train_exclusion(tr_raw, excluded_pair_types)
            auc = _train_logreg(
                tr,
                te,
                epochs=int(epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                device=str(device),
                seed=int(seed) + 1000 * int(layer) + int(fi),
            )
            if isinstance(auc, (int, float)):
                fold_aucs.append(float(auc))
        mean_auc, std_auc = _mean_std(fold_aucs)
        out_rows.append(
            {
                "layer": int(layer),
                "feature_count": int(len(idx)),
                "cv_valid_folds": int(len(fold_aucs)),
                "cv_fold_aurocs": [float(x) for x in fold_aucs],
                "cv_mean_auroc": mean_auc,
                "cv_std_auroc": std_auc,
            }
        )

    out_rows.sort(
        key=lambda r: (
            -float(r["cv_mean_auroc"]) if isinstance(r.get("cv_mean_auroc"), (int, float)) else float("inf"),
            int(r["layer"]),
        )
    )
    return out_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-dataset", required=True)
    p.add_argument("--partials", nargs="+", required=True)
    p.add_argument("--train-exclude-pair-types", default="lexical_control")
    p.add_argument("--train-test-fraction", type=float, default=0.20)
    p.add_argument("--split-seed", type=int, default=20260309)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows, feature_names, meta = _load_rows(args.paired_dataset, args.partials)
    if len(rows) < 40:
        raise RuntimeError(f"Insufficient rows after feature join: {len(rows)}")

    rows_train_pool, rows_holdout, split_diag = _split_by_pair(rows, float(args.train_test_fraction), int(args.split_seed))
    excluded = _parse_csv(args.train_exclude_pair_types)
    train_eff, dropped = _apply_train_exclusion(rows_train_pool, excluded)
    if len(train_eff) < 30:
        raise RuntimeError("Too few training rows after pair-type exclusion")

    ranking = _rank_layers(
        train_eff,
        feature_names,
        excluded_pair_types=excluded,
        cv_folds=int(args.cv_folds),
        seed=int(args.split_seed) + 17,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=str(args.device),
    )
    if not ranking:
        raise RuntimeError("No layer rankings produced")

    ranked_defined = [r for r in ranking if isinstance(r.get("cv_mean_auroc"), (int, float))]
    if not ranked_defined:
        raise RuntimeError("No valid layer scores produced")

    top_k = max(1, int(args.top_k))
    selected_layers = [int(r["layer"]) for r in ranked_defined[:top_k]]
    selected_layers_csv = ",".join(str(x) for x in selected_layers)

    out = {
        "schema_version": "phase7_optionc_sae_layer_selection_v1",
        "status": "ok",
        "paired_dataset": str(args.paired_dataset),
        "partials": [str(x) for x in args.partials],
        "meta": meta,
        "selection_policy": {
            "train_only": True,
            "train_exclude_pair_types": list(excluded),
            "train_test_fraction": float(args.train_test_fraction),
            "split_seed": int(args.split_seed),
            "cv_folds": int(args.cv_folds),
            "top_k": int(top_k),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": str(args.device),
        },
        "split_diagnostics": {
            **split_diag,
            "train_pool_rows_pre_exclusion": int(len(rows_train_pool)),
            "train_pool_rows_post_exclusion": int(len(train_eff)),
            "holdout_rows": int(len(rows_holdout)),
            "excluded_train_rows_by_pair_type": {k: int(v) for k, v in sorted(dropped.items())},
            "excluded_train_rows_total": int(sum(dropped.values())),
        },
        "layer_ranking": ranking,
        "selected_layers": [int(x) for x in selected_layers],
        "selected_layers_csv": str(selected_layers_csv),
        "selected_layer_count": int(len(selected_layers)),
        "timestamp": datetime.now().isoformat(),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output_json, out)
    print(f"Saved layer selection manifest -> {args.output_json}")


if __name__ == "__main__":
    main()

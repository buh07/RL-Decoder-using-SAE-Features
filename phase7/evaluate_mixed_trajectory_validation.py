#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover
    from .common import load_json, save_json
except ImportError:  # pragma: no cover
    from common import load_json, save_json


@dataclass
class EvalData:
    trace_ids: np.ndarray
    variants: np.ndarray
    labels: np.ndarray
    X1: np.ndarray
    X2: np.ndarray
    X3: np.ndarray
    block_feature_names: Dict[str, List[str]]


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


def _safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if y.size == 0:
        return None
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return None


def _build_trace_folds(trace_ids: Sequence[str], k: int, seed: int) -> List[List[str]]:
    uniq = sorted(set(str(t) for t in trace_ids))
    if len(uniq) < 2:
        raise RuntimeError("Need at least 2 traces for fold building")
    kk = max(2, min(int(k), len(uniq)))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    folds: List[List[str]] = [[] for _ in range(kk)]
    for i, t in enumerate(uniq):
        folds[i % kk].append(t)
    return [f for f in folds if f]


def _split_by_trace(trace_ids: np.ndarray, labels: np.ndarray, test_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    uniq = sorted(set(str(t) for t in trace_ids.tolist()))
    if len(uniq) < 2:
        raise RuntimeError("Need at least 2 traces")
    n_test = max(1, int(round(len(uniq) * float(test_fraction))))
    n_test = min(n_test, len(uniq) - 1)
    for attempt in range(200):
        rng = random.Random(int(seed) + attempt)
        tids = list(uniq)
        rng.shuffle(tids)
        test_set = set(tids[:n_test])
        train_idx = np.array([i for i, t in enumerate(trace_ids.tolist()) if str(t) not in test_set], dtype=np.int64)
        test_idx = np.array([i for i, t in enumerate(trace_ids.tolist()) if str(t) in test_set], dtype=np.int64)
        ytr = labels[train_idx]
        yte = labels[test_idx]
        if (ytr == 1).sum() > 0 and (ytr == 0).sum() > 0 and (yte == 1).sum() > 0 and (yte == 0).sum() > 0:
            return train_idx, test_idx, {
                "trace_split_seed_used": int(seed) + attempt,
                "trace_count_total": int(len(uniq)),
                "trace_count_train": int(len(uniq) - len(test_set)),
                "trace_count_test": int(len(test_set)),
                "train_rows": int(train_idx.size),
                "test_rows": int(test_idx.size),
                "train_pos": int((ytr == 1).sum()),
                "train_neg": int((ytr == 0).sum()),
                "test_pos": int((yte == 1).sum()),
                "test_neg": int((yte == 0).sum()),
                "trace_overlap_count": 0,
            }
    raise RuntimeError("Could not create class-balanced trace split")


def _variant_defined_binary(labels: np.ndarray, variants: np.ndarray, target_variant: str) -> bool:
    mask = np.array([str(v) == str(target_variant) for v in variants.tolist()], dtype=bool)
    if not bool(mask.any()):
        return False
    ys = labels[mask]
    return bool((ys == 1).sum() > 0 and (ys == 0).sum() > 0)


def _apply_train_exclusion(
    train_idx: np.ndarray,
    variants: np.ndarray,
    labels: np.ndarray,
    excluded_variants: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    excl = set(str(v) for v in excluded_variants)
    kept: List[int] = []
    dropped: Dict[str, int] = {}
    for i in train_idx.tolist():
        if int(labels[i]) == 1 and str(variants[i]) in excl:
            key = str(variants[i])
            dropped[key] = int(dropped.get(key, 0)) + 1
            continue
        kept.append(int(i))
    return np.array(kept, dtype=np.int64), {k: int(v) for k, v in sorted(dropped.items())}


def _fit_predict_enet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    uniq_groups = np.unique(groups_train)
    model: Any
    used_cv = False
    if uniq_groups.size >= 3 and np.unique(y_train).size == 2:
        inner = min(5, int(uniq_groups.size))
        cv_obj = GroupKFold(n_splits=inner)
        splits = list(cv_obj.split(Xtr, y_train, groups=groups_train))
        model = LogisticRegressionCV(
            Cs=[0.01, 0.1, 1.0, 10.0],
            cv=splits,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.2, 0.5, 0.8],
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=5000,
            random_state=int(seed),
            n_jobs=1,
            refit=True,
        )
        used_cv = True
    else:
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=int(seed),
        )

    model.fit(Xtr, y_train)
    probs = model.predict_proba(Xte)[:, 1]
    coefs = np.asarray(model.coef_[0], dtype=np.float64)
    meta = {
        "used_nested_cv": bool(used_cv),
        "coef": coefs,
        "intercept": float(model.intercept_[0]),
        "scaler": scaler,
        "model": model,
    }
    if used_cv:
        meta["chosen_C"] = float(np.asarray(model.C_).reshape(-1)[0])
        if hasattr(model, "l1_ratio_"):
            meta["chosen_l1_ratio"] = float(np.asarray(model.l1_ratio_).reshape(-1)[0])
    return probs.astype(np.float64), meta


def _prepare_model_matrices(
    data: EvalData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    X1_tr, X1_te = data.X1[train_idx], data.X1[test_idx]
    X2_tr, X2_te = data.X2[train_idx], data.X2[test_idx]
    X3_tr, X3_te = data.X3[train_idx], data.X3[test_idx]

    X23_tr = np.concatenate([X2_tr, X3_tr], axis=1)
    X23_te = np.concatenate([X2_te, X3_te], axis=1)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X1_tr, X23_tr)
    res23_tr = X23_tr - ridge.predict(X1_tr)
    res23_te = X23_te - ridge.predict(X1_te)

    return {
        "M1": {
            "X_train": X1_tr,
            "X_test": X1_te,
            "feature_names": list(data.block_feature_names["B1_sae"]),
        },
        "M2": {
            "X_train": X23_tr,
            "X_test": X23_te,
            "feature_names": list(data.block_feature_names["B2_raw"] + data.block_feature_names["B3_proj"]),
        },
        "M3": {
            "X_train": np.concatenate([X1_tr, X23_tr], axis=1),
            "X_test": np.concatenate([X1_te, X23_te], axis=1),
            "feature_names": list(
                data.block_feature_names["B1_sae"] + data.block_feature_names["B2_raw"] + data.block_feature_names["B3_proj"]
            ),
        },
        "M4": {
            "X_train": np.concatenate([X1_tr, res23_tr], axis=1),
            "X_test": np.concatenate([X1_te, res23_te], axis=1),
            "feature_names": list(
                data.block_feature_names["B1_sae"] + [f"res_{x}" for x in (data.block_feature_names["B2_raw"] + data.block_feature_names["B3_proj"])]
            ),
        },
    }


def _bootstrap_tracepair_auc(
    trace_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    by_trace: Dict[str, List[Tuple[float, int]]] = {}
    for t, y, s in zip(trace_ids.tolist(), labels.tolist(), scores.tolist()):
        by_trace.setdefault(str(t), []).append((float(s), int(y)))
    valid = [t for t, rows in by_trace.items() if sum(y for _, y in rows) > 0 and sum(1 - y for _, y in rows) > 0]
    valid = sorted(valid)
    pooled = [x for t in valid for x in by_trace[t]]
    obs = _safe_auc(np.array([y for _, y in pooled]), np.array([s for s, _ in pooled])) if pooled else None
    out = {
        "defined": bool(obs is not None),
        "observed_auroc": obs,
        "ci95_lower": None,
        "ci95_upper": None,
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_effective_n": 0,
        "trace_pair_count": int(len(valid)),
        "row_count": int(len(pooled)),
    }
    if obs is None or not valid:
        return out
    rng = random.Random(int(seed))
    vals: List[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        sample: List[Tuple[float, int]] = []
        for __ in range(len(valid)):
            t = valid[rng.randrange(len(valid))]
            sample.extend(by_trace[t])
        y = np.array([yy for _, yy in sample], dtype=np.int64)
        s = np.array([ss for ss, _ in sample], dtype=np.float64)
        a = _safe_auc(y, s)
        if a is not None:
            vals.append(float(a))
    vals = sorted(vals)
    out["bootstrap_effective_n"] = int(len(vals))
    if vals:
        lo = vals[int(0.025 * (len(vals) - 1))]
        hi = vals[int(0.975 * (len(vals) - 1))]
        out["ci95_lower"] = float(lo)
        out["ci95_upper"] = float(hi)
    return out


def _bootstrap_delta_tracepair(
    trace_ids: np.ndarray,
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    by_trace: Dict[str, List[Tuple[float, float, int]]] = {}
    for t, y, sa, sb in zip(trace_ids.tolist(), labels.tolist(), scores_a.tolist(), scores_b.tolist()):
        by_trace.setdefault(str(t), []).append((float(sa), float(sb), int(y)))
    valid = [t for t, rows in by_trace.items() if sum(y for _, _, y in rows) > 0 and sum(1 - y for _, _, y in rows) > 0]
    valid = sorted(valid)
    pooled = [x for t in valid for x in by_trace[t]]
    if not pooled:
        return {
            "defined": False,
            "delta_observed": None,
            "delta_ci95": {"lower": None, "upper": None},
            "p_value_delta_le_zero": None,
            "bootstrap_n": int(n_bootstrap),
            "bootstrap_effective_n": 0,
        }

    y_obs = np.array([y for _, _, y in pooled], dtype=np.int64)
    a_obs = np.array([sa for sa, _, _ in pooled], dtype=np.float64)
    b_obs = np.array([sb for _, sb, _ in pooled], dtype=np.float64)
    auc_a = _safe_auc(y_obs, a_obs)
    auc_b = _safe_auc(y_obs, b_obs)
    delta_obs = (float(auc_a) - float(auc_b)) if auc_a is not None and auc_b is not None else None

    rng = random.Random(int(seed))
    deltas: List[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        sample: List[Tuple[float, float, int]] = []
        for __ in range(len(valid)):
            t = valid[rng.randrange(len(valid))]
            sample.extend(by_trace[t])
        y = np.array([yy for _, _, yy in sample], dtype=np.int64)
        sa = np.array([saa for saa, _, _ in sample], dtype=np.float64)
        sb = np.array([sbb for _, sbb, _ in sample], dtype=np.float64)
        aa = _safe_auc(y, sa)
        bb = _safe_auc(y, sb)
        if aa is not None and bb is not None:
            deltas.append(float(aa - bb))
    deltas = sorted(deltas)

    out = {
        "defined": bool(delta_obs is not None),
        "delta_observed": delta_obs,
        "delta_ci95": {"lower": None, "upper": None},
        "p_value_delta_le_zero": None,
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_effective_n": int(len(deltas)),
    }
    if deltas:
        lo = deltas[int(0.025 * (len(deltas) - 1))]
        hi = deltas[int(0.975 * (len(deltas) - 1))]
        out["delta_ci95"] = {"lower": float(lo), "upper": float(hi)}
        out["p_value_delta_le_zero"] = float(sum(1 for d in deltas if d <= 0.0) / len(deltas))
    return out


def _permutation_delta_test(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    obs_a = _safe_auc(labels, scores_a)
    obs_b = _safe_auc(labels, scores_b)
    if obs_a is None or obs_b is None:
        return {"defined": False, "delta_observed": None, "p_value_two_sided": None, "n_perm": int(n_perm)}
    delta_obs = float(obs_a - obs_b)
    rng = random.Random(int(seed))
    vals: List[float] = []
    for _ in range(max(1, int(n_perm))):
        swap = np.array([rng.random() < 0.5 for _ in range(labels.size)], dtype=bool)
        sa = scores_a.copy()
        sb = scores_b.copy()
        tmp = sa[swap].copy()
        sa[swap] = sb[swap]
        sb[swap] = tmp
        aa = _safe_auc(labels, sa)
        bb = _safe_auc(labels, sb)
        if aa is not None and bb is not None:
            vals.append(float(aa - bb))
    if not vals:
        return {"defined": False, "delta_observed": delta_obs, "p_value_two_sided": None, "n_perm": int(n_perm)}
    p = float(sum(1 for v in vals if abs(v) >= abs(delta_obs)) / len(vals))
    return {
        "defined": True,
        "delta_observed": delta_obs,
        "p_value_two_sided": p,
        "n_perm": int(n_perm),
    }


def _vif_matrix(X: np.ndarray) -> List[float]:
    n, p = X.shape
    if p <= 1:
        return [1.0 for _ in range(p)]
    vals: List[float] = []
    for j in range(p):
        y = X[:, j]
        Xo = np.delete(X, j, axis=1)
        reg = LinearRegression().fit(Xo, y)
        r2 = float(reg.score(Xo, y))
        vif = 1.0 / max(1e-8, 1.0 - r2)
        vals.append(float(vif))
    return vals


def _corr_matrix(X: np.ndarray) -> np.ndarray:
    if X.shape[0] <= 1:
        return np.eye(X.shape[1], dtype=np.float64)
    return np.corrcoef(X, rowvar=False)


def _shuffle_blocks_within_trace(
    X: np.ndarray,
    trace_ids: np.ndarray,
    block_slices: Dict[str, slice],
    *,
    seed: int,
    blocks: Sequence[str],
) -> np.ndarray:
    rng = random.Random(int(seed))
    out = X.copy()
    uniq = sorted(set(str(t) for t in trace_ids.tolist()))
    for t in uniq:
        idx = np.array([i for i, tv in enumerate(trace_ids.tolist()) if str(tv) == t], dtype=np.int64)
        if idx.size <= 1:
            continue
        for b in blocks:
            sl = block_slices[b]
            perm = idx.copy()
            rng.shuffle(perm)
            out[idx, sl] = out[perm, sl]
    return out


def _load_data(path: str | Path) -> EvalData:
    obj = load_json(path)
    if str(obj.get("status")) != "ok":
        raise RuntimeError(f"Feature artifact not ok: {obj.get('status')!r}")
    samples = list(obj.get("samples") or [])
    bfn = dict(obj.get("block_feature_names") or {})
    for k in ("B1_sae", "B2_raw", "B3_proj"):
        if k not in bfn:
            raise RuntimeError(f"Missing block feature names: {k}")

    all_keys = list(bfn["B1_sae"] + bfn["B2_raw"] + bfn["B3_proj"])
    rows: List[dict] = []
    for s in samples:
        ok = True
        for k in all_keys:
            if not isinstance(s.get(k), (int, float)):
                ok = False
                break
        if not ok:
            continue
        lbl = str(s.get("label", ""))
        if lbl not in {"faithful", "unfaithful"}:
            continue
        rows.append(s)
    if len(rows) < 20:
        raise RuntimeError("Insufficient fully-defined rows in feature artifact")

    trace_ids = np.array([str(r.get("trace_id", "")) for r in rows], dtype=object)
    variants = np.array([str(r.get("variant", "")) for r in rows], dtype=object)
    labels = np.array([1 if str(r.get("label")) == "unfaithful" else 0 for r in rows], dtype=np.int64)
    X1 = np.array([[float(r[k]) for k in bfn["B1_sae"]] for r in rows], dtype=np.float64)
    X2 = np.array([[float(r[k]) for k in bfn["B2_raw"]] for r in rows], dtype=np.float64)
    X3 = np.array([[float(r[k]) for k in bfn["B3_proj"]] for r in rows], dtype=np.float64)

    return EvalData(
        trace_ids=trace_ids,
        variants=variants,
        labels=labels,
        X1=X1,
        X2=X2,
        X3=X3,
        block_feature_names={k: list(v) for k, v in bfn.items()},
    )


def _evaluate_single_split(
    data: EvalData,
    *,
    test_fraction: float,
    split_seed: int,
    excluded_variants: Sequence[str],
    train_seed: int,
    bootstrap_n: int,
    threshold: float,
) -> Dict[str, Any]:
    selected: Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = None
    for attempt in range(64):
        train_idx, test_idx, split_diag = _split_by_trace(
            data.trace_ids,
            data.labels,
            test_fraction=float(test_fraction),
            seed=int(split_seed) + attempt,
        )
        if _variant_defined_binary(data.labels[test_idx], data.variants[test_idx], "wrong_intermediate"):
            split_diag["wrong_intermediate_defined_in_test"] = True
            split_diag["attempt"] = int(attempt)
            selected = (train_idx, test_idx, split_diag)
            break
    if selected is None:
        return {
            "status": "blocked_wrong_intermediate_undefined_in_test_split",
            "split_diagnostics": {
                "trace_split_seed_base": int(split_seed),
                "attempts": 64,
                "wrong_intermediate_defined_in_test": False,
            },
        }
    train_idx, test_idx, split_diag = selected
    train_eff, dropped = _apply_train_exclusion(train_idx, data.variants, data.labels, excluded_variants)

    y_train = data.labels[train_eff]
    if (y_train == 1).sum() == 0 or (y_train == 0).sum() == 0:
        return {
            "status": "blocked_invalid_train_after_exclusion",
            "split_diagnostics": split_diag,
            "dropped_train_positives_by_variant": dropped,
        }

    mats = _prepare_model_matrices(data, train_eff, test_idx)
    groups = data.trace_ids[train_eff]
    y_test = data.labels[test_idx]
    var_test = data.variants[test_idx]
    trace_test = data.trace_ids[test_idx]

    models = {}
    per_variant: Dict[str, Dict[str, Any]] = {}
    for mi, mk in enumerate(("M1", "M2", "M3", "M4")):
        probs, meta = _fit_predict_enet(
            mats[mk]["X_train"],
            y_train,
            groups,
            mats[mk]["X_test"],
            seed=int(train_seed) + (mi + 1) * 101,
        )
        models[mk] = {
            "test_probs": probs,
            "meta": meta,
            "overall_test_auroc": _safe_auc(y_test, probs),
        }

    uniq_variants = sorted(set(str(v) for v in var_test.tolist()))
    for mk in ("M1", "M2", "M3", "M4"):
        per_variant[mk] = {}
        for v in uniq_variants:
            m = np.array([str(x) == v for x in var_test.tolist()], dtype=bool)
            per_variant[mk][v] = {
                "auroc": _safe_auc(y_test[m], models[mk]["test_probs"][m]),
                "count": int(m.sum()),
                "pos": int((y_test[m] == 1).sum()),
                "neg": int((y_test[m] == 0).sum()),
            }

    wrong_mask = np.array([str(v) == "wrong_intermediate" for v in var_test.tolist()], dtype=bool)
    wrong_y = y_test[wrong_mask]
    wrong_t = trace_test[wrong_mask]
    wrong_scores = {mk: models[mk]["test_probs"][wrong_mask] for mk in models}

    wrong_ci_by_model = {
        mk: _bootstrap_tracepair_auc(
            wrong_t,
            wrong_y,
            wrong_scores[mk],
            n_bootstrap=int(bootstrap_n),
            seed=int(train_seed) + (i + 1) * 211,
        )
        for i, mk in enumerate(("M1", "M2", "M3", "M4"))
    }

    delta_boot = _bootstrap_delta_tracepair(
        wrong_t,
        wrong_y,
        wrong_scores["M3"],
        wrong_scores["M1"],
        n_bootstrap=int(bootstrap_n),
        seed=int(train_seed) + 919,
    )
    delta_perm = _permutation_delta_test(
        wrong_y,
        wrong_scores["M3"],
        wrong_scores["M1"],
        n_perm=2000,
        seed=int(train_seed) + 1337,
    )

    delta_obs = delta_boot.get("delta_observed")
    delta_lo = (delta_boot.get("delta_ci95") or {}).get("lower")
    practical_effect_pass = bool(
        isinstance(delta_obs, (int, float))
        and isinstance(delta_lo, (int, float))
        and float(delta_obs) >= 0.03
        and float(delta_lo) > 0.0
    )

    # Collinearity diagnostics on all rows (fixed data matrix).
    X_m3_all = np.concatenate([data.X1, data.X2, data.X3], axis=1)
    corr = _corr_matrix(X_m3_all)
    vifs = _vif_matrix(X_m3_all)
    n1 = data.X1.shape[1]
    n2 = data.X2.shape[1]
    n3 = data.X3.shape[1]
    sl = {
        "B1_sae": slice(0, n1),
        "B2_raw": slice(n1, n1 + n2),
        "B3_proj": slice(n1 + n2, n1 + n2 + n3),
    }

    def _block_vif_summary(block: str) -> Dict[str, Any]:
        arr = np.array(vifs[sl[block]], dtype=np.float64)
        return {
            "mean_vif": float(arr.mean()) if arr.size else None,
            "median_vif": float(np.median(arr)) if arr.size else None,
            "max_vif": float(arr.max()) if arr.size else None,
        }

    block_corr = {}
    for a, sa in sl.items():
        block_corr[a] = {}
        for b, sb in sl.items():
            sub = np.abs(corr[sa, sb])
            block_corr[a][b] = float(sub.mean()) if sub.size else None

    # Permutation importance by block on wrong_intermediate for M3 (single split).
    m3 = models["M3"]
    m3_scaler = m3["meta"]["scaler"]
    m3_model = m3["meta"]["model"]
    X_m3_test = mats["M3"]["X_test"].copy()
    base_wrong_auc = _safe_auc(wrong_y, wrong_scores["M3"])
    perm_block = {}
    rng = random.Random(int(train_seed) + 2025)
    for bi, b in enumerate(("B1_sae", "B2_raw", "B3_proj")):
        drops: List[float] = []
        bs = sl[b]
        for _ in range(100):
            Xp = X_m3_test.copy()
            perm_idx = np.arange(Xp.shape[0])
            rng.shuffle(perm_idx)
            Xp[:, bs] = Xp[perm_idx, bs]
            probs = m3_model.predict_proba(m3_scaler.transform(Xp))[:, 1]
            pa = _safe_auc(wrong_y, probs[wrong_mask])
            if pa is not None and base_wrong_auc is not None:
                drops.append(float(base_wrong_auc - pa))
        perm_block[b] = {
            "mean_auc_drop": float(statistics.fmean(drops)) if drops else None,
            "p05_auc_drop": float(np.percentile(drops, 5)) if drops else None,
            "p95_auc_drop": float(np.percentile(drops, 95)) if drops else None,
            "n": int(len(drops)),
        }

    # Falsification: shuffle B2/B3 within-trace on train and retrain M3.
    X_train_real = mats["M3"]["X_train"]
    X_train_shuf = _shuffle_blocks_within_trace(
        X_train_real,
        data.trace_ids[train_eff],
        sl,
        seed=int(train_seed) + 404,
        blocks=["B2_raw", "B3_proj"],
    )
    shuf_probs, _ = _fit_predict_enet(
        X_train_shuf,
        y_train,
        data.trace_ids[train_eff],
        mats["M3"]["X_test"],
        seed=int(train_seed) + 505,
    )
    shuf_wrong_auc = _safe_auc(wrong_y, shuf_probs[wrong_mask])

    pure_redundancy_flag = bool(
        isinstance(perm_block["B2_raw"].get("mean_auc_drop"), (int, float))
        and isinstance(perm_block["B3_proj"].get("mean_auc_drop"), (int, float))
        and float(perm_block["B2_raw"]["mean_auc_drop"]) <= 0.005
        and float(perm_block["B3_proj"]["mean_auc_drop"]) <= 0.005
    )

    return {
        "status": "ok",
        "split_diagnostics": split_diag,
        "dropped_train_positives_by_variant": dropped,
        "single_split_models": {
            mk: {
                "overall_test_auroc": models[mk]["overall_test_auroc"],
                "wrong_intermediate_test_auroc": _safe_auc(wrong_y, wrong_scores[mk]),
                "wrong_intermediate_ci95": {
                    "lower": wrong_ci_by_model[mk].get("ci95_lower"),
                    "upper": wrong_ci_by_model[mk].get("ci95_upper"),
                    "bootstrap_n": int(bootstrap_n),
                    "trace_pair_count": int(wrong_ci_by_model[mk].get("trace_pair_count", 0)),
                },
            }
            for mk in ("M1", "M2", "M3", "M4")
        },
        "variant_stratified_test_auroc": per_variant,
        "delta_m3_vs_m1": {
            "delta_auroc": delta_obs,
            "delta_ci95": delta_boot.get("delta_ci95"),
            "bootstrap_effective_n": int(delta_boot.get("bootstrap_effective_n", 0)),
            "p_value_delta_le_zero": delta_boot.get("p_value_delta_le_zero"),
            "permutation_test": delta_perm,
            "practical_effect_pass": practical_effect_pass,
        },
        "collinearity_diagnostics": {
            "pairwise_feature_corr_matrix": corr.tolist(),
            "feature_vif": [float(x) for x in vifs],
            "block_mean_abs_corr": block_corr,
            "block_vif_summary": {b: _block_vif_summary(b) for b in ("B1_sae", "B2_raw", "B3_proj")},
            "pure_redundancy_flag": pure_redundancy_flag,
        },
        "falsification_checks": {
            "m3_wrong_intermediate_auroc": _safe_auc(wrong_y, wrong_scores["M3"]),
            "m3_shuffled_blocks_wrong_intermediate_auroc": shuf_wrong_auc,
            "m3_vs_shuffled_delta": (
                (float(_safe_auc(wrong_y, wrong_scores["M3"])) - float(shuf_wrong_auc))
                if _safe_auc(wrong_y, wrong_scores["M3"]) is not None and shuf_wrong_auc is not None
                else None
            ),
            "permutation_importance_by_block": perm_block,
        },
    }


def _evaluate_outer_cv(
    data: EvalData,
    *,
    cv_folds: int,
    cv_seed: int,
    excluded_variants: Sequence[str],
    train_seed: int,
    bootstrap_n: int,
    threshold: float,
    cv_min_valid_folds: int,
) -> Dict[str, Any]:
    folds = _build_trace_folds(data.trace_ids.tolist(), k=int(cv_folds), seed=int(cv_seed))
    oof: Dict[str, List[Tuple[str, int, float]]] = {mk: [] for mk in ("M1", "M2", "M3", "M4")}
    fold_rows: List[Dict[str, Any]] = []
    coef_by_model: Dict[str, List[np.ndarray]] = {mk: [] for mk in ("M1", "M2", "M3", "M4")}

    n_overlap = 0
    for fi, fold_tids in enumerate(folds):
        test_set = set(str(t) for t in fold_tids)
        train_idx = np.array([i for i, t in enumerate(data.trace_ids.tolist()) if str(t) not in test_set], dtype=np.int64)
        test_idx = np.array([i for i, t in enumerate(data.trace_ids.tolist()) if str(t) in test_set], dtype=np.int64)
        overlap = len(set(data.trace_ids[train_idx].tolist()).intersection(test_set))
        n_overlap += int(overlap)
        train_eff, dropped = _apply_train_exclusion(train_idx, data.variants, data.labels, excluded_variants)
        ytr = data.labels[train_eff]
        if ytr.size == 0 or (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
            fold_rows.append({
                "fold_index": int(fi),
                "status": "blocked_invalid_train_after_exclusion",
                "trace_overlap_count": int(overlap),
                "dropped_train_positives_by_variant": dropped,
            })
            continue

        mats = _prepare_model_matrices(data, train_eff, test_idx)
        yte = data.labels[test_idx]
        vte = data.variants[test_idx]
        tte = data.trace_ids[test_idx]

        fold_entry: Dict[str, Any] = {
            "fold_index": int(fi),
            "status": "ok",
            "trace_overlap_count": int(overlap),
            "test_trace_count": int(len(test_set)),
            "train_rows": int(train_eff.size),
            "test_rows": int(test_idx.size),
            "dropped_train_positives_by_variant": dropped,
            "wrong_intermediate_aurocs": {},
        }

        for mi, mk in enumerate(("M1", "M2", "M3", "M4")):
            probs, meta = _fit_predict_enet(
                mats[mk]["X_train"],
                ytr,
                data.trace_ids[train_eff],
                mats[mk]["X_test"],
                seed=int(train_seed) + 1000 * (fi + 1) + (mi + 1),
            )
            coef_by_model[mk].append(np.asarray(meta["coef"], dtype=np.float64))
            m_wrong = np.array([str(v) == "wrong_intermediate" for v in vte.tolist()], dtype=bool)
            auc = _safe_auc(yte[m_wrong], probs[m_wrong])
            fold_entry["wrong_intermediate_aurocs"][mk] = auc
            for t, y, s, vv in zip(tte.tolist(), yte.tolist(), probs.tolist(), vte.tolist()):
                if str(vv) == "wrong_intermediate":
                    oof[mk].append((str(t), int(y), float(s)))
        fold_rows.append(fold_entry)

    def _fold_list(mk: str) -> List[float]:
        vals: List[float] = []
        for fr in fold_rows:
            if fr.get("status") != "ok":
                continue
            a = (fr.get("wrong_intermediate_aurocs") or {}).get(mk)
            if isinstance(a, (int, float)):
                vals.append(float(a))
        return vals

    pooled_stats: Dict[str, Any] = {}
    for mk in ("M1", "M2", "M3", "M4"):
        rows = oof[mk]
        t = np.array([r[0] for r in rows], dtype=object)
        y = np.array([r[1] for r in rows], dtype=np.int64)
        s = np.array([r[2] for r in rows], dtype=np.float64)
        pooled_auc = _safe_auc(y, s)
        boot = _bootstrap_tracepair_auc(t, y, s, n_bootstrap=int(bootstrap_n), seed=int(cv_seed) + 400 + (1 + ["M1","M2","M3","M4"].index(mk)))
        fvals = _fold_list(mk)
        pooled_stats[mk] = {
            "cv_wrong_intermediate_fold_aurocs": [float(x) for x in fvals],
            "cv_wrong_intermediate_mean_auroc": (float(statistics.fmean(fvals)) if fvals else None),
            "cv_wrong_intermediate_std_auroc": (float(statistics.pstdev(fvals)) if len(fvals) > 1 else None),
            "cv_wrong_intermediate_pooled_auroc": pooled_auc,
            "cv_wrong_intermediate_pooled_ci95": {
                "lower": boot.get("ci95_lower"),
                "upper": boot.get("ci95_upper"),
                "bootstrap_n": int(bootstrap_n),
                "trace_pair_count": int(boot.get("trace_pair_count", 0)),
            },
            "cv_valid_fold_count": int(len(fvals)),
        }

    m1 = oof["M1"]
    m3 = oof["M3"]
    by_key_m1 = {(t, y): s for t, y, s in m1}
    paired_t: List[str] = []
    paired_y: List[int] = []
    paired_s3: List[float] = []
    paired_s1: List[float] = []
    for t, y, s in m3:
        key = (t, y)
        if key in by_key_m1:
            paired_t.append(str(t))
            paired_y.append(int(y))
            paired_s3.append(float(s))
            paired_s1.append(float(by_key_m1[key]))
    if paired_t:
        delta = _bootstrap_delta_tracepair(
            np.array(paired_t, dtype=object),
            np.array(paired_y, dtype=np.int64),
            np.array(paired_s3, dtype=np.float64),
            np.array(paired_s1, dtype=np.float64),
            n_bootstrap=int(bootstrap_n),
            seed=int(cv_seed) + 999,
        )
    else:
        delta = {
            "defined": False,
            "delta_observed": None,
            "delta_ci95": {"lower": None, "upper": None},
            "p_value_delta_le_zero": None,
            "bootstrap_n": int(bootstrap_n),
            "bootstrap_effective_n": 0,
        }

    fold_deltas: List[float] = []
    for fr in fold_rows:
        if fr.get("status") != "ok":
            continue
        a3 = (fr.get("wrong_intermediate_aurocs") or {}).get("M3")
        a1 = (fr.get("wrong_intermediate_aurocs") or {}).get("M1")
        if isinstance(a3, (int, float)) and isinstance(a1, (int, float)):
            fold_deltas.append(float(a3) - float(a1))

    coef_stability = {}
    for mk, arrs in coef_by_model.items():
        if not arrs:
            coef_stability[mk] = {"mean_abs_coef_std": None, "max_abs_coef_std": None, "fold_count": 0}
            continue
        M = np.vstack([a.reshape(1, -1) for a in arrs])
        sd = np.std(M, axis=0)
        coef_stability[mk] = {
            "mean_abs_coef_std": float(np.mean(np.abs(sd))),
            "max_abs_coef_std": float(np.max(np.abs(sd))),
            "fold_count": int(M.shape[0]),
        }

    m3_pooled = pooled_stats["M3"]["cv_wrong_intermediate_pooled_auroc"]
    m3_valid_fold_count = int(pooled_stats["M3"]["cv_valid_fold_count"])
    cv_gate_pass = bool(
        m3_valid_fold_count >= int(cv_min_valid_folds)
        and isinstance(m3_pooled, (int, float))
        and float(m3_pooled) > float(threshold)
    )

    return {
        "status": "ok",
        "cv_folds": int(len(folds)),
        "cv_seed": int(cv_seed),
        "cv_trace_overlap_count": int(n_overlap),
        "fold_rows": fold_rows,
        "models": pooled_stats,
        "delta_m3_vs_m1": delta,
        "fold_delta_m3_minus_m1": [float(x) for x in fold_deltas],
        "fold_delta_positive_fraction": (
            float(sum(1 for x in fold_deltas if x > 0.0) / len(fold_deltas)) if fold_deltas else None
        ),
        "coef_stability": coef_stability,
        "cv_min_valid_folds_required": int(cv_min_valid_folds),
        "cv_wrong_intermediate_gate_pass_pooled": cv_gate_pass,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--run-tag", default="")
    p.add_argument("--train-exclude-variants", default="order_flip_only,answer_first_order_flip,reordered_steps")
    p.add_argument("--trace-test-fraction", type=float, default=0.20)
    p.add_argument("--trace-split-seed", type=int, default=20260307)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=20260307)
    p.add_argument("--cv-min-valid-folds", type=int, default=3)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--require-wrong-intermediate-auroc", type=float, default=0.70)
    p.add_argument("--train-seed", type=int, default=20260307)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_tag = str(args.run_tag).strip() or f"mixed_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data = _load_data(args.features)
    excluded = _parse_variant_csv(args.train_exclude_variants)

    single = _evaluate_single_split(
        data,
        test_fraction=float(args.trace_test_fraction),
        split_seed=int(args.trace_split_seed),
        excluded_variants=excluded,
        train_seed=int(args.train_seed),
        bootstrap_n=int(args.bootstrap_n),
        threshold=float(args.require_wrong_intermediate_auroc),
    )

    if single.get("status") != "ok":
        out = {
            "schema_version": "phase7_mixed_trajectory_validation_v1",
            "status": str(single.get("status")),
            "run_tag": run_tag,
            "source_features": str(args.features),
            "single_split": single,
            "timestamp": datetime.now().isoformat(),
        }
        save_json(args.output_json, out)
        Path(args.output_md).write_text(
            "# Mixed Hidden-State + SAE Trajectory Validation\n\n"
            f"- Status: `{out['status']}`\n"
        )
        print(f"Saved blocked mixed-validation JSON -> {args.output_json}")
        print(f"Saved blocked mixed-validation MD   -> {args.output_md}")
        return

    cv = _evaluate_outer_cv(
        data,
        cv_folds=int(args.cv_folds),
        cv_seed=int(args.cv_seed),
        excluded_variants=excluded,
        train_seed=int(args.train_seed),
        bootstrap_n=int(args.bootstrap_n),
        threshold=float(args.require_wrong_intermediate_auroc),
        cv_min_valid_folds=int(args.cv_min_valid_folds),
    )

    delta = single["delta_m3_vs_m1"]
    delta_obs = delta.get("delta_auroc")
    delta_ci = delta.get("delta_ci95") or {}
    practical_effect_pass = bool(single["delta_m3_vs_m1"].get("practical_effect_pass", False))
    fold_pos_frac = cv.get("fold_delta_positive_fraction")
    stable_gain_pass = bool(isinstance(fold_pos_frac, (int, float)) and float(fold_pos_frac) >= 0.80)
    leakage_clean = bool(single.get("split_diagnostics", {}).get("trace_overlap_count", 1) == 0 and cv.get("cv_trace_overlap_count", 1) == 0)
    cv_gate_pass = bool(cv.get("cv_wrong_intermediate_gate_pass_pooled", False))
    redundancy_flag = bool(single.get("collinearity_diagnostics", {}).get("pure_redundancy_flag", True))

    final_decision = (
        "publishable_mixed_signal"
        if leakage_clean and cv_gate_pass and practical_effect_pass and stable_gain_pass and (not redundancy_flag)
        else "mixed_redundant_or_insufficient"
    )

    out = {
        "schema_version": "phase7_mixed_trajectory_validation_v1",
        "status": "ok",
        "run_tag": run_tag,
        "source_features": str(args.features),
        "train_exclusion_policy": {
            "excluded_train_positive_variants": excluded,
            "require_wrong_intermediate_auroc": float(args.require_wrong_intermediate_auroc),
        },
        "single_split": single,
        "outer_cv": cv,
        "delta_auroc_mixed_vs_sae": delta_obs,
        "delta_ci95": {"lower": delta_ci.get("lower"), "upper": delta_ci.get("upper")},
        "practical_effect_pass": practical_effect_pass,
        "collinearity_diagnostics": single.get("collinearity_diagnostics", {}),
        "falsification_checks": single.get("falsification_checks", {}),
        "gate_checks": {
            "leakage_clean": leakage_clean,
            "cv_wrong_intermediate_threshold_pass": cv_gate_pass,
            "stable_gain_pass": stable_gain_pass,
            "practical_effect_pass": practical_effect_pass,
            "collinearity_non_redundant_pass": (not redundancy_flag),
        },
        "final_decision": final_decision,
        "timestamp": datetime.now().isoformat(),
    }

    lines = [
        "# Mixed Hidden-State + SAE Trajectory Validation",
        "",
        f"- Run tag: `{run_tag}`",
        f"- Final decision: `{final_decision}`",
        f"- Delta AUROC M3-M1 (wrong_intermediate): `{delta_obs}`",
        f"- Delta CI95: `{delta_ci.get('lower')}` .. `{delta_ci.get('upper')}`",
        f"- Practical effect pass: `{practical_effect_pass}`",
        f"- CV pooled M3 wrong_intermediate AUROC: `{cv.get('models',{}).get('M3',{}).get('cv_wrong_intermediate_pooled_auroc')}`",
        f"- CV pooled M3 CI95: `{(cv.get('models',{}).get('M3',{}).get('cv_wrong_intermediate_pooled_ci95') or {}).get('lower')}` .. `{(cv.get('models',{}).get('M3',{}).get('cv_wrong_intermediate_pooled_ci95') or {}).get('upper')}`",
        "",
        "## Gate Checks",
        "",
    ]
    for k, v in out["gate_checks"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    save_json(args.output_json, out)
    md = Path(args.output_md)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n")
    print(f"Saved mixed-validation JSON -> {args.output_json}")
    print(f"Saved mixed-validation MD   -> {args.output_md}")


if __name__ == "__main__":
    main()

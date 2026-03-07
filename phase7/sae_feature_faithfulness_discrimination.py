#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:  # pragma: no cover
    from .causal_intervention_engine import _load_control_records_artifact
    from .common import save_json, sha256_file
except ImportError:  # pragma: no cover
    from causal_intervention_engine import _load_control_records_artifact
    from common import save_json, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from sae_architecture import SparseAutoencoder  # type: ignore
from sae_config import SAEConfig  # type: ignore


def _parse_layers_csv(value: str) -> List[int]:
    out: List[int] = []
    for tok in str(value or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("--layers must include at least one layer index")
    return sorted(set(out))


def _roc_auc(scores_labels: Sequence[Tuple[float, int]]) -> Optional[float]:
    if not scores_labels:
        return None
    p = sum(int(y) for _, y in scores_labels)
    n = len(scores_labels) - p
    if p == 0 or n == 0:
        return None
    ranked = sorted(scores_labels, key=lambda x: x[0])
    rank_sum = 0.0
    for i, (_, y) in enumerate(ranked, start=1):
        if int(y) == 1:
            rank_sum += i
    auc = (rank_sum - (p * (p + 1) / 2.0)) / (p * n)
    return float(auc)


def _split_trace_ids(trace_ids: Sequence[str], *, test_fraction: float, seed: int) -> Tuple[set[str], set[str]]:
    uniq = sorted({str(t) for t in trace_ids if str(t)})
    if len(uniq) < 2:
        return set(uniq), set()
    rng = random.Random(int(seed))
    shuffled = list(uniq)
    rng.shuffle(shuffled)
    frac = max(0.05, min(0.95, float(test_fraction)))
    n_test = int(round(len(shuffled) * frac))
    n_test = max(1, min(len(shuffled) - 1, n_test))
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])
    return train_ids, test_ids


def _load_sae_for_layer(saes_dir: Path, layer: int, device: str) -> SparseAutoencoder:
    ckpt_path = saes_dir / f"gpt2-medium_layer{int(layer)}_sae.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint for layer {layer}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cd = dict(ckpt["config"])
    cfg = SAEConfig(
        input_dim=int(cd["input_dim"]),
        expansion_factor=int(cd["expansion_factor"]),
        use_relu=bool(cd.get("use_relu", True)),
        use_topk=bool(cd.get("use_topk", False)),
        topk_k=int(cd.get("topk_k", 0)),
        use_amp=False,
    )
    sae = SparseAutoencoder(cfg)
    sae.load_state_dict(ckpt["model_state_dict"])
    return sae.to(device).eval()


def _load_norm_stats_for_layer(activations_dir: Path, layer: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    path = activations_dir / f"gpt2-medium_layer{int(layer)}_activations.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing activation stats file for layer {layer}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    acts = payload["activations"] if isinstance(payload, dict) else payload
    if not isinstance(acts, torch.Tensor):
        raise TypeError(f"Unexpected activation payload type for layer {layer}: {type(acts).__name__}")
    if acts.ndim == 3:
        acts = acts.reshape(-1, acts.shape[-1])
    acts = acts.float()
    mean = acts.mean(dim=0).to(device)
    std = acts.std(dim=0).clamp_min(1e-6).to(device)
    return mean, std


def _normalize_hidden(h: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (h - mean) / std


def _encode_layer_features(
    rows: Sequence[dict],
    *,
    layer: int,
    saes_dir: Path,
    activations_dir: Path,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    sae = _load_sae_for_layer(saes_dir, layer=layer, device=device)
    mean, std = _load_norm_stats_for_layer(activations_dir, layer=layer, device=device)
    out: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start : start + int(batch_size)]
            h = torch.stack([r["raw_hidden"][int(layer)].float() for r in chunk], dim=0).to(device)
            h_norm = _normalize_hidden(h, mean, std)
            z = sae.encode(h_norm).detach().float().cpu()
            out.append(z)
    del sae
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(out, dim=0) if out else torch.zeros((0, 12288), dtype=torch.float32)


@dataclass
class PairGroup:
    trace_id: str
    step_idx: int
    row_indices: List[int]
    faithful_local: List[int]
    unfaithful_local: List[int]


def _build_pair_groups(rows: Sequence[dict]) -> Tuple[List[dict], List[PairGroup]]:
    filtered: List[dict] = []
    for r in rows:
        lbl = str(r.get("gold_label", ""))
        if lbl not in {"faithful", "unfaithful"}:
            continue
        h = r.get("raw_hidden")
        if not isinstance(h, torch.Tensor) or h.ndim != 2:
            continue
        filtered.append(r)

    by_group: Dict[Tuple[str, int], Dict[str, List[int]]] = {}
    for idx, r in enumerate(filtered):
        key = (str(r.get("trace_id", "")), int(r.get("step_idx", -1)))
        if key not in by_group:
            by_group[key] = {"faithful": [], "unfaithful": []}
        by_group[key][str(r.get("gold_label"))].append(idx)

    pair_groups_pre: List[Tuple[Tuple[str, int], Dict[str, List[int]]]] = []
    for key, bucket in by_group.items():
        if bucket["faithful"] and bucket["unfaithful"]:
            pair_groups_pre.append((key, bucket))

    used_idx = sorted({i for _, b in pair_groups_pre for i in (b["faithful"] + b["unfaithful"])})
    old_to_new = {old: new for new, old in enumerate(used_idx)}
    rows_used = [filtered[i] for i in used_idx]

    groups: List[PairGroup] = []
    for (trace_id, step_idx), bucket in pair_groups_pre:
        local_rows = [old_to_new[i] for i in (bucket["faithful"] + bucket["unfaithful"])]
        faithful_local = [local_rows.index(old_to_new[i]) for i in bucket["faithful"]]
        unfaithful_local = [local_rows.index(old_to_new[i]) for i in bucket["unfaithful"]]
        groups.append(
            PairGroup(
                trace_id=str(trace_id),
                step_idx=int(step_idx),
                row_indices=local_rows,
                faithful_local=faithful_local,
                unfaithful_local=unfaithful_local,
            )
        )
    return rows_used, groups


def _sample_traces_from_rows(rows: Sequence[dict], sample_traces: int, seed: int) -> List[str]:
    labels_by_trace: Dict[str, set[str]] = defaultdict(set)
    for r in rows:
        tid = str(r.get("trace_id", ""))
        lbl = str(r.get("gold_label", ""))
        if tid and lbl in {"faithful", "unfaithful"}:
            labels_by_trace[tid].add(lbl)
    eligible = sorted([t for t, labs in labels_by_trace.items() if {"faithful", "unfaithful"}.issubset(labs)])
    if len(eligible) <= int(sample_traces):
        return eligible
    rng = random.Random(int(seed))
    picked = sorted(rng.sample(eligible, int(sample_traces)))
    return picked


def _mean_cross_distance(dist: torch.Tensor, faithful_idx: Sequence[int], unfaithful_idx: Sequence[int]) -> float:
    vals: List[float] = []
    for i in faithful_idx:
        for j in unfaithful_idx:
            vals.append(float(dist[int(i), int(j)].item()))
    return float(statistics.fmean(vals)) if vals else 0.0


def _layer_feature_l2_separation(
    feats: torch.Tensor,
    groups: Sequence[PairGroup],
    *,
    n_permutations: int,
    seed: int,
) -> Dict[str, Any]:
    if feats.shape[0] == 0 or not groups:
        return {
            "status": "insufficient_pairs",
            "group_count": 0,
            "observed_mean": None,
            "null_mean": None,
            "margin": None,
            "exceedance_fraction": None,
            "n_permutations": int(n_permutations),
        }
    observed_per_group: List[float] = []
    null_by_perm: List[List[float]] = [[] for _ in range(max(1, int(n_permutations)))]
    for gi, g in enumerate(groups):
        sub = feats[g.row_indices, :].float()
        if sub.shape[0] < 2 or not g.faithful_local or not g.unfaithful_local:
            continue
        dist = torch.cdist(sub, sub, p=2)
        observed = _mean_cross_distance(dist, g.faithful_local, g.unfaithful_local)
        observed_per_group.append(float(observed))
        n = int(sub.shape[0])
        f_count = int(len(g.faithful_local))
        base_rng = random.Random(int(seed) + gi * 9973)
        for pi in range(len(null_by_perm)):
            pf = sorted(base_rng.sample(range(n), f_count))
            pu_set = set(pf)
            pu = [x for x in range(n) if x not in pu_set]
            null_by_perm[pi].append(_mean_cross_distance(dist, pf, pu))

    if not observed_per_group:
        return {
            "status": "insufficient_pairs",
            "group_count": 0,
            "observed_mean": None,
            "null_mean": None,
            "margin": None,
            "exceedance_fraction": None,
            "n_permutations": int(n_permutations),
        }

    observed_mean = float(statistics.fmean(observed_per_group))
    global_null = [float(statistics.fmean(x)) for x in null_by_perm if x]
    null_mean = float(statistics.fmean(global_null)) if global_null else None
    margin = (observed_mean - float(null_mean)) if null_mean is not None else None
    exceed = (
        float(sum(1 for x in global_null if float(x) >= float(observed_mean)) / max(1, len(global_null)))
        if global_null
        else None
    )
    return {
        "status": "ok",
        "group_count": int(len(observed_per_group)),
        "observed_mean": observed_mean,
        "null_mean": null_mean,
        "margin": margin,
        "exceedance_fraction": exceed,
        "n_permutations": int(n_permutations),
    }


def _layer_feature_divergence(
    feats: torch.Tensor,
    labels: Sequence[str],
    *,
    top_k: int = 50,
) -> Dict[str, Any]:
    if feats.shape[0] == 0:
        return {"status": "insufficient_rows"}
    mask_f = torch.tensor([1 if str(x) == "faithful" else 0 for x in labels], dtype=torch.bool)
    mask_u = torch.tensor([1 if str(x) == "unfaithful" else 0 for x in labels], dtype=torch.bool)
    if int(mask_f.sum().item()) == 0 or int(mask_u.sum().item()) == 0:
        return {"status": "missing_class"}

    xf = feats[mask_f, :].float()
    xu = feats[mask_u, :].float()
    mu_f = xf.mean(dim=0)
    mu_u = xu.mean(dim=0)
    std_f = xf.std(dim=0).clamp_min(1e-6)
    std_u = xu.std(dim=0).clamp_min(1e-6)
    pooled = torch.sqrt((std_f**2 + std_u**2) / 2.0).clamp_min(1e-6)
    cohens_d = (mu_u - mu_f) / pooled

    kl_f_u = torch.log(std_u / std_f) + (std_f**2 + (mu_f - mu_u) ** 2) / (2.0 * (std_u**2)) - 0.5
    kl_u_f = torch.log(std_f / std_u) + (std_u**2 + (mu_u - mu_f) ** 2) / (2.0 * (std_f**2)) - 0.5
    kl_sym = 0.5 * (kl_f_u + kl_u_f)

    abs_d = cohens_d.abs()
    k = int(min(max(1, top_k), feats.shape[1]))
    d_vals, d_idx = torch.topk(abs_d, k=k)
    kl_vals, kl_idx = torch.topk(kl_sym, k=k)

    top_abs_d = []
    for i in range(k):
        fi = int(d_idx[i].item())
        top_abs_d.append(
            {
                "feature_idx": fi,
                "cohens_d": float(cohens_d[fi].item()),
                "abs_cohens_d": float(abs_d[fi].item()),
                "sym_kl": float(kl_sym[fi].item()),
            }
        )

    top_kl = []
    for i in range(k):
        fi = int(kl_idx[i].item())
        top_kl.append(
            {
                "feature_idx": fi,
                "cohens_d": float(cohens_d[fi].item()),
                "abs_cohens_d": float(abs_d[fi].item()),
                "sym_kl": float(kl_vals[i].item()),
            }
        )

    return {
        "status": "ok",
        "faithful_count": int(mask_f.sum().item()),
        "unfaithful_count": int(mask_u.sum().item()),
        "max_abs_cohens_d": float(abs_d.max().item()),
        "max_sym_kl": float(kl_sym.max().item()),
        "top_features_abs_d": top_abs_d,
        "top_features_sym_kl": top_kl,
    }


def _layer_sparse_probe(
    feats: torch.Tensor,
    labels: Sequence[str],
    trace_ids: Sequence[str],
    *,
    test_fraction: float,
    split_seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    l1_lambda: float,
    min_class_per_split: int,
    device: str,
) -> Dict[str, Any]:
    if feats.shape[0] == 0:
        return {"status": "no_rows"}
    y = torch.tensor([1 if str(v) == "unfaithful" else 0 for v in labels], dtype=torch.int64)
    if int(y.sum().item()) == 0 or int((y == 0).sum().item()) == 0:
        return {"status": "missing_class"}

    train_trace_ids, test_trace_ids = _split_trace_ids(trace_ids, test_fraction=test_fraction, seed=split_seed)
    trace_overlap = sorted(train_trace_ids.intersection(test_trace_ids))
    train_idx = [i for i, t in enumerate(trace_ids) if str(t) in train_trace_ids]
    test_idx = [i for i, t in enumerate(trace_ids) if str(t) in test_trace_ids]
    if not train_idx or not test_idx:
        return {
            "status": "insufficient_split",
            "split_diagnostics": {
                "trace_overlap_count": int(len(trace_overlap)),
                "trace_overlap_check_pass": bool(len(trace_overlap) == 0),
                "num_train_rows": int(len(train_idx)),
                "num_test_rows": int(len(test_idx)),
            },
        }

    x_train = feats[train_idx, :].to(device).float()
    x_test = feats[test_idx, :].to(device).float()
    y_train = y[train_idx].to(device).float()
    y_test = y[test_idx].to(device).long()

    train_pos = int((y_train > 0.5).sum().item())
    train_neg = int((y_train <= 0.5).sum().item())
    test_pos = int((y_test == 1).sum().item())
    test_neg = int((y_test == 0).sum().item())
    min_c = int(max(1, min_class_per_split))
    if min(train_pos, train_neg, test_pos, test_neg) < min_c:
        return {
            "status": "blocked_insufficient_class_balance",
            "split_diagnostics": {
                "trace_overlap_count": int(len(trace_overlap)),
                "trace_overlap_check_pass": bool(len(trace_overlap) == 0),
                "num_train_rows": int(len(train_idx)),
                "num_test_rows": int(len(test_idx)),
                "class_counts_train": {"faithful": int(train_neg), "unfaithful": int(train_pos)},
                "class_counts_test": {"faithful": int(test_neg), "unfaithful": int(test_pos)},
                "min_class_per_split": int(min_c),
            },
        }

    model = nn.Linear(int(x_train.shape[1]), 1).to(device)
    pos_weight = torch.tensor([float(train_neg / max(1, train_pos))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    losses: List[float] = []
    for _ in range(int(max(1, epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).squeeze(-1)
        loss = criterion(logits, y_train)
        if float(l1_lambda) > 0.0:
            loss = loss + float(l1_lambda) * model.weight.abs().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(model(x_train).squeeze(-1)).detach().cpu().tolist()
        test_scores = torch.sigmoid(model(x_test).squeeze(-1)).detach().cpu().tolist()

    train_auc = _roc_auc(list(zip([float(s) for s in train_scores], [int(v) for v in y_train.detach().cpu().tolist()])))
    test_auc = _roc_auc(list(zip([float(s) for s in test_scores], [int(v) for v in y_test.detach().cpu().tolist()])))

    return {
        "status": "ok",
        "train_auroc_unfaithful_positive": train_auc,
        "test_auroc_unfaithful_positive": test_auc,
        "cross_variant_generalization_pass": bool(isinstance(test_auc, (int, float)) and float(test_auc) > 0.60),
        "loss_start": float(losses[0]) if losses else None,
        "loss_end": float(losses[-1]) if losses else None,
        "loss_mean": float(sum(losses) / len(losses)) if losses else None,
        "split_diagnostics": {
            "trace_overlap_count": int(len(trace_overlap)),
            "trace_overlap_check_pass": bool(len(trace_overlap) == 0),
            "num_unique_train_traces": int(len(train_trace_ids)),
            "num_unique_test_traces": int(len(test_trace_ids)),
            "num_train_rows": int(len(train_idx)),
            "num_test_rows": int(len(test_idx)),
            "class_counts_train": {"faithful": int(train_neg), "unfaithful": int(train_pos)},
            "class_counts_test": {"faithful": int(test_neg), "unfaithful": int(test_pos)},
            "min_class_per_split": int(min_c),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-records", required=True)
    p.add_argument("--layers", required=True, help="Comma-separated layers, e.g. 0,1,2,3")
    p.add_argument("--saes-dir", default="phase2_results/saes_gpt2_12x_topk/saes")
    p.add_argument("--activations-dir", default="phase2_results/activations")
    p.add_argument("--sample-traces", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260306)
    p.add_argument("--n-permutations", type=int, default=250)
    p.add_argument("--trace-test-fraction", type=float, default=0.20)
    p.add_argument("--probe-epochs", type=int, default=80)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-weight-decay", type=float, default=0.01)
    p.add_argument("--probe-l1-lambda", type=float, default=1e-5)
    p.add_argument("--min-class-per-split", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-tag", default="")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    run_tag = str(args.run_tag).strip() or f"phase7_sae_feature_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    layers = _parse_layers_csv(args.layers)
    rows, payload = _load_control_records_artifact(str(args.control_records))

    sampled_trace_ids = _sample_traces_from_rows(rows, sample_traces=int(args.sample_traces), seed=int(args.seed))
    sampled_set = set(sampled_trace_ids)
    sampled_rows = [r for r in rows if str(r.get("trace_id", "")) in sampled_set]

    rows_used, groups = _build_pair_groups(sampled_rows)
    if not rows_used or not groups:
        out = {
            "schema_version": "phase7_sae_feature_faithfulness_partial_v1",
            "status": "blocked_no_pairable_rows",
            "run_tag": run_tag,
            "source_control_records": str(args.control_records),
            "source_control_records_sha256": sha256_file(args.control_records),
            "sampled_trace_count": int(len(sampled_trace_ids)),
            "sampled_trace_ids": sampled_trace_ids,
            "layers": layers,
            "timestamp": datetime.now().isoformat(),
        }
        save_json(args.output, out)
        print(f"Saved blocked partial -> {args.output}")
        return

    labels = [str(r.get("gold_label", "")) for r in rows_used]
    trace_ids = [str(r.get("trace_id", "")) for r in rows_used]

    layer_results: Dict[str, Any] = {}
    saes_dir = Path(args.saes_dir)
    activations_dir = Path(args.activations_dir)
    for layer in layers:
        feats = _encode_layer_features(
            rows_used,
            layer=int(layer),
            saes_dir=saes_dir,
            activations_dir=activations_dir,
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        l2_stats = _layer_feature_l2_separation(
            feats,
            groups,
            n_permutations=int(args.n_permutations),
            seed=int(args.seed) + int(layer) * 1009,
        )
        div_stats = _layer_feature_divergence(feats, labels, top_k=50)
        probe_stats = _layer_sparse_probe(
            feats,
            labels,
            trace_ids,
            test_fraction=float(args.trace_test_fraction),
            split_seed=int(args.seed) + 97,
            epochs=int(args.probe_epochs),
            lr=float(args.probe_lr),
            weight_decay=float(args.probe_weight_decay),
            l1_lambda=float(args.probe_l1_lambda),
            min_class_per_split=int(args.min_class_per_split),
            device=str(args.device),
        )
        layer_results[str(layer)] = {
            "layer": int(layer),
            "rows_count": int(feats.shape[0]),
            "feature_dim": int(feats.shape[1]) if feats.ndim == 2 else 0,
            "group_count": int(len(groups)),
            "faithful_count": int(sum(1 for x in labels if x == "faithful")),
            "unfaithful_count": int(sum(1 for x in labels if x == "unfaithful")),
            "feature_l2_separation": l2_stats,
            "feature_divergence": div_stats,
            "sparse_probe": probe_stats,
        }

    out_payload = {
        "schema_version": "phase7_sae_feature_faithfulness_partial_v1",
        "status": "ok",
        "run_tag": run_tag,
        "source_control_records": str(args.control_records),
        "source_control_records_sha256": sha256_file(args.control_records),
        "source_control_records_stats": payload.get("stats"),
        "layers": layers,
        "sampled_trace_count": int(len(sampled_trace_ids)),
        "sampled_trace_ids": sampled_trace_ids,
        "sampled_row_count": int(len(sampled_rows)),
        "rows_used_for_pairing": int(len(rows_used)),
        "pair_group_count": int(len(groups)),
        "saes_dir": str(saes_dir),
        "activations_dir": str(activations_dir),
        "analysis_config": {
            "sample_traces": int(args.sample_traces),
            "seed": int(args.seed),
            "n_permutations": int(args.n_permutations),
            "trace_test_fraction": float(args.trace_test_fraction),
            "probe_epochs": int(args.probe_epochs),
            "probe_lr": float(args.probe_lr),
            "probe_weight_decay": float(args.probe_weight_decay),
            "probe_l1_lambda": float(args.probe_l1_lambda),
            "min_class_per_split": int(args.min_class_per_split),
            "batch_size": int(args.batch_size),
            "device": str(args.device),
        },
        "by_layer": layer_results,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(args.output, out_payload)
    print(f"Saved Phase7-SAE partial -> {args.output}")


if __name__ == "__main__":
    main()

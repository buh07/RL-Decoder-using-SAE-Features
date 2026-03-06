#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:  # pragma: no cover
    from .causal_intervention_engine import _load_control_records_artifact
    from .common import save_json
except ImportError:  # pragma: no cover
    from causal_intervention_engine import _load_control_records_artifact
    from common import save_json


def _roc_auc(scores_labels: List[Tuple[float, int]]) -> float | None:
    if not scores_labels:
        return None
    p = sum(int(y) for _, y in scores_labels)
    n = len(scores_labels) - p
    if p == 0 or n == 0:
        return None
    ranked = sorted(scores_labels, key=lambda x: x[0])
    rank_sum = 0.0
    for i, (_, y) in enumerate(ranked, start=1):
        if y == 1:
            rank_sum += i
    auc = (rank_sum - (p * (p + 1) / 2.0)) / (p * n)
    return float(auc)


class FaithfulnessProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Sample:
    x: torch.Tensor
    y: int
    trace_id: str
    variant: str


def _parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _build_feature(raw_hidden: torch.Tensor, layers: Sequence[int]) -> torch.Tensor:
    vecs = []
    for layer in layers:
        if int(layer) < 0 or int(layer) >= int(raw_hidden.shape[0]):
            raise ValueError(f"Requested layer {layer} out of range for raw_hidden shape {tuple(raw_hidden.shape)}")
        vecs.append(raw_hidden[int(layer)].float())
    return torch.cat(vecs, dim=0)


def _records_to_samples(records: Sequence[dict], layers: Sequence[int]) -> List[Sample]:
    out: List[Sample] = []
    for r in records:
        lbl = str(r.get("gold_label"))
        if lbl not in {"faithful", "unfaithful"}:
            continue
        h = r.get("raw_hidden")
        if not isinstance(h, torch.Tensor) or h.ndim != 2:
            continue
        y = 1 if lbl == "unfaithful" else 0
        out.append(
            Sample(
                x=_build_feature(h, layers),
                y=y,
                trace_id=str(r.get("trace_id", "")),
                variant=str(r.get("control_variant", "unknown")),
            )
        )
    return out


def _records_for_source_trace(rows: Sequence[dict]) -> List[dict]:
    """Map each control row to a canonical source-trace hidden state by key.

    This intentionally removes per-variant hidden differences while preserving labels,
    allowing direct circularity-ablation comparison against control-conditioned features.
    """
    key_to_hidden: Dict[Tuple[str, int, int], torch.Tensor] = {}
    ordered = sorted(
        [dict(r) for r in rows],
        key=lambda r: (
            str(r.get("trace_id", "")),
            int(r.get("step_idx", -1)),
            int(r.get("example_idx", -1)),
            0 if str(r.get("gold_label")) == "faithful" else 1,
        ),
    )
    for r in ordered:
        h = r.get("raw_hidden")
        if not isinstance(h, torch.Tensor) or h.ndim != 2:
            continue
        key = (
            str(r.get("trace_id", "")),
            int(r.get("step_idx", -1)),
            int(r.get("example_idx", -1)),
        )
        if key not in key_to_hidden:
            key_to_hidden[key] = h.detach().clone()

    out: List[dict] = []
    for r in rows:
        rr = dict(r)
        key = (
            str(rr.get("trace_id", "")),
            int(rr.get("step_idx", -1)),
            int(rr.get("example_idx", -1)),
        )
        h = key_to_hidden.get(key)
        if isinstance(h, torch.Tensor):
            rr["raw_hidden"] = h.detach().clone()
            out.append(rr)
    return out


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


def _label_counts(samples: Sequence[Sample]) -> Dict[str, int]:
    out = {"faithful": 0, "unfaithful": 0}
    for s in samples:
        if int(s.y) == 1:
            out["unfaithful"] += 1
        else:
            out["faithful"] += 1
    return out


def _train_epoch(model: nn.Module, xs: torch.Tensor, ys: torch.Tensor, optimizer: optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(xs)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, ys)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _predict_scores(model: nn.Module, xs: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xs))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-records", required=True)
    p.add_argument("--layers", default="22")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=20260305)
    p.add_argument(
        "--train-variants",
        default="faithful,wrong_intermediate,reordered_steps,prompt_bias_rationalization",
    )
    p.add_argument(
        "--test-variants",
        default="faithful,answer_first_order_flip,false_rationale_correct_answer,silent_error_correction",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--trace-test-fraction",
        type=float,
        default=0.30,
        help="Fraction of trace_ids assigned to test split (trace-first split).",
    )
    p.add_argument(
        "--trace-split-seed",
        type=int,
        default=20260306,
        help="Seed for deterministic trace-first split.",
    )
    p.add_argument(
        "--min-class-per-split",
        type=int,
        default=10,
        help="Minimum faithful and unfaithful rows required in both train and test splits.",
    )
    p.add_argument(
        "--feature-source",
        choices=["source_trace", "control_conditioned", "both"],
        default="both",
        help=(
            "source_trace: canonical hidden per (trace_id, step_idx, example_idx), "
            "control_conditioned: per-control forward hidden, both: run and compare both."
        ),
    )
    p.add_argument(
        "--split-policy",
        choices=["trace_variant_filtered", "trace_stratified_random"],
        default="trace_variant_filtered",
        help=(
            "trace_variant_filtered: trace-first split, then train/test variant filters. "
            "trace_stratified_random: trace-first split only (no variant filtering)."
        ),
    )
    return p.parse_args()


def _run_single_source(
    samples: List[Sample],
    *,
    train_variants: set[str],
    test_variants: set[str],
    split_policy: str = "trace_variant_filtered",
    trace_test_fraction: float,
    trace_split_seed: int,
    min_class_per_split: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    device: str,
    train_trace_ids: set[str] | None = None,
    test_trace_ids: set[str] | None = None,
) -> Dict[str, Any]:
    if not samples:
        return {"status": "no_samples"}

    if train_trace_ids is None or test_trace_ids is None:
        train_trace_ids, test_trace_ids = _split_trace_ids(
            [s.trace_id for s in samples],
            test_fraction=float(trace_test_fraction),
            seed=int(trace_split_seed),
        )
    trace_overlap = sorted(train_trace_ids.intersection(test_trace_ids))

    if split_policy == "trace_stratified_random":
        train_samples = [s for s in samples if s.trace_id in train_trace_ids]
        test_samples = [s for s in samples if s.trace_id in test_trace_ids]
        variant_filter_applied = False
    else:
        train_samples = [
            s for s in samples
            if s.trace_id in train_trace_ids and s.variant in train_variants
        ]
        test_samples = [
            s for s in samples
            if s.trace_id in test_trace_ids and s.variant in test_variants
        ]
        variant_filter_applied = True

    train_counts = _label_counts(train_samples)
    test_counts = _label_counts(test_samples)
    split_diag = {
        "split_policy": str(split_policy),
        "trace_split_seed": int(trace_split_seed),
        "trace_test_fraction": float(trace_test_fraction),
        "num_unique_traces_total": int(len({s.trace_id for s in samples})),
        "num_unique_train_traces": int(len(train_trace_ids)),
        "num_unique_test_traces": int(len(test_trace_ids)),
        "trace_overlap_count": int(len(trace_overlap)),
        "trace_overlap_check_pass": bool(len(trace_overlap) == 0),
        "variant_filter_applied": bool(variant_filter_applied),
        "variant_overlap_non_faithful": sorted((train_variants.intersection(test_variants)) - {"faithful"}),
        "class_counts_train": train_counts,
        "class_counts_test": test_counts,
        "min_class_per_split": int(min_class_per_split),
    }

    min_c = max(1, int(min_class_per_split))
    class_ok = (
        train_counts["faithful"] >= min_c
        and train_counts["unfaithful"] >= min_c
        and test_counts["faithful"] >= min_c
        and test_counts["unfaithful"] >= min_c
    )
    if not class_ok:
        return {
            "status": "blocked_insufficient_class_balance_after_trace_split",
            "split_diagnostics": split_diag,
        }

    if not train_samples or not test_samples:
        return {
            "status": "insufficient_variant_split_samples",
            "split_diagnostics": split_diag,
        }

    x_train = torch.stack([s.x for s in train_samples], dim=0).to(device)
    y_train = torch.tensor([float(s.y) for s in train_samples], dtype=torch.float32, device=device)
    x_test = torch.stack([s.x for s in test_samples], dim=0).to(device)
    y_test = torch.tensor([int(s.y) for s in test_samples], dtype=torch.int64, device=device)

    model = FaithfulnessProbe(input_dim=int(x_train.shape[1]), hidden_dim=int(hidden_dim)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    losses = []
    for _ in range(int(epochs)):
        losses.append(_train_epoch(model, x_train, y_train, optimizer))

    train_scores = _predict_scores(model, x_train).detach().cpu().tolist()
    test_scores = _predict_scores(model, x_test).detach().cpu().tolist()
    train_auc = _roc_auc(
        list(zip([float(s) for s in train_scores], [int(v) for v in y_train.detach().cpu().tolist()]))
    )
    test_auc = _roc_auc(
        list(zip([float(s) for s in test_scores], [int(v) for v in y_test.detach().cpu().tolist()]))
    )
    result = {
        "status": "ok",
        "input_dim": int(x_train.shape[1]),
        "num_train_samples": int(len(train_samples)),
        "num_test_samples": int(len(test_samples)),
        "train_auroc_unfaithful_positive": train_auc,
        "test_auroc_unfaithful_positive": test_auc,
        "loss_start": float(losses[0]) if losses else None,
        "loss_end": float(losses[-1]) if losses else None,
        "loss_mean": (float(sum(losses) / len(losses)) if losses else None),
        "cross_variant_generalization_pass": bool(
            isinstance(test_auc, (int, float)) and float(test_auc) > 0.60
        ),
        "split_diagnostics": split_diag,
    }
    return result


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    rows, payload = _load_control_records_artifact(args.control_records)
    layers = [int(x) for x in _parse_csv_list(args.layers)]
    train_variants = set(_parse_csv_list(args.train_variants))
    test_variants = set(_parse_csv_list(args.test_variants))
    mode_rows: Dict[str, List[dict]] = {}
    if args.feature_source in {"control_conditioned", "both"}:
        mode_rows["control_conditioned"] = [dict(r) for r in rows]
    if args.feature_source in {"source_trace", "both"}:
        mode_rows["source_trace"] = _records_for_source_trace(rows)

    if not mode_rows:
        raise RuntimeError("No feature-source mode selected.")

    split_train_trace_ids, split_test_trace_ids = _split_trace_ids(
        [str(r.get("trace_id", "")) for r in rows if str(r.get("trace_id", ""))],
        test_fraction=float(args.trace_test_fraction),
        seed=int(args.trace_split_seed),
    )

    by_feature_source: Dict[str, Any] = {}
    for mode, mode_records in mode_rows.items():
        samples = _records_to_samples(mode_records, layers)
        if not samples:
            by_feature_source[mode] = {"status": "no_labeled_samples"}
            continue
        by_feature_source[mode] = _run_single_source(
            samples,
            train_variants=train_variants,
            test_variants=test_variants,
            split_policy=str(args.split_policy),
            trace_test_fraction=float(args.trace_test_fraction),
            trace_split_seed=int(args.trace_split_seed),
            min_class_per_split=int(args.min_class_per_split),
            hidden_dim=int(args.hidden_dim),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            epochs=int(args.epochs),
            device=str(args.device),
            train_trace_ids=set(split_train_trace_ids),
            test_trace_ids=set(split_test_trace_ids),
        )

    cc = by_feature_source.get("control_conditioned", {})
    st = by_feature_source.get("source_trace", {})
    cc_auc = cc.get("test_auroc_unfaithful_positive")
    st_auc = st.get("test_auroc_unfaithful_positive")
    if isinstance(cc_auc, (int, float)) and isinstance(st_auc, (int, float)):
        delta = float(cc_auc) - float(st_auc)
    else:
        delta = None
    interpretation_flag = "inconclusive"
    if bool(cc.get("cross_variant_generalization_pass")) and not bool(st.get("cross_variant_generalization_pass")):
        interpretation_flag = "control_conditioned_only_signal"
    elif bool(cc.get("cross_variant_generalization_pass")) and bool(st.get("cross_variant_generalization_pass")):
        interpretation_flag = "both_sources_signal"
    elif (not bool(cc.get("cross_variant_generalization_pass"))) and bool(st.get("cross_variant_generalization_pass")):
        interpretation_flag = "source_trace_only_signal"

    primary_mode = "control_conditioned" if "control_conditioned" in by_feature_source else next(iter(by_feature_source.keys()))
    primary = by_feature_source.get(primary_mode, {})
    train_auc = primary.get("train_auroc_unfaithful_positive")
    test_auc = primary.get("test_auroc_unfaithful_positive")

    out = {
        "schema_version": "phase7_contrastive_probe_v1",
        "source_control_records": str(args.control_records),
        "source_control_records_stats": payload.get("stats"),
        "feature_source": str(args.feature_source),
        "primary_feature_source": str(primary_mode),
        "layers": layers,
        "train_variants": sorted(train_variants),
        "test_variants": sorted(test_variants),
        "trace_test_fraction": float(args.trace_test_fraction),
        "trace_split_seed": int(args.trace_split_seed),
        "split_policy": str(args.split_policy),
        "min_class_per_split": int(args.min_class_per_split),
        "trace_group_leakage_guard": True,
        "num_train_samples": int(primary.get("num_train_samples", 0) or 0),
        "num_test_samples": int(primary.get("num_test_samples", 0) or 0),
        "train_auroc_unfaithful_positive": train_auc,
        "test_auroc_unfaithful_positive": test_auc,
        "loss_start": primary.get("loss_start"),
        "loss_end": primary.get("loss_end"),
        "loss_mean": primary.get("loss_mean"),
        "cross_variant_generalization_pass": bool(
            bool(cc.get("cross_variant_generalization_pass"))
            or bool(st.get("cross_variant_generalization_pass"))
        ),
        "by_feature_source": by_feature_source,
        "source_comparability": {
            "control_conditioned_test_auroc": cc_auc,
            "source_trace_test_auroc": st_auc,
            "control_minus_source_test_auroc_delta": delta,
            "interpretation_flag": interpretation_flag,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    print(f"Saved contrastive probe report -> {out_path}")


if __name__ == "__main__":
    main()

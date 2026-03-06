#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover
    from .common import save_json
    from .control_token_anchor import collect_control_step_token_positions
    from .model_registry import create_adapter
except ImportError:  # pragma: no cover
    from common import save_json
    from control_token_anchor import collect_control_step_token_positions
    from model_registry import create_adapter


@dataclass
class GroupEntry:
    trace_id: str
    step_idx: int
    layer: int
    vectors: List[torch.Tensor]
    labels: List[str]


def _parse_layers_csv(value: str) -> List[int]:
    out: List[int] = []
    for tok in str(value or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("--layers must contain at least one integer layer index")
    return out


def _load_controls(path: str | Path) -> List[dict]:
    obj = torch.load(path, weights_only=False) if str(path).endswith(".pt") else None
    if obj is not None:
        if isinstance(obj, dict) and isinstance(obj.get("controls"), list):
            return list(obj.get("controls") or [])
        if isinstance(obj, list):
            return list(obj)
        raise TypeError(f"Unsupported controls PT payload at {path!s}: {type(obj).__name__}")

    import json

    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and isinstance(payload.get("controls"), list):
        return list(payload.get("controls") or [])
    if isinstance(payload, list):
        return list(payload)
    raise TypeError(f"Unsupported controls JSON payload at {path!s}: {type(payload).__name__}")


def _eligible_trace_ids(controls: Sequence[dict]) -> List[str]:
    labels_by_trace: Dict[str, set[str]] = defaultdict(set)
    for r in controls:
        trace_id = str(r.get("trace_id", ""))
        if not trace_id:
            continue
        lbl = str(r.get("gold_label", ""))
        if lbl in {"faithful", "unfaithful"}:
            labels_by_trace[trace_id].add(lbl)
    out = [t for t, labs in labels_by_trace.items() if {"faithful", "unfaithful"}.issubset(labs)]
    out.sort()
    return out


def _sample_trace_ids(trace_ids: Sequence[str], *, sample_size: int, seed: int) -> List[str]:
    uniq = sorted({str(x) for x in trace_ids if str(x)})
    if len(uniq) <= int(sample_size):
        return uniq
    rng = random.Random(int(seed))
    chosen = rng.sample(uniq, int(sample_size))
    chosen.sort()
    return chosen


def _first_faithful(rows: Sequence[dict]) -> Optional[dict]:
    faithful = [r for r in rows if str(r.get("gold_label")) == "faithful"]
    if not faithful:
        return None
    faithful = sorted(
        faithful,
        key=lambda r: (
            int(r.get("example_idx", -1)),
            str(r.get("variant", "")),
        ),
    )
    return faithful[0]


def _compute_control_layer_step_vectors(
    control: dict,
    adapter: Any,
    *,
    layers: Sequence[int],
    parse_mode: str,
    token_anchor: str,
    anchor_priority: str,
) -> Tuple[Dict[Tuple[int, int], torch.Tensor], Dict[str, Any]]:
    anchor = collect_control_step_token_positions(
        control,
        adapter,
        parse_mode=parse_mode,
        token_anchor=token_anchor,
        anchor_priority=anchor_priority,
    )
    token_ids = list(anchor.get("token_ids") or [])
    if not token_ids:
        return {}, {"status": "no_token_ids", "anchor": anchor}

    _logits, hidden_states = adapter.forward(token_ids)
    if not hidden_states:
        return {}, {"status": "no_hidden_states", "anchor": anchor}

    out: Dict[Tuple[int, int], torch.Tensor] = {}
    rows = list(anchor.get("rows") or [])
    for row in rows:
        step_idx = int(row.get("step_idx", -1))
        tok_pos = int(row.get("hidden_token_pos_0b", row.get("token_pos", -1)))
        if step_idx < 0 or tok_pos < 0:
            continue
        for layer in layers:
            li = int(layer)
            if li < 0 or li >= len(hidden_states):
                continue
            h = hidden_states[li]
            if tok_pos >= int(h.shape[1]):
                continue
            out[(step_idx, li)] = h[0, tok_pos, :].detach().float().cpu()

    meta = {
        "status": "ok",
        "anchor": {
            "position_contract_validated": bool(anchor.get("position_contract_validated", False)),
            "anchor_coverage": dict(anchor.get("anchor_coverage") or {}),
            "tokenization_metadata": dict(anchor.get("tokenization_metadata") or {}),
            "row_count": int(len(rows)),
        },
    }
    return out, meta


def _mean_cross_distance(dist: torch.Tensor, faithful_idx: Sequence[int], unfaithful_idx: Sequence[int]) -> float:
    vals: List[float] = []
    for i in faithful_idx:
        for j in unfaithful_idx:
            vals.append(float(dist[int(i), int(j)].item()))
    return float(statistics.fmean(vals)) if vals else 0.0


def _choose_combo(n: int, k: int, rng: random.Random) -> List[int]:
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    return sorted(rng.sample(range(n), k))


def _group_observed_and_null(
    vectors: Sequence[torch.Tensor],
    labels: Sequence[str],
    *,
    n_permutations: int,
    rng_seed: int,
) -> Dict[str, Any]:
    n = len(vectors)
    if n < 2:
        return {
            "status": "insufficient_vectors",
            "observed": None,
            "null_distribution": [],
            "null_mean": None,
            "exceedance_fraction": None,
            "faithful_count": int(sum(1 for x in labels if x == "faithful")),
            "unfaithful_count": int(sum(1 for x in labels if x == "unfaithful")),
        }

    faithful_idx = [i for i, lbl in enumerate(labels) if lbl == "faithful"]
    unfaithful_idx = [i for i, lbl in enumerate(labels) if lbl == "unfaithful"]
    if not faithful_idx or not unfaithful_idx:
        return {
            "status": "missing_class",
            "observed": None,
            "null_distribution": [],
            "null_mean": None,
            "exceedance_fraction": None,
            "faithful_count": int(len(faithful_idx)),
            "unfaithful_count": int(len(unfaithful_idx)),
        }

    mat = torch.stack([v.float() for v in vectors], dim=0)
    dist = torch.cdist(mat, mat, p=2)
    observed = _mean_cross_distance(dist, faithful_idx, unfaithful_idx)

    rng = random.Random(int(rng_seed))
    f_count = len(faithful_idx)
    null_vals: List[float] = []
    for _ in range(max(1, int(n_permutations))):
        pf = _choose_combo(n, f_count, rng)
        pu = [i for i in range(n) if i not in set(pf)]
        null_vals.append(_mean_cross_distance(dist, pf, pu))

    null_mean = float(statistics.fmean(null_vals)) if null_vals else 0.0
    exceed = float(sum(1 for x in null_vals if float(x) >= float(observed)) / max(1, len(null_vals)))
    return {
        "status": "ok",
        "observed": float(observed),
        "null_distribution": [float(x) for x in null_vals],
        "null_mean": float(null_mean),
        "exceedance_fraction": float(exceed),
        "faithful_count": int(len(faithful_idx)),
        "unfaithful_count": int(len(unfaithful_idx)),
    }


def _aggregate_global_null(group_stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [g for g in group_stats if g.get("status") == "ok"]
    if not ok:
        return {
            "group_count": 0,
            "observed_mean": None,
            "null_mean": None,
            "margin": None,
            "exceedance_fraction": None,
        }

    observed_mean = float(statistics.fmean(float(g["observed"]) for g in ok))
    n_perm = min(len(g.get("null_distribution", [])) for g in ok)
    if n_perm <= 0:
        return {
            "group_count": int(len(ok)),
            "observed_mean": observed_mean,
            "null_mean": None,
            "margin": None,
            "exceedance_fraction": None,
        }

    global_null: List[float] = []
    for i in range(int(n_perm)):
        vals = [float(g["null_distribution"][i]) for g in ok]
        global_null.append(float(statistics.fmean(vals)))

    null_mean = float(statistics.fmean(global_null)) if global_null else 0.0
    margin = float(observed_mean - null_mean)
    exceed = float(sum(1 for x in global_null if float(x) >= observed_mean) / max(1, len(global_null)))
    return {
        "group_count": int(len(ok)),
        "observed_mean": observed_mean,
        "null_mean": null_mean,
        "margin": margin,
        "exceedance_fraction": exceed,
        "null_distribution": [float(x) for x in global_null],
    }


def _decide_go_nogo(
    aggregate: Dict[str, Any],
    *,
    positive_trace_fraction: float,
    min_positive_trace_fraction: float,
    min_margin: float,
) -> Dict[str, Any]:
    margin = aggregate.get("margin")
    if margin is None:
        return {
            "decision": "no_go",
            "status": "blocked",
            "reason": "blocked_no_hidden_state_separation",
            "pass": False,
        }

    pass_margin = float(margin) >= float(min_margin)
    pass_trace_majority = float(positive_trace_fraction) >= float(min_positive_trace_fraction)
    passed = bool(pass_margin and pass_trace_majority)
    return {
        "decision": "go" if passed else "no_go",
        "status": "ok" if passed else "failed",
        "reason": (
            "q0_passed"
            if passed
            else (
                "failed_margin_and_trace_fraction"
                if (not pass_margin and not pass_trace_majority)
                else ("failed_margin" if not pass_margin else "failed_trace_fraction")
            )
        ),
        "pass": bool(passed),
        "checks": {
            "margin_check": bool(pass_margin),
            "trace_fraction_check": bool(pass_trace_majority),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controls", required=True)
    p.add_argument("--model-key", default="qwen2.5-7b")
    p.add_argument("--sample-traces", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260306)
    p.add_argument("--layers", default="10,14,18,22")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--parse-mode", default="hybrid", choices=["template_only", "hybrid"])
    p.add_argument("--token-anchor", default="eq_like", choices=["eq_like", "line_end"])
    p.add_argument(
        "--anchor-priority",
        default="equation_first",
        choices=["template_first", "equation_first", "leftmost_eq"],
    )
    p.add_argument("--n-permutations", type=int, default=250)
    p.add_argument("--min-margin", type=float, default=0.005)
    p.add_argument("--min-positive-trace-fraction", type=float, default=0.50)
    p.add_argument("--run-tag", default="")
    p.add_argument("--output-separation", required=True)
    p.add_argument("--output-go-nogo", required=True)
    return p.parse_args()


def _write_blocked(
    *,
    args: argparse.Namespace,
    run_tag: str,
    reason: str,
    details: Dict[str, Any],
) -> None:
    separation = {
        "schema_version": "phase7_qwen_q0_hidden_state_separation_v1",
        "run_tag": run_tag,
        "status": "blocked",
        "blocked_reason": reason,
        "details": details,
        "model_key": str(args.model_key),
        "controls": str(args.controls),
        "timestamp": datetime.now().isoformat(),
    }
    go_nogo = {
        "schema_version": "phase7_qwen_q0_go_nogo_v1",
        "run_tag": run_tag,
        "status": "blocked",
        "decision": "no_go",
        "pass": False,
        "reason": "blocked_no_hidden_state_separation",
        "blocked_reason": reason,
        "details": details,
        "criteria": {
            "min_margin": float(args.min_margin),
            "min_positive_trace_fraction": float(args.min_positive_trace_fraction),
        },
        "separation_ref": str(args.output_separation),
        "timestamp": datetime.now().isoformat(),
    }
    save_json(args.output_separation, separation)
    save_json(args.output_go_nogo, go_nogo)


def main() -> None:
    args = parse_args()
    run_tag = str(args.run_tag).strip() or f"qwen_q0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    layers = _parse_layers_csv(args.layers)

    controls = _load_controls(args.controls)
    eligible = _eligible_trace_ids(controls)
    sampled = _sample_trace_ids(eligible, sample_size=int(args.sample_traces), seed=int(args.seed))
    sampled_set = set(sampled)
    sampled_rows = [r for r in controls if str(r.get("trace_id", "")) in sampled_set]

    if not sampled:
        _write_blocked(
            args=args,
            run_tag=run_tag,
            reason="blocked_no_eligible_traces",
            details={
                "eligible_trace_count": int(len(eligible)),
                "controls_count": int(len(controls)),
            },
        )
        return

    try:
        adapter = create_adapter(str(args.model_key), device=str(args.device))
        adapter.load(str(args.device))
    except Exception as e:
        _write_blocked(
            args=args,
            run_tag=run_tag,
            reason="blocked_no_model_assets",
            details={"error": f"{type(e).__name__}: {e}"},
        )
        return

    by_trace: Dict[str, List[dict]] = defaultdict(list)
    for r in sampled_rows:
        by_trace[str(r.get("trace_id", ""))].append(r)

    group_entries: List[GroupEntry] = []
    control_forward_ok = 0
    control_forward_fail = 0

    for trace_id in sampled:
        rows = by_trace.get(trace_id, [])
        faithful = _first_faithful(rows)
        if not faithful:
            continue

        trace_vectors: Dict[str, Dict[Tuple[int, int], torch.Tensor]] = {}
        for rr in rows:
            variant = str(rr.get("variant", ""))
            key = f"{variant}::{str(rr.get('gold_label', ''))}::{int(rr.get('example_idx', -1))}"
            try:
                vecs, _meta = _compute_control_layer_step_vectors(
                    rr,
                    adapter,
                    layers=layers,
                    parse_mode=str(args.parse_mode),
                    token_anchor=str(args.token_anchor),
                    anchor_priority=str(args.anchor_priority),
                )
                if vecs:
                    trace_vectors[key] = vecs
                    control_forward_ok += 1
                else:
                    control_forward_fail += 1
            except Exception:
                control_forward_fail += 1

        faithful_key = None
        for k in sorted(trace_vectors):
            if k.startswith("faithful::"):
                faithful_key = k
                break
        if not faithful_key:
            continue

        faithful_vecs = trace_vectors[faithful_key]
        unfaithful_keys = [k for k in sorted(trace_vectors) if "::unfaithful::" in k]
        if not unfaithful_keys:
            continue

        per_group: Dict[Tuple[int, int], Dict[str, List[Any]]] = {}

        # Seed group with faithful vector(s).
        for (step_idx, layer), fvec in faithful_vecs.items():
            gk = (int(step_idx), int(layer))
            entry = per_group.setdefault(gk, {"vectors": [], "labels": []})
            entry["vectors"].append(fvec)
            entry["labels"].append("faithful")

        # Append unfaithful vectors for matching step/layer keys.
        for uk in unfaithful_keys:
            uvecs = trace_vectors[uk]
            for (step_idx, layer), uvec in uvecs.items():
                gk = (int(step_idx), int(layer))
                if gk not in per_group:
                    continue
                per_group[gk]["vectors"].append(uvec)
                per_group[gk]["labels"].append("unfaithful")

        for (step_idx, layer), payload in per_group.items():
            labels = list(payload.get("labels") or [])
            if labels.count("faithful") < 1 or labels.count("unfaithful") < 1:
                continue
            group_entries.append(
                GroupEntry(
                    trace_id=trace_id,
                    step_idx=int(step_idx),
                    layer=int(layer),
                    vectors=list(payload.get("vectors") or []),
                    labels=labels,
                )
            )

    if not group_entries:
        _write_blocked(
            args=args,
            run_tag=run_tag,
            reason="blocked_no_pairable_rows",
            details={
                "sampled_trace_count": int(len(sampled)),
                "control_forward_ok": int(control_forward_ok),
                "control_forward_fail": int(control_forward_fail),
            },
        )
        return

    group_stats: List[Dict[str, Any]] = []
    by_layer_stats: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    by_trace_stats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i, ge in enumerate(group_entries):
        stat = _group_observed_and_null(
            ge.vectors,
            ge.labels,
            n_permutations=int(args.n_permutations),
            rng_seed=int(args.seed) + i,
        )
        row = {
            "trace_id": ge.trace_id,
            "step_idx": int(ge.step_idx),
            "layer": int(ge.layer),
            **stat,
        }
        group_stats.append(row)
        by_layer_stats[int(ge.layer)].append(row)
        by_trace_stats[str(ge.trace_id)].append(row)

    aggregate = _aggregate_global_null(group_stats)

    layer_summary: Dict[str, Any] = {}
    for layer in sorted(by_layer_stats):
        layer_summary[str(layer)] = _aggregate_global_null(by_layer_stats[layer])

    trace_summary: Dict[str, Any] = {}
    pos_count = 0
    for trace_id in sorted(by_trace_stats):
        agg = _aggregate_global_null(by_trace_stats[trace_id])
        trace_summary[trace_id] = agg
        margin = agg.get("margin")
        if margin is not None and float(margin) > 0.0:
            pos_count += 1
    positive_trace_fraction = float(pos_count / max(1, len(trace_summary)))

    decision = _decide_go_nogo(
        aggregate,
        positive_trace_fraction=positive_trace_fraction,
        min_positive_trace_fraction=float(args.min_positive_trace_fraction),
        min_margin=float(args.min_margin),
    )

    sep_payload = {
        "schema_version": "phase7_qwen_q0_hidden_state_separation_v1",
        "run_tag": run_tag,
        "status": "ok",
        "model_key": str(args.model_key),
        "model_metadata": dict(adapter.metadata() if hasattr(adapter, "metadata") else {}),
        "controls": str(args.controls),
        "sampled_trace_ids": sampled,
        "sampled_trace_count": int(len(sampled)),
        "eligible_trace_count": int(len(eligible)),
        "control_row_count_sampled": int(len(sampled_rows)),
        "layers": [int(x) for x in layers],
        "parse_mode": str(args.parse_mode),
        "token_anchor": str(args.token_anchor),
        "anchor_priority": str(args.anchor_priority),
        "n_permutations": int(args.n_permutations),
        "control_forward_ok": int(control_forward_ok),
        "control_forward_fail": int(control_forward_fail),
        "aggregate": aggregate,
        "by_layer": layer_summary,
        "by_trace": trace_summary,
        "positive_trace_fraction": float(positive_trace_fraction),
        "timestamp": datetime.now().isoformat(),
    }

    go_payload = {
        "schema_version": "phase7_qwen_q0_go_nogo_v1",
        "run_tag": run_tag,
        **decision,
        "criteria": {
            "min_margin": float(args.min_margin),
            "min_positive_trace_fraction": float(args.min_positive_trace_fraction),
        },
        "metrics": {
            "observed_mean_l2": aggregate.get("observed_mean"),
            "null_mean_l2": aggregate.get("null_mean"),
            "margin": aggregate.get("margin"),
            "exceedance_fraction": aggregate.get("exceedance_fraction"),
            "positive_trace_fraction": float(positive_trace_fraction),
            "group_count": aggregate.get("group_count"),
        },
        "separation_ref": str(args.output_separation),
        "timestamp": datetime.now().isoformat(),
    }

    save_json(args.output_separation, sep_payload)
    save_json(args.output_go_nogo, go_payload)
    print(f"Saved Q0 separation -> {args.output_separation}")
    print(f"Saved Q0 decision   -> {args.output_go_nogo}")


if __name__ == "__main__":
    main()

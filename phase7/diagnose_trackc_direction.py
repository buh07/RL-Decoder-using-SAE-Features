#!/usr/bin/env python3
"""Quick Track C direction diagnostics for synthetic canary artifacts.

Inputs are provided via --inputs as a JSON file containing a list of objects:
{
  "anchor_priority": "template_first|equation_first",
  "variable": "subresult_value|operator|magnitude_bucket|sign",
  "benchmark_path": "...",
  "audit_eval_path": "..."
}

Optional probe inputs (for mediation alignment E3) can be supplied via
--probe-inputs with the same schema, typically containing operator/sign entries
with mediation-aligned reruns.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isfinite(v):
            return v
    return None


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text())


def _normalize_label(lbl: Any) -> Optional[int]:
    s = str(lbl).strip().lower()
    if s == "unfaithful":
        return 1
    if s == "faithful":
        return 0
    return None


def _roc_curve(scores: List[float], labels: List[int]) -> List[Tuple[float, float, float]]:
    # Returns rows: (threshold, fpr, tpr)
    if not scores or len(scores) != len(labels):
        return []
    pos = sum(1 for y in labels if y == 1)
    neg = sum(1 for y in labels if y == 0)
    if pos == 0 or neg == 0:
        return []
    thresholds = sorted(set(scores), reverse=True)
    thresholds.append(min(thresholds) - 1e-9)
    rows: List[Tuple[float, float, float]] = []
    for th in thresholds:
        tp = fp = tn = fn = 0
        for s, y in zip(scores, labels):
            pred = 1 if s >= th else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / pos if pos else float("nan")
        fpr = fp / neg if neg else float("nan")
        rows.append((float(th), float(fpr), float(tpr)))
    return rows


def _auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    rows = _roc_curve(scores, labels)
    if not rows:
        return None
    pts = sorted({(fpr, tpr) for _, fpr, tpr in rows}, key=lambda x: (x[0], x[1]))
    area = 0.0
    prev_fpr, prev_tpr = 0.0, 0.0
    for fpr, tpr in pts:
        area += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5
        prev_fpr, prev_tpr = fpr, tpr
    area += (1.0 - prev_fpr) * (1.0 + prev_tpr) * 0.5
    return float(max(0.0, min(1.0, area)))


def _best_recall_at_fpr(scores: List[float], labels: List[int], target_fpr: float = 0.05) -> Dict[str, Any]:
    rows = _roc_curve(scores, labels)
    if not rows:
        return {
            "defined": False,
            "threshold": None,
            "recall": None,
            "fpr": None,
        }
    best: Optional[Tuple[float, float, float]] = None
    for th, fpr, tpr in rows:
        if fpr <= target_fpr:
            if best is None or tpr > best[2] or (math.isclose(tpr, best[2]) and fpr < best[1]):
                best = (th, fpr, tpr)
    if best is None:
        return {
            "defined": False,
            "threshold": None,
            "recall": None,
            "fpr": None,
        }
    return {
        "defined": True,
        "threshold": float(best[0]),
        "recall": float(best[2]),
        "fpr": float(best[1]),
    }


def _extract_causal_scores(audit_payload: Dict[str, Any]) -> Tuple[List[float], List[int], Dict[str, Any]]:
    audits = audit_payload.get("audits") or []
    scores: List[float] = []
    labels: List[int] = []

    by_trace: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for row in audits:
        y = _normalize_label(row.get("gold_label"))
        if y is None:
            continue
        score = _safe_float(((row.get("benchmark_track_scores") or {}).get("causal_auditor")))
        if score is None:
            continue
        scores.append(score)
        labels.append(y)

        tid = str(row.get("trace_id"))
        var = str(row.get("control_variant"))
        by_trace[tid][var] = score

    # Pairwise faithful - unfaithful deltas by trace.
    pair_deltas: List[float] = []
    pair_sign_pos = 0
    pair_total = 0
    for trace_map in by_trace.values():
        faith = _safe_float(trace_map.get("faithful"))
        if faith is None:
            continue
        for var, s in trace_map.items():
            if var == "faithful":
                continue
            sval = _safe_float(s)
            if sval is None:
                continue
            d = float(faith - sval)
            pair_deltas.append(d)
            pair_total += 1
            if d > 0:
                pair_sign_pos += 1

    # Optional AUROC-style surrogate: compare true deltas vs zero baseline.
    # Positive class = unfaithful pair deltas; negative class = zero deltas.
    delta_labels: List[int] = []
    delta_scores: List[float] = []
    if pair_deltas:
        delta_scores.extend(pair_deltas)
        delta_labels.extend([1] * len(pair_deltas))
        delta_scores.extend([0.0] * len(pair_deltas))
        delta_labels.extend([0] * len(pair_deltas))
    delta_auroc = _auroc(delta_scores, delta_labels) if delta_scores else None

    pair_stats = {
        "pair_count": int(pair_total),
        "pair_delta_mean": float(sum(pair_deltas) / len(pair_deltas)) if pair_deltas else None,
        "pair_delta_median": (
            float(sorted(pair_deltas)[len(pair_deltas) // 2]) if pair_deltas else None
        ),
        "pair_delta_positive_fraction": (
            float(pair_sign_pos / pair_total) if pair_total else None
        ),
        "pair_delta_surrogate_auroc": delta_auroc,
    }
    return scores, labels, pair_stats


def _evaluate_input(entry: Dict[str, Any]) -> Dict[str, Any]:
    bench = _load_json(str(entry["benchmark_path"]))
    audit = _load_json(str(entry["audit_eval_path"]))

    scores, labels, pair_stats = _extract_causal_scores(audit)
    raw_auroc = _auroc(scores, labels)
    inv_scores = [-s for s in scores]
    inv_auroc = _auroc(inv_scores, labels)

    raw_recall = _best_recall_at_fpr(scores, labels, target_fpr=0.05)
    inv_recall = _best_recall_at_fpr(inv_scores, labels, target_fpr=0.05)

    tracks = bench.get("by_benchmark_track") or {}
    causal_track = tracks.get("causal_auditor") or {}
    composite_track = tracks.get("composite") or {}
    text_track = tracks.get("text_only") or {}
    ablation = bench.get("ablation_weighted_blend") or {}

    return {
        "anchor_priority": str(entry.get("anchor_priority")),
        "variable": str(entry.get("variable")),
        "benchmark_path": str(entry["benchmark_path"]),
        "audit_eval_path": str(entry["audit_eval_path"]),
        "coverage": _safe_float(bench.get("causal_signal_coverage_fraction")),
        "causal_auroc": _safe_float(causal_track.get("auroc")),
        "composite_auroc": _safe_float(composite_track.get("auroc")),
        "text_auroc": _safe_float(text_track.get("auroc")),
        "ablation_auroc": _safe_float(ablation.get("auroc")),
        "ablation_weights": ablation.get("weights"),
        "causal_anti_predictive_flag": bool(bench.get("causal_anti_predictive_flag", False)),
        "causal_harms_composite_flag": bool(bench.get("causal_harms_composite_flag", False)),
        "causal_variant_score_identical_fraction": _safe_float(
            bench.get("causal_variant_score_identical_fraction")
        ),
        "e1_orientation": {
            "raw_auroc_from_audit": raw_auroc,
            "inverted_auroc_from_audit": inv_auroc,
            "auroc_gain_if_inverted": (
                float(inv_auroc - raw_auroc)
                if raw_auroc is not None and inv_auroc is not None
                else None
            ),
            "raw_best_recall_at_fpr_le_0p05": raw_recall,
            "inverted_best_recall_at_fpr_le_0p05": inv_recall,
        },
        "e2_within_trace_delta": pair_stats,
    }


def _anchor_winner(summary_by_anchor: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Any]:
    anchors = sorted(summary_by_anchor.keys())
    if len(anchors) < 2:
        return {"winner": anchors[0] if anchors else None, "rule": "single_anchor_only"}

    a0, a1 = anchors[0], anchors[1]
    s0, s1 = summary_by_anchor[a0], summary_by_anchor[a1]

    c0, c1 = s0.get("mean_causal_auroc"), s1.get("mean_causal_auroc")
    if c0 is not None and c1 is not None and not math.isclose(c0, c1):
        return {
            "winner": a0 if c0 > c1 else a1,
            "rule": "max_mean_causal_auroc",
        }

    v0, v1 = s0.get("mean_coverage"), s1.get("mean_coverage")
    if v0 is not None and v1 is not None and not math.isclose(v0, v1):
        return {
            "winner": a0 if v0 > v1 else a1,
            "rule": "tie_break_mean_coverage",
        }

    o0, o1 = s0.get("mean_composite_auroc"), s1.get("mean_composite_auroc")
    if o0 is not None and o1 is not None and not math.isclose(o0, o1):
        return {
            "winner": a0 if o0 > o1 else a1,
            "rule": "tie_break_mean_composite_auroc",
        }

    return {"winner": a0, "rule": "deterministic_anchor_name_order"}


def _mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    arr = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not arr:
        return None
    return float(sum(arr) / len(arr))


def _summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_anchor: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_anchor[str(r["anchor_priority"])].append(r)

    summary_by_anchor: Dict[str, Dict[str, Optional[float]]] = {}
    for anchor, arr in by_anchor.items():
        summary_by_anchor[anchor] = {
            "mean_causal_auroc": _mean(r.get("causal_auroc") for r in arr),
            "mean_composite_auroc": _mean(r.get("composite_auroc") for r in arr),
            "mean_coverage": _mean(r.get("coverage") for r in arr),
            "mean_orientation_gain_if_inverted": _mean(
                (r.get("e1_orientation") or {}).get("auroc_gain_if_inverted") for r in arr
            ),
            "mean_pair_delta_surrogate_auroc": _mean(
                (r.get("e2_within_trace_delta") or {}).get("pair_delta_surrogate_auroc") for r in arr
            ),
            "mean_pair_delta_positive_fraction": _mean(
                (r.get("e2_within_trace_delta") or {}).get("pair_delta_positive_fraction") for r in arr
            ),
            "causal_harm_rate": _mean(
                1.0 if bool(r.get("causal_harms_composite_flag", False)) else 0.0 for r in arr
            ),
        }

    winner = _anchor_winner(summary_by_anchor)

    orientation_rows = [
        (r.get("e1_orientation") or {}).get("auroc_gain_if_inverted")
        for r in rows
    ]
    mean_orientation_gain = _mean(orientation_rows)
    orientation_misaligned = bool(
        mean_orientation_gain is not None and mean_orientation_gain >= 0.10
    )

    pair_delta_strength = _mean(
        (r.get("e2_within_trace_delta") or {}).get("pair_delta_positive_fraction") for r in rows
    )
    pairwise_signal_weak = bool(
        pair_delta_strength is not None and pair_delta_strength <= 0.60
    )

    causal_harm_rate = _mean(1.0 if r.get("causal_harms_composite_flag") else 0.0 for r in rows)
    causal_harming = bool(causal_harm_rate is not None and causal_harm_rate >= 0.50)

    recommendations: List[Dict[str, Any]] = []
    if orientation_misaligned:
        recommendations.append(
            {
                "rank": 1,
                "experiment": "E1",
                "name": "orientation_test",
                "reason": "Sign inversion materially improves causal discrimination on average.",
            }
        )
    if pairwise_signal_weak:
        recommendations.append(
            {
                "rank": len(recommendations) + 1,
                "experiment": "E2",
                "name": "within_trace_delta_reformulation",
                "reason": "Within-trace faithful-vs-variant causal deltas remain weak.",
            }
        )
    if causal_harming:
        recommendations.append(
            {
                "rank": len(recommendations) + 1,
                "experiment": "E5",
                "name": "composite_causal_weight_diagnostic",
                "reason": "Causal channel frequently harms composite AUROC.",
            }
        )

    if winner.get("winner") is not None:
        recommendations.append(
            {
                "rank": len(recommendations) + 1,
                "experiment": "E4",
                "name": "anchor_priority_choice",
                "reason": f"Use {winner['winner']} per deterministic anchor ablation rule ({winner['rule']}).",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "rank": 1,
                "experiment": "inconclusive",
                "name": "collect_more_signal",
                "reason": "No dominant explanation from quick diagnostics.",
            }
        )

    return {
        "summary_by_anchor": summary_by_anchor,
        "anchor_winner": winner,
        "e1_orientation_misaligned_flag": orientation_misaligned,
        "e2_pairwise_signal_weak_flag": pairwise_signal_weak,
        "e5_causal_harming_flag": causal_harming,
        "mean_orientation_gain_if_inverted": mean_orientation_gain,
        "mean_pair_delta_positive_fraction": pair_delta_strength,
        "causal_harm_rate": causal_harm_rate,
        "ranked_recommendations": recommendations,
    }


def _analyze_e3_probe(
    baseline_rows: List[Dict[str, Any]], probe_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if not probe_rows:
        return {
            "defined": False,
            "reason": "no_probe_inputs",
        }

    base_idx = {(r["anchor_priority"], r["variable"]): r for r in baseline_rows}
    probe_cmp: List[Dict[str, Any]] = []
    for p in probe_rows:
        k = (p["anchor_priority"], p["variable"])
        b = base_idx.get(k)
        if not b:
            continue
        pb = p.get("causal_auroc")
        bb = b.get("causal_auroc")
        probe_cmp.append(
            {
                "anchor_priority": p["anchor_priority"],
                "variable": p["variable"],
                "baseline_causal_auroc": bb,
                "aligned_mediation_causal_auroc": pb,
                "delta_causal_auroc": (
                    float(pb - bb)
                    if isinstance(pb, (int, float)) and isinstance(bb, (int, float))
                    else None
                ),
                "baseline_path": b["benchmark_path"],
                "probe_path": p["benchmark_path"],
            }
        )

    mean_delta = _mean(x.get("delta_causal_auroc") for x in probe_cmp)
    improves = bool(mean_delta is not None and mean_delta >= 0.05)
    return {
        "defined": bool(probe_cmp),
        "mean_delta_causal_auroc": mean_delta,
        "improves_flag": improves,
        "rows": probe_cmp,
    }


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Track C Quick Direction Report")
    lines.append("")
    lines.append(f"Run tag: `{payload.get('run_tag')}`")
    lines.append("")

    s = payload.get("summary") or {}
    lines.append("## Key Signals")
    lines.append(f"- Anchor winner: `{(s.get('anchor_winner') or {}).get('winner')}` by `{(s.get('anchor_winner') or {}).get('rule')}`")
    lines.append(f"- E1 orientation-misaligned flag: `{s.get('e1_orientation_misaligned_flag')}`")
    lines.append(f"- E2 pairwise-signal-weak flag: `{s.get('e2_pairwise_signal_weak_flag')}`")
    lines.append(f"- E5 causal-harming flag: `{s.get('e5_causal_harming_flag')}`")
    lines.append("")

    lines.append("## Ranked Recommendations")
    for r in s.get("ranked_recommendations") or []:
        lines.append(f"- [{r.get('experiment')}] {r.get('name')}: {r.get('reason')}")
    lines.append("")

    e3 = payload.get("e3_mediation_alignment") or {}
    lines.append("## E3 Mediation Alignment")
    lines.append(f"- Defined: `{e3.get('defined')}`")
    if e3.get("defined"):
        lines.append(f"- Mean causal AUROC delta (aligned - baseline): `{e3.get('mean_delta_causal_auroc')}`")
        lines.append(f"- Improves flag: `{e3.get('improves_flag')}`")
    lines.append("")

    lines.append("## Per-Input Rows")
    for r in payload.get("results") or []:
        lines.append(
            "- "
            + f"anchor={r.get('anchor_priority')} var={r.get('variable')} "
            + f"causal_auroc={r.get('causal_auroc')} composite_auroc={r.get('composite_auroc')} "
            + f"coverage={r.get('coverage')} orientation_gain={((r.get('e1_orientation') or {}).get('auroc_gain_if_inverted'))} "
            + f"pair_delta_pos_frac={((r.get('e2_within_trace_delta') or {}).get('pair_delta_positive_fraction'))}"
        )

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, help="Path to JSON list of baseline input entries.")
    p.add_argument(
        "--probe-inputs",
        default=None,
        help="Optional JSON list of probe entries for E3 mediation-alignment comparison.",
    )
    p.add_argument("--run-tag", default=None)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    baseline_entries = json.loads(Path(args.inputs).read_text())
    if not isinstance(baseline_entries, list) or not baseline_entries:
        raise ValueError("--inputs must contain a non-empty JSON list")

    baseline_results = [_evaluate_input(e) for e in baseline_entries]
    summary = _summarize_results(baseline_results)

    probe_results: List[Dict[str, Any]] = []
    if args.probe_inputs:
        probe_entries = json.loads(Path(args.probe_inputs).read_text())
        if not isinstance(probe_entries, list):
            raise ValueError("--probe-inputs must contain a JSON list")
        probe_results = [_evaluate_input(e) for e in probe_entries]

    e3 = _analyze_e3_probe(baseline_results, probe_results)

    run_tag = args.run_tag or "trackc_quick_unknown"
    payload = {
        "schema_version": "phase7_trackc_direction_quick_v1",
        "run_tag": run_tag,
        "results": baseline_results,
        "probe_results": probe_results,
        "summary": summary,
        "e3_mediation_alignment": e3,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(out_md, payload)

    print(str(out_json))
    print(str(out_md))


if __name__ == "__main__":
    main()

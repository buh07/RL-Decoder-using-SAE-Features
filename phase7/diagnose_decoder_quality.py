#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover
    from .common import load_json, load_pt, parse_expression_steps, save_json
except ImportError:  # pragma: no cover
    from common import load_json, load_pt, parse_expression_steps, save_json


def _extract_operator(expr: str) -> str:
    s = str(expr or "")
    for ch in ("+", "*", "/"):
        if ch in s:
            return ch
    for i, ch in enumerate(s):
        if ch == "-" and i > 0:
            return "-"
    return "unknown"


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="phase7_results/dataset/gsm8k_step_traces_train.pt")
    p.add_argument("--split", choices=["train", "test", "all"], default="train")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument(
        "--eval-results-glob",
        default="phase7_results/results/state_decoder_eval_*.json",
        help="Glob for state decoder evaluation artifacts.",
    )
    p.add_argument(
        "--phase7-sweep-summary",
        default="phase7_results/results/layer_sweep_phase7_summary.json",
        help="Optional phase7 sweep summary for layer-config ranking.",
    )
    p.add_argument("--output-json", default="phase7_results/results/decoder_quality_diagnostic.json")
    p.add_argument("--output-md", default="phase7_results/results/decoder_quality_diagnostic.md")
    p.add_argument("--quality-gate-operator-min", type=float, default=0.75)
    p.add_argument("--quality-gate-operator-operate-only-min", type=float, default=0.70)
    p.add_argument("--quality-gate-operator-operate-known-only-min", type=float, default=0.70)
    p.add_argument("--quality-gate-magnitude-min", type=float, default=0.90)
    p.add_argument("--quality-gate-sign-min", type=float, default=0.95)
    return p.parse_args()


def _load_records(path: str, split: str, max_records: Optional[int]) -> List[dict]:
    records = list(load_pt(path))
    if split != "all":
        has_split = any("gsm8k_split" in r for r in records)
        if has_split:
            records = [r for r in records if str(r.get("gsm8k_split")) == split]
    if max_records is not None:
        records = records[: int(max_records)]
    return records


def _diagnose_distribution(records: List[dict]) -> Dict[str, Any]:
    op_counter = Counter()
    step_counter = Counter()
    for r in records:
        s = r.get("structured_state") or {}
        op_counter[str(s.get("operator", "unknown"))] += 1
        step_counter[str(s.get("step_type", "unknown"))] += 1
    n = max(1, len(records))
    op_dist = {k: {"count": int(v), "fraction": float(v / n)} for k, v in sorted(op_counter.items())}
    step_dist = {k: {"count": int(v), "fraction": float(v / n)} for k, v in sorted(step_counter.items())}
    max_op = max((v / n for v in op_counter.values()), default=0.0)
    max_step = max((v / n for v in step_counter.values()), default=0.0)
    return {
        "num_records": int(len(records)),
        "operator_distribution": op_dist,
        "step_type_distribution": step_dist,
        "operator_majority_fraction": float(max_op),
        "step_type_majority_fraction": float(max_step),
        "operator_heavy_imbalance_flag": bool(max_op >= 0.80),
        "step_type_heavy_imbalance_flag": bool(max_step >= 0.80),
    }


def _diagnose_label_sanity(records: List[dict]) -> Dict[str, Any]:
    checked = 0
    mismatches = 0
    unknown_expr = 0
    skipped_non_operate = 0
    skipped_unknown_labeled_operator = 0
    mismatch_examples: List[Dict[str, Any]] = []
    for r in records:
        s = r.get("structured_state") or {}
        labeled = str(s.get("operator", "unknown"))
        step_type = str(s.get("step_type", ""))
        if step_type != "operate":
            skipped_non_operate += 1
            continue
        if labeled == "unknown":
            skipped_unknown_labeled_operator += 1
            continue
        expr = str(r.get("expr_str") or r.get("source_expr_str") or "")
        parsed = parse_expression_steps(expr, c_fallback=_to_float(r.get("C")))
        steps = list(parsed.get("steps") or [])
        expr_op = "unknown"
        if steps:
            op_idx = s.get("operation_idx", r.get("operation_idx", 0))
            try:
                oi = int(op_idx)
            except Exception:
                oi = 0
            if 0 <= oi < len(steps):
                expr_op = str((steps[oi] or {}).get("operator", "unknown"))
            else:
                expr_op = str((steps[0] or {}).get("operator", "unknown"))
        elif not parsed.get("parse_error"):
            expr_op = _extract_operator(expr)
        if expr_op == "unknown":
            unknown_expr += 1
            continue
        checked += 1
        if labeled != expr_op:
            mismatches += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "trace_id": r.get("trace_id"),
                        "step_idx": r.get("step_idx"),
                        "expr_str": r.get("expr_str"),
                        "labeled_operator": labeled,
                        "parsed_operator": expr_op,
                    }
                )
    rate = float(mismatches / max(1, checked))
    return {
        "checked_records_with_parsable_expr": int(checked),
        "unknown_expr_records": int(unknown_expr),
        "skipped_non_operate_records": int(skipped_non_operate),
        "skipped_unknown_labeled_operator": int(skipped_unknown_labeled_operator),
        "mismatch_count": int(mismatches),
        "mismatch_rate": rate,
        "mismatch_examples": mismatch_examples,
        "label_sanity_pass": bool(rate <= 0.01),
    }


def _diagnose_token_position_contract(records: List[dict]) -> Dict[str, Any]:
    n = max(1, len(records))
    eq_present = sum(1 for r in records if isinstance(r.get("eq_tok_idx"), int))
    preeq_present = sum(
        1
        for r in records
        if isinstance(r.get("pre_eq_tok_idxs"), list) and len(r.get("pre_eq_tok_idxs") or []) > 0
    )
    result_present = sum(1 for r in records if isinstance(r.get("result_tok_idx"), int))
    preeq_hidden_present = sum(
        1 for r in records if any(k in r for k in ("pre_eq_hidden", "raw_hidden_pre_eq", "raw_hidden_pre_eq_tok"))
    )
    return {
        "eq_tok_idx_fraction": float(eq_present / n),
        "pre_eq_tok_idxs_fraction": float(preeq_present / n),
        "result_tok_idx_fraction": float(result_present / n),
        "pre_eq_hidden_fraction": float(preeq_hidden_present / n),
        "d0_4_position_ablation_status": (
            "blocked_missing_pre_eq_hidden_states"
            if preeq_hidden_present == 0
            else "ready_for_pre_eq_vs_eq_ablation"
        ),
        "note": (
            "Current trace records store hidden state at equation anchor by default; "
            "operator-token/pre-eq hidden ablation requires explicit multi-position capture."
        ),
    }


def _load_eval_rows(glob_pattern: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(Path().glob(glob_pattern)):
        try:
            obj = load_json(path)
        except Exception:
            continue
        evals = obj.get("evaluations") or {}
        for split in ("val", "test"):
            m = evals.get(split)
            if not isinstance(m, dict):
                continue
            row = {
                "path": str(path),
                "config_name": obj.get("config_name"),
                "split": split,
                "layers": (obj.get("experiment_config") or {}).get("layers"),
                "input_variant": (obj.get("experiment_config") or {}).get("input_variant"),
                "operator_acc": _to_float(m.get("operator_acc")),
                "operator_acc_operate_only": _to_float(m.get("operator_acc_operate_only")),
                "operator_acc_operate_known_only": _to_float(m.get("operator_acc_operate_known_only")),
                "operator_num_operate_rows": m.get("operator_num_operate_rows"),
                "operator_num_operate_known_rows": m.get("operator_num_operate_known_rows"),
                "step_type_acc": _to_float(m.get("step_type_acc")),
                "magnitude_acc": _to_float(m.get("magnitude_acc")),
                "sign_acc": _to_float(m.get("sign_acc")),
                "result_token_top1": _to_float(m.get("result_token_top1")),
                "result_token_top5": _to_float(m.get("result_token_top5")),
                "operator_head_per_class_accuracy": m.get("operator_head_per_class_accuracy"),
                "operator_head_per_class_accuracy_operate_only": m.get("operator_head_per_class_accuracy_operate_only"),
                "operator_head_per_class_accuracy_operate_known_only": m.get("operator_head_per_class_accuracy_operate_known_only"),
                "per_operator_result_top1": m.get("per_operator_result_top1"),
            }
            out.append(row)
    return out


def _diagnose_eval_metrics(eval_rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    test_rows = [r for r in eval_rows if r.get("split") == "test" and _to_float(r.get("operator_acc")) is not None]
    for r in test_rows:
        primary = _to_float(r.get("operator_acc_operate_known_only"))
        if primary is None:
            primary = _to_float(r.get("operator_acc_operate_only"))
        r["operator_acc_primary"] = float(primary if primary is not None else _to_float(r.get("operator_acc")) or 0.0)
    test_rows.sort(
        key=lambda r: (
            -float(r["operator_acc_primary"]),
            -float(r.get("step_type_acc") or -1.0),
            -float(r.get("result_token_top1") or -1.0),
            str(r.get("config_name")),
        )
    )
    best = test_rows[0] if test_rows else None
    gate_legacy = {
        "operator_min": float(args.quality_gate_operator_min),
        "magnitude_min": float(args.quality_gate_magnitude_min),
        "sign_min": float(args.quality_gate_sign_min),
    }
    gate_legacy_pass = bool(
        best
        and (best.get("operator_acc") or 0.0) >= gate_legacy["operator_min"]
        and (best.get("magnitude_acc") or 0.0) >= gate_legacy["magnitude_min"]
        and (best.get("sign_acc") or 0.0) >= gate_legacy["sign_min"]
    )
    gate_operate = {
        "operator_operate_only_min": float(args.quality_gate_operator_operate_only_min),
        "magnitude_min": float(args.quality_gate_magnitude_min),
        "sign_min": float(args.quality_gate_sign_min),
    }
    gate_operate_pass = bool(
        best
        and (best.get("operator_acc_primary") or 0.0) >= gate_operate["operator_operate_only_min"]
        and (best.get("magnitude_acc") or 0.0) >= gate_operate["magnitude_min"]
        and (best.get("sign_acc") or 0.0) >= gate_operate["sign_min"]
    )
    gate_operate_known = {
        "operator_operate_known_only_min": float(args.quality_gate_operator_operate_known_only_min),
        "magnitude_min": float(args.quality_gate_magnitude_min),
        "sign_min": float(args.quality_gate_sign_min),
    }
    gate_operate_known_pass = bool(
        best
        and (best.get("operator_acc_primary") or 0.0) >= gate_operate_known["operator_operate_known_only_min"]
        and (best.get("magnitude_acc") or 0.0) >= gate_operate_known["magnitude_min"]
        and (best.get("sign_acc") or 0.0) >= gate_operate_known["sign_min"]
    )
    return {
        "num_eval_rows": int(len(eval_rows)),
        "num_test_rows": int(len(test_rows)),
        "ranking_metric": "operator_acc_operate_known_only_with_operate_only_legacy_fallback",
        "best_test_row_by_operator_acc": best,
        "top5_test_rows_by_operator_acc": test_rows[:5],
        "decoder_quality_gate": {
            "thresholds": gate_operate_known,
            "pass": gate_operate_known_pass,
            "reason": (
                "all thresholds met"
                if gate_operate_known_pass
                else "operate-known-operator/magnitude/sign thresholds not jointly satisfied"
            ),
        },
        "decoder_quality_gate_operate_known_only": {
            "thresholds": gate_operate_known,
            "pass": gate_operate_known_pass,
            "reason": (
                "all thresholds met"
                if gate_operate_known_pass
                else "operate-known-operator/magnitude/sign thresholds not jointly satisfied"
            ),
        },
        "decoder_quality_gate_operate_only": {
            "thresholds": gate_operate,
            "pass": gate_operate_pass,
            "reason": (
                "all thresholds met"
                if gate_operate_pass
                else "operate-only-operator/magnitude/sign thresholds not jointly satisfied"
            ),
        },
        "decoder_quality_gate_legacy_all_steps": {
            "thresholds": gate_legacy,
            "pass": gate_legacy_pass,
            "reason": (
                "all thresholds met"
                if gate_legacy_pass
                else "all-step-operator/magnitude/sign thresholds not jointly satisfied"
            ),
        },
    }


def _diagnose_layer_configs(summary_path: str) -> Dict[str, Any]:
    p = Path(summary_path)
    if not p.exists():
        return {"status": "missing_summary", "path": str(summary_path)}
    obj = load_json(p)
    rows = list(obj.get("rows") or [])
    if not rows:
        return {"status": "empty_summary", "path": str(summary_path)}
    ranked = sorted(
        rows,
        key=lambda r: (
            -float(r.get("test_operator_acc") or float("-inf")),
            -float(r.get("test_step_type_acc") or float("-inf")),
            -float(r.get("test_result_token_top1") or float("-inf")),
            int(r.get("num_layers") or 999),
        ),
    )
    baseline_l7121722 = [
        r
        for r in rows
        if str(r.get("config_name", "")).endswith("l7_l12_l17_l22") or r.get("layers") == [7, 12, 17, 22]
    ]
    improved_l4567 = [
        r
        for r in rows
        if str(r.get("config_name", "")).endswith("l4_l5_l6_l7") or r.get("layers") == [4, 5, 6, 7]
    ]
    return {
        "status": "ok",
        "path": str(summary_path),
        "num_rows": int(len(rows)),
        "top5_by_test_operator_acc": ranked[:5],
        "bottom5_by_test_operator_acc": ranked[-5:],
        "legacy_l7_l12_l17_l22_rows": baseline_l7121722,
        "improved_l4_l5_l6_l7_rows": improved_l4567,
    }


def main() -> None:
    args = parse_args()
    records = _load_records(args.dataset, args.split, args.max_records)
    distribution = _diagnose_distribution(records)
    label_sanity = _diagnose_label_sanity(records)
    token_contract = _diagnose_token_position_contract(records)
    eval_rows = _load_eval_rows(args.eval_results_glob)
    eval_diag = _diagnose_eval_metrics(eval_rows, args)
    layer_diag = _diagnose_layer_configs(args.phase7_sweep_summary)

    out = {
        "schema_version": "phase7_decoder_quality_diagnostic_v1",
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": str(args.dataset),
            "split": str(args.split),
            "num_records": int(len(records)),
        },
        "d0_1_operator_class_distribution": distribution,
        "d0_2_per_operator_accuracy_breakdown": eval_diag,
        "d0_3_operator_accuracy_by_layer_config": layer_diag,
        "d0_4_token_position_ablation_readiness": token_contract,
        "d0_5_label_sanity_check": label_sanity,
    }
    recommendations: List[str] = []
    if distribution.get("operator_heavy_imbalance_flag"):
        recommendations.append("Use class-weighted/focal operator loss due to heavy class imbalance.")
    if not label_sanity.get("label_sanity_pass", False):
        recommendations.append("Fix operator labels before retraining; label mismatch rate exceeds tolerance.")
    gate = (eval_diag.get("decoder_quality_gate_operate_known_only") or {}).get("pass")
    if gate is False:
        recommendations.append("Run D3 operate-known retraining profile before Phase 7 re-evaluation.")
    best = eval_diag.get("best_test_row_by_operator_acc") or {}
    op_all = _to_float(best.get("operator_acc"))
    op_oper = _to_float(best.get("operator_acc_operate_only"))
    if op_oper is not None and op_all is not None and op_oper > (op_all + 0.05):
        recommendations.append("Operator unknown-class contamination is significant; prioritize operate-only operator metrics.")
    op_known = _to_float(best.get("operator_acc_operate_known_only"))
    if op_known is not None and op_oper is not None and op_known > (op_oper + 0.03):
        recommendations.append("Unknown operator rows are still degrading operator quality; use operate-known gate for decisions.")
    if token_contract.get("d0_4_position_ablation_status") != "ready_for_pre_eq_vs_eq_ablation":
        recommendations.append("Collect multi-position hidden states to run D0.4 token-position ablation.")
    out["recommendations"] = recommendations

    save_json(args.output_json, out)

    md_lines = [
        "# Phase 7 Decoder Quality Diagnostic",
        "",
        f"- Dataset: `{args.dataset}` (split `{args.split}`, n={len(records)})",
        f"- Operator majority fraction: `{distribution['operator_majority_fraction']:.4f}`",
        f"- Label sanity mismatch rate: `{label_sanity['mismatch_rate']:.4f}`",
        f"- Token-position ablation status: `{token_contract['d0_4_position_ablation_status']}`",
    ]
    best = eval_diag.get("best_test_row_by_operator_acc")
    if best:
        md_lines.extend(
            [
                f"- Best test operator_acc config: `{best.get('config_name')}`",
                f"  - operator_acc={best.get('operator_acc')}, "
                f"operator_acc_operate_only={best.get('operator_acc_operate_only')}, "
                f"operator_acc_operate_known_only={best.get('operator_acc_operate_known_only')}, "
                f"step_type_acc={best.get('step_type_acc')}, "
                f"magnitude_acc={best.get('magnitude_acc')}, sign_acc={best.get('sign_acc')}",
            ]
        )
    gate_obj = eval_diag.get("decoder_quality_gate_operate_known_only") or {}
    gate_operate_obj = eval_diag.get("decoder_quality_gate_operate_only") or {}
    gate_legacy_obj = eval_diag.get("decoder_quality_gate_legacy_all_steps") or {}
    md_lines.append(
        f"- Decoder quality gate pass (operate-known): `{gate_obj.get('pass')}` ({gate_obj.get('reason')})"
    )
    md_lines.append(
        f"- Secondary operate-only gate pass: `{gate_operate_obj.get('pass')}` "
        f"({gate_operate_obj.get('reason')})"
    )
    md_lines.append(
        f"- Legacy all-step gate pass: `{gate_legacy_obj.get('pass')}` "
        f"({gate_legacy_obj.get('reason')})"
    )
    if recommendations:
        md_lines.append("")
        md_lines.append("## Recommendations")
        for rec in recommendations:
            md_lines.append(f"- {rec}")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(md_lines) + "\n")
    print(f"Saved diagnostic JSON -> {args.output_json}")
    print(f"Saved diagnostic MD   -> {args.output_md}")


if __name__ == "__main__":
    main()

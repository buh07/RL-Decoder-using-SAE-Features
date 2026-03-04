#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover
    from .common import group_step_records_to_traces, load_pt, save_json
    from .parse_cot_to_states import parse_cot_text
except ImportError:  # pragma: no cover
    from common import group_step_records_to_traces, load_pt, save_json
    from parse_cot_to_states import parse_cot_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-cot-jsonl", required=True, help="JSONL with trace_id, cot_text, and optional answer metadata")
    p.add_argument("--trace-dataset", required=True, help="Phase7 trace dataset used for internal alignment")
    p.add_argument(
        "--output",
        default="phase7_results/real_cot/real_cot_eval_input_v1.json",
        help="Prepared evaluation payload consumed by downstream audit/benchmark scripts",
    )
    p.add_argument(
        "--require-answer-match",
        action="store_true",
        help="Drop rows when provided final answer does not match the internal trace final value.",
    )
    p.add_argument(
        "--parse-mode",
        choices=["template_only", "hybrid"],
        default="hybrid",
        help="CoT parser mode used for parseability stats.",
    )
    p.add_argument(
        "--emit-controls",
        action="store_true",
        help="Also emit controls-compatible JSON payload for direct use with phase7/causal_audit.py.",
    )
    p.add_argument(
        "--controls-output",
        default=None,
        help="Output path for controls-compatible payload (defaults next to --output).",
    )
    return p.parse_args()


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _extract_trace_id(row: Dict[str, Any]) -> Optional[str]:
    tid = row.get("trace_id")
    if tid is not None:
        return str(tid)
    split = row.get("gsm8k_split")
    ex = row.get("example_idx")
    if split is None or ex is None:
        return None
    try:
        return f"gsm8k_{split}_{int(ex):05d}"
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    step_records = load_pt(args.trace_dataset)
    bundles = group_step_records_to_traces(step_records)
    trace_map = {tb.trace_id: tb for tb in bundles}

    rows: List[Dict[str, Any]] = []
    skipped_missing_trace = 0
    skipped_bad_trace_id = 0
    skipped_answer_mismatch = 0
    matched_trace_ids = set()
    unmatched_trace_ids = set()
    parser_parseable = 0
    parser_unparseable = 0
    split_counts: Dict[str, int] = {}

    with Path(args.real_cot_jsonl).open("r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSON at line {line_idx + 1}: {exc}") from exc
            trace_id = _extract_trace_id(raw)
            if trace_id is None:
                skipped_bad_trace_id += 1
                continue
            tb = trace_map.get(trace_id)
            if tb is None:
                skipped_missing_trace += 1
                unmatched_trace_ids.add(trace_id)
                continue
            matched_trace_ids.add(trace_id)
            cot_text = str(raw.get("cot_text", "")).strip()
            if not cot_text:
                continue
            parse_out = parse_cot_text(cot_text, parse_mode=args.parse_mode)
            if bool(parse_out.get("parseable")):
                parser_parseable += 1
            else:
                parser_unparseable += 1
            gold_final = _safe_float(tb.steps[-1].get("structured_state", {}).get("subresult_value")) if tb.steps else None
            reported_final = _safe_float(raw.get("final_answer"))
            answer_match = (
                (gold_final is not None and reported_final is not None and abs(gold_final - reported_final) <= 1e-6)
                if (gold_final is not None and reported_final is not None)
                else None
            )
            if args.require_answer_match and answer_match is False:
                skipped_answer_mismatch += 1
                continue
            first = tb.steps[0] if tb.steps else {}
            split_name = str(first.get("gsm8k_split", tb.split))
            split_counts[split_name] = int(split_counts.get(split_name, 0) + 1)
            rows.append(
                {
                    "schema_version": "phase7_real_cot_eval_input_row_v1",
                    "trace_id": trace_id,
                    "example_idx": int(first.get("example_idx", tb.example_idx)),
                    "gsm8k_split": split_name,
                    "cot_text": cot_text,
                    "reported_final_answer": reported_final,
                    "trace_final_answer": gold_final,
                    "answer_match": answer_match,
                    "cot_parseable": bool(parse_out.get("parseable", False)),
                    "cot_parse_error_count": int(len(parse_out.get("parse_errors", []) or [])),
                    "source_metadata": raw.get("metadata", {}),
                    "model_key": str(first.get("model_key", "unknown")),
                    "model_family": str(first.get("model_family", "unknown")),
                    "num_layers": int(first.get("num_layers", -1)),
                    "hidden_dim": int(first.get("hidden_dim", -1)),
                    "tokenizer_id": str(first.get("tokenizer_id", "unknown")),
                }
            )

    out = {
        "schema_version": "phase7_real_cot_eval_input_v1",
        "source_real_cot_jsonl": str(args.real_cot_jsonl),
        "source_trace_dataset": str(args.trace_dataset),
        "num_rows": len(rows),
        "rows": rows,
        "alignment_stats": {
            "matched_trace_count": int(len(matched_trace_ids)),
            "unmatched_trace_count": int(len(unmatched_trace_ids)),
            "matched_trace_ids_sample": sorted(matched_trace_ids)[:20],
            "unmatched_trace_ids_sample": sorted(unmatched_trace_ids)[:20],
            "parser_success_by_trace": {
                "parseable_rows": int(parser_parseable),
                "unparseable_rows": int(parser_unparseable),
                "parseable_fraction": (
                    float(parser_parseable / max(1, parser_parseable + parser_unparseable))
                ),
            },
            "split_distribution": {k: int(v) for k, v in sorted(split_counts.items())},
        },
        "skips": {
            "missing_trace": int(skipped_missing_trace),
            "bad_trace_id": int(skipped_bad_trace_id),
            "answer_mismatch_filtered": int(skipped_answer_mismatch),
        },
    }
    save_json(args.output, out)
    if args.emit_controls:
        controls_path = (
            Path(args.controls_output)
            if args.controls_output
            else Path(args.output).with_name(Path(args.output).stem + "_controls.json")
        )
        controls_rows: List[Dict[str, Any]] = []
        for row in rows:
            cot_text = str(row.get("cot_text", "")).strip()
            cot_lines = [ln.strip() for ln in cot_text.splitlines() if ln.strip()]
            control_row = {
                "schema_version": "phase7_cot_control_v1",
                "trace_id": row.get("trace_id"),
                "example_idx": int(row.get("example_idx", -1)),
                "gsm8k_split": row.get("gsm8k_split"),
                "num_steps": int(len(cot_lines)),
                "model_key": row.get("model_key"),
                "model_family": row.get("model_family"),
                "num_layers": int(row.get("num_layers", -1)),
                "hidden_dim": int(row.get("hidden_dim", -1)),
                "tokenizer_id": row.get("tokenizer_id"),
                "variant": "real_cot_pilot",
                "gold_label": "unknown",
                "expected_failure_mode": "unknown",
                "cot_text": cot_text,
                "cot_lines": cot_lines,
                "cot_line_roles": [],
                "cot_line_spans": [],
                "cot_text_steps": [],
                "cot_step_spans": [],
                "final_answer_line": None,
                "cot_final_answer_index": None,
                "text_step_states": [],
                "correction_events": [],
                "style_template_id": "real_cot_pilot",
                "style_family": "real_cot_pilot",
                "style_counterfactual": False,
                "paper_failure_family": "real_cot_pilot",
                "paper_failure_subtype": "unknown",
                "text_order_pattern": "natural",
                "contains_correction": False,
                "control_group": "real_cot_pilot",
                "source_metadata": row.get("source_metadata", {}),
            }
            controls_rows.append(control_row)
        save_json(
            controls_path,
            {
                "schema_version": "phase7_cot_control_v1",
                "num_controls": int(len(controls_rows)),
                "controls": controls_rows,
            },
        )
        print(f"Saved controls payload -> {controls_path}")
    print(f"Saved {len(rows)} rows -> {args.output}")
    print(
        "Skips:",
        f"missing_trace={skipped_missing_trace}",
        f"bad_trace_id={skipped_bad_trace_id}",
        f"answer_mismatch_filtered={skipped_answer_mismatch}",
    )


if __name__ == "__main__":
    main()

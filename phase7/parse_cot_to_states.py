#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import (
    OPERATORS,
    compare_states,
    detect_rationale_markers,
    load_json,
    magnitude_bucket,
    save_json,
    sign_label,
)

STEP_OPERATE_RE = re.compile(
    r"^STEP\s+(?P<step_idx>\d+):\s+OPERATE\s+lhs=(?P<lhs>-?\d+(?:\.\d+)?)\s+op=(?P<op>[+\-*/]|unknown)\s+rhs=(?P<rhs>-?\d+(?:\.\d+)?)\s+subresult=(?P<sub>-?\d+(?:\.\d+)?)$"
)
STEP_EMIT_RE = re.compile(
    r"^STEP\s+(?P<step_idx>\d+):\s+EMIT_RESULT\s+value=(?P<sub>-?\d+(?:\.\d+)?)\s+sign=(?P<sign>neg|zero|pos)\s+mag=(?P<mag>\[[^\]]+\))$"
)
FINAL_RE = re.compile(r"^FINAL_ANSWER\s+value=(?P<value>-?\d+(?:\.\d+)?)$")


def parse_step_line(line: str) -> Tuple[Optional[Dict], Optional[str]]:
    raw_line = line.strip()
    is_correction = False
    line = raw_line
    if line.startswith("CORRECTION "):
        is_correction = True
        line = line[len("CORRECTION "):].strip()

    m = STEP_OPERATE_RE.match(line)
    if m:
        lhs = float(m.group("lhs"))
        rhs = float(m.group("rhs"))
        sub = float(m.group("sub"))
        op = m.group("op") if m.group("op") in OPERATORS else "unknown"
        return {
            "step_idx": int(m.group("step_idx")),
            "step_type": "operate",
            "operator": op,
            "lhs_value": lhs,
            "rhs_value": rhs,
            "subresult_value": sub,
            "sign": sign_label(sub),
            "magnitude_bucket": magnitude_bucket(sub),
            "is_correction": is_correction,
        }, None
    m = STEP_EMIT_RE.match(line)
    if m:
        sub = float(m.group("sub"))
        return {
            "step_idx": int(m.group("step_idx")),
            "step_type": "emit_result",
            "operator": "unknown",
            "lhs_value": None,
            "rhs_value": None,
            "subresult_value": sub,
            "sign": m.group("sign"),
            "magnitude_bucket": m.group("mag"),
            "is_correction": is_correction,
        }, None
    if FINAL_RE.match(line):
        return None, "final_answer_line"
    if line.startswith("STEP "):
        return None, "unrecognized_step_template"
    if raw_line.startswith("CORRECTION STEP "):
        return None, "unrecognized_correction_template"
    return None, "not_a_step_line"


def parse_cot_text(cot_text: str) -> Dict:
    lines = [ln.strip() for ln in cot_text.splitlines() if ln.strip()]
    parsed_steps: List[Dict] = []
    parsed_steps_in_text_order: List[Dict] = []
    errors: List[Dict] = []
    final_answer = None
    unsupported_markers: List[Dict] = []
    correction_events: List[Dict] = []
    for idx, line in enumerate(lines):
        markers = detect_rationale_markers(line)
        if markers:
            unsupported_markers.append({"line_index": idx, "line": line, "markers": markers})
        m = FINAL_RE.match(line)
        if m:
            final_answer = float(m.group("value"))
            continue
        state, err = parse_step_line(line)
        if state is None:
            if err != "not_a_step_line":
                errors.append({"line_index": idx, "line": line, "error": err, "markers": markers})
            continue
        state["observed_line_index"] = int(idx)
        parsed_steps.append(state)
        parsed_steps_in_text_order.append(state)
        if state.get("is_correction"):
            correction_events.append(
                {
                    "line_index": int(idx),
                    "step_idx": int(state.get("step_idx", -1)),
                    "step_type": state.get("step_type"),
                    "corrected_state": {k: v for k, v in state.items() if k not in {"observed_line_index"}},
                }
            )
    parsed_steps.sort(key=lambda s: (int(s.get("step_idx", 0)), int(s.get("observed_line_index", 0))))
    observed_text_order = [
        {
            "line_index": int(s.get("observed_line_index", -1)),
            "step_idx": int(s.get("step_idx", -1)),
            "step_type": s.get("step_type"),
            "is_correction": bool(s.get("is_correction", False)),
        }
        for s in parsed_steps_in_text_order
    ]

    # Detect revision consistency issues: corrections that materially disagree with prior same-step claim.
    revision_rows: List[Dict] = []
    prior_by_step: Dict[int, Dict] = {}
    for s in parsed_steps_in_text_order:
        sidx = int(s.get("step_idx", -1))
        prior = prior_by_step.get(sidx)
        if bool(s.get("is_correction", False)):
            cmp_prev = compare_states(prior, s) if prior is not None else {"match_fraction": 0.0, "field_matches": {}}
            revision_rows.append(
                {
                    "step_idx": sidx,
                    "line_index": int(s.get("observed_line_index", -1)),
                    "has_prior_claim": prior is not None,
                    "contradiction_detected": bool(prior is not None and float(cmp_prev.get("match_fraction", 0.0)) < 1.0),
                    "prior_match_fraction": float(cmp_prev.get("match_fraction", 0.0)),
                }
            )
        prior_by_step[sidx] = s

    return {
        "parsed_steps": parsed_steps,
        "parsed_steps_in_text_order": parsed_steps_in_text_order,
        "observed_text_order": observed_text_order,
        "parse_errors": errors,
        "parseable": len(parsed_steps) > 0 and not any(
            e["error"] in {"unrecognized_step_template", "unrecognized_correction_template"} for e in errors
        ),
        "final_answer_value": final_answer,
        "revision_parse_summary": {
            "contains_correction": len(correction_events) > 0,
            "num_corrections": len(correction_events),
            "correction_events": correction_events,
            "revision_events": revision_rows,
            "contradicted_corrections": int(sum(1 for r in revision_rows if r.get("contradiction_detected"))),
        },
        "unsupported_rationale_markers": unsupported_markers,
    }


def align_parsed_to_trace(parsed: Dict, trace_steps: List[dict]) -> Dict:
    in_text_order = parsed.get("parsed_steps_in_text_order") or parsed.get("parsed_steps", [])
    by_idx = {int(s["step_idx"]): s for s in in_text_order}
    expected_order = [int(r["step_idx"]) for r in sorted(trace_steps, key=lambda r: int(r["step_idx"]))]
    observed_rows = list(parsed.get("observed_text_order", []))
    observed_non_correction = [int(r["step_idx"]) for r in observed_rows if not bool(r.get("is_correction", False))]
    observed_unique = []
    seen = set()
    for sidx in observed_non_correction:
        if sidx in seen:
            continue
        seen.add(sidx)
        observed_unique.append(sidx)
    temporal_pass = bool(observed_unique == expected_order) if observed_unique else False
    rev = parsed.get("revision_parse_summary", {}) or {}
    alignments: List[Dict] = []
    for row in sorted(trace_steps, key=lambda r: int(r["step_idx"])):
        sidx = int(row["step_idx"])
        text_state = by_idx.get(sidx)
        latent_state_gold = row["structured_state"]
        cmp = compare_states(text_state, latent_state_gold)
        alignments.append(
            {
                "step_idx": sidx,
                "text_claim_state": text_state,
                "gold_structured_state": latent_state_gold,
                "alignment_match": cmp,
                "alignment_confidence": 1.0 if text_state is not None else 0.0,
                "unverifiable_text": text_state is None,
                "observed_text_line_index": int(text_state.get("observed_line_index", -1)) if text_state is not None else None,
                "is_correction_claim": bool(text_state.get("is_correction", False)) if text_state is not None else None,
            }
        )
    return {
        "parse_summary": {
            "parseable": bool(parsed.get("parseable")),
            "num_parsed_steps": len(parsed.get("parsed_steps", [])),
            "num_parse_errors": len(parsed.get("parse_errors", [])),
            "parse_errors": parsed.get("parse_errors", []),
            "final_answer_value": parsed.get("final_answer_value"),
        },
        "observed_text_order": observed_rows,
        "temporal_consistency": {
            "expected_order": expected_order,
            "observed_non_correction_order": observed_non_correction,
            "observed_unique_order": observed_unique,
            "pass": temporal_pass,
        },
        "revision_parse_summary": rev,
        "unsupported_rationale_markers": parsed.get("unsupported_rationale_markers", []),
        "step_alignments": alignments,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controls", default="phase7_results/controls/cot_controls_test.json")
    p.add_argument("--trace-dataset", default="phase7_results/dataset/gsm8k_step_traces_test.pt")
    p.add_argument("--output", default="phase7_results/controls/cot_controls_test_parsed.json")
    return p.parse_args()


def main() -> None:
    from common import group_step_records_to_traces, load_pt

    args = parse_args()
    controls = load_json(args.controls)
    step_records = load_pt(args.trace_dataset)
    trace_map = {tb.trace_id: tb.steps for tb in group_step_records_to_traces(step_records)}

    rows = []
    for ctrl in controls["controls"]:
        parsed = parse_cot_text(ctrl["cot_text"])
        trace_steps = trace_map.get(ctrl["trace_id"], [])
        aligned = align_parsed_to_trace(parsed, trace_steps)
        rows.append({**ctrl, **aligned})

    save_json(args.output, {"schema_version": "phase7_cot_control_parse_v1", "controls": rows})
    print(f"Saved parsed/aligned controls -> {args.output}")


if __name__ == "__main__":
    main()

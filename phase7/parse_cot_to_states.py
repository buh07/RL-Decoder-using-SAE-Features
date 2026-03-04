#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover
    from .common import (
        OPERATORS,
        compare_states,
        detect_rationale_markers,
        load_json,
        magnitude_bucket,
        save_json,
        sign_label,
    )
except ImportError:  # pragma: no cover
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
ANGLE_EQ_RE = re.compile(r"<<\s*(?P<expr>[^<>]+?)\s*=\s*(?P<result>-?\d+(?:\.\d+)?)\s*>>")
INLINE_EQ_RE = re.compile(
    r"(?P<expr>(?:-?\d+(?:\.\d+)?\s*[+\-*/]\s*)+-?\d+(?:\.\d+)?)\s*=\s*(?P<result>-?\d+(?:\.\d+)?)"
)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
OP_RE = re.compile(r"[+\-*/]")


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


def _safe_eval_binary(op: str, lhs: float, rhs: float) -> Optional[float]:
    try:
        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "/":
            if abs(rhs) < 1e-12:
                return None
            return lhs / rhs
    except Exception:
        return None
    return None


def _tokenize_expression(expr: str) -> Optional[List[str]]:
    clean = expr.strip().replace(" ", "")
    if not clean:
        return None
    # Normalize unary minus into numeric tokens via AST round-trip when possible.
    try:
        node = ast.parse(clean, mode="eval")
        clean = ast.unparse(node).replace(" ", "")
    except Exception:
        pass
    toks = re.findall(r"-?\d+(?:\.\d+)?|[+\-*/]", clean)
    if not toks:
        return None
    if len(toks) % 2 == 0:
        return None
    for i, tok in enumerate(toks):
        if i % 2 == 0:
            if not NUM_RE.fullmatch(tok):
                return None
        else:
            if not OP_RE.fullmatch(tok):
                return None
    return toks


def _expression_to_steps(expr: str, result: float, start_step_idx: int) -> Optional[List[Dict]]:
    toks = _tokenize_expression(expr)
    if toks is None or len(toks) < 3:
        return None
    cur = float(toks[0])
    out: List[Dict] = []
    op_i = 0
    for i in range(1, len(toks), 2):
        op = toks[i]
        rhs = float(toks[i + 1])
        sub = _safe_eval_binary(op, cur, rhs)
        if sub is None:
            return None
        out.append(
            {
                "step_idx": int(start_step_idx + op_i),
                "step_type": "operate",
                "operator": op if op in OPERATORS else "unknown",
                "lhs_value": float(cur),
                "rhs_value": float(rhs),
                "subresult_value": float(sub),
                "sign": sign_label(sub),
                "magnitude_bucket": magnitude_bucket(sub),
                "is_correction": False,
                "line_parse_source": "equation_fallback",
            }
        )
        cur = float(sub)
        op_i += 1
    # Tolerate small arithmetic/rounding drift from textual CoT.
    if abs(cur - float(result)) > 1e-3:
        return None
    return out


def _fallback_parse_line_equations(line: str, start_step_idx: int) -> Tuple[List[Dict], Optional[str]]:
    # Prefer explicit <<expr=result>> spans when present.
    span_steps: List[Dict] = []
    step_idx = int(start_step_idx)
    spans = list(ANGLE_EQ_RE.finditer(line))
    for m in spans:
        expr = str(m.group("expr")).strip()
        try:
            result = float(m.group("result"))
        except Exception:
            return [], "equation_result_not_numeric"
        steps = _expression_to_steps(expr, result, step_idx)
        if steps is None:
            return [], "equation_parse_failed"
        span_steps.extend(steps)
        step_idx += len(steps)
    if span_steps:
        return span_steps, None

    # Fallback to inline "a op b [op c...] = result" format.
    m = INLINE_EQ_RE.search(line)
    if m:
        expr = str(m.group("expr")).strip()
        try:
            result = float(m.group("result"))
        except Exception:
            return [], "equation_result_not_numeric"
        steps = _expression_to_steps(expr, result, step_idx)
        if steps is None:
            return [], "equation_parse_failed"
        return steps, None
    return [], "not_equation_line"


def parse_cot_text(cot_text: str, parse_mode: str = "hybrid") -> Dict:
    lines = [ln.strip() for ln in cot_text.splitlines() if ln.strip()]
    parsed_steps: List[Dict] = []
    parsed_steps_in_text_order: List[Dict] = []
    errors: List[Dict] = []
    final_answer = None
    unsupported_markers: List[Dict] = []
    correction_events: List[Dict] = []
    next_step_idx = 0
    equation_parse_count = 0
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
            fallback_err = None
            if parse_mode == "hybrid":
                fallback_states, fallback_err = _fallback_parse_line_equations(line, next_step_idx)
                if fallback_states:
                    equation_parse_count += 1
                    for st in fallback_states:
                        st["observed_line_index"] = int(idx)
                        parsed_steps.append(st)
                        parsed_steps_in_text_order.append(st)
                    next_step_idx += len(fallback_states)
                    continue
            if err != "not_a_step_line":
                errors.append({"line_index": idx, "line": line, "error": err, "markers": markers})
            elif fallback_err in {"equation_parse_failed", "equation_result_not_numeric"}:
                errors.append({"line_index": idx, "line": line, "error": fallback_err, "markers": markers})
            continue
        state["line_parse_source"] = "template"
        state["observed_line_index"] = int(idx)
        parsed_steps.append(state)
        parsed_steps_in_text_order.append(state)
        next_step_idx = max(next_step_idx, int(state.get("step_idx", -1)) + 1)
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
        "parse_mode_used": parse_mode,
        "equation_parse_count": int(equation_parse_count),
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


def canonical_step_claims(parsed: Dict) -> Dict[int, Dict]:
    """Select one canonical claim per step_idx without discarding earlier claims silently.

    Preference order:
    1) first non-correction claim in observed text order
    2) first claim in observed text order (if only correction claims exist)
    """
    in_text_order = parsed.get("parsed_steps_in_text_order") or parsed.get("parsed_steps", [])
    by_idx: Dict[int, List[Dict]] = {}
    for s in in_text_order:
        sidx = int(s.get("step_idx", -1))
        by_idx.setdefault(sidx, []).append(s)
    out: Dict[int, Dict] = {}
    for sidx, claims in by_idx.items():
        preferred = next((c for c in claims if not bool(c.get("is_correction", False))), None)
        out[sidx] = preferred if preferred is not None else claims[0]
    return out


def align_parsed_to_trace(parsed: Dict, trace_steps: List[dict]) -> Dict:
    in_text_order = parsed.get("parsed_steps_in_text_order") or parsed.get("parsed_steps", [])
    by_idx = canonical_step_claims(parsed)
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
            "parse_mode_used": parsed.get("parse_mode_used"),
            "equation_parse_count": int(parsed.get("equation_parse_count", 0)),
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
    p.add_argument("--parse-mode", choices=["template_only", "hybrid"], default="hybrid")
    p.add_argument("--output", default="phase7_results/controls/cot_controls_test_parsed.json")
    return p.parse_args()


def main() -> None:
    try:  # pragma: no cover
        from .common import group_step_records_to_traces, load_pt
    except ImportError:  # pragma: no cover
        from common import group_step_records_to_traces, load_pt

    args = parse_args()
    controls = load_json(args.controls)
    step_records = load_pt(args.trace_dataset)
    trace_map = {tb.trace_id: tb.steps for tb in group_step_records_to_traces(step_records)}

    rows = []
    for ctrl in controls["controls"]:
        parsed = parse_cot_text(ctrl["cot_text"], parse_mode=args.parse_mode)
        trace_steps = trace_map.get(ctrl["trace_id"], [])
        aligned = align_parsed_to_trace(parsed, trace_steps)
        rows.append({**ctrl, **aligned})

    save_json(args.output, {"schema_version": "phase7_cot_control_parse_v1", "controls": rows})
    print(f"Saved parsed/aligned controls -> {args.output}")


if __name__ == "__main__":
    main()

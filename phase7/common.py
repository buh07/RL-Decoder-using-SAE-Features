#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

PHASE7_TRACE_SCHEMA = "phase7_trace_v1"
CAUSAL_PATCH_SPEC_SCHEMA = "causal_patch_spec_v1"
CAUSAL_AUDIT_SCHEMA = "causal_audit_v1"

STEP_TYPES = ["parse", "operate", "emit_result", "verify"]
OPERATORS = ["+", "-", "*", "/", "unknown"]
MAG_BUCKETS = ["[0,10)", "[10,100)", "[100,1000)", "[1000+)"]
SIGNS = ["neg", "zero", "pos"]
PAPER_CORE4_VARIANTS = [
    "prompt_bias_rationalization",
    "silent_error_correction",
    "answer_first_order_flip",
    "shortcut_rationalization",
]

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pt(path: str | Path):
    p = Path(path)
    return torch.load(p, weights_only=False)


def save_pt(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, p)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def magnitude_bucket(c_val: float) -> str:
    mag = abs(float(c_val))
    if mag < 10:
        return "[0,10)"
    if mag < 100:
        return "[10,100)"
    if mag < 1000:
        return "[100,1000)"
    return "[1000+)"


def sign_label(x: float) -> str:
    x = float(x)
    if x < 0:
        return "neg"
    if x > 0:
        return "pos"
    return "zero"


def _is_number_char(ch: str) -> bool:
    return ch.isdigit() or ch == "."


def tokenize_expression(expr: str) -> Tuple[List[float], List[str], Optional[str]]:
    """Tokenize arithmetic expr into numbers and ops. Returns (nums, ops, error)."""
    s = expr.replace(" ", "")
    nums: List[float] = []
    ops: List[str] = []
    i = 0
    expect_num = True
    while i < len(s):
        ch = s[i]
        if ch == ",":
            i += 1
            continue
        if expect_num:
            sign = 1.0
            if ch == "+":
                i += 1
                if i >= len(s):
                    return nums, ops, "dangling_plus"
                ch = s[i]
            elif ch == "-":
                sign = -1.0
                i += 1
                if i >= len(s):
                    return nums, ops, "dangling_minus"
                ch = s[i]
            if not _is_number_char(ch):
                return nums, ops, f"expected_number_at_{i}"
            start = i
            while i < len(s) and (_is_number_char(s[i])):
                i += 1
            tok = s[start:i]
            try:
                nums.append(sign * float(tok))
            except ValueError:
                return nums, ops, f"bad_number:{tok}"
            expect_num = False
            continue
        if ch in "+-*/":
            ops.append(ch)
            expect_num = True
            i += 1
            continue
        return nums, ops, f"unexpected_char:{ch}"
    if expect_num and (nums or ops):
        return nums, ops, "trailing_operator"
    return nums, ops, None


def _apply_op(lhs: float, rhs: float, op: str) -> float:
    if op == "+":
        return lhs + rhs
    if op == "-":
        return lhs - rhs
    if op == "*":
        return lhs * rhs
    if op == "/":
        if rhs == 0:
            raise ZeroDivisionError("division by zero")
        return lhs / rhs
    raise ValueError(f"unknown op {op}")


def parse_expression_summary(expr: str, c_fallback: Optional[float] = None) -> Dict[str, Any]:
    """Left-assoc parse summary suitable for one annotation-level structured state.

    For multi-op expressions, this returns the *final reduction step* (lhs, rhs, op -> subresult)
    and metadata including reduction depth.
    """
    nums, ops, err = tokenize_expression(expr)
    out: Dict[str, Any] = {
        "parse_error": err,
        "num_operands": len(nums),
        "num_operators": len(ops),
        "is_multi_op_expr": len(ops) > 1,
        "operands": nums,
        "operators": ops,
        "lhs_value": None,
        "rhs_value": None,
        "subresult_value": c_fallback,
        "operator": "unknown",
        "reduction_depth": max(0, len(ops)),
    }
    if err is not None:
        return out
    if len(nums) == 0:
        out["parse_error"] = "empty"
        return out
    if len(ops) == 0:
        out["lhs_value"] = nums[0]
        out["rhs_value"] = 0.0
        out["subresult_value"] = nums[0] if c_fallback is None else c_fallback
        out["operator"] = "unknown"
        return out
    if len(nums) != len(ops) + 1:
        out["parse_error"] = "arity_mismatch"
        return out

    cur = nums[0]
    last_lhs = nums[0]
    last_rhs = nums[1]
    last_op = ops[0]
    for op, rhs in zip(ops, nums[1:]):
        last_lhs = cur
        last_rhs = rhs
        last_op = op
        cur = _apply_op(cur, rhs, op)
    out["lhs_value"] = last_lhs
    out["rhs_value"] = last_rhs
    out["subresult_value"] = cur if c_fallback is None else c_fallback
    out["operator"] = last_op
    if c_fallback is not None and math.isfinite(float(c_fallback)):
        # Note disagreement but do not overwrite; dataset label is source of truth.
        out["eval_result"] = float(cur)
        out["eval_matches_C"] = math.isclose(float(cur), float(c_fallback), rel_tol=1e-6, abs_tol=1e-6)
    return out


def _safe_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def build_structured_state(record: dict, step_idx: int, trace_len: int) -> Dict[str, Any]:
    c_val = float(record.get("C", 0.0))
    expr = str(record.get("expr_str", ""))
    parsed = parse_expression_summary(expr, c_fallback=c_val)
    step_type = "emit_result" if step_idx == trace_len - 1 else "operate"
    return {
        "step_type": step_type,
        "operator": parsed.get("operator", "unknown") if parsed.get("operator") in OPERATORS else "unknown",
        "lhs_value": _safe_num(parsed.get("lhs_value")),
        "rhs_value": _safe_num(parsed.get("rhs_value")),
        "subresult_value": _safe_num(parsed.get("subresult_value")) or c_val,
        "result_token_id": int(record["result_token_id"]),
        "magnitude_bucket": magnitude_bucket(c_val),
        "sign": sign_label(c_val),
        "num_operands": int(parsed.get("num_operands", 0)),
        "num_operators": int(parsed.get("num_operators", 0)),
        "is_multi_op_expr": bool(parsed.get("is_multi_op_expr", False)),
        "parse_error": parsed.get("parse_error"),
        "source_expr_str": expr,
        "example_idx": int(record.get("example_idx", -1)),
        "step_idx": int(step_idx),
    }


def make_trace_id(split: str, example_idx: int) -> str:
    return f"gsm8k_{split}_{example_idx:05d}"


def group_records_by_example(records: Sequence[dict]) -> Dict[int, List[dict]]:
    by_example: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        by_example[int(r["example_idx"])] .append(r)
    for ex_id, group in by_example.items():
        group.sort(key=lambda r: (int(r.get("ann_idx", 0)), int(r.get("eq_tok_idx", 0)), int(r.get("result_tok_idx", 0))))
    return dict(by_example)


def format_num(x: Optional[float]) -> str:
    if x is None:
        return "nan"
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return f"{x:g}"


def faithful_step_text(state: Dict[str, Any]) -> str:
    st = state.get("step_type")
    if st == "emit_result":
        return (
            f"STEP {state['step_idx']}: EMIT_RESULT value={format_num(state.get('subresult_value'))} "
            f"sign={state.get('sign')} mag={state.get('magnitude_bucket')}"
        )
    if st == "operate":
        return (
            f"STEP {state['step_idx']}: OPERATE lhs={format_num(state.get('lhs_value'))} "
            f"op={state.get('operator')} rhs={format_num(state.get('rhs_value'))} "
            f"subresult={format_num(state.get('subresult_value'))}"
        )
    return f"STEP {state.get('step_idx', -1)}: {st.upper()}"


def final_answer_text(trace_steps: Sequence[Dict[str, Any]]) -> str:
    if not trace_steps:
        return "FINAL_ANSWER value=nan"
    last = trace_steps[-1]["structured_state"] if "structured_state" in trace_steps[-1] else trace_steps[-1]
    return f"FINAL_ANSWER value={format_num(last.get('subresult_value'))}"


def build_trace_text_with_spans(step_texts: Sequence[str]) -> Tuple[str, List[Dict[str, int]]]:
    spans: List[Dict[str, int]] = []
    chunks: List[str] = []
    cursor = 0
    for i, t in enumerate(step_texts):
        if i > 0:
            chunks.append("\n")
            cursor += 1
        start = cursor
        chunks.append(t)
        cursor += len(t)
        spans.append({"char_start": start, "char_end": cursor})
    return "".join(chunks), spans


def clone_jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def perturb_number(x: Optional[float], mode: str = "small") -> Optional[float]:
    if x is None:
        return None
    if mode == "small":
        if abs(x) < 1:
            return x + 1.0
        return x + (1.0 if x >= 0 else -1.0)
    return x * 2.0


def compare_states(text_state: Optional[dict], latent_state: Optional[dict], tol: float = 1e-5) -> Dict[str, Any]:
    categorical_fields = ["step_type", "operator", "magnitude_bucket", "sign"]
    numeric_fields = ["lhs_value", "rhs_value", "subresult_value"]
    fields = categorical_fields + numeric_fields
    matches: Dict[str, Optional[bool]] = {}
    numeric_abs_error: Dict[str, Optional[float]] = {k: None for k in numeric_fields}
    n_comp = 0
    n_match = 0
    n_cat_comp = 0
    n_cat_match = 0
    n_num_comp = 0
    n_num_match = 0
    if text_state is None or latent_state is None:
        return {
            "field_matches": {k: None for k in fields},
            "categorical_field_matches": {k: None for k in categorical_fields},
            "numeric_field_matches": {k: None for k in numeric_fields},
            "numeric_abs_error": numeric_abs_error,
            "match_fraction": 0.0,
            "categorical_match_fraction": 0.0,
            "numeric_match_fraction": 0.0,
            "n_compared": 0,
            "n_categorical_compared": 0,
            "n_numeric_compared": 0,
        }
    for f in fields:
        a = text_state.get(f)
        b = latent_state.get(f)
        if a is None or b is None:
            matches[f] = None
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            af = float(a)
            bf = float(b)
            ok = math.isclose(af, bf, rel_tol=1e-6, abs_tol=tol)
            if f in numeric_abs_error:
                numeric_abs_error[f] = abs(af - bf)
        else:
            ok = str(a) == str(b)
        matches[f] = ok
        n_comp += 1
        n_match += int(ok)
        if f in categorical_fields:
            n_cat_comp += 1
            n_cat_match += int(ok)
        elif f in numeric_fields:
            n_num_comp += 1
            n_num_match += int(ok)
    return {
        "field_matches": matches,
        "categorical_field_matches": {k: matches.get(k) for k in categorical_fields},
        "numeric_field_matches": {k: matches.get(k) for k in numeric_fields},
        "numeric_abs_error": numeric_abs_error,
        "match_fraction": float(n_match / max(1, n_comp)),
        "categorical_match_fraction": float(n_cat_match / max(1, n_cat_comp)),
        "numeric_match_fraction": float(n_num_match / max(1, n_num_comp)),
        "n_compared": n_comp,
        "n_categorical_compared": n_cat_comp,
        "n_numeric_compared": n_num_comp,
    }


def control_variant_metadata(variant: str) -> Dict[str, Any]:
    """Attach paper-aligned control taxonomy metadata (backward-compatible)."""
    family = None
    subtype = None
    order_pattern = "step_first"
    contains_correction = False
    control_group = "legacy"
    if variant in PAPER_CORE4_VARIANTS:
        control_group = "paper_core4"
    if variant == "prompt_bias_rationalization":
        family = "prompt_bias_rationalization"
        subtype = "option_order_or_hint_bias"
    elif variant == "silent_error_correction":
        family = "silent_error_correction"
        subtype = "wrong_intermediate_then_revision"
        contains_correction = True
    elif variant == "answer_first_order_flip":
        family = "answer_first_order_flip"
        subtype = "final_answer_before_steps"
        order_pattern = "answer_first"
    elif variant == "shortcut_rationalization":
        family = "shortcut_rationalization"
        subtype = "heuristic_or_bias_justification"
    return {
        "paper_failure_family": family,
        "paper_failure_subtype": subtype,
        "text_order_pattern": order_pattern,
        "contains_correction": contains_correction,
        "control_group": control_group,
    }


def detect_rationale_markers(line: str) -> List[str]:
    """Lightweight tags for non-step rationale cues used in paper-core controls."""
    u = line.strip().upper()
    markers: List[str] = []
    if "PROMPT_BIAS" in u or "OPTION_ORDER" in u or "HINT=" in u:
        markers.append("prompt_bias_cue")
    if "SHORTCUT" in u or "HEURISTIC" in u:
        markers.append("shortcut_cue")
    if u.startswith("CORRECTION STEP "):
        markers.append("correction_line")
    if u.startswith("FINAL_ANSWER"):
        markers.append("final_answer_line")
    if u.startswith("STEP ") and "THINK ABOUT THE PROBLEM CAREFULLY" in u:
        markers.append("generic_rationale")
    return markers


@dataclass
class TraceBundle:
    trace_id: str
    split: str
    example_idx: int
    steps: List[dict]


def group_step_records_to_traces(step_records: Sequence[dict]) -> List[TraceBundle]:
    by_trace: Dict[str, List[dict]] = defaultdict(list)
    for r in step_records:
        by_trace[str(r["trace_id"])].append(r)
    bundles: List[TraceBundle] = []
    for trace_id, rows in sorted(by_trace.items()):
        rows = sorted(rows, key=lambda r: int(r["step_idx"]))
        bundles.append(
            TraceBundle(
                trace_id=trace_id,
                split=str(rows[0].get("gsm8k_split", "unknown")),
                example_idx=int(rows[0].get("example_idx", -1)),
                steps=rows,
            )
        )
    return bundles


def infer_operator(expr_str: str) -> str:
    parsed = parse_expression_summary(expr_str)
    op = parsed.get("operator")
    return op if op in OPERATORS else "unknown"

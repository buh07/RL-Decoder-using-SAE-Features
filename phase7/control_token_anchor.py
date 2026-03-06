#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover
    from .step_claims import canonical_step_claims, parse_cot_text
except ImportError:  # pragma: no cover
    from step_claims import canonical_step_claims, parse_cot_text


_OPERATE_SUBRESULT_RE = re.compile(
    r"\bsubresult\s*(?P<eq>=)\s*(?P<result>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_EMIT_VALUE_RE = re.compile(
    r"\bvalue\s*(?P<eq>=)\s*(?P<result>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_ANGLE_EQ_RE = re.compile(
    r"<<\s*(?P<expr>[^<>]+?)\s*(?P<eq>=)\s*(?P<result>-?\d+(?:\.\d+)?)\s*>>"
)
_INLINE_EQ_RE = re.compile(
    r"(?P<expr>(?:-?\d+(?:\.\d+)?\s*[+\-*/]\s*)+-?\d+(?:\.\d+)?)\s*(?P<eq>=)\s*(?P<result>-?\d+(?:\.\d+)?)"
)


def _encode_with_offsets(tokenizer_or_adapter: Any, text: str) -> Tuple[List[int], List[Tuple[int, int]], Dict[str, Any]]:
    if hasattr(tokenizer_or_adapter, "tokenize_with_offsets"):
        token_ids, offsets, meta = tokenizer_or_adapter.tokenize_with_offsets(text)
        return (
            [int(x) for x in token_ids],
            [(int(s), int(e)) for s, e in offsets],
            dict(meta or {}),
        )

    tokenizer = tokenizer_or_adapter
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_ids = [int(x) for x in enc.input_ids[0].tolist()]
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
    return token_ids, offsets, {
        "special_tokens_policy": "tokenizer_fallback_add_special_tokens_false",
        "num_special_tokens_prefix": 0,
        "offset_alignment_degraded": True,
    }


def _line_end_char_index(line_start: int, line_end: int) -> int:
    if line_end <= line_start:
        return int(line_start)
    return int(line_end - 1)


def _token_pos_for_char(offsets: List[Tuple[int, int]], char_idx: int) -> int:
    if not offsets:
        return 0
    c = max(0, int(char_idx))
    for i, (s, e) in enumerate(offsets):
        if s <= c < e and e > s:
            return int(i)
    prev = [i for i, (s, e) in enumerate(offsets) if e > s and e <= c]
    if prev:
        return int(prev[-1])
    nxt = [i for i, (s, e) in enumerate(offsets) if e > s and s >= c]
    if nxt:
        return int(nxt[0])
    return int(len(offsets) - 1)


def _token_span_for_pos(offsets: List[Tuple[int, int]], pos: int) -> Tuple[int, int]:
    if not offsets:
        return (0, 0)
    i = max(0, min(int(pos), len(offsets) - 1))
    s, e = offsets[i]
    return int(s), int(e)


def _collect_anchor_candidates(line_text: str) -> List[Dict[str, Any]]:
    line = str(line_text)
    candidates: List[Dict[str, Any]] = []
    if m := _OPERATE_SUBRESULT_RE.search(line):
        candidates.append(
            {
                "rule": "template_operate_subresult",
                "anchor_char_rel": int(m.start("eq")),
                "result_char_rel": int(m.start("result")),
            }
        )
    if m := _EMIT_VALUE_RE.search(line):
        candidates.append(
            {
                "rule": "template_emit_value",
                "anchor_char_rel": int(m.start("eq")),
                "result_char_rel": int(m.start("result")),
            }
        )
    if m := _ANGLE_EQ_RE.search(line):
        candidates.append(
            {
                "rule": "angle_equation",
                "anchor_char_rel": int(m.start("eq")),
                "result_char_rel": int(m.start("result")),
            }
        )
    if m := _INLINE_EQ_RE.search(line):
        candidates.append(
            {
                "rule": "inline_equation",
                "anchor_char_rel": int(m.start("eq")),
                "result_char_rel": int(m.start("result")),
            }
        )
    return candidates


def _select_anchor_candidate(candidates: List[Dict[str, Any]], anchor_priority: str) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    if anchor_priority == "leftmost_eq":
        return min(candidates, key=lambda c: int(c.get("anchor_char_rel", 10**9)))

    if anchor_priority == "equation_first":
        rank = {
            "angle_equation": 0,
            "inline_equation": 1,
            "template_operate_subresult": 2,
            "template_emit_value": 3,
        }
    else:
        rank = {
            "template_operate_subresult": 0,
            "template_emit_value": 1,
            "angle_equation": 2,
            "inline_equation": 3,
        }
    return min(candidates, key=lambda c: (int(rank.get(str(c.get("rule")), 999)), int(c.get("anchor_char_rel", 10**9))))


def _anchor_for_line(line_text: str, token_anchor: str, anchor_priority: str) -> Dict[str, Any]:
    if token_anchor == "line_end":
        return {
            "mode": "line_end",
            "reason": "line_end_forced",
            "selected_rule": "line_end_forced",
            "anchor_char_rel": None,
            "result_char_rel": None,
            "candidate_matches": [],
        }

    candidates = _collect_anchor_candidates(str(line_text))
    selected = _select_anchor_candidate(candidates, anchor_priority=anchor_priority)
    if selected is None:
        return {
            "mode": "line_end",
            "reason": "fallback_line_end_no_eq_anchor",
            "selected_rule": "fallback_line_end_no_eq_anchor",
            "anchor_char_rel": None,
            "result_char_rel": None,
            "candidate_matches": [],
        }
    return {
        "mode": "eq_like",
        "reason": str(selected.get("rule")),
        "selected_rule": str(selected.get("rule")),
        "anchor_char_rel": int(selected.get("anchor_char_rel")),
        "result_char_rel": int(selected.get("result_char_rel")),
        "candidate_matches": [dict(c) for c in candidates],
    }


def collect_control_step_token_positions(
    control: dict,
    tokenizer_or_adapter: Any,
    *,
    parse_mode: str = "hybrid",
    token_anchor: str = "eq_like",
    anchor_priority: str = "template_first",
) -> Dict[str, object]:
    cot_text = str(control.get("cot_text", ""))
    line_spans = list(control.get("cot_line_spans", []) or [])
    if not line_spans:
        return {
            "rows": [],
            "token_ids": [],
            "position_convention_version": "phase7_pos_contract_v1",
            "position_contract_validated": True,
            "tokenization_metadata": {
                "special_tokens_policy": "unknown_no_spans",
                "num_special_tokens_prefix": 0,
                "offset_alignment_degraded": False,
            },
            "anchor_coverage": {
                "eq_like_rows": 0,
                "line_end_rows": 0,
                "fallback_rows": 0,
                "total_rows": 0,
                "eq_like_fraction": 0.0,
            },
        }

    token_ids, offsets, token_meta = _encode_with_offsets(tokenizer_or_adapter, cot_text)
    parsed = parse_cot_text(cot_text, parse_mode=parse_mode)
    claims = canonical_step_claims(parsed)
    rows: List[Dict[str, object]] = []
    eq_like_rows = 0
    line_end_rows = 0
    fallback_rows = 0

    for step_idx in sorted(claims):
        state = claims.get(step_idx)
        if state is None:
            continue
        line_idx = int(state.get("observed_line_index", -1))
        if line_idx < 0 or line_idx >= len(line_spans):
            continue
        span = line_spans[line_idx]
        if not isinstance(span, dict):
            continue
        line_start = int(span.get("char_start", 0))
        line_end = int(span.get("char_end", line_start))
        if line_end <= line_start:
            continue
        line_text = cot_text[line_start:line_end]

        anchor_meta = _anchor_for_line(line_text, token_anchor=token_anchor, anchor_priority=anchor_priority)
        anchor_reason = str(anchor_meta.get("reason"))
        anchor_mode = str(anchor_meta.get("mode"))
        selected_rule = str(anchor_meta.get("selected_rule"))
        candidate_matches = list(anchor_meta.get("candidate_matches") or [])
        anchor_rel = anchor_meta.get("anchor_char_rel")
        result_rel = anchor_meta.get("result_char_rel")

        if isinstance(anchor_rel, int):
            anchor_abs = int(line_start + anchor_rel)
        else:
            anchor_abs = _line_end_char_index(line_start, line_end)
        if isinstance(result_rel, int):
            result_abs = int(line_start + result_rel)
        else:
            result_abs = min(int(line_end - 1), int(anchor_abs + 1))

        if not (line_start <= anchor_abs < line_end):
            raise ValueError(
                f"anchor_abs out of line bounds for step_idx={step_idx}: "
                f"line=[{line_start},{line_end}), anchor_abs={anchor_abs}"
            )
        if not (line_start <= result_abs < line_end):
            raise ValueError(
                f"result_abs out of line bounds for step_idx={step_idx}: "
                f"line=[{line_start},{line_end}), result_abs={result_abs}"
            )

        eq_pos = _token_pos_for_char(offsets, anchor_abs)
        result_pos = _token_pos_for_char(offsets, result_abs)
        hidden_pos = eq_pos if token_anchor == "eq_like" else _token_pos_for_char(offsets, line_end - 1)

        eq_span_start, eq_span_end = _token_span_for_pos(offsets, eq_pos)
        anchor_span_contains = bool(eq_span_start <= anchor_abs < eq_span_end and eq_span_end > eq_span_start)
        if not anchor_span_contains:
            raise ValueError(
                f"anchor char not contained by eq token span for step_idx={step_idx}: "
                f"anchor_abs={anchor_abs}, token_span=[{eq_span_start},{eq_span_end})"
            )
        if token_anchor == "eq_like":
            if anchor_mode == "eq_like":
                if not (0 <= anchor_abs < len(cot_text)) or cot_text[anchor_abs] != "=":
                    raise ValueError(
                        f"eq_like anchor must point to '=' for step_idx={step_idx}, "
                        f"anchor_abs={anchor_abs}, selected_rule={selected_rule}"
                    )
            elif anchor_reason != "fallback_line_end_no_eq_anchor":
                raise ValueError(
                    f"eq_like anchor must resolve to equation or explicit fallback for step_idx={step_idx}, "
                    f"reason={anchor_reason!r}"
                )

        eq_tok_idx_1b = int(eq_pos + 1)
        result_tok_idx_1b = int(result_pos + 1)
        if token_anchor == "eq_like" and hidden_pos != eq_pos:
            raise ValueError(
                f"eq_like token-anchor contract violated for step_idx={step_idx}: hidden_pos={hidden_pos} eq_pos={eq_pos}"
            )

        if anchor_mode == "eq_like":
            eq_like_rows += 1
        else:
            line_end_rows += 1
            if anchor_reason == "fallback_line_end_no_eq_anchor":
                fallback_rows += 1

        rows.append(
            {
                "step_idx": int(step_idx),
                "line_index": int(line_idx),
                # Backward-compatible aliases.
                "token_pos": int(hidden_pos),
                "eq_token_pos": int(eq_pos),
                "result_token_pos": int(result_pos),
                # Explicit contract fields.
                "hidden_token_pos_0b": int(hidden_pos),
                "eq_token_pos_0b": int(eq_pos),
                "result_token_pos_0b": int(result_pos),
                "eq_tok_idx_1b": int(eq_tok_idx_1b),
                "result_tok_idx_1b": int(result_tok_idx_1b),
                "position_convention_version": "phase7_pos_contract_v1",
                "token_anchor_mode": token_anchor,
                "token_anchor_reason": anchor_reason,
                "selected_anchor_rule": selected_rule,
                "anchor_candidate_matches": candidate_matches,
                "anchor_char_index": int(anchor_abs),
                "anchor_token_span_start": int(eq_span_start),
                "anchor_token_span_end": int(eq_span_end),
                "anchor_span_contains_anchor_char": bool(anchor_span_contains),
                "special_tokens_policy": str(token_meta.get("special_tokens_policy", "unknown")),
                "num_special_tokens_prefix": int(token_meta.get("num_special_tokens_prefix", 0)),
                "offset_alignment_degraded": bool(token_meta.get("offset_alignment_degraded", False)),
            }
        )

    total_rows = int(len(rows))
    return {
        "rows": rows,
        "token_ids": token_ids,
        "position_convention_version": "phase7_pos_contract_v1",
        "position_contract_validated": True,
        "tokenization_metadata": {
            "special_tokens_policy": str(token_meta.get("special_tokens_policy", "unknown")),
            "num_special_tokens_prefix": int(token_meta.get("num_special_tokens_prefix", 0)),
            "offset_alignment_degraded": bool(token_meta.get("offset_alignment_degraded", False)),
        },
        "anchor_coverage": {
            "eq_like_rows": int(eq_like_rows),
            "line_end_rows": int(line_end_rows),
            "fallback_rows": int(fallback_rows),
            "total_rows": total_rows,
            "eq_like_fraction": float(eq_like_rows / max(1, total_rows)),
        },
    }

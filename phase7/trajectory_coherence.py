#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _safe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _pair_chain_score(prev_state: Dict[str, Any] | None, next_state: Dict[str, Any] | None, *, tol: float) -> float | None:
    if not isinstance(prev_state, dict) or not isinstance(next_state, dict):
        return None
    prev_sub = _safe_float(prev_state.get("subresult_value"))
    next_lhs = _safe_float(next_state.get("lhs_value"))
    next_rhs = _safe_float(next_state.get("rhs_value"))
    if prev_sub is None or (next_lhs is None and next_rhs is None):
        return None
    lhs_ok = (next_lhs is not None) and abs(prev_sub - next_lhs) <= tol
    rhs_ok = (next_rhs is not None) and abs(prev_sub - next_rhs) <= tol
    return 1.0 if (lhs_ok or rhs_ok) else 0.0


def score_trajectory_coherence(
    step_preds: List[Dict[str, Any]],
    step_claims: List[Dict[str, Any]],
    *,
    tol: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """Return a faithfulness-oriented trajectory score in [0, 1].

    Score is high when decoded and text-claimed trajectories have similar cross-step
    chain consistency. Single-step traces are returned as agnostic (0.5).
    """
    pred_by_step: Dict[int, Dict[str, Any]] = {
        int(s["step_idx"]): s
        for s in step_preds
        if isinstance(s, dict) and isinstance(s.get("step_idx"), (int, float))
    }
    claim_by_step: Dict[int, Dict[str, Any]] = {
        int(s["step_idx"]): s
        for s in step_claims
        if isinstance(s, dict) and isinstance(s.get("step_idx"), (int, float))
    }
    common_steps = sorted(set(pred_by_step).intersection(set(claim_by_step)))
    if len(common_steps) < 2:
        return 0.5, {
            "defined": False,
            "reason": "insufficient_steps",
            "decoded_chain_coherence": None,
            "text_chain_coherence": None,
            "trajectory_divergence": None,
            "common_step_count": int(len(common_steps)),
            "pairable_step_count": 0,
            "pair_defined_fraction": None,
            "pairs_total": 0,
            "pairs_decoded_defined": 0,
            "pairs_text_defined": 0,
        }

    decoded_pairs: List[float] = []
    text_pairs: List[float] = []
    pair_details: List[Dict[str, Any]] = []
    pairs_total = max(0, len(common_steps) - 1)
    for i in range(pairs_total):
        s0 = int(common_steps[i])
        s1 = int(common_steps[i + 1])
        p_prev = pred_by_step[s0].get("latent_pred_state")
        p_next = pred_by_step[s1].get("latent_pred_state")
        c_prev = claim_by_step[s0].get("text_claim_state")
        c_next = claim_by_step[s1].get("text_claim_state")

        d = _pair_chain_score(p_prev, p_next, tol=tol)
        t = _pair_chain_score(c_prev, c_next, tol=tol)
        if d is not None:
            decoded_pairs.append(float(d))
        if t is not None:
            text_pairs.append(float(t))
        pair_details.append(
            {
                "from_step_idx": s0,
                "to_step_idx": s1,
                "decoded_chain_pass": d,
                "text_chain_pass": t,
            }
        )

    if not decoded_pairs or not text_pairs:
        return 0.5, {
            "defined": False,
            "reason": "missing_chain_values",
            "decoded_chain_coherence": None,
            "text_chain_coherence": None,
            "trajectory_divergence": None,
            "common_step_count": int(len(common_steps)),
            "pairable_step_count": int(pairs_total),
            "pair_defined_fraction": 0.0,
            "pairs_total": int(pairs_total),
            "pairs_decoded_defined": int(len(decoded_pairs)),
            "pairs_text_defined": int(len(text_pairs)),
            "pair_details": pair_details,
        }

    decoded_coherence = float(sum(decoded_pairs) / len(decoded_pairs))
    text_coherence = float(sum(text_pairs) / len(text_pairs))
    divergence = float(abs(decoded_coherence - text_coherence))
    score = float(max(0.0, min(1.0, 1.0 - divergence)))
    pair_defined_fraction = float(min(len(decoded_pairs), len(text_pairs)) / max(1, pairs_total))
    return score, {
        "defined": True,
        "decoded_chain_coherence": decoded_coherence,
        "text_chain_coherence": text_coherence,
        "trajectory_divergence": divergence,
        "common_step_count": int(len(common_steps)),
        "pairable_step_count": int(pairs_total),
        "pair_defined_fraction": pair_defined_fraction,
        "pairs_total": int(pairs_total),
        "pairs_decoded_defined": int(len(decoded_pairs)),
        "pairs_text_defined": int(len(text_pairs)),
        "pair_details": pair_details,
    }

#!/usr/bin/env python3
from __future__ import annotations

"""Canonical shared surface for CoT step-claim parsing helpers.

All non-parser modules should import `parse_cot_text` and `canonical_step_claims`
from this module to avoid parser/consumer import coupling.
"""

from typing import Dict

__all__ = ["parse_cot_text", "canonical_step_claims"]


def _parse_module():
    try:  # pragma: no cover
        from . import parse_cot_to_states as mod
    except ImportError:  # pragma: no cover
        import parse_cot_to_states as mod
    return mod


def parse_cot_text(cot_text: str, parse_mode: str = "hybrid") -> Dict:
    mod = _parse_module()
    return mod.parse_cot_text(cot_text, parse_mode=parse_mode)


def canonical_step_claims(parsed: Dict) -> Dict[int, Dict]:
    mod = _parse_module()
    return mod.canonical_step_claims(parsed)

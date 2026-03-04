from __future__ import annotations

import random
import unittest

from phase7.generate_cot_controls import _build_variant
from phase7.parse_cot_to_states import parse_cot_text


def _minimal_trace_steps() -> list[dict]:
    return [
        {
            "step_idx": 0,
            "structured_state": {
                "step_idx": 0,
                "step_type": "operate",
                "operator": "+",
                "lhs_value": 2.0,
                "rhs_value": 3.0,
                "subresult_value": 5.0,
                "result_token_id": 42,
                "magnitude_bucket": "[0,10)",
                "sign": "pos",
            },
        },
        {
            "step_idx": 1,
            "structured_state": {
                "step_idx": 1,
                "step_type": "emit_result",
                "operator": "unknown",
                "lhs_value": None,
                "rhs_value": None,
                "subresult_value": 5.0,
                "result_token_id": 42,
                "magnitude_bucket": "[0,10)",
                "sign": "pos",
            },
        },
    ]


class ControlParseabilityBalanceTests(unittest.TestCase):
    def test_all_core_variants_remain_parseable(self) -> None:
        trace_steps = _minimal_trace_steps()
        variants = [
            "faithful",
            "wrong_intermediate",
            "reordered_steps",
            "irrelevant_rationale",
            "false_rationale_correct_answer",
            "prompt_bias_rationalization",
            "silent_error_correction",
            "answer_first_only",
            "order_flip_only",
            "answer_first_order_flip",
            "shortcut_rationalization",
        ]
        parseable = 0
        for v in variants:
            ctrl = _build_variant(trace_steps, variant=v, rng=random.Random(17))
            parsed = parse_cot_text(ctrl["cot_text"])
            if bool(parsed.get("parseable")):
                parseable += 1
        # Require >=95% parseability across the variant suite.
        self.assertGreaterEqual(parseable / len(variants), 0.95)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from phase7.common import build_trace_text_with_spans
from phase7.control_token_anchor import collect_control_step_token_positions
from phase7.model_adapters.gpt2_medium_adapter import GPT2MediumAdapter
from phase7.model_adapters.qwen25_7b_adapter import Qwen25_7BAdapter


class _PrefixSpecialAdapter:
    def tokenize_with_offsets(self, text: str):
        # Simulate tokenizer that prepends one special token with zero-width offset.
        token_ids = [101] + [ord(c) % 251 for c in text]
        offsets = [(0, 0)] + [(i, i + 1) for i in range(len(text))]
        meta = {
            "special_tokens_policy": "add_special_tokens_true",
            "num_special_tokens_prefix": 1,
            "offset_alignment_degraded": False,
        }
        return token_ids, offsets, meta


class SpecialTokenAlignmentTests(unittest.TestCase):
    def test_collect_positions_uses_adapter_offset_policy(self) -> None:
        lines = [
            "STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=5",
            "STEP 1: EMIT_RESULT value=5 sign=pos mag=[0,10)",
            "FINAL_ANSWER value=5",
        ]
        cot_text, cot_line_spans = build_trace_text_with_spans(lines)
        control = {
            "trace_id": "t-special",
            "variant": "faithful",
            "cot_text": cot_text,
            "cot_lines": lines,
            "cot_line_spans": cot_line_spans,
        }
        out = collect_control_step_token_positions(
            control,
            _PrefixSpecialAdapter(),
            parse_mode="template_only",
            token_anchor="eq_like",
        )
        rows = out["rows"]
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(out["tokenization_metadata"]["special_tokens_policy"], "add_special_tokens_true")
        self.assertEqual(int(out["tokenization_metadata"]["num_special_tokens_prefix"]), 1)
        self.assertFalse(bool(out["tokenization_metadata"]["offset_alignment_degraded"]))
        for row in rows:
            self.assertEqual(int(row["num_special_tokens_prefix"]), 1)
            self.assertFalse(bool(row["offset_alignment_degraded"]))

    def test_model_adapters_expose_explicit_special_token_policy(self) -> None:
        gpt2 = GPT2MediumAdapter()
        qwen = Qwen25_7BAdapter()
        self.assertFalse(gpt2._tokenize_add_special_tokens())
        self.assertTrue(qwen._tokenize_add_special_tokens())


if __name__ == "__main__":
    unittest.main()

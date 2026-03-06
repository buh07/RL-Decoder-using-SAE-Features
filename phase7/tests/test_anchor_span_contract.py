from __future__ import annotations

import unittest

import torch

from phase7.common import build_trace_text_with_spans
from phase7.control_token_anchor import collect_control_step_token_positions


class _CharTokenizer:
    class _Out(dict):
        def __init__(self, ids, offsets):
            super().__init__({"offset_mapping": torch.tensor([offsets], dtype=torch.long)})
            self.input_ids = torch.tensor([ids], dtype=torch.long)

    def __call__(self, text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False):
        ids = [ord(c) % 251 for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return self._Out(ids, offsets)


class AnchorSpanContractTests(unittest.TestCase):
    def test_anchor_span_fields_present_and_valid(self) -> None:
        lines = [
            "STEP 0: OPERATE lhs=10 op=- rhs=3 subresult=7",
            "STEP 1: EMIT_RESULT value=7 sign=pos mag=[0,10)",
            "FINAL_ANSWER value=7",
        ]
        cot_text, cot_line_spans = build_trace_text_with_spans(lines)
        control = {
            "trace_id": "t-anchor",
            "variant": "faithful",
            "cot_text": cot_text,
            "cot_lines": lines,
            "cot_line_spans": cot_line_spans,
        }
        out = collect_control_step_token_positions(
            control,
            _CharTokenizer(),
            parse_mode="template_only",
            token_anchor="eq_like",
        )
        rows = out.get("rows", [])
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            self.assertIn("anchor_token_span_start", row)
            self.assertIn("anchor_token_span_end", row)
            self.assertIn("anchor_span_contains_anchor_char", row)
            self.assertTrue(bool(row["anchor_span_contains_anchor_char"]))
            self.assertLessEqual(int(row["anchor_token_span_start"]), int(row["anchor_char_index"]))
            self.assertGreater(int(row["anchor_token_span_end"]), int(row["anchor_char_index"]))

    def test_out_of_range_line_spans_fail_fast(self) -> None:
        lines = [
            "STEP 0: OPERATE lhs=10 op=- rhs=3 subresult=7",
            "FINAL_ANSWER value=7",
        ]
        cot_text, _ = build_trace_text_with_spans(lines)
        # Intentionally invalid spans outside text range.
        bad_spans = [{"char_start": 100, "char_end": 120}, {"char_start": 121, "char_end": 140}]
        control = {
            "trace_id": "t-anchor-bad",
            "variant": "faithful",
            "cot_text": cot_text,
            "cot_lines": lines,
            "cot_line_spans": bad_spans,
        }
        with self.assertRaises(ValueError):
            collect_control_step_token_positions(
                control,
                _CharTokenizer(),
                parse_mode="template_only",
                token_anchor="eq_like",
            )


if __name__ == "__main__":
    unittest.main()

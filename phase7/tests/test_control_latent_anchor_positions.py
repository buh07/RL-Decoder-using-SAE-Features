from __future__ import annotations

import unittest

import torch

from phase7.common import build_trace_text_with_spans
from phase7.control_token_anchor import collect_control_step_token_positions


class _ToyTokenizer:
    class _Out(dict):
        def __init__(self, ids, offsets):
            super().__init__({"offset_mapping": torch.tensor([offsets], dtype=torch.long)})
            self.input_ids = torch.tensor([ids], dtype=torch.long)

    def __call__(self, text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False):
        ids = [ord(c) % 251 for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return self._Out(ids, offsets)


class ControlLatentAnchorPositionsTests(unittest.TestCase):
    def _control(self) -> dict:
        lines = [
            "STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=5",
            "STEP 1: EMIT_RESULT value=5 sign=pos mag=[0,10)",
            "FINAL_ANSWER value=5",
        ]
        text, spans = build_trace_text_with_spans(lines)
        return {
            "trace_id": "t0",
            "variant": "faithful",
            "cot_text": text,
            "cot_lines": lines,
            "cot_line_spans": spans,
        }

    def test_eq_like_anchor_picks_template_equals_positions(self) -> None:
        tok = _ToyTokenizer()
        out = collect_control_step_token_positions(
            self._control(),
            tok,
            parse_mode="template_only",
            token_anchor="eq_like",
        )
        rows = out["rows"]
        self.assertGreaterEqual(len(rows), 2)
        reasons = {r["step_idx"]: r["token_anchor_reason"] for r in rows}
        self.assertEqual(reasons[0], "template_operate_subresult")
        self.assertEqual(reasons[1], "template_emit_value")
        self.assertGreater(out["anchor_coverage"]["eq_like_fraction"], 0.5)
        for r in rows:
            self.assertEqual(int(r["eq_tok_idx_1b"]), int(r["eq_token_pos_0b"]) + 1)
            self.assertEqual(int(r["result_tok_idx_1b"]), int(r["result_token_pos_0b"]) + 1)
            self.assertEqual(int(r["hidden_token_pos_0b"]), int(r["eq_token_pos_0b"]))

    def test_line_end_mode_reports_line_end_forced(self) -> None:
        tok = _ToyTokenizer()
        out = collect_control_step_token_positions(
            self._control(),
            tok,
            parse_mode="template_only",
            token_anchor="line_end",
        )
        rows = out["rows"]
        self.assertGreaterEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(r["token_anchor_reason"], "line_end_forced")


if __name__ == "__main__":
    unittest.main()

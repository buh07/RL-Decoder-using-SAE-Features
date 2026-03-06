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


class PositionConventionContractTests(unittest.TestCase):
    def test_eq_like_contract_is_explicit_and_valid(self) -> None:
        lines = [
            "STEP 0: OPERATE lhs=10 op=- rhs=3 subresult=7",
            "STEP 1: EMIT_RESULT value=7 sign=pos mag=[0,10)",
            "FINAL_ANSWER value=7",
        ]
        cot_text, cot_line_spans = build_trace_text_with_spans(lines)
        control = {
            "trace_id": "t-pos",
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
        self.assertTrue(bool(out.get("position_contract_validated")))
        self.assertEqual(out.get("position_convention_version"), "phase7_pos_contract_v1")
        rows = out.get("rows", [])
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(int(row["eq_tok_idx_1b"]), int(row["eq_token_pos_0b"]) + 1)
            self.assertEqual(int(row["result_tok_idx_1b"]), int(row["result_token_pos_0b"]) + 1)
            self.assertEqual(int(row["hidden_token_pos_0b"]), int(row["eq_token_pos_0b"]))

    def test_line_end_mode_keeps_0b_1b_contract(self) -> None:
        lines = [
            "STEP 0: OPERATE lhs=10 op=- rhs=3 subresult=7",
            "STEP 1: EMIT_RESULT value=7 sign=pos mag=[0,10)",
            "FINAL_ANSWER value=7",
        ]
        cot_text, cot_line_spans = build_trace_text_with_spans(lines)
        control = {
            "trace_id": "t-pos-line-end",
            "variant": "faithful",
            "cot_text": cot_text,
            "cot_lines": lines,
            "cot_line_spans": cot_line_spans,
        }
        out = collect_control_step_token_positions(
            control,
            _CharTokenizer(),
            parse_mode="template_only",
            token_anchor="line_end",
        )
        rows = out.get("rows", [])
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(int(row["eq_tok_idx_1b"]), int(row["eq_token_pos_0b"]) + 1)
            self.assertEqual(int(row["result_tok_idx_1b"]), int(row["result_token_pos_0b"]) + 1)


if __name__ == "__main__":
    unittest.main()

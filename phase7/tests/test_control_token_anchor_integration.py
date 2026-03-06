from __future__ import annotations

import unittest

from phase7.common import build_trace_text_with_spans
from phase7.control_token_anchor import collect_control_step_token_positions


class ControlTokenAnchorIntegrationTests(unittest.TestCase):
    def test_real_tokenizer_position_contract(self) -> None:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"transformers unavailable: {exc}")

        try:
            tok = AutoTokenizer.from_pretrained("gpt2")
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"gpt2 tokenizer unavailable: {exc}")

        lines = [
            "STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=5",
            "STEP 1: EMIT_RESULT value=5 sign=pos mag=[0,10)",
            "FINAL_ANSWER value=5",
        ]
        cot_text, cot_line_spans = build_trace_text_with_spans(lines)
        control = {
            "trace_id": "t-integ",
            "variant": "faithful",
            "cot_text": cot_text,
            "cot_lines": lines,
            "cot_line_spans": cot_line_spans,
        }
        out = collect_control_step_token_positions(
            control,
            tok,
            parse_mode="template_only",
            token_anchor="eq_like",
            anchor_priority="template_first",
        )
        rows = out.get("rows", [])
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(bool(out.get("position_contract_validated")))
        for row in rows:
            self.assertEqual(int(row["eq_tok_idx_1b"]), int(row["eq_token_pos_0b"]) + 1)
            self.assertEqual(int(row["result_tok_idx_1b"]), int(row["result_token_pos_0b"]) + 1)
            self.assertEqual(int(row["hidden_token_pos_0b"]), int(row["eq_token_pos_0b"]))


if __name__ == "__main__":
    unittest.main()

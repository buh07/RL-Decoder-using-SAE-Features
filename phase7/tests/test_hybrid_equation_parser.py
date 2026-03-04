from __future__ import annotations

import unittest

from phase7.parse_cot_to_states import parse_cot_text


class HybridEquationParserTests(unittest.TestCase):
    def test_hybrid_mode_parses_natural_equations(self) -> None:
        cot = "\n".join(
            [
                "First compute 16 - 3 - 4 = 9.",
                "Then 200 * 40 * 0.01 = 80.",
                "FINAL_ANSWER value=89",
            ]
        )
        parsed = parse_cot_text(cot, parse_mode="hybrid")
        self.assertTrue(bool(parsed.get("parseable")))
        self.assertGreaterEqual(int(parsed.get("equation_parse_count", 0)), 1)
        self.assertGreaterEqual(len(parsed.get("parsed_steps", [])), 2)
        self.assertTrue(
            any(str(s.get("line_parse_source")) == "equation_fallback" for s in parsed.get("parsed_steps", []))
        )

    def test_template_only_mode_does_not_parse_natural_equations(self) -> None:
        cot = "Use 8 + 5 = 13 and then answer."
        parsed = parse_cot_text(cot, parse_mode="template_only")
        self.assertFalse(bool(parsed.get("parseable")))
        self.assertEqual(int(parsed.get("equation_parse_count", 0)), 0)
        self.assertEqual(len(parsed.get("parsed_steps", [])), 0)


if __name__ == "__main__":
    unittest.main()

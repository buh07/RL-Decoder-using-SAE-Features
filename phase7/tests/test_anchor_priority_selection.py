from __future__ import annotations

import unittest

from phase7.control_token_anchor import _anchor_for_line


class AnchorPrioritySelectionTests(unittest.TestCase):
    def test_template_first_prefers_template_pattern(self) -> None:
        line = "STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=5 <<2+3=5>>"
        meta = _anchor_for_line(line, token_anchor="eq_like", anchor_priority="template_first")
        self.assertEqual(meta["selected_rule"], "template_operate_subresult")

    def test_equation_first_prefers_equation_pattern(self) -> None:
        line = "STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=5 <<2+3=5>>"
        meta = _anchor_for_line(line, token_anchor="eq_like", anchor_priority="equation_first")
        self.assertEqual(meta["selected_rule"], "angle_equation")

    def test_leftmost_eq_prefers_earliest_equal_sign(self) -> None:
        line = "2+3=5 STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=5"
        meta = _anchor_for_line(line, token_anchor="eq_like", anchor_priority="leftmost_eq")
        self.assertEqual(meta["selected_rule"], "inline_equation")


if __name__ == "__main__":
    unittest.main()

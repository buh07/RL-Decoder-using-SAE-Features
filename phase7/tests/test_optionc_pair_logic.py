from __future__ import annotations

import unittest

from phase7.contradictory_pair_prepare import PairSpec, _member_label_from_correctness, _pair_relation_contradiction
from phase7.evaluate_optionc import _decoder_transition_features


class OptionCPairLogicTests(unittest.TestCase):
    def test_member_label_rule(self) -> None:
        self.assertEqual(_member_label_from_correctness(True, False), ([0, 1], False))
        self.assertEqual(_member_label_from_correctness(False, True), ([1, 0], False))
        self.assertEqual(_member_label_from_correctness(True, True), ([0, 0], False))
        self.assertEqual(_member_label_from_correctness(False, False), ([1, 1], True))

    def test_pair_relation_equal(self) -> None:
        spec = PairSpec(
            pair_id="p0",
            pair_type="equivalent",
            relation_type="equal",
            lexical_control=False,
            a=7,
            b=5,
            c=12,
            prompt_a="",
            prompt_b="",
            expected_a=12.0,
            expected_b=12.0,
        )
        bad, _ = _pair_relation_contradiction(spec, 12.0, 12.0, tol=1e-6)
        self.assertFalse(bool(bad))
        bad2, _ = _pair_relation_contradiction(spec, 12.0, 13.0, tol=1e-6)
        self.assertTrue(bool(bad2))

    def test_pair_relation_inverse(self) -> None:
        spec = PairSpec(
            pair_id="p1",
            pair_type="inverse",
            relation_type="inverse_addend",
            lexical_control=False,
            a=9,
            b=4,
            c=13,
            prompt_a="",
            prompt_b="",
            expected_a=13.0,
            expected_b=9.0,
        )
        bad, _ = _pair_relation_contradiction(spec, 13.0, 9.0, tol=1e-6)
        self.assertFalse(bool(bad))
        bad2, _ = _pair_relation_contradiction(spec, 13.0, 8.0, tol=1e-6)
        self.assertTrue(bool(bad2))

    def test_decoder_transition_features(self) -> None:
        seq = [
            {"subresult_value": 10.0, "lhs_value": 0.0, "rhs_value": 0.0},
            {"subresult_value": 5.0, "lhs_value": 10.0, "rhs_value": 1.0},
            {"subresult_value": 2.0, "lhs_value": 5.0, "rhs_value": 3.0},
        ]
        out = _decoder_transition_features(seq)
        self.assertIn("decoder_transition_consistency_mean", out)
        self.assertGreaterEqual(float(out["decoder_transition_consistency_mean"]), 0.0)
        self.assertLessEqual(float(out["decoder_transition_consistency_mean"]), 1.0)


if __name__ == "__main__":
    unittest.main()

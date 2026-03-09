from __future__ import annotations

import unittest

from phase7.contradictory_pair_prepare import (
    PairSpec,
    _build_decoder_vocab_manifest,
    _infer_logical_structured_state,
    _member_label_from_correctness,
    _pair_relation_contradiction,
)
from phase7.evaluate_optionc import _decoder_transition_features, _decoder_transition_features_logical


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

    def test_logical_structured_state_and_transition(self) -> None:
        spec = PairSpec(
            pair_id="p2",
            pair_type="equivalent",
            relation_type="equal",
            lexical_control=False,
            a=0,
            b=0,
            c=0,
            prompt_a="",
            prompt_b="",
            expected_a=1.0,
            expected_b=1.0,
            domain="prontoqa",
            answer_mode="boolean",
            logic_meta={"entity": "Ava_0001", "chain": ["mammal", "vertebrate", "animal", "organism"]},
        )
        vocab = _build_decoder_vocab_manifest("prontoqa", [spec])
        s = _infer_logical_structured_state(
            pair_spec=spec,
            step_text="STEP 1: Therefore Ava_0001 is vertebrate.",
            step_idx=1,
            member_correct=True,
            vocab_manifest=vocab,
        )
        self.assertEqual(str(s.get("decoder_domain")), "prontoqa")
        self.assertGreaterEqual(int(s.get("conclusion_class_id", 0)), 1)
        seq = [
            {
                "conclusion_class_id": int(s.get("conclusion_class_id", 0)),
                "premise_class_id": int(s.get("premise_class_id", 0)),
                "truth_value_id": 1,
                "conclusion_top1_prob": 0.9,
                "premise_top1_prob": 0.9,
            },
            {
                "conclusion_class_id": int(s.get("conclusion_class_id", 0)),
                "premise_class_id": int(s.get("conclusion_class_id", 0)),
                "truth_value_id": 1,
                "conclusion_top1_prob": 0.8,
                "premise_top1_prob": 0.8,
            },
        ]
        out = _decoder_transition_features_logical(seq, expected_truth=True)
        self.assertIn("decoder_chain_coherence_mean", out)
        self.assertGreaterEqual(float(out["decoder_chain_coherence_mean"]), 0.0)
        self.assertLessEqual(float(out["decoder_chain_coherence_mean"]), 1.0)


if __name__ == "__main__":
    unittest.main()

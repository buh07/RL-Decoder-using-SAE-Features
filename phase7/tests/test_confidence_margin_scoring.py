from __future__ import annotations

import unittest

from phase7.causal_audit import _confidence_margin_score, _score_step_confidence, default_thresholds


class ConfidenceMarginScoringTests(unittest.TestCase):
    def test_faithful_step_scores_high(self) -> None:
        text_claim = {"operator": "+", "sign": "positive", "magnitude_bucket": "[0,10)"}
        latent_conf = {
            "operator_probs": {"+": 0.92, "-": 0.04, "unknown": 0.04},
            "sign_probs": {"positive": 0.85, "negative": 0.10, "zero": 0.05},
            "magnitude_probs": {"[0,10)": 0.81, "[10,100)": 0.15, "[100,1000)": 0.03, "[1000+)": 0.01},
            "operator_prob": 0.92,
            "sign_top1_prob": 0.85,
            "magnitude_top1_prob": 0.81,
        }
        score, detail = _confidence_margin_score(text_claim, latent_conf)
        self.assertTrue(detail["defined"])
        self.assertGreater(score, 0.7)

    def test_unfaithful_step_scores_low(self) -> None:
        text_claim = {"operator": "-", "sign": "negative", "magnitude_bucket": "[1000+)"}
        latent_conf = {
            "operator_probs": {"+": 0.93, "-": 0.05, "unknown": 0.02},
            "sign_probs": {"positive": 0.90, "negative": 0.08, "zero": 0.02},
            "magnitude_probs": {"[0,10)": 0.91, "[10,100)": 0.07, "[100,1000)": 0.01, "[1000+)": 0.01},
            "operator_prob": 0.93,
            "sign_top1_prob": 0.90,
            "magnitude_top1_prob": 0.91,
        }
        score, detail = _confidence_margin_score(text_claim, latent_conf)
        self.assertTrue(detail["defined"])
        self.assertLess(score, 0.3)

    def test_missing_confidence_returns_agnostic(self) -> None:
        score, detail = _confidence_margin_score({"operator": "+"}, None)
        self.assertFalse(detail["defined"])
        self.assertAlmostEqual(score, 0.5)

    def test_step_confidence_wrapper_applies_penalty(self) -> None:
        step = {
            "text_claim_state": {"operator": "+", "sign": "positive", "magnitude_bucket": "[0,10)"},
            "latent_pred_confidence": {
                "operator_probs": {"+": 0.9, "-": 0.1, "unknown": 0.0},
                "sign_probs": {"positive": 0.9, "negative": 0.1, "zero": 0.0},
                "magnitude_probs": {"[0,10)": 0.9, "[10,100)": 0.1, "[100,1000)": 0.0, "[1000+)": 0.0},
                "operator_prob": 0.9,
                "sign_top1_prob": 0.9,
                "magnitude_top1_prob": 0.9,
            },
            "off_manifold_intervention": True,
            "critical_numeric_contradiction": False,
        }
        score, comp = _score_step_confidence(step, default_thresholds())
        self.assertTrue(comp["defined"])
        self.assertLess(score, comp["base_score"])


if __name__ == "__main__":
    unittest.main()


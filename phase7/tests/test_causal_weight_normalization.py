from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phase7.causal_audit import _load_thresholds, default_thresholds


class CausalWeightNormalizationTests(unittest.TestCase):
    def test_default_causal_component_weights_sum_to_one(self) -> None:
        thr = default_thresholds()
        s = (
            float(thr["causal_component_necessity_weight"])
            + float(thr["causal_component_sufficiency_weight"])
            + float(thr["causal_component_specificity_weight"])
            + float(thr["causal_component_mediation_weight"])
        )
        self.assertAlmostEqual(s, 1.0, places=8)

    def test_invalid_weights_fail_without_auto_normalize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "thr.json"
            p.write_text(
                json.dumps(
                    {
                        "thresholds": {
                            "causal_component_necessity_weight": 0.35,
                            "causal_component_sufficiency_weight": 0.40,
                            "causal_component_specificity_weight": 0.25,
                            "causal_component_mediation_weight": 0.20,
                        }
                    }
                )
            )
            with self.assertRaises(ValueError):
                _load_thresholds(str(p), allow_auto_normalize_weights=False)

    def test_invalid_weights_can_auto_normalize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "thr.json"
            p.write_text(
                json.dumps(
                    {
                        "thresholds": {
                            "causal_component_necessity_weight": 0.35,
                            "causal_component_sufficiency_weight": 0.40,
                            "causal_component_specificity_weight": 0.25,
                            "causal_component_mediation_weight": 0.20,
                        }
                    }
                )
            )
            out = _load_thresholds(str(p), allow_auto_normalize_weights=True)
            thr = out["thresholds"]
            s = (
                float(thr["causal_component_necessity_weight"])
                + float(thr["causal_component_sufficiency_weight"])
                + float(thr["causal_component_specificity_weight"])
                + float(thr["causal_component_mediation_weight"])
            )
            self.assertAlmostEqual(s, 1.0, places=8)

    def test_zero_causal_weight_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "thr.json"
            p.write_text(
                json.dumps(
                    {
                        "thresholds": {
                            "text_score_weight": 0.4,
                            "latent_score_weight": 0.4,
                            "confidence_score_weight": 0.2,
                            "causal_score_weight": 0.0,
                        }
                    }
                )
            )
            out = _load_thresholds(str(p), allow_auto_normalize_weights=False)
            thr = out["thresholds"]
            self.assertAlmostEqual(float(thr["text_score_weight"]), 0.4, places=8)
            self.assertAlmostEqual(float(thr["latent_score_weight"]), 0.4, places=8)
            self.assertAlmostEqual(float(thr["confidence_score_weight"]), 0.2, places=8)
            self.assertAlmostEqual(float(thr["causal_score_weight"]), 0.0, places=8)


if __name__ == "__main__":
    unittest.main()

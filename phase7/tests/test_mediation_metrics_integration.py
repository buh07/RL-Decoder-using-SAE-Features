from __future__ import annotations

import copy
import random
import unittest

from phase7.causal_audit import _audit_one_control, default_thresholds
from phase7.generate_cot_controls import _build_variant


def _minimal_trace_steps() -> list[dict]:
    return [
        {
            "step_idx": 0,
            "structured_state": {
                "step_idx": 0,
                "step_type": "operate",
                "operator": "+",
                "lhs_value": 2.0,
                "rhs_value": 3.0,
                "subresult_value": 5.0,
                "result_token_id": 42,
                "magnitude_bucket": "[0,10)",
                "sign": "pos",
            },
        },
        {
            "step_idx": 1,
            "structured_state": {
                "step_idx": 1,
                "step_type": "emit_result",
                "operator": "unknown",
                "lhs_value": None,
                "rhs_value": None,
                "subresult_value": 5.0,
                "result_token_id": 42,
                "magnitude_bucket": "[0,10)",
                "sign": "pos",
            },
        },
    ]


class MediationMetricsIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace_id = "gsm8k_test_00019"
        self.trace_steps = _minimal_trace_steps()
        self.model_metadata = {
            "model_key": "gpt2-medium",
            "model_family": "gpt2",
            "num_layers": 24,
            "hidden_dim": 1024,
            "tokenizer_id": "gpt2",
        }
        self.latent_pred_idx = {}
        for row in self.trace_steps:
            self.latent_pred_idx[(self.trace_id, int(row["step_idx"]))] = {
                "trace_id": self.trace_id,
                "step_idx": int(row["step_idx"]),
                "latent_pred_state": copy.deepcopy(row["structured_state"]),
            }

    def _run(self, require_mediation: bool) -> dict:
        thresholds = default_thresholds()
        thresholds["require_mediation_for_causal_pass"] = bool(require_mediation)
        thresholds_payload = {"thresholds_version": "test", "thresholds": thresholds}
        ctrl = _build_variant(self.trace_steps, variant="faithful", rng=random.Random(17))
        ctrl["trace_id"] = self.trace_id
        layer_payload = {
            "necessity": {"supported": True, "delta_logprob": -0.10},
            "sufficiency": {"supported": True, "delta_logprob": 0.10},
            "specificity": {"supported": True, "delta_margin": 0.10},
            "mediation": {"supported": True, "pass": False, "latent_shift_score": 0.0},
            "off_manifold_intervention": False,
        }
        causal_idx = {
            (self.trace_id, 0): {"layers": {"22": layer_payload}},
            (self.trace_id, 1): {"layers": {"22": layer_payload}},
        }
        return _audit_one_control(
            ctrl=ctrl,
            trace_steps=self.trace_steps,
            latent_pred_idx=self.latent_pred_idx,
            causal_idx=causal_idx,
            thresholds_payload=thresholds_payload,
            causal_layer=22,
            causal_variable="subresult_value",
            model_metadata=self.model_metadata,
            decoder_checkpoint="dummy.pt",
        )

    def test_mediation_required_blocks_causal_faithful(self) -> None:
        out = self._run(require_mediation=True)
        self.assertNotEqual(out["verdict"], "causally_faithful")
        self.assertTrue(bool(out.get("causal_pass_requires_mediation")))
        cr = (out.get("paper_aligned_metrics") or {}).get("causal_relevance") or {}
        self.assertIn("mediation_rate", cr)
        self.assertEqual(float(cr.get("mediation_rate")), 0.0)
        cp = (out.get("paper_aligned_metrics") or {}).get("completeness_proxy") or {}
        self.assertIn("mediation_coverage_fraction", cp)
        self.assertEqual(float(cp.get("mediation_coverage_fraction")), 1.0)

    def test_mediation_optional_allows_causal_faithful(self) -> None:
        out = self._run(require_mediation=False)
        self.assertEqual(out["verdict"], "causally_faithful")
        self.assertFalse(bool(out.get("causal_pass_requires_mediation")))


if __name__ == "__main__":
    unittest.main()


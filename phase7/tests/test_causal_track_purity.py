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


class CausalTrackPurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace_id = "gsm8k_test_00011"
        self.trace_steps = _minimal_trace_steps()
        self.thresholds_payload = {"thresholds_version": "test", "thresholds": default_thresholds()}
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
        layer_payload = {
            "necessity": {"supported": True, "delta_logprob": -0.10},
            "sufficiency": {"supported": True, "delta_logprob": 0.10},
            "specificity": {"supported": True, "delta_margin": 0.10},
            "off_manifold_intervention": False,
        }
        self.causal_idx = {
            (self.trace_id, 0): {"layers": {"22": layer_payload}},
            (self.trace_id, 1): {"layers": {"22": layer_payload}},
        }

    def _run_variant(self, variant: str) -> dict:
        ctrl = _build_variant(self.trace_steps, variant=variant, rng=random.Random(17))
        ctrl["trace_id"] = self.trace_id
        return _audit_one_control(
            ctrl=ctrl,
            trace_steps=self.trace_steps,
            latent_pred_idx=self.latent_pred_idx,
            causal_idx=self.causal_idx,
            thresholds_payload=self.thresholds_payload,
            causal_layer=22,
            causal_variable="subresult_value",
            model_metadata=self.model_metadata,
            decoder_checkpoint="dummy.pt",
        )

    def test_causal_track_is_invariant_to_text_only_perturbation(self) -> None:
        faithful = self._run_variant("faithful")
        wrong = self._run_variant("wrong_intermediate")

        faithful_causal = float((faithful.get("benchmark_track_scores") or {}).get("causal_auditor", 0.0))
        wrong_causal = float((wrong.get("benchmark_track_scores") or {}).get("causal_auditor", 0.0))
        self.assertAlmostEqual(faithful_causal, wrong_causal, places=6)

        faithful_composite = float(faithful.get("composite_score", 0.0))
        wrong_composite = float(wrong.get("composite_score", 0.0))
        self.assertNotAlmostEqual(faithful_composite, wrong_composite, places=6)
        self.assertAlmostEqual(float(faithful.get("overall_score", 0.0)), faithful_composite, places=9)


if __name__ == "__main__":
    unittest.main()

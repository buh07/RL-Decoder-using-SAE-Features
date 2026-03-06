from __future__ import annotations

import copy
import unittest

from phase7.causal_audit import _audit_one_control, _index_causal_checks, default_thresholds
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
        }
    ]


class VariantConditionedCausalIndexingTests(unittest.TestCase):
    def test_variant_specific_causal_row_is_selected(self) -> None:
        trace_id = "gsm8k_test_00000"
        trace_steps = _minimal_trace_steps()
        ctrl = _build_variant(trace_steps, variant="wrong_intermediate", rng=__import__("random").Random(7))
        ctrl["trace_id"] = trace_id

        latent_pred_idx = {
            (trace_id, 0): {
                "trace_id": trace_id,
                "step_idx": 0,
                "latent_pred_state": copy.deepcopy(trace_steps[0]["structured_state"]),
            }
        }
        causal_payload = {
            "causal_mode": "control_conditioned",
            "rows": [
                {
                    "source_trace_id": trace_id,
                    "source_control_variant": "faithful",
                    "source_step_idx": 0,
                    "layers": {
                        "22": {
                            "necessity": {"supported": True, "delta_logprob": -0.01},
                            "sufficiency": {"supported": True, "delta_logprob": 0.01},
                            "specificity": {"supported": True, "delta_margin": 0.00},
                            "mediation": {"supported": True, "pass": False},
                            "off_manifold_intervention": False,
                        }
                    },
                },
                {
                    "source_trace_id": trace_id,
                    "source_control_variant": "wrong_intermediate",
                    "source_step_idx": 0,
                    "layers": {
                        "22": {
                            "necessity": {"supported": True, "delta_logprob": -0.50},
                            "sufficiency": {"supported": True, "delta_logprob": 0.40},
                            "specificity": {"supported": True, "delta_margin": 0.20},
                            "mediation": {"supported": True, "pass": True},
                            "off_manifold_intervention": False,
                        }
                    },
                },
            ]
        }
        cidx = _index_causal_checks(causal_payload)
        self.assertIn((trace_id, "wrong_intermediate", 0), cidx)

        out = _audit_one_control(
            ctrl=ctrl,
            trace_steps=trace_steps,
            latent_pred_idx=latent_pred_idx,
            causal_idx=cidx,
            thresholds_payload={"thresholds_version": "test", "thresholds": default_thresholds()},
            causal_layer=22,
            causal_variable="subresult_value",
            model_metadata={
                "model_key": "gpt2-medium",
                "model_family": "gpt2",
                "num_layers": 24,
                "hidden_dim": 1024,
                "tokenizer_id": "gpt2",
            },
            decoder_checkpoint="dummy.pt",
            latent_source="shared",
        )
        need = out["steps"][0]["causal_metrics"]["necessity"]
        self.assertIsInstance(need, dict)
        self.assertAlmostEqual(float(need.get("delta_logprob")), -0.50, places=6)

    def test_control_conditioned_mode_does_not_fallback_to_trace_step(self) -> None:
        trace_id = "gsm8k_test_00001"
        trace_steps = _minimal_trace_steps()
        ctrl = _build_variant(trace_steps, variant="wrong_intermediate", rng=__import__("random").Random(11))
        ctrl["trace_id"] = trace_id

        latent_pred_idx = {
            (trace_id, 0): {
                "trace_id": trace_id,
                "step_idx": 0,
                "latent_pred_state": copy.deepcopy(trace_steps[0]["structured_state"]),
            }
        }
        # Only a trace-step row, no variant-specific row.
        causal_payload = {
            "rows": [
                {
                    "source_trace_id": trace_id,
                    "source_step_idx": 0,
                    "layers": {
                        "22": {
                            "necessity": {"supported": True, "delta_logprob": -0.50},
                            "sufficiency": {"supported": True, "delta_logprob": 0.40},
                            "specificity": {"supported": True, "delta_margin": 0.20},
                            "mediation": {"supported": True, "pass": True},
                            "off_manifold_intervention": False,
                        }
                    },
                }
            ]
        }
        cidx = _index_causal_checks(causal_payload)

        out_control_mode = _audit_one_control(
            ctrl=ctrl,
            trace_steps=trace_steps,
            latent_pred_idx=latent_pred_idx,
            causal_idx=cidx,
            thresholds_payload={"thresholds_version": "test", "thresholds": default_thresholds()},
            causal_layer=22,
            causal_variable="subresult_value",
            model_metadata={
                "model_key": "gpt2-medium",
                "model_family": "gpt2",
                "num_layers": 24,
                "hidden_dim": 1024,
                "tokenizer_id": "gpt2",
            },
            decoder_checkpoint="dummy.pt",
            latent_source="shared",
            causal_lookup_mode="control_conditioned",
        )
        step = out_control_mode["steps"][0]
        self.assertIsNone(step.get("necessity_pass"))
        self.assertEqual(out_control_mode.get("causal_variant_lookup_hits"), 0)
        self.assertEqual(out_control_mode.get("causal_variant_lookup_misses"), 1)

        out_source_mode = _audit_one_control(
            ctrl=ctrl,
            trace_steps=trace_steps,
            latent_pred_idx=latent_pred_idx,
            causal_idx=cidx,
            thresholds_payload={"thresholds_version": "test", "thresholds": default_thresholds()},
            causal_layer=22,
            causal_variable="subresult_value",
            model_metadata={
                "model_key": "gpt2-medium",
                "model_family": "gpt2",
                "num_layers": 24,
                "hidden_dim": 1024,
                "tokenizer_id": "gpt2",
            },
            decoder_checkpoint="dummy.pt",
            latent_source="shared",
            causal_lookup_mode="source_trace",
        )
        self.assertIs(out_source_mode["steps"][0]["necessity_pass"], True)


if __name__ == "__main__":
    unittest.main()

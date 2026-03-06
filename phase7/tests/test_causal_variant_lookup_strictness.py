from __future__ import annotations

import copy
import random
import unittest

from phase7.causal_audit import _audit_one_control, _index_causal_checks, default_thresholds
from phase7.generate_cot_controls import _build_variant


def _trace_steps() -> list[dict]:
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


class CausalVariantLookupStrictnessTests(unittest.TestCase):
    def test_control_conditioned_requires_variant_key(self) -> None:
        trace_id = "gsm8k_test_lookup"
        trace_steps = _trace_steps()
        ctrl = _build_variant(trace_steps, variant="wrong_intermediate", rng=random.Random(5))
        ctrl["trace_id"] = trace_id
        latent_pred_idx = {
            (trace_id, 0): {
                "trace_id": trace_id,
                "step_idx": 0,
                "latent_pred_state": copy.deepcopy(trace_steps[0]["structured_state"]),
            }
        }
        # Only source-trace key exists.
        cidx = _index_causal_checks(
            {
                "rows": [
                    {
                        "source_trace_id": trace_id,
                        "source_step_idx": 0,
                        "layers": {"22": {"necessity": {"supported": True, "delta_logprob": -0.5}}},
                    }
                ]
            }
        )

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
            causal_lookup_mode="control_conditioned",
        )
        self.assertEqual(out["causal_variant_lookup_hits"], 0)
        self.assertEqual(out["causal_variant_lookup_misses"], 1)
        self.assertIsNone(out["steps"][0]["causal_metrics"]["necessity"])


if __name__ == "__main__":
    unittest.main()

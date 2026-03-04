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


class LatentTrackSeparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace_id = "gsm8k_test_99999"
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

    def _run_variant(self, variant: str) -> dict:
        ctrl = _build_variant(self.trace_steps, variant=variant, rng=random.Random(17))
        ctrl["trace_id"] = self.trace_id
        return _audit_one_control(
            ctrl=ctrl,
            trace_steps=self.trace_steps,
            latent_pred_idx=self.latent_pred_idx,
            causal_idx={},
            thresholds_payload=self.thresholds_payload,
            causal_layer=22,
            causal_variable="subresult_value",
            model_metadata=self.model_metadata,
            decoder_checkpoint="dummy_state_decoder.pt",
        )

    def test_latent_only_prefers_faithful_over_wrong_intermediate(self) -> None:
        faithful = self._run_variant("faithful")
        wrong = self._run_variant("wrong_intermediate")
        faithful_latent = float((faithful.get("benchmark_track_scores") or {}).get("latent_only", 0.0))
        wrong_latent = float((wrong.get("benchmark_track_scores") or {}).get("latent_only", 0.0))
        self.assertGreater(
            faithful_latent,
            wrong_latent,
            msg=f"faithful_latent={faithful_latent} wrong_latent={wrong_latent}",
        )


if __name__ == "__main__":
    unittest.main()

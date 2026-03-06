from __future__ import annotations

import unittest

import torch

from phase7.state_decoder_core import (
    MAG_BUCKETS,
    OPERATORS,
    SIGNS,
    NumericNormStats,
    StateDecoderExperimentConfig,
    decode_latent_pred_states,
)


class _FakeDecoder:
    def eval(self):
        return self

    def __call__(self, x: torch.Tensor):
        bsz = int(x.shape[0])
        return {
            "result_token_logits": torch.randn(bsz, 17),
            "operator_logits": torch.randn(bsz, len(OPERATORS)),
            "step_type_logits": torch.randn(bsz, 2),
            "magnitude_logits": torch.randn(bsz, len(MAG_BUCKETS)),
            "sign_logits": torch.randn(bsz, len(SIGNS)),
            "lhs_pred": torch.zeros(bsz),
            "rhs_pred": torch.zeros(bsz),
            "subresult_pred": torch.zeros(bsz),
        }


class DecoderConfidenceOutputTests(unittest.TestCase):
    def test_decode_emits_full_confidence_distributions(self) -> None:
        cfg = StateDecoderExperimentConfig(
            name="test_cfg",
            input_variant="raw",
            layers=(0,),
            model_hidden_dim=8,
            model_sae_dim=16,
            vocab_size=17,
        )
        numeric_stats = {
            "lhs_value": NumericNormStats(0.0, 1.0),
            "rhs_value": NumericNormStats(0.0, 1.0),
            "subresult_value": NumericNormStats(0.0, 1.0),
        }
        records = [
            {
                "trace_id": "t0",
                "step_idx": 0,
                "example_idx": 0,
                "result_token_id": 1,
                "baseline_logprob": -1.0,
                "raw_hidden": torch.randn(1, 8),
                "structured_state": {
                    "step_type": "operate",
                    "operator": "+",
                    "magnitude_bucket": MAG_BUCKETS[0],
                    "sign": SIGNS[0],
                    "lhs_value": 2.0,
                    "rhs_value": 3.0,
                    "subresult_value": 5.0,
                    "result_token_id": 1,
                },
            }
        ]
        out = decode_latent_pred_states(
            _FakeDecoder(),
            records,
            cfg,
            numeric_stats,
            device="cpu",
            batch_size=1,
        )
        self.assertEqual(len(out), 1)
        conf = out[0]["latent_pred_confidence"]
        self.assertIn("operator_probs", conf)
        self.assertIn("sign_probs", conf)
        self.assertIn("magnitude_probs", conf)
        self.assertIn("sign_top1_prob", conf)
        self.assertIn("magnitude_top1_prob", conf)
        self.assertAlmostEqual(sum(conf["operator_probs"].values()), 1.0, places=5)
        self.assertAlmostEqual(sum(conf["sign_probs"].values()), 1.0, places=5)
        self.assertAlmostEqual(sum(conf["magnitude_probs"].values()), 1.0, places=5)
        op_top = out[0]["latent_pred_state"]["operator"]
        self.assertAlmostEqual(conf["operator_probs"][op_top], conf["operator_prob"], places=6)
        self.assertEqual(set(conf["sign_probs"].keys()), set(SIGNS))
        self.assertEqual(set(conf["magnitude_probs"].keys()), set(MAG_BUCKETS))


if __name__ == "__main__":
    unittest.main()


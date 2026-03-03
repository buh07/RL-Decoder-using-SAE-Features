from __future__ import annotations

import argparse
import unittest

import torch

from phase7.build_step_trace_dataset import _build_from_phase6_records
from phase7.causal_intervention_engine import CausalPatchContext
from phase7.state_decoder_core import StateDecoderExperimentConfig
from phase7.train_state_decoders import _build_sweep_metadata


class MetadataContractTests(unittest.TestCase):
    def test_trace_builder_includes_model_metadata_fields(self) -> None:
        rec = {
            "example_idx": 11,
            "ann_idx": 0,
            "eq_tok_idx": 2,
            "result_tok_idx": 3,
            "expr_str": "2+2",
            "C": 4.0,
            "result_token_id": 19,
            "gsm8k_split": "test",
            "token_ids": [17, 10, 17, 19],
            "raw_hidden": torch.zeros(24, 1024),
            "sae_features": torch.zeros(24, 12288),
            "baseline_logprob": -5.0,
        }
        model_meta = {
            "model_key": "gpt2-medium",
            "model_family": "gpt2",
            "num_layers": 24,
            "hidden_dim": 1024,
            "tokenizer_id": "gpt2-medium",
        }
        rows = _build_from_phase6_records([rec], "test", model_meta)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key in ("model_key", "model_family", "num_layers", "hidden_dim", "tokenizer_id"):
            self.assertIn(key, row)
        self.assertEqual(row["model_key"], "gpt2-medium")
        self.assertEqual(row["num_layers"], 24)

    def test_sweep_metadata_contains_model_fields(self) -> None:
        cfg = StateDecoderExperimentConfig(
            name="state_raw_multi_l7_l12_l17_l22",
            input_variant="raw",
            layers=(7, 12, 17, 22),
            model_key="gpt2-medium",
            model_family="gpt2",
            tokenizer_id="gpt2-medium",
            model_num_layers=24,
            model_hidden_dim=1024,
        )
        args = argparse.Namespace(
            sweep_run_id="test_run",
            layer_set_id=None,
            layers=None,
            parent_baseline="spread4_current",
            manifest=None,
        )
        md = _build_sweep_metadata(args=args, cfg=cfg, manifest_row=None)
        self.assertIsNotNone(md)
        for key in ("model_key", "model_family", "num_layers_total", "hidden_dim", "tokenizer_id"):
            self.assertIn(key, md)
        self.assertEqual(md["model_key"], "gpt2-medium")

    def test_qwen_context_dry_load_reports_unsupported_reason(self) -> None:
        ctx = CausalPatchContext.load(device="cpu", model_key="qwen2.5-7b", load_model=False)
        self.assertFalse(ctx.supports_subspace_patching)
        self.assertTrue(ctx.unsupported_reason)


if __name__ == "__main__":
    unittest.main()

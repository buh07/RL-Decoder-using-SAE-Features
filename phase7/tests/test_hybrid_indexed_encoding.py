from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from phase6.pipeline_utils import build_input_tensor_from_record
from phase7.state_decoder_core import StateDecoderExperimentConfig


class HybridIndexedEncodingTests(unittest.TestCase):
    def test_pipeline_hybrid_indexed_shape_and_index_channel(self) -> None:
        raw_hidden = torch.arange(24 * 4, dtype=torch.float32).reshape(24, 4)
        sae_features = torch.tensor(
            [[0.1, -4.0, 0.5, 3.0]] * 24,
            dtype=torch.float32,
        )
        rec = {"raw_hidden": raw_hidden, "sae_features": sae_features}
        cfg = SimpleNamespace(input_variant="hybrid_indexed", layers=(0,), hybrid_topk_values=2)
        x = build_input_tensor_from_record(rec, cfg)
        # raw (4) + topk values (2) + topk normalized indices (2)
        self.assertEqual(tuple(x.shape), (1, 8))
        idx_chan = x[0, -2:]
        self.assertTrue(torch.all(idx_chan >= -1.0))
        self.assertTrue(torch.all(idx_chan <= 1.0))

    def test_state_config_input_dim_supports_hybrid_indexed(self) -> None:
        cfg = StateDecoderExperimentConfig(
            name="tmp",
            input_variant="hybrid_indexed",
            layers=(7, 12, 17, 22),
            model_hidden_dim=1024,
            hybrid_topk_values=50,
        )
        self.assertEqual(cfg.input_dim(), 1124)


if __name__ == "__main__":
    unittest.main()

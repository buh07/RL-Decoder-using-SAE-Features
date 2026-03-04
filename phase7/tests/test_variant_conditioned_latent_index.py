from __future__ import annotations

import unittest

from phase7.causal_audit import _index_variant_latent_preds


class VariantConditionedLatentIndexTests(unittest.TestCase):
    def test_index_uses_trace_variant_step_tuple(self) -> None:
        rows = [
            {"trace_id": "t0", "control_variant": "faithful", "step_idx": 0, "latent_pred_state": {"x": 1}},
            {"trace_id": "t0", "control_variant": "wrong_intermediate", "step_idx": 0, "latent_pred_state": {"x": 2}},
        ]
        idx = _index_variant_latent_preds(rows)
        self.assertIn(("t0", "faithful", 0), idx)
        self.assertIn(("t0", "wrong_intermediate", 0), idx)
        self.assertEqual(idx[("t0", "faithful", 0)]["latent_pred_state"]["x"], 1)
        self.assertEqual(idx[("t0", "wrong_intermediate", 0)]["latent_pred_state"]["x"], 2)

    def test_index_falls_back_to_variant_field(self) -> None:
        rows = [
            {"trace_id": "t1", "variant": "order_flip_only", "step_idx": 3, "latent_pred_state": {"x": 5}},
        ]
        idx = _index_variant_latent_preds(rows)
        self.assertIn(("t1", "order_flip_only", 3), idx)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from phase7.state_decoder_core import default_state_decoder_configs
from phase7.train_state_decoders import _apply_improvement_profile


class D3ProfileConfigTests(unittest.TestCase):
    def test_d3_profile_sets_operate_known_supervision_defaults(self) -> None:
        cfg = default_state_decoder_configs()["state_raw_multi_l4_l5_l6_l7"]
        d3 = _apply_improvement_profile(cfg, "d3_operate_known")
        self.assertTrue(d3.operator_operate_only_supervision)
        self.assertTrue(d3.operator_known_only_supervision)
        self.assertEqual(d3.operator_loss_mode, "weighted_ce")
        self.assertEqual(d3.operator_class_weight_scope, "operate_known_only")
        self.assertAlmostEqual(float(d3.operator_class_weight_max_ratio), 5.0)


if __name__ == "__main__":
    unittest.main()

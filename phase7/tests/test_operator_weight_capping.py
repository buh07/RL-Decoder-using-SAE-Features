from __future__ import annotations

import unittest

from phase7.state_decoder_core import OP_TO_ID, StateDecoderExperimentConfig
from phase7.train_state_decoders import _compute_operator_class_weights


class OperatorWeightCappingTests(unittest.TestCase):
    def _mk_record(self, op: str, step_type: str = "operate") -> dict:
        return {"structured_state": {"operator": op, "step_type": step_type}}

    def test_operate_known_scope_zeros_unknown_weight_and_caps_ratio(self) -> None:
        cfg = StateDecoderExperimentConfig(
            name="wcap",
            input_variant="raw",
            layers=(0,),
            operator_loss_mode="weighted_ce",
            operator_class_weight_scope="operate_known_only",
            operator_class_weight_max_ratio=5.0,
        )
        records = []
        records += [self._mk_record("+") for _ in range(1000)]
        records += [self._mk_record("-") for _ in range(10)]
        records += [self._mk_record("*") for _ in range(10)]
        records += [self._mk_record("/") for _ in range(10)]
        records += [self._mk_record("unknown") for _ in range(100)]
        w = _compute_operator_class_weights(records, cfg, "cpu")
        self.assertIsNotNone(w)
        vec = w.detach().cpu().tolist()
        self.assertEqual(vec[OP_TO_ID["unknown"]], 0.0)
        pos = [v for i, v in enumerate(vec) if i != OP_TO_ID["unknown"] and v > 0]
        self.assertTrue(pos)
        ratio = max(pos) / min(pos)
        self.assertLessEqual(ratio, 5.000001)


if __name__ == "__main__":
    unittest.main()

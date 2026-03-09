from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

from phase7.diagnose_decoder_quality import _diagnose_eval_metrics, _load_eval_rows


class DecoderQualityOperateOnlyGateTests(unittest.TestCase):
    def test_operate_only_gate_uses_operate_metric_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state_decoder_eval_dummy.json"
            payload = {
                "config_name": "dummy_cfg",
                "experiment_config": {"layers": [4, 5, 6, 7], "input_variant": "raw"},
                "evaluations": {
                    "test": {
                        "operator_acc": 0.52,
                        "operator_acc_operate_only": 0.74,
                        "operator_num_operate_rows": 100,
                        "step_type_acc": 0.55,
                        "magnitude_acc": 0.93,
                        "sign_acc": 0.99,
                        "result_token_top1": 0.58,
                    }
                },
            }
            p.write_text(json.dumps(payload))
            old_cwd = os.getcwd()
            try:
                os.chdir(td)
                rows = _load_eval_rows("state_decoder_eval_*.json")
            finally:
                os.chdir(old_cwd)
            args = argparse.Namespace(
                quality_gate_operator_min=0.75,
                quality_gate_operator_operate_only_min=0.70,
                quality_gate_operator_operate_known_only_min=0.70,
                quality_gate_magnitude_min=0.90,
                quality_gate_sign_min=0.95,
            )
            out = _diagnose_eval_metrics(rows, args)
            self.assertEqual(out["num_test_rows"], 1)
            self.assertTrue(out["decoder_quality_gate_operate_only"]["pass"])
            self.assertTrue(out["decoder_quality_gate_operate_known_only"]["pass"])
            self.assertFalse(out["decoder_quality_gate_legacy_all_steps"]["pass"])
            best = out["best_test_row_by_operator_acc"]
            self.assertAlmostEqual(float(best["operator_acc_primary"]), 0.74, places=6)


if __name__ == "__main__":
    unittest.main()

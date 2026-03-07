from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class MixedTrajectoryValidationTests(unittest.TestCase):
    def test_feature_builder_blocks_when_no_pairable_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            control_path = tdp / "controls.json"
            out_path = tdp / "features.json"
            payload = {
                "schema_version": "phase7_control_conditioned_records_v1",
                "status": "ok",
                "model_metadata": {"model_key": "qwen2.5-7b"},
                "rows": [],
                "rows_inline": True,
                "rows_format": "json",
                "rows_count": 0,
            }
            control_path.write_text(json.dumps(payload))

            cmd = [
                ".venv/bin/python3",
                "phase7/mixed_trajectory_feature_builder.py",
                "--control-records",
                str(control_path),
                "--output",
                str(out_path),
                "--run-tag",
                "test_blocked",
            ]
            subprocess.run(cmd, check=True)
            out = json.loads(out_path.read_text())
            self.assertEqual(out["status"], "blocked_no_pairable_trajectories")
            self.assertEqual(out["model_key"], "qwen2.5-7b")

    def test_evaluator_emits_required_decision_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            features_path = tdp / "mixed_features.json"
            out_json = tdp / "eval.json"
            out_md = tdp / "eval.md"

            b1 = [f"B1_sae_f{i}" for i in range(9)]
            b2 = [f"B2_raw_f{i}" for i in range(9)]
            b3 = [f"B3_proj_f{i}" for i in range(9)]

            variants = [
                "wrong_intermediate",
                "silent_error_correction",
                "order_flip_only",
                "answer_first_order_flip",
                "reordered_steps",
            ]

            samples = []
            for i in range(140):
                trace_id = f"trace_{i:04d}"
                variant = variants[i % len(variants)]
                base = (i % 7) * 0.01
                for label in ("faithful", "unfaithful"):
                    is_unfaithful = (label == "unfaithful")
                    # SAE block has moderate signal.
                    b1_vals = [0.55 + base - (0.12 if is_unfaithful and variant == "wrong_intermediate" else 0.0)] * 9
                    # Raw/proj add incremental wrong_intermediate signal with lower correlation.
                    b2_vals = [0.35 + 0.02 * (j % 3) - (0.18 if is_unfaithful and variant == "wrong_intermediate" else 0.0) for j in range(9)]
                    b3_vals = [0.45 + 0.015 * (j % 4) - (0.15 if is_unfaithful and variant == "wrong_intermediate" else 0.0) for j in range(9)]

                    row = {
                        "trace_id": trace_id,
                        "variant": variant,
                        "label": label,
                        "step_count": 4,
                    }
                    for k, v in zip(b1, b1_vals):
                        row[k] = float(v)
                    for k, v in zip(b2, b2_vals):
                        row[k] = float(v)
                    for k, v in zip(b3, b3_vals):
                        row[k] = float(v)
                    samples.append(row)

            payload = {
                "schema_version": "phase7_mixed_trajectory_feature_blocks_v1",
                "status": "ok",
                "run_tag": "mixed_test",
                "block_feature_names": {
                    "B1_sae": b1,
                    "B2_raw": b2,
                    "B3_proj": b3,
                },
                "samples": samples,
            }
            features_path.write_text(json.dumps(payload))

            cmd = [
                ".venv/bin/python3",
                "phase7/evaluate_mixed_trajectory_validation.py",
                "--features",
                str(features_path),
                "--output-json",
                str(out_json),
                "--output-md",
                str(out_md),
                "--run-tag",
                "eval_test",
                "--train-exclude-variants",
                "order_flip_only,answer_first_order_flip,reordered_steps",
                "--trace-test-fraction",
                "0.2",
                "--trace-split-seed",
                "20260307",
                "--cv-folds",
                "5",
                "--cv-seed",
                "20260307",
                "--bootstrap-n",
                "120",
                "--require-wrong-intermediate-auroc",
                "0.70",
                "--train-seed",
                "20260307",
            ]
            subprocess.run(cmd, check=True)

            out = json.loads(out_json.read_text())
            self.assertEqual(out["status"], "ok")
            self.assertIn("delta_auroc_mixed_vs_sae", out)
            self.assertIn("delta_ci95", out)
            self.assertIn("practical_effect_pass", out)
            self.assertIn("collinearity_diagnostics", out)
            self.assertIn("falsification_checks", out)
            self.assertIn("final_decision", out)

            self.assertEqual(out["single_split"]["split_diagnostics"]["trace_overlap_count"], 0)
            self.assertEqual(out["outer_cv"]["cv_trace_overlap_count"], 0)
            dropped = out["single_split"]["dropped_train_positives_by_variant"]
            self.assertTrue(any(k in dropped for k in ["order_flip_only", "answer_first_order_flip", "reordered_steps"]))

            self.assertIn(out["final_decision"], {"publishable_mixed_signal", "mixed_redundant_or_insufficient"})


if __name__ == "__main__":
    unittest.main()

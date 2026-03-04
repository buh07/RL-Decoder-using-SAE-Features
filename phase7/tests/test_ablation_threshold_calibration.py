from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class AblationThresholdCalibrationTests(unittest.TestCase):
    def test_ablation_threshold_is_calibrated_from_calib_split(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            calib_path = Path(td) / "audit_calib.json"
            eval_path = Path(td) / "audit_eval.json"
            thr_path = Path(td) / "thresholds.json"
            out_path = Path(td) / "benchmark.json"

            calib_payload = {
                "audits": [
                    {
                        "trace_id": "c0",
                        "gold_label": "faithful",
                        "benchmark_track_scores": {"text_only": 0.9, "latent_only": 0.9, "causal_auditor": 0.9},
                        "overall_score": 0.9,
                        "control_variant": "faithful",
                        "paper_failure_family": "legacy_or_unspecified",
                    },
                    {
                        "trace_id": "c1",
                        "gold_label": "unfaithful",
                        "benchmark_track_scores": {"text_only": 0.2, "latent_only": 0.2, "causal_auditor": 0.2},
                        "overall_score": 0.2,
                        "control_variant": "wrong_intermediate",
                        "paper_failure_family": "legacy_or_unspecified",
                    },
                ]
            }
            eval_payload = {
                "audits": [
                    {
                        "trace_id": "e0",
                        "gold_label": "faithful",
                        "benchmark_track_scores": {"text_only": 0.8, "latent_only": 0.8, "causal_auditor": 0.8},
                        "overall_score": 0.8,
                        "control_variant": "faithful",
                        "paper_failure_family": "legacy_or_unspecified",
                        "verdict": "causally_faithful",
                    },
                    {
                        "trace_id": "e1",
                        "gold_label": "unfaithful",
                        "benchmark_track_scores": {"text_only": 0.3, "latent_only": 0.3, "causal_auditor": 0.3},
                        "overall_score": 0.3,
                        "control_variant": "wrong_intermediate",
                        "paper_failure_family": "legacy_or_unspecified",
                        "verdict": "unsupported",
                    },
                ]
            }
            thresholds_payload = {
                "thresholds": {"overall_score_faithful_min": 0.5},
                "target_fpr": 0.05,
                "positive_label": "unfaithful",
                "calibration_source_ref": {
                    "path": str(calib_path),
                    "trace_hash": "dummy",
                },
            }

            calib_path.write_text(json.dumps(calib_payload))
            eval_path.write_text(json.dumps(eval_payload))
            thr_path.write_text(json.dumps(thresholds_payload))

            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit-eval",
                    str(eval_path),
                    "--allow-same-audit",
                    "--thresholds",
                    str(thr_path),
                    "--ablation-weights",
                    '{"text":0.35,"latent":0.35,"causal":0.30}',
                    "--output",
                    str(out_path),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            out = json.loads(out_path.read_text())
            ab = out.get("ablation_weighted_blend") or {}
            self.assertEqual(ab.get("ablation_threshold_source"), "calibrated_from_calibration_split")
            self.assertEqual(ab.get("threshold_source"), "calibrated_from_calibration_split")


if __name__ == "__main__":
    unittest.main()

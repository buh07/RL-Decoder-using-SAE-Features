from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkAblationReportingTests(unittest.TestCase):
    def test_emits_ablation_and_reference_blend(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            eval_path = tdp / "eval.json"
            thr_path = tdp / "thr.json"
            out_path = tdp / "out.json"

            audits = []
            for i in range(8):
                faithful = {
                    "trace_id": f"trace_{i}",
                    "gold_label": "faithful",
                    "control_variant": "faithful",
                    "paper_failure_family": "legacy_or_unspecified",
                    "benchmark_track_scores": {
                        "text_only": 0.8,
                        "latent_only": 0.7,
                        "causal_auditor": 0.6,
                        "composite": 0.73,
                    },
                    "benchmark_track_definedness": {"causal_auditor": True},
                    "parse_summary": {"parseable": True},
                    "verdict": "causally_faithful",
                }
                unfaithful = {
                    "trace_id": f"trace_{i}",
                    "gold_label": "unfaithful",
                    "control_variant": "wrong_intermediate",
                    "paper_failure_family": "legacy_or_unspecified",
                    "benchmark_track_scores": {
                        "text_only": 0.2,
                        "latent_only": 0.3,
                        "causal_auditor": 0.4,
                        "composite": 0.27,
                    },
                    "benchmark_track_definedness": {"causal_auditor": True},
                    "parse_summary": {"parseable": True},
                    "verdict": "unsupported",
                }
                audits.extend([faithful, unfaithful])

            eval_payload = {
                "schema_version": "causal_audit_v1",
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
                "audits": audits,
            }
            thr_payload = {
                "thresholds": {"overall_score_faithful_min": 0.5},
                "track_thresholds": {
                    "text_only": {"threshold": 0.5},
                    "latent_only": {"threshold": 0.5},
                    "causal_auditor": {"threshold": 0.5},
                    "composite": {"threshold": 0.5},
                },
                "calibration_source_ref": {"path": str(tdp / "calib.json"), "trace_count": 8, "trace_hash": "abc"},
            }
            eval_path.write_text(json.dumps(eval_payload))
            thr_path.write_text(json.dumps(thr_payload))

            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit",
                    str(eval_path),
                    "--thresholds",
                    str(thr_path),
                    "--benchmark-scope",
                    "synthetic_controls",
                    "--external-validity-status",
                    "not_tested",
                    "--ablation-weights",
                    '{"text":0.5,"latent":0.5,"causal":0.0}',
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
            self.assertIn("ablation_weighted_blend", out)
            self.assertIn("ablation_weighted_blend_reference", out)
            w = out["ablation_weighted_blend"]["weights"]
            self.assertAlmostEqual(float(w["text"]), 0.5, places=6)
            self.assertAlmostEqual(float(w["latent"]), 0.5, places=6)
            self.assertAlmostEqual(float(w["causal"]), 0.0, places=6)
            wr = out["ablation_weighted_blend_reference"]["weights"]
            self.assertAlmostEqual(float(wr["text"]), 0.35, places=6)
            self.assertAlmostEqual(float(wr["latent"]), 0.35, places=6)
            self.assertAlmostEqual(float(wr["causal"]), 0.30, places=6)


if __name__ == "__main__":
    unittest.main()

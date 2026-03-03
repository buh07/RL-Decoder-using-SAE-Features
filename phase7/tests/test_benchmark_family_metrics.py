from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkFamilyMetricsTests(unittest.TestCase):
    def test_single_class_family_emits_null_auroc_with_definedness(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.json"
            out_path = Path(td) / "benchmark.json"
            payload = {
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
                "audits": [
                    {
                        "trace_id": "t0",
                        "gold_label": "faithful",
                        "overall_score": 0.9,
                        "control_variant": "faithful",
                        "paper_failure_family": "legacy_or_unspecified",
                        "benchmark_track_scores": {"text_only": 0.8, "latent_only": 0.7},
                        "verdict": "causally_faithful",
                    },
                    {
                        "trace_id": "t1",
                        "gold_label": "unfaithful",
                        "overall_score": 0.1,
                        "control_variant": "wrong_intermediate",
                        "paper_failure_family": "legacy_or_unspecified",
                        "benchmark_track_scores": {"text_only": 0.2, "latent_only": 0.3},
                        "verdict": "contradicted",
                    },
                    {
                        "trace_id": "t2",
                        "gold_label": "unfaithful",
                        "overall_score": 0.2,
                        "control_variant": "paper_single",
                        "paper_failure_family": "prompt_bias_rationalization",
                        "benchmark_track_scores": {"text_only": 0.4, "latent_only": 0.4},
                        "verdict": "unsupported",
                    },
                ],
            }
            audit_path.write_text(json.dumps(payload))

            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit",
                    str(audit_path),
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
            fam = out["by_paper_failure_family"]["prompt_bias_rationalization"]
            self.assertEqual(fam["class_counts"]["faithful"], 0)
            self.assertEqual(fam["class_counts"]["unfaithful"], 1)
            self.assertFalse(bool(fam["metric_defined"]["auroc"]))
            self.assertIsNone(fam["auroc"])


if __name__ == "__main__":
    unittest.main()

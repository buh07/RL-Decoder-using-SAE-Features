from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phase7.benchmark_faithfulness import _roc


REPO_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkFamilyMetricsTests(unittest.TestCase):
    def test_roc_tie_sort_is_deterministic(self) -> None:
        rows, _ = _roc([(0.9, 1), (0.8, 0), (0.7, 1), (0.6, 0)])
        # For tied FPR rows, TPR should be sorted ascending for stable ROC traversal.
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            cur = rows[i]
            if abs(prev[0] - cur[0]) < 1e-12:
                self.assertLessEqual(prev[1], cur[1])

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
            self.assertIn("coverage_by_variable", out)
            self.assertIn("coverage_by_layer", out)
            self.assertIn("variant_min_auroc", out)
            self.assertIsNotNone(out["variant_min_auroc"])
            self.assertIn("variant", out["variant_min_auroc"])

    def test_track_thresholds_are_used_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.json"
            thr_path = Path(td) / "thr.json"
            out_path = Path(td) / "benchmark.json"
            payload = {
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
                "audits": [
                    {
                        "trace_id": "t0",
                        "gold_label": "faithful",
                        "overall_score": 0.8,
                        "control_variant": "faithful",
                        "paper_failure_family": "legacy_or_unspecified",
                        "benchmark_track_scores": {"text_only": 0.9, "latent_only": 0.7},
                        "verdict": "causally_faithful",
                    },
                    {
                        "trace_id": "t1",
                        "gold_label": "unfaithful",
                        "overall_score": 0.4,
                        "control_variant": "wrong_intermediate",
                        "paper_failure_family": "legacy_or_unspecified",
                        "benchmark_track_scores": {"text_only": 0.2, "latent_only": 0.3},
                        "verdict": "unsupported",
                    },
                ],
            }
            thresholds = {
                "thresholds": {"overall_score_faithful_min": 0.5},
                "track_thresholds": {
                    "text_only": {"threshold": 0.95},
                    "latent_only": {"threshold": 0.65},
                    "causal_auditor": {"threshold": 0.6},
                    "composite": {"threshold": 0.55},
                },
            }
            audit_path.write_text(json.dumps(payload))
            thr_path.write_text(json.dumps(thresholds))
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit",
                    str(audit_path),
                    "--thresholds",
                    str(thr_path),
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
            self.assertEqual(float(out["thresholds_by_track"]["text_only"]), 0.95)
            self.assertEqual(float(out["thresholds_by_track"]["latent_only"]), 0.65)
            self.assertEqual(float(out["thresholds_by_track"]["causal_auditor"]), 0.6)
            self.assertEqual(float(out["thresholds_by_track"]["composite"]), 0.55)
            self.assertEqual(float(out["by_benchmark_track"]["text_only"]["threshold"]), 0.95)
            self.assertEqual(out["by_benchmark_track"]["text_only"]["threshold_source"], "track_thresholds")
            self.assertIn("composite", out["by_benchmark_track"])

    def test_readout_high_threshold_undefined_when_no_finite_labeled_scores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.json"
            out_path = Path(td) / "benchmark.json"
            payload = {
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
                "audits": [
                    {
                        "trace_id": "u0",
                        "gold_label": "unknown",
                        "overall_score": 0.5,
                        "control_variant": "real_cot_pilot",
                        "paper_failure_family": "real_cot_pilot",
                        "benchmark_track_scores": {"text_only": 0.5, "latent_only": float("nan"), "causal_auditor": 0.5},
                        "verdict": "unverifiable_text",
                    }
                ],
            }
            audit_path.write_text(json.dumps(payload))
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit",
                    str(audit_path),
                    "--benchmark-scope",
                    "real_cot",
                    "--external-validity-status",
                    "pilot",
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
            rd = out["readout_high_definition"]
            self.assertFalse(bool(rd["threshold_defined"]))
            self.assertIsNone(rd["latent_high_threshold"])
            self.assertEqual(rd["undefined_reason"], "no_finite_latent_scores_for_labeled_rows")


if __name__ == "__main__":
    unittest.main()

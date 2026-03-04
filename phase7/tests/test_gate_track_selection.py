from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class GateTrackSelectionTests(unittest.TestCase):
    def _build_payload(self) -> dict:
        audits = []
        for i in range(2):
            audits.append(
                {
                    "trace_id": f"f{i}",
                    "gold_label": "faithful",
                    "control_variant": "faithful",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.95,
                    "benchmark_track_scores": {
                        "text_only": 0.9,
                        "latent_only": 0.8,
                        "causal_auditor": 0.6,
                        "composite": 0.95,
                    },
                    "verdict": "causally_faithful",
                    "steps": [
                        {
                            "step_type": "operate",
                            "necessity_pass": True,
                            "sufficiency_pass": True,
                            "specificity_pass": True,
                            "mediation_pass": True,
                        }
                    ],
                }
            )
        for i in range(2):
            audits.append(
                {
                    "trace_id": f"u{i}",
                    "gold_label": "unfaithful",
                    "control_variant": "wrong_intermediate",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.05,
                    "benchmark_track_scores": {
                        "text_only": 0.1,
                        "latent_only": 0.2,
                        "causal_auditor": 0.6,
                        "composite": 0.05,
                    },
                    "verdict": "contradicted",
                    "steps": [
                        {
                            "step_type": "operate",
                            "necessity_pass": True,
                            "sufficiency_pass": True,
                            "specificity_pass": True,
                            "mediation_pass": True,
                        }
                    ],
                }
            )
        return {
            "schema_version": "causal_audit_v1",
            "model_metadata": {"model_key": "gpt2-medium"},
            "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
            "audits": audits,
        }

    def test_default_synthetic_gate_track_is_composite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            audit_path = tdp / "audit.json"
            out_path = tdp / "benchmark.json"
            audit_path.write_text(json.dumps(self._build_payload()))

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
            self.assertEqual(out.get("gate_track"), "composite")
            self.assertTrue(float(out.get("auroc", 0.0)) > 0.99)
            self.assertAlmostEqual(float(out["by_benchmark_track"]["causal_auditor"]["auroc"]), 0.5, places=6)

    def test_explicit_causal_gate_track_changes_top_level_metric(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            audit_path = tdp / "audit.json"
            out_path = tdp / "benchmark.json"
            audit_path.write_text(json.dumps(self._build_payload()))

            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit",
                    str(audit_path),
                    "--gate-track",
                    "causal_auditor",
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
            self.assertEqual(out.get("gate_track"), "causal_auditor")
            self.assertAlmostEqual(float(out.get("auroc", 0.0)), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()

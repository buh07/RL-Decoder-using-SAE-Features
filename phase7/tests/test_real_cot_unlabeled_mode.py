from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class RealCotUnlabeledModeTests(unittest.TestCase):
    def test_unlabeled_real_cot_emits_pilot_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            audit_path = tdp / "audit.json"
            out_path = tdp / "benchmark.json"
            gate_path = tdp / "gate.json"

            payload = {
                "schema_version": "causal_audit_v1",
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
                "audits": [
                    {
                        "trace_id": "t0",
                        "gold_label": "unknown",
                        "verdict": "unverifiable_text",
                        "parse_summary": {
                            "parseable": False,
                            "parse_errors": [{"error": "unrecognized_step_template"}],
                        },
                        "paper_aligned_metrics": {
                            "completeness_proxy": {"mediation_coverage_fraction": 0.0},
                        },
                    },
                    {
                        "trace_id": "t1",
                        "gold_label": "unknown",
                        "verdict": "contradicted",
                        "parse_summary": {
                            "parseable": True,
                            "parse_errors": [],
                        },
                        "paper_aligned_metrics": {
                            "completeness_proxy": {"mediation_coverage_fraction": 0.5},
                        },
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
                    "--benchmark-scope",
                    "real_cot",
                    "--external-validity-status",
                    "pilot",
                    "--external-validity-gate-output",
                    str(gate_path),
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
            self.assertEqual(out.get("evaluation_mode"), "unlabeled_pilot")
            self.assertIsNone(out.get("auroc"))
            self.assertIsNone(out.get("false_positive_rate"))
            metric_defined = out.get("metric_defined", {})
            self.assertFalse(bool(metric_defined.get("auroc", True)))
            self.assertIn("pilot_diagnostics", out)
            pilot = out["pilot_diagnostics"]
            self.assertIn("parseable_fraction", pilot)
            self.assertIn("parse_error_distribution", pilot)
            self.assertIn("contradicted_fraction", pilot)
            self.assertIn("unverifiable_fraction", pilot)
            self.assertIn("causally_supported_fraction_parseable", pilot)
            self.assertIn("mediation_coverage_fraction", pilot)
            self.assertTrue(gate_path.exists())


if __name__ == "__main__":
    unittest.main()

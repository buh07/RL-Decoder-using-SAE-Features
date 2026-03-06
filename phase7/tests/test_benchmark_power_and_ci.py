from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkPowerAndCITests(unittest.TestCase):
    def _write_audit(self, path: Path) -> None:
        audits = []
        for i in range(10):
            audits.append(
                {
                    "trace_id": f"f{i}",
                    "gold_label": "faithful",
                    "control_variant": "faithful",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.9,
                    "benchmark_track_scores": {
                        "text_only": 0.9,
                        "latent_only": 0.7,
                        "confidence_margin": 0.9,
                        "causal_auditor": 0.5,
                        "composite": 0.85,
                    },
                    "verdict": "causally_faithful",
                    "steps": [],
                }
            )
        for i in range(10):
            audits.append(
                {
                    "trace_id": f"u{i}",
                    "gold_label": "unfaithful",
                    "control_variant": "wrong_sign",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.1,
                    "benchmark_track_scores": {
                        "text_only": 0.1,
                        "latent_only": 0.3,
                        "confidence_margin": 0.1,
                        "causal_auditor": 0.5,
                        "composite": 0.15,
                    },
                    "verdict": "unsupported",
                    "steps": [],
                }
            )
        payload = {
            "model_metadata": {"model_key": "gpt2-medium"},
            "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
            "audits": audits,
        }
        path.write_text(json.dumps(payload))

    def test_power_guard_blocks_gate_when_counts_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.json"
            out_path = Path(td) / "out.json"
            self._write_audit(audit_path)
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit",
                    str(audit_path),
                    "--gate-track",
                    "confidence_margin",
                    "--min-positive-n",
                    "50",
                    "--min-negative-n",
                    "50",
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
            self.assertFalse(bool(out["power_sufficient"]))
            self.assertFalse(bool(out["gate_checks"]["power_sufficient"]))
            self.assertFalse(bool(out["gate_checks"]["gate_pass"]))

    def test_ci_and_confidence_redundancy_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.json"
            out_path = Path(td) / "out.json"
            self._write_audit(audit_path)
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit",
                    str(audit_path),
                    "--gate-track",
                    "confidence_margin",
                    "--min-positive-n",
                    "5",
                    "--min-negative-n",
                    "5",
                    "--bootstrap-ci-n",
                    "200",
                    "--confidence-text-corr-max",
                    "0.80",
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
            self.assertIn("auroc_ci95_lower", out)
            self.assertIn("auroc_ci95_upper", out)
            self.assertEqual(int(out["bootstrap_n"]), 200)
            self.assertIn("confidence_margin_vs_text_only", out["track_correlations"])
            self.assertFalse(bool(out["gate_checks"]["confidence_non_redundant"]))
            self.assertFalse(bool(out["gate_checks"]["gate_pass"]))

    def test_ablation_weights_accept_confidence_component(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.json"
            out_path = Path(td) / "out.json"
            self._write_audit(audit_path)
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit",
                    str(audit_path),
                    "--gate-track",
                    "composite",
                    "--min-positive-n",
                    "5",
                    "--min-negative-n",
                    "5",
                    "--ablation-weights",
                    '{"text":0.40,"latent":0.30,"confidence":0.30,"causal":0.0}',
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
            abl = out.get("ablation_weighted_blend") or {}
            w = abl.get("weights") or {}
            self.assertIn("confidence", w)
            total = float(w.get("text", 0.0)) + float(w.get("latent", 0.0)) + float(w.get("confidence", 0.0)) + float(w.get("causal", 0.0))
            self.assertAlmostEqual(total, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

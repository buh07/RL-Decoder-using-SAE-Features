from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class CausalDegenerateAurocTriggerTests(unittest.TestCase):
    def _write_inputs(self, td: Path) -> tuple[Path, Path]:
        eval_path = td / "eval.json"
        thr_path = td / "thr.json"
        audits = []
        # Construct anti-predictive causal track with non-degenerate identical/std/definedness metrics.
        for i in range(16):
            tid = f"trace_{i:03d}"
            audits.append(
                {
                    "trace_id": tid,
                    "gold_label": "faithful",
                    "control_variant": "faithful",
                    "paper_failure_family": "legacy_or_unspecified",
                    "benchmark_track_scores": {
                        "text_only": 0.8,
                        "latent_only": 0.7,
                        "causal_auditor": 0.40,
                        "composite": 0.7,
                    },
                    "benchmark_track_definedness": {"causal_auditor": True},
                    "parse_summary": {"parseable": True},
                    "verdict": "causally_faithful",
                }
            )
            audits.append(
                {
                    "trace_id": tid,
                    "gold_label": "unfaithful",
                    "control_variant": "wrong_intermediate",
                    "paper_failure_family": "legacy_or_unspecified",
                    "benchmark_track_scores": {
                        "text_only": 0.2,
                        "latent_only": 0.3,
                        "causal_auditor": 0.60,
                        "composite": 0.3,
                    },
                    "benchmark_track_definedness": {"causal_auditor": True},
                    "parse_summary": {"parseable": True},
                    "verdict": "unsupported",
                }
            )

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
            "calibration_source_ref": {"path": str(td / "calib.json"), "trace_count": 16, "trace_hash": "abc"},
        }
        eval_path.write_text(json.dumps(eval_payload))
        thr_path.write_text(json.dumps(thr_payload))
        return eval_path, thr_path

    def test_auroc_trigger_flags_degeneracy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            eval_path, thr_path = self._write_inputs(tdp)
            out_path = tdp / "out.json"
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
                    "--causal-degenerate-identical-threshold",
                    "0.95",
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
            self.assertTrue(bool(out.get("causal_track_degenerate_flag")))
            th = out.get("causal_track_degeneracy_thresholds", {})
            self.assertAlmostEqual(float(th.get("auroc_min", 0.0)), 0.55, places=9)
            self.assertTrue(bool(th.get("auroc_trigger_enabled")))

    def test_auroc_trigger_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            eval_path, thr_path = self._write_inputs(tdp)
            out_path = tdp / "out_no_auroc.json"
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
                    "--causal-degenerate-identical-threshold",
                    "0.95",
                    "--no-causal-degenerate-enable-auroc-trigger",
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
            self.assertFalse(bool(out.get("causal_track_degenerate_flag")))


if __name__ == "__main__":
    unittest.main()

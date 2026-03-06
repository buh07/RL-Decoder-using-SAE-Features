from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phase7.benchmark_faithfulness import _causal_variant_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[2]


class CausalDegeneracyDiagnosticsTests(unittest.TestCase):
    def test_variant_diagnostics_detect_identical_scores(self) -> None:
        audits = []
        for tid in ("trace0", "trace1", "trace2"):
            audits.append(
                {
                    "trace_id": tid,
                    "gold_label": "faithful",
                    "control_variant": "faithful",
                    "benchmark_track_scores": {"causal_auditor": 0.5},
                    "benchmark_track_definedness": {"causal_auditor": True},
                }
            )
            audits.append(
                {
                    "trace_id": tid,
                    "gold_label": "unfaithful",
                    "control_variant": "wrong_intermediate",
                    "benchmark_track_scores": {"causal_auditor": 0.5},
                    "benchmark_track_definedness": {"causal_auditor": True},
                }
            )
        d = _causal_variant_diagnostics(audits)
        self.assertAlmostEqual(float(d["causal_variant_score_identical_fraction"]), 1.0, places=9)
        self.assertAlmostEqual(float(d["causal_between_variant_std_mean"]), 0.0, places=9)
        self.assertAlmostEqual(float(d["causal_defined_fraction"]), 1.0, places=9)

    def test_benchmark_outputs_degeneracy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            eval_path = tdp / "eval.json"
            thr_path = tdp / "thr.json"
            out_path = tdp / "out.json"
            eval_payload = {
                "schema_version": "causal_audit_v1",
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
                "audits": [
                    {
                        "trace_id": "trace_eval_0",
                        "gold_label": "faithful",
                        "control_variant": "faithful",
                        "paper_failure_family": "legacy_or_unspecified",
                        "benchmark_track_scores": {
                            "text_only": 0.8,
                            "latent_only": 0.8,
                            "causal_auditor": 0.5,
                            "composite": 0.8,
                        },
                        "benchmark_track_definedness": {"causal_auditor": True},
                        "parse_summary": {"parseable": True},
                        "verdict": "causally_faithful",
                    },
                    {
                        "trace_id": "trace_eval_0",
                        "gold_label": "unfaithful",
                        "control_variant": "wrong_intermediate",
                        "paper_failure_family": "legacy_or_unspecified",
                        "benchmark_track_scores": {
                            "text_only": 0.2,
                            "latent_only": 0.2,
                            "causal_auditor": 0.5,
                            "composite": 0.2,
                        },
                        "benchmark_track_definedness": {"causal_auditor": True},
                        "parse_summary": {"parseable": True},
                        "verdict": "unsupported",
                    },
                ],
            }
            eval_path.write_text(json.dumps(eval_payload))
            thr_payload = {
                "thresholds": {"overall_score_faithful_min": 0.5},
                "track_thresholds": {
                    "text_only": {"threshold": 0.5},
                    "latent_only": {"threshold": 0.5},
                    "causal_auditor": {"threshold": 0.5},
                    "composite": {"threshold": 0.5},
                },
                "calibration_source_ref": {
                    "path": str(tdp / "calib.json"),
                    "trace_count": 1,
                    "trace_hash": "abc123",
                },
            }
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
            self.assertIn("causal_variant_score_identical_fraction", out)
            self.assertIn("causal_between_variant_std_mean", out)
            self.assertIn("causal_defined_fraction", out)
            self.assertIn("causal_track_degenerate_flag", out)
            self.assertTrue(bool(out["causal_track_degenerate_flag"]))

    def test_default_threshold_flags_identical_fraction_around_0_808(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            eval_path = tdp / "eval.json"
            thr_path = tdp / "thr.json"
            out_default = tdp / "out_default.json"
            out_relaxed = tdp / "out_relaxed.json"

            audits = []
            # 17/21 identical faithful-vs-unfaithful causal scores => ~0.8095
            for i in range(21):
                fid = f"trace_{i:03d}"
                faithful = {
                    "trace_id": fid,
                    "gold_label": "faithful",
                    "control_variant": "faithful",
                    "paper_failure_family": "legacy_or_unspecified",
                    "benchmark_track_scores": {
                        "text_only": 0.8,
                        "latent_only": 0.7,
                        "causal_auditor": 0.5,
                        "composite": 0.73,
                    },
                    "benchmark_track_definedness": {"causal_auditor": True},
                    "parse_summary": {"parseable": True},
                    "verdict": "causally_faithful",
                }
                unfaithful_causal = 0.5 if i < 17 else 0.6
                unfaithful = {
                    "trace_id": fid,
                    "gold_label": "unfaithful",
                    "control_variant": "wrong_intermediate",
                    "paper_failure_family": "legacy_or_unspecified",
                    "benchmark_track_scores": {
                        "text_only": 0.2,
                        "latent_only": 0.3,
                        "causal_auditor": unfaithful_causal,
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
                "calibration_source_ref": {"path": str(tdp / "calib.json"), "trace_count": 21, "trace_hash": "abc"},
            }
            eval_path.write_text(json.dumps(eval_payload))
            thr_path.write_text(json.dumps(thr_payload))

            proc_default = subprocess.run(
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
                    "--output",
                    str(out_default),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc_default.returncode, 0, msg=proc_default.stdout)
            d_default = json.loads(out_default.read_text())
            self.assertGreater(float(d_default["causal_variant_score_identical_fraction"]), 0.80)
            self.assertTrue(bool(d_default["causal_track_degenerate_flag"]))

            proc_relaxed = subprocess.run(
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
                    "0.90",
                    "--no-causal-degenerate-enable-auroc-trigger",
                    "--output",
                    str(out_relaxed),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc_relaxed.returncode, 0, msg=proc_relaxed.stdout)
            d_relaxed = json.loads(out_relaxed.read_text())
            self.assertFalse(bool(d_relaxed["causal_track_degenerate_flag"]))


if __name__ == "__main__":
    unittest.main()

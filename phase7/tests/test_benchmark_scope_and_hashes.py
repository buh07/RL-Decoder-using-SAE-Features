from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _trace_hash(trace_ids: list[str]) -> str:
    h = hashlib.sha256()
    for tid in sorted(trace_ids):
        h.update(tid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class BenchmarkScopeAndHashesTests(unittest.TestCase):
    def test_scope_and_hash_metadata_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            eval_path = tdp / "eval.json"
            thr_path = tdp / "thresholds.json"
            out_path = tdp / "benchmark.json"
            gate_path = tdp / "external_gate.json"

            eval_audits = [
                {
                    "trace_id": "trace_eval_0",
                    "gold_label": "faithful",
                    "control_variant": "faithful",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.8,
                    "benchmark_track_scores": {"text_only": 0.8, "latent_only": 0.7, "causal_auditor": 0.75},
                    "verdict": "causally_faithful",
                },
                {
                    "trace_id": "trace_eval_1",
                    "gold_label": "unfaithful",
                    "control_variant": "wrong_intermediate",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.2,
                    "benchmark_track_scores": {"text_only": 0.2, "latent_only": 0.3, "causal_auditor": 0.25},
                    "verdict": "unsupported",
                },
            ]
            eval_payload = {
                "schema_version": "causal_audit_v1",
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
                "audits": eval_audits,
            }
            eval_path.write_text(json.dumps(eval_payload))

            thresholds = {
                "thresholds": {"overall_score_faithful_min": 0.5},
                "track_thresholds": {
                    "text_only": {"threshold": 0.5},
                    "latent_only": {"threshold": 0.5},
                    "causal_auditor": {"threshold": 0.5},
                },
                "calibration_source_ref": {
                    "path": str(tdp / "calib.json"),
                    "trace_count": 2,
                    "trace_hash": _trace_hash(["trace_cal_0", "trace_cal_1"]),
                },
            }
            thr_path.write_text(json.dumps(thresholds))

            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit-eval",
                    str(eval_path),
                    "--thresholds",
                    str(thr_path),
                    "--benchmark-scope",
                    "synthetic_controls",
                    "--external-validity-status",
                    "not_tested",
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
            self.assertEqual(out["benchmark_scope"], "synthetic_controls")
            self.assertEqual(out["external_validity_status"], "not_tested")
            self.assertTrue(bool(out["leakage_check_pass"]))
            self.assertEqual(out["gate_track"], "composite")
            self.assertIn("No direct natural-language CoT generalization claim", out["scope_disclaimer"])
            self.assertIn("evaluation_trace_hash", out)
            self.assertIn("calibration_source_ref", out)
            self.assertIn("composite", out["by_benchmark_track"])

            gate = json.loads(gate_path.read_text())
            self.assertFalse(bool(gate["externally_supported_claims"]))
            self.assertTrue(bool(gate["requires_real_cot_pilot"]))


if __name__ == "__main__":
    unittest.main()

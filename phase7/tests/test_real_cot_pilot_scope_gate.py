from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _trace_hash(trace_ids: list[str]) -> str:
    h = hashlib.sha256()
    for tid in sorted(trace_ids):
        h.update(tid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class RealCotPilotScopeGateTests(unittest.TestCase):
    def test_prepare_real_cot_eval_emits_alignment_stats(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            trace_path = tdp / "trace.pt"
            real_cot_path = tdp / "real_cot.jsonl"
            out_path = tdp / "real_cot_eval_input_v1.json"

            rows = [
                {
                    "trace_id": "gsm8k_test_00001",
                    "example_idx": 1,
                    "gsm8k_split": "test",
                    "step_idx": 0,
                    "structured_state": {
                        "step_idx": 0,
                        "step_type": "emit_result",
                        "operator": "unknown",
                        "lhs_value": None,
                        "rhs_value": None,
                        "subresult_value": 5.0,
                        "result_token_id": 42,
                        "magnitude_bucket": "[0,10)",
                        "sign": "pos",
                    },
                    "model_key": "gpt2-medium",
                    "model_family": "gpt2",
                    "num_layers": 24,
                    "hidden_dim": 1024,
                    "tokenizer_id": "gpt2",
                    "raw_hidden": torch.zeros(24, 1024),
                }
            ]
            torch.save(rows, trace_path)
            real_cot_path.write_text(
                "\n".join(
                    [
                        json.dumps({"trace_id": "gsm8k_test_00001", "cot_text": "STEP 0: EMIT_RESULT value=5 sign=pos mag=[0,10)"}),
                        json.dumps({"trace_id": "gsm8k_test_99999", "cot_text": "STEP 0: EMIT_RESULT value=1 sign=pos mag=[0,10)"}),
                    ]
                )
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/prepare_real_cot_eval.py",
                    "--real-cot-jsonl",
                    str(real_cot_path),
                    "--trace-dataset",
                    str(trace_path),
                    "--output",
                    str(out_path),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            payload = json.loads(out_path.read_text())
            self.assertEqual(payload["schema_version"], "phase7_real_cot_eval_input_v1")
            stats = payload.get("alignment_stats", {})
            self.assertEqual(int(stats.get("matched_trace_count", -1)), 1)
            self.assertEqual(int(stats.get("unmatched_trace_count", -1)), 1)
            self.assertIn("parser_success_by_trace", stats)
            self.assertIn("split_distribution", stats)

    def test_benchmark_real_cot_requires_scope_status_and_writes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            eval_path = tdp / "eval.json"
            thr_path = tdp / "thresholds.json"
            out_path = tdp / "real_benchmark.json"
            auto_gate = tdp / "real_benchmark_external_validity_gate.json"

            eval_audits = [
                {
                    "trace_id": "trace_eval_0",
                    "gold_label": "faithful",
                    "control_variant": "faithful",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.9,
                    "benchmark_track_scores": {"text_only": 0.8, "latent_only": 0.7, "causal_auditor": 0.75},
                    "verdict": "causally_faithful",
                    "parse_summary": {"parseable": 1, "total_steps": 1},
                },
                {
                    "trace_id": "trace_eval_1",
                    "gold_label": "unfaithful",
                    "control_variant": "wrong_intermediate",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.1,
                    "benchmark_track_scores": {"text_only": 0.2, "latent_only": 0.3, "causal_auditor": 0.25},
                    "verdict": "unsupported",
                    "parse_summary": {"parseable": 1, "total_steps": 1},
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

            fail_proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit-eval",
                    str(eval_path),
                    "--thresholds",
                    str(thr_path),
                    "--benchmark-scope",
                    "real_cot",
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
            self.assertNotEqual(fail_proc.returncode, 0)

            ok_proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit-eval",
                    str(eval_path),
                    "--thresholds",
                    str(thr_path),
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
            self.assertEqual(ok_proc.returncode, 0, msg=ok_proc.stdout)
            self.assertTrue(auto_gate.exists())
            out = json.loads(out_path.read_text())
            self.assertEqual(out["benchmark_scope"], "real_cot")
            self.assertEqual(out["external_validity_status"], "pilot")

            gate = json.loads(auto_gate.read_text())
            self.assertTrue(bool(gate["externally_supported_claims"]))
            self.assertTrue(bool(gate["has_real_cot_pilot"]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from phase7.generate_cot_controls import _build_variant
from phase7.parse_cot_to_states import align_parsed_to_trace, parse_cot_text


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _minimal_trace_steps() -> list[dict]:
    return [
        {
            "step_idx": 0,
            "structured_state": {
                "step_idx": 0,
                "step_type": "operate",
                "operator": "+",
                "lhs_value": 2.0,
                "rhs_value": 3.0,
                "subresult_value": 5.0,
                "result_token_id": 42,
                "magnitude_bucket": "[0,10)",
                "sign": "pos",
            },
        },
        {
            "step_idx": 1,
            "structured_state": {
                "step_idx": 1,
                "step_type": "emit_result",
                "operator": "unknown",
                "lhs_value": None,
                "rhs_value": None,
                "subresult_value": 5.0,
                "result_token_id": 42,
                "magnitude_bucket": "[0,10)",
                "sign": "pos",
            },
        },
    ]


class Phase7ControlsAndAuditTests(unittest.TestCase):
    def test_reordered_steps_preserves_ids_and_temporal_failure(self) -> None:
        trace_steps = _minimal_trace_steps()
        out = _build_variant(trace_steps, variant="reordered_steps", rng=random.Random(17))
        states = out["text_step_states"]
        self.assertEqual({int(s["step_idx"]) for s in states}, {0, 1})
        self.assertEqual([int(s["step_idx"]) for s in states], [1, 0])

        parsed = parse_cot_text(out["cot_text"])
        aligned = align_parsed_to_trace(parsed, trace_steps)
        self.assertFalse(bool(aligned["temporal_consistency"]["pass"]))

    def test_answer_first_order_flip_temporal_failure(self) -> None:
        trace_steps = _minimal_trace_steps()
        out = _build_variant(trace_steps, variant="answer_first_order_flip", rng=random.Random(17))
        self.assertEqual(out["cot_line_roles"][0], "final_answer")
        self.assertEqual({int(s["step_idx"]) for s in out["text_step_states"]}, {0, 1})

        parsed = parse_cot_text(out["cot_text"])
        aligned = align_parsed_to_trace(parsed, trace_steps)
        self.assertFalse(bool(aligned["temporal_consistency"]["pass"]))

    def test_calibrate_thresholds_module_mode_without_embedded_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.json"
            out_path = Path(td) / "thr.json"
            payload = {
                "audits": [
                    {"gold_label": "faithful", "overall_score": 0.9},
                    {"gold_label": "unfaithful", "overall_score": 0.1},
                ],
                "summary": {},
            }
            audit_path.write_text(json.dumps(payload))
            proc = _run(
                [
                    sys.executable,
                    "-m",
                    "phase7.calibrate_audit_thresholds",
                    "--audit",
                    str(audit_path),
                    "--output",
                    str(out_path),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertTrue(out_path.exists())
            out = json.loads(out_path.read_text())
            self.assertIn("thresholds", out)
            self.assertIn("overall_score_faithful_min", out["thresholds"])

    def test_causal_engine_strict_fail_empty_model_slice(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "trace.pt"
            out_path = Path(td) / "dryrun.json"
            rows = [
                {
                    "gsm8k_split": "test",
                    "model_key": "gpt2-medium",
                    "raw_hidden": torch.zeros(24, 1024),
                }
            ]
            torch.save(rows, ds_path)
            proc = _run(
                [
                    sys.executable,
                    "phase7/causal_intervention_engine.py",
                    "--trace-dataset",
                    str(ds_path),
                    "--model-key",
                    "qwen2.5-7b",
                    "--dry-run",
                    "--output",
                    str(out_path),
                ]
            )
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertTrue(out_path.exists())
            out = json.loads(out_path.read_text())
            self.assertEqual(out.get("status"), "error_no_records_for_model_key")

    def test_causal_engine_strict_fail_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "trace.pt"
            out_path = Path(td) / "dryrun.json"
            rows = [
                {
                    "gsm8k_split": "test",
                    "model_key": "qwen2.5-7b",
                    "model_family": "qwen2.5",
                    "num_layers": 28,
                    "hidden_dim": 3584,
                    "tokenizer_id": "Qwen/Qwen2.5-7B",
                    "raw_hidden": torch.zeros(24, 1024),
                }
            ]
            torch.save(rows, ds_path)
            proc = _run(
                [
                    sys.executable,
                    "phase7/causal_intervention_engine.py",
                    "--trace-dataset",
                    str(ds_path),
                    "--model-key",
                    "qwen2.5-7b",
                    "--dry-run",
                    "--output",
                    str(out_path),
                ]
            )
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertTrue(out_path.exists())
            out = json.loads(out_path.read_text())
            self.assertEqual(out.get("status"), "error_model_data_mismatch")


if __name__ == "__main__":
    unittest.main()

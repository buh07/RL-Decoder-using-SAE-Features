from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from phase7.common import parse_expression_steps
from phase7.train_state_decoders import _validate_schema_versions


REPO_ROOT = Path(__file__).resolve().parents[2]


class MultiOpTraceExpansionTests(unittest.TestCase):
    def test_build_v2_expanded_emits_multiop_steps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            train_src = tdp / "train.pt"
            test_src = tdp / "test.pt"
            out_dir = tdp / "out"

            train_rows = [
                {
                    "gsm8k_split": "train",
                    "example_idx": 10,
                    "ann_idx": 0,
                    "expr_str": "2+3*4",
                    "C": 20.0,  # left-assoc parse: (2+3)*4
                    "result_token_id": 42,
                    "raw_hidden": torch.zeros(24, 1024),
                    "model_key": "gpt2-medium",
                    "model_family": "gpt2",
                    "num_layers": 24,
                    "hidden_dim": 1024,
                    "tokenizer_id": "gpt2-medium",
                }
            ]
            test_rows = [
                {
                    "gsm8k_split": "test",
                    "example_idx": 11,
                    "ann_idx": 0,
                    "expr_str": "5-1",
                    "C": 4.0,
                    "result_token_id": 43,
                    "raw_hidden": torch.zeros(24, 1024),
                    "model_key": "gpt2-medium",
                    "model_family": "gpt2",
                    "num_layers": 24,
                    "hidden_dim": 1024,
                    "tokenizer_id": "gpt2-medium",
                }
            ]
            torch.save(train_rows, train_src)
            torch.save(test_rows, test_src)

            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/build_step_trace_dataset.py",
                    "--phase6-train",
                    str(train_src),
                    "--phase6-test",
                    str(test_src),
                    "--output-dir",
                    str(out_dir),
                    "--model-key",
                    "gpt2-medium",
                    "--state-ontology",
                    "v2_expanded",
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)

            built_train = torch.load(out_dir / "gsm8k_step_traces_train.pt", weights_only=False)
            self.assertTrue(built_train)
            self.assertTrue(all(r.get("schema_version") == "phase7_trace_v2" for r in built_train))

            # For 2+3*4 there should be two operate rows + one emit_result row.
            self.assertEqual(len(built_train), 3)
            self.assertEqual([r["structured_state"]["step_type"] for r in built_train], ["operate", "operate", "emit_result"])
            self.assertEqual([int(r["operation_idx"]) for r in built_train], [0, 1, 2])
            self.assertEqual([int(r["operation_count"]) for r in built_train], [2, 2, 2])
            self.assertTrue(all(r.get("state_ontology_version") == "v2_expanded" for r in built_train))

            summary = json.loads((out_dir / "build_summary.json").read_text())
            self.assertEqual(summary.get("schema_version"), "phase7_trace_v2")
            self.assertEqual(summary.get("state_ontology"), "v2_expanded")
            self.assertIn("2", summary["splits"]["train"]["operation_count_distribution"])

    def test_mixed_schema_rejected_by_default(self) -> None:
        rows = [
            {"schema_version": "phase7_trace_v1"},
            {"schema_version": "phase7_trace_v2"},
        ]
        with self.assertRaises(RuntimeError):
            _validate_schema_versions(rows, allow_mixed_schema=False)
        v = _validate_schema_versions(rows, allow_mixed_schema=True)
        self.assertEqual(v, "phase7_trace_v1")

    def test_parse_expression_steps_runtime_error_is_reported(self) -> None:
        parsed = parse_expression_steps("4/0+1", c_fallback=0.0)
        self.assertTrue(bool(parsed.get("parse_error")))
        self.assertIn("eval_error", str(parsed.get("parse_error")))

    def test_parse_expression_steps_respects_precedence_and_parentheses(self) -> None:
        parsed = parse_expression_steps("3*(2/3)", c_fallback=None)
        self.assertIsNone(parsed.get("parse_error"))
        steps = list(parsed.get("steps") or [])
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["operator"], "/")
        self.assertEqual(steps[1]["operator"], "*")
        self.assertAlmostEqual(float(steps[-1]["subresult_value"]), 2.0, places=6)

        parsed2 = parse_expression_steps("(2*111)-50", c_fallback=None)
        self.assertIsNone(parsed2.get("parse_error"))
        steps2 = list(parsed2.get("steps") or [])
        self.assertEqual([s["operator"] for s in steps2], ["*", "-"])
        self.assertAlmostEqual(float(steps2[-1]["subresult_value"]), 172.0, places=6)

        parsed3 = parse_expression_steps("560//10", c_fallback=None)
        self.assertIsNone(parsed3.get("parse_error"))
        steps3 = list(parsed3.get("steps") or [])
        self.assertEqual(len(steps3), 1)
        self.assertEqual(steps3[0]["operator"], "/")
        self.assertAlmostEqual(float(steps3[0]["subresult_value"]), 56.0, places=6)


if __name__ == "__main__":
    unittest.main()

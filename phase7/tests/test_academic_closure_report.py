from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class AcademicClosureReportTests(unittest.TestCase):
    def test_report_builder_emits_weakness_answers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            run_tag = "testrun"
            synth_path = td_path / f"faithfulness_benchmark_academic_{run_tag}_raw_every2_even.json"
            qwen_path = td_path / "qwen.json"
            out_path = td_path / "report.json"
            synth = {
                "by_benchmark_track": {"composite": {"auroc": 0.86}, "causal_auditor": {"auroc": 0.67}},
                "auroc": 0.86,
                "false_positive_rate": 0.03,
                "recall_at_gate": 0.2,
                "causal_signal_coverage_fraction": 0.4,
            }
            synth_path.write_text(json.dumps(synth))
            qwen_path.write_text(json.dumps({"model_metadata": {"model_key": "qwen2.5-7b"}}))
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/build_academic_closure_report.py",
                    "--run-tag",
                    run_tag,
                    "--synthetic-benchmarks-glob",
                    str(synth_path),
                    "--qwen-benchmark",
                    str(qwen_path),
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
            self.assertIn("weakness_answers", out)
            self.assertEqual(out["weakness_answers"]["B_threshold_utility"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class InvocationModeTests(unittest.TestCase):
    def _run(self, cmd: list[str]) -> None:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")

    def test_script_mode_help(self) -> None:
        scripts = [
            "phase7/train_state_decoders.py",
            "phase7/evaluate_state_decoders.py",
            "phase7/causal_intervention_engine.py",
            "phase7/causal_audit.py",
            "phase7/build_step_trace_dataset.py",
            "phase7/calibrate_audit_thresholds.py",
            "phase7/benchmark_faithfulness.py",
            "phase7/generate_cot_controls.py",
            "phase7/parse_cot_to_states.py",
        ]
        for script in scripts:
            with self.subTest(script=script):
                self._run([sys.executable, script, "--help"])

    def test_module_mode_help(self) -> None:
        modules = [
            "phase7.train_state_decoders",
            "phase7.evaluate_state_decoders",
            "phase7.causal_intervention_engine",
            "phase7.causal_audit",
            "phase7.build_step_trace_dataset",
            "phase7.calibrate_audit_thresholds",
            "phase7.benchmark_faithfulness",
            "phase7.generate_cot_controls",
            "phase7.parse_cot_to_states",
        ]
        for module in modules:
            with self.subTest(module=module):
                self._run([sys.executable, "-m", module, "--help"])


if __name__ == "__main__":
    unittest.main()

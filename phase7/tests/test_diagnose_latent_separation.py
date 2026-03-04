from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class DiagnoseLatentSeparationTests(unittest.TestCase):
    def test_empty_controls_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            controls = Path(td) / "controls.json"
            controls.write_text(json.dumps({"schema_version": "phase7_cot_control_v1", "controls": []}))
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/diagnose_latent_separation.py",
                    "--controls",
                    str(controls),
                    "--trace-dataset",
                    "does_not_matter.pt",
                    "--state-decoder-checkpoint",
                    "does_not_matter.pt",
                    "--device",
                    "cpu",
                    "--output-json",
                    str(Path(td) / "diag.json"),
                    "--output-report",
                    str(Path(td) / "diag.md"),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertIn("No controls found", proc.stdout)


if __name__ == "__main__":
    unittest.main()


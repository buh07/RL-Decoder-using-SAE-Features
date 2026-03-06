from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from phase7.common import sha256_file


class SplitManifestReuseHashGuardTest(unittest.TestCase):
    def test_reuse_then_regenerate_on_hash_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit = root / "audit.json"
            prefix = root / "split"
            manifest = root / "split_split_manifest.json"
            calib = root / "split_calib.json"
            evalp = root / "split_eval.json"

            payload = {
                "schema_version": "causal_audit_v1",
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"x": 1},
                "audits": [
                    {"trace_id": "t1", "gold_label": "faithful", "control_variant": "faithful"},
                    {"trace_id": "t2", "gold_label": "unfaithful", "control_variant": "wrong_intermediate"},
                ],
            }
            audit.write_text(json.dumps(payload, indent=2))

            cmd = [
                ".venv/bin/python3",
                "phase7/split_audit_dataset.py",
                "--audit",
                str(audit),
                "--seed",
                "20260303",
                "--calib-fraction",
                "0.30",
                "--output-prefix",
                str(prefix),
            ]
            subprocess.run(cmd, check=True)
            self.assertTrue(manifest.exists())
            self.assertTrue(calib.exists())
            self.assertTrue(evalp.exists())

            mtime_calib_1 = calib.stat().st_mtime
            mtime_eval_1 = evalp.stat().st_mtime
            mtime_manifest_1 = manifest.stat().st_mtime

            cmd_reuse = [
                ".venv/bin/python3",
                "phase7/split_audit_dataset.py",
                "--audit",
                str(audit),
                "--seed",
                "20260303",
                "--calib-fraction",
                "0.30",
                "--output-prefix",
                str(prefix),
                "--reuse-manifest-if-compatible",
                str(manifest),
                "--source-audit-hash",
                str(sha256_file(audit)),
            ]
            subprocess.run(cmd_reuse, check=True)
            self.assertEqual(mtime_calib_1, calib.stat().st_mtime)
            self.assertEqual(mtime_eval_1, evalp.stat().st_mtime)
            self.assertEqual(mtime_manifest_1, manifest.stat().st_mtime)

            payload["audits"].append(
                {"trace_id": "t3", "gold_label": "faithful", "control_variant": "faithful"}
            )
            audit.write_text(json.dumps(payload, indent=2))
            subprocess.run(cmd_reuse[:-1] + [str(sha256_file(audit))], check=True)
            self.assertGreaterEqual(calib.stat().st_mtime, mtime_calib_1)
            self.assertGreaterEqual(evalp.stat().st_mtime, mtime_eval_1)
            self.assertGreaterEqual(manifest.stat().st_mtime, mtime_manifest_1)


if __name__ == "__main__":
    unittest.main()

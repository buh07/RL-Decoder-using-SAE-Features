from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class StrictExternalLabelPolicyKeywordTests(unittest.TestCase):
    def test_keyword_proxy_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_path = td_path / "labels.jsonl"
            out_path = td_path / "ingest.json"
            rows = [
                {
                    "trace_id": "r0",
                    "cot_text": "Compute 1+1=2.",
                    "label": "faithful",
                    "failure_family": "real_cot_pilot",
                    "label_source": "automatic_label",
                    "label_definition": "auto_label_by_regex_match",
                    "annotation_origin": "benchmark_gold",
                }
            ]
            src_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/ingest_public_cot_benchmark.py",
                    "--source",
                    "local_jsonl",
                    "--local-jsonl",
                    str(src_path),
                    "--strict-external-labels",
                    "--output",
                    str(out_path),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertIn("Strict external-label policy rejected", proc.stdout)


if __name__ == "__main__":
    unittest.main()

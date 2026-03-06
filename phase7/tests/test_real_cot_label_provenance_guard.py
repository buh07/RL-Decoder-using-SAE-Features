from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class RealCotLabelProvenanceGuardTests(unittest.TestCase):
    def test_empty_local_jsonl_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_path = td_path / "empty.jsonl"
            out_path = td_path / "ingest.json"
            src_path.write_text("")
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/ingest_public_cot_benchmark.py",
                    "--source",
                    "local_jsonl",
                    "--local-jsonl",
                    str(src_path),
                    "--output",
                    str(out_path),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout)

    def test_label_provenance_emitted_for_nonempty_local_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_path = td_path / "public.jsonl"
            out_path = td_path / "ingest.json"
            controls_path = td_path / "controls.json"
            rows = [
                {
                    "trace_id": "r0",
                    "cot_text": "Compute 2+2=4.",
                    "label": "faithful",
                    "failure_family": "real_cot_pilot",
                },
                {
                    "trace_id": "r1",
                    "cot_text": "Compute 2+2=5.",
                    "label": "unfaithful",
                    "failure_family": "wrong_intermediate",
                },
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
                    "--output",
                    str(out_path),
                    "--controls-output",
                    str(controls_path),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            out = json.loads(out_path.read_text())
            self.assertEqual(out["num_rows"], 2)
            self.assertIn("label_provenance", out)
            self.assertIn("external_benchmark_label", out["label_provenance"]["label_sources_present"])
            self.assertTrue(controls_path.exists())

    def test_strict_external_label_policy_rejects_proxy_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_path = td_path / "public_proxy.jsonl"
            out_path = td_path / "ingest.json"
            rows = [
                {
                    "trace_id": "r0",
                    "cot_text": "Compute 2+2=4.",
                    "label": "faithful",
                    "failure_family": "real_cot_pilot",
                    "label_source": "answer_match_proxy",
                    "label_definition": "proxy_answer_match",
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

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phase7.common import load_rows_payload, sha256_file, write_rows_sidecar


class RowsSidecarJsonlGzTest(unittest.TestCase):
    def test_sidecar_roundtrip(self) -> None:
        rows = [
            {"trace_id": "t1", "step_idx": 1, "x": 1.5},
            {"trace_id": "t1", "step_idx": 2, "x": -3.0},
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "payload.json"
            payload = write_rows_sidecar(out, rows, rows_format="jsonl.gz", rows_inline=False)
            self.assertFalse(bool(payload.get("rows_inline")))
            self.assertEqual("jsonl.gz", payload.get("rows_format"))
            rows_path = Path(str(payload.get("rows_path")))
            self.assertTrue(rows_path.exists())
            self.assertEqual(str(sha256_file(rows_path)), str(payload.get("rows_sha256")))
            loaded = load_rows_payload(payload, base_path=out)
            self.assertEqual(rows, loaded)

    def test_inline_roundtrip(self) -> None:
        rows = [{"k": 1}, {"k": 2}]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "payload.json"
            payload = write_rows_sidecar(out, rows, rows_format="json", rows_inline=True)
            self.assertTrue(bool(payload.get("rows_inline")))
            self.assertEqual("json", payload.get("rows_format"))
            self.assertIsNone(payload.get("rows_path"))
            loaded = load_rows_payload(payload, base_path=out)
            self.assertEqual(rows, loaded)


if __name__ == "__main__":
    unittest.main()

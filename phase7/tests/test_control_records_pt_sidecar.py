from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

from phase7.causal_intervention_engine import (
    _load_control_records_artifact,
    _save_control_records_artifact,
)


class ControlRecordsPtSidecarTest(unittest.TestCase):
    def test_tensor_rows_roundtrip_pt_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            controls = root / "controls.json"
            trace_dataset = root / "trace.pt"
            output = root / "records.json"
            controls.write_text(json.dumps({"controls": []}))
            torch.save([], trace_dataset)

            args = Namespace(
                controls=str(controls),
                trace_dataset=str(trace_dataset),
                parse_mode="hybrid",
                token_anchor="eq_like",
                anchor_priority="template_first",
                control_sampling_policy="stratified_trace_variant",
                seed=17,
                max_records_cap=12000,
                rows_format="jsonl.gz",
            )
            model_spec = SimpleNamespace(
                model_key="gpt2-medium",
                model_family="gpt2",
                num_layers=24,
                hidden_dim=1024,
                tokenizer_id="gpt2",
            )
            rows = [
                {
                    "trace_id": "t1",
                    "step_idx": 1,
                    "raw_hidden": torch.randn(24, 8),
                    "structured_state": {"step_type": "operate"},
                }
            ]
            payload = _save_control_records_artifact(
                str(output),
                records=rows,
                stats={"controls_total": 1, "controls_used": 1},
                model_spec=model_spec,
                args=args,
            )
            self.assertEqual("pt", payload.get("rows_format"))
            self.assertTrue(Path(str(payload.get("rows_path"))).exists())

            loaded_rows, loaded_payload = _load_control_records_artifact(str(output))
            self.assertEqual("pt", loaded_payload.get("rows_format"))
            self.assertEqual(1, len(loaded_rows))
            self.assertTrue(isinstance(loaded_rows[0].get("raw_hidden"), torch.Tensor))
            self.assertEqual(tuple(rows[0]["raw_hidden"].shape), tuple(loaded_rows[0]["raw_hidden"].shape))


if __name__ == "__main__":
    unittest.main()

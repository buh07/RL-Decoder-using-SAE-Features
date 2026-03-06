from __future__ import annotations

import unittest

from phase7.causal_audit import _index_causal_checks


class CausalIndexModeInvariantsTests(unittest.TestCase):
    def test_control_conditioned_indexes_only_variant_triples(self) -> None:
        payload = {
            "causal_mode": "control_conditioned",
            "rows": [
                {
                    "source_trace_id": "t1",
                    "source_control_variant": "faithful",
                    "source_step_idx": 0,
                    "layers": {},
                },
                {
                    "source_trace_id": "t1",
                    "source_control_variant": "wrong_intermediate",
                    "source_step_idx": 0,
                    "layers": {},
                },
            ],
        }
        idx = _index_causal_checks(payload)
        self.assertIn(("t1", "faithful", 0), idx)
        self.assertIn(("t1", "wrong_intermediate", 0), idx)
        self.assertNotIn(("t1", 0), idx)

    def test_control_conditioned_missing_variant_fails(self) -> None:
        payload = {
            "causal_mode": "control_conditioned",
            "rows": [
                {
                    "source_trace_id": "t1",
                    "source_step_idx": 0,
                    "layers": {},
                }
            ],
        }
        with self.assertRaises(ValueError):
            _index_causal_checks(payload)

    def test_source_trace_variant_tagged_rows_fail(self) -> None:
        payload = {
            "causal_mode": "source_trace",
            "rows": [
                {
                    "source_trace_id": "t1",
                    "source_control_variant": "faithful",
                    "source_step_idx": 0,
                    "layers": {},
                }
            ],
        }
        with self.assertRaises(ValueError):
            _index_causal_checks(payload)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import unittest

from phase7.causal_intervention_engine import select_matched_donor


def _row(trace_id: str, step_idx: int, example_idx: int, subresult: float) -> dict:
    return {
        "trace_id": trace_id,
        "step_idx": step_idx,
        "example_idx": example_idx,
        "structured_state": {
            "step_type": "operate",
            "operator": "+",
            "magnitude_bucket": "[0,10)",
            "subresult_value": subresult,
        },
    }


class DonorIdentityExclusionTests(unittest.TestCase):
    def test_source_identity_key_is_excluded_even_with_distinct_object(self) -> None:
        source = _row("t0", 0, 100, 5.0)
        source_clone_same_key = copy.deepcopy(source)
        donor_ok = _row("t1", 0, 200, 7.0)
        rows = [source_clone_same_key, donor_ok]
        donor = select_matched_donor(rows, source, variable="subresult_value", seed=17)
        self.assertIsNotNone(donor)
        self.assertEqual(str(donor.get("trace_id")), "t1")


if __name__ == "__main__":
    unittest.main()

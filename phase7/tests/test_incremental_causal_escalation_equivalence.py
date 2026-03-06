from __future__ import annotations

import unittest

from phase7.causal_intervention_engine import _causal_row_identity_key


class IncrementalCausalEscalationEquivalenceTest(unittest.TestCase):
    def test_incremental_merge_matches_one_shot_keys(self) -> None:
        variable = "subresult_value"
        existing = [
            {
                "source_trace_id": "t1",
                "source_control_variant": "faithful",
                "source_step_idx": 1,
                "layers": {},
            },
            {
                "source_trace_id": "t2",
                "source_control_variant": "wrong_intermediate",
                "source_step_idx": 2,
                "layers": {},
            },
        ]
        staged_new = [
            {
                "source_trace_id": "t2",
                "source_control_variant": "wrong_intermediate",
                "source_step_idx": 2,
                "layers": {},
            },
            {
                "source_trace_id": "t3",
                "source_control_variant": "reordered_steps",
                "source_step_idx": 1,
                "layers": {},
            },
        ]

        existing_keys = {_causal_row_identity_key(r, variable=variable) for r in existing}
        incremental_rows = list(existing)
        for row in staged_new:
            k = _causal_row_identity_key(row, variable=variable)
            if k in existing_keys:
                continue
            incremental_rows.append(row)
            existing_keys.add(k)

        one_shot = list(existing)
        seen = {_causal_row_identity_key(r, variable=variable) for r in one_shot}
        for row in staged_new:
            k = _causal_row_identity_key(row, variable=variable)
            if k not in seen:
                one_shot.append(row)
                seen.add(k)

        inc_keys = {_causal_row_identity_key(r, variable=variable) for r in incremental_rows}
        one_keys = {_causal_row_identity_key(r, variable=variable) for r in one_shot}
        self.assertEqual(one_keys, inc_keys)
        self.assertEqual(len(one_shot), len(incremental_rows))


if __name__ == "__main__":
    unittest.main()

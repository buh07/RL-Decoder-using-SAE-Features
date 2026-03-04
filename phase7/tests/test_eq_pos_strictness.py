from __future__ import annotations

import unittest

from phase7.causal_intervention_engine import _record_eq_pos


class EqPosStrictnessTests(unittest.TestCase):
    def test_valid_eq_pos(self) -> None:
        rec = {"eq_tok_idx": 2, "token_ids": [10, 11, 12]}
        self.assertEqual(_record_eq_pos(rec), 1)

    def test_missing_eq_tok_idx_raises(self) -> None:
        rec = {"token_ids": [10, 11, 12]}
        with self.assertRaises(KeyError):
            _record_eq_pos(rec)

    def test_non_positive_eq_tok_idx_raises(self) -> None:
        rec = {"eq_tok_idx": 0, "token_ids": [10, 11, 12]}
        with self.assertRaises(ValueError):
            _record_eq_pos(rec)

    def test_out_of_bounds_eq_tok_idx_raises(self) -> None:
        rec = {"eq_tok_idx": 5, "token_ids": [10, 11, 12]}
        with self.assertRaises(ValueError):
            _record_eq_pos(rec)


if __name__ == "__main__":
    unittest.main()

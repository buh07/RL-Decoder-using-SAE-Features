from __future__ import annotations

import unittest

from phase7.evaluate_state_decoders import _filter_records_by_split as eval_filter
from phase7.train_state_decoders import _filter_records_by_split as train_filter


class SplitFieldStrictnessTests(unittest.TestCase):
    def test_train_filter_strict_missing_split_field(self) -> None:
        rows = [{"example_idx": 1}, {"example_idx": 2}]
        with self.assertRaises(RuntimeError):
            train_filter(rows, "train", allow_missing_split_field=False)

    def test_train_filter_allow_legacy(self) -> None:
        rows = [{"example_idx": 1}, {"example_idx": 2}]
        out = train_filter(rows, "train", allow_missing_split_field=True)
        self.assertEqual(len(out), 2)

    def test_eval_filter_strict_no_matching_split(self) -> None:
        rows = [{"gsm8k_split": "train", "example_idx": 1}]
        with self.assertRaises(RuntimeError):
            eval_filter(rows, "test", allow_missing_split_field=False)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from unittest import mock

from phase7.evaluate_state_decoders import parse_args


class SaliencySplitSelectionTests(unittest.TestCase):
    def test_saliency_split_defaults_to_test(self) -> None:
        argv = ["prog", "--checkpoint", "dummy.pt"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.saliency_split, "test")
        self.assertEqual(args.grad_saliency_max_records, 256)


if __name__ == "__main__":
    unittest.main()

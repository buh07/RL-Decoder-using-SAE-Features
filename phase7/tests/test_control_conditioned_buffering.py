from __future__ import annotations

import unittest

from phase7.causal_intervention_engine import _iter_record_buffers


class ControlConditionedBufferingTests(unittest.TestCase):
    def test_iter_record_buffers_chunks_exactly(self) -> None:
        rows = [{"i": i} for i in range(10)]
        chunks = list(_iter_record_buffers(rows, buffer_size=4))
        self.assertEqual([len(c) for c in chunks], [4, 4, 2])
        flattened = [r["i"] for c in chunks for r in c]
        self.assertEqual(flattened, list(range(10)))

    def test_iter_record_buffers_rejects_non_positive(self) -> None:
        with self.assertRaises(ValueError):
            list(_iter_record_buffers([{"i": 0}], buffer_size=0))


if __name__ == "__main__":
    unittest.main()

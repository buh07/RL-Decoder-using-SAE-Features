from __future__ import annotations

import random
import unittest

from phase7.common import perturb_number


class PerturbNumberSymmetryTests(unittest.TestCase):
    def test_small_mode_direction_is_balanced(self) -> None:
        rng = random.Random(1234)
        pos_up = 0
        pos_down = 0
        for _ in range(400):
            base = 10.0
            out = perturb_number(base, mode="small", rng=rng)
            if out is None:
                continue
            if out > base:
                pos_up += 1
            elif out < base:
                pos_down += 1
        total = pos_up + pos_down
        self.assertGreater(total, 0)
        ratio_up = pos_up / total
        self.assertGreater(ratio_up, 0.35)
        self.assertLess(ratio_up, 0.65)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from phase7.causal_intervention_engine import _next_max_records_stage


class CoverageEscalationPolicyTests(unittest.TestCase):
    def test_escalates_through_stages_until_cap(self) -> None:
        self.assertEqual(
            _next_max_records_stage(
                1200,
                coverage_fraction=0.10,
                controls_used_fraction=0.10,
                target_controls_used_fraction=0.35,
                max_records_cap=12000,
            ),
            2200,
        )
        self.assertEqual(
            _next_max_records_stage(
                2200,
                coverage_fraction=0.10,
                controls_used_fraction=0.20,
                target_controls_used_fraction=0.35,
                max_records_cap=12000,
            ),
            6000,
        )
        self.assertEqual(
            _next_max_records_stage(
                6000,
                coverage_fraction=0.20,
                controls_used_fraction=0.20,
                target_controls_used_fraction=0.35,
                max_records_cap=12000,
            ),
            12000,
        )
        self.assertIsNone(
            _next_max_records_stage(
                12000,
                coverage_fraction=0.10,
                controls_used_fraction=0.10,
                target_controls_used_fraction=0.35,
                max_records_cap=12000,
            )
        )

    def test_no_escalation_when_target_met(self) -> None:
        self.assertIsNone(
            _next_max_records_stage(
                1200,
                coverage_fraction=0.40,
                controls_used_fraction=0.35,
                target_controls_used_fraction=0.35,
                max_records_cap=12000,
            )
        )

    def test_cap_constrains_next_stage(self) -> None:
        self.assertEqual(
            _next_max_records_stage(
                2200,
                coverage_fraction=0.10,
                controls_used_fraction=0.10,
                target_controls_used_fraction=0.35,
                max_records_cap=6000,
            ),
            6000,
        )
        self.assertIsNone(
            _next_max_records_stage(
                6000,
                coverage_fraction=0.10,
                controls_used_fraction=0.10,
                target_controls_used_fraction=0.35,
                max_records_cap=6000,
            )
        )


if __name__ == "__main__":
    unittest.main()

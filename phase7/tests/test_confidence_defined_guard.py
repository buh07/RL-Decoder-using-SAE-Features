from __future__ import annotations

import unittest

from phase7.causal_audit import _evaluate_confidence_defined_guard, default_thresholds


class ConfidenceDefinedGuardTests(unittest.TestCase):
    def test_guard_not_required_when_confidence_weight_zero(self) -> None:
        thr = {"thresholds": default_thresholds()}
        audits = [
            {"benchmark_track_definedness": {"confidence_margin": False}},
            {"benchmark_track_definedness": {"confidence_margin": False}},
        ]
        out = _evaluate_confidence_defined_guard(audits, thr, required_fraction=0.95)
        self.assertFalse(bool(out.get("guard_required")))
        self.assertIsNone(out.get("guard_pass"))

    def test_guard_required_and_passes(self) -> None:
        t = default_thresholds()
        t["confidence_score_weight"] = 0.2
        thr = {"thresholds": t}
        audits = [
            {"benchmark_track_definedness": {"confidence_margin": True}}
            for _ in range(10)
        ]
        out = _evaluate_confidence_defined_guard(audits, thr, required_fraction=0.95)
        self.assertTrue(bool(out.get("guard_required")))
        self.assertTrue(bool(out.get("guard_pass")))
        self.assertAlmostEqual(float(out.get("confidence_defined_fraction")), 1.0, places=8)

    def test_guard_required_and_fails(self) -> None:
        t = default_thresholds()
        t["confidence_score_weight"] = 0.2
        thr = {"thresholds": t}
        audits = [
            {"benchmark_track_definedness": {"confidence_margin": True}}
            for _ in range(8)
        ] + [
            {"benchmark_track_definedness": {"confidence_margin": False}}
            for _ in range(2)
        ]
        out = _evaluate_confidence_defined_guard(audits, thr, required_fraction=0.95)
        self.assertTrue(bool(out.get("guard_required")))
        self.assertFalse(bool(out.get("guard_pass")))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from phase7.causal_audit import _resolve_causal_lookup_mode


class CausalModeContractTests(unittest.TestCase):
    def test_accepts_matching_required_mode(self) -> None:
        mode = _resolve_causal_lookup_mode(
            {"causal_mode": "control_conditioned"},
            required_mode="control_conditioned",
            source_path="dummy.json",
        )
        self.assertEqual(mode, "control_conditioned")

    def test_normalizes_unknown_mode_to_source_trace(self) -> None:
        mode = _resolve_causal_lookup_mode({"causal_mode": "unknown_mode"}, required_mode=None)
        self.assertEqual(mode, "source_trace")

    def test_raises_on_mode_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_causal_lookup_mode(
                {"causal_mode": "source_trace"},
                required_mode="control_conditioned",
                source_path="causal.json",
            )


if __name__ == "__main__":
    unittest.main()

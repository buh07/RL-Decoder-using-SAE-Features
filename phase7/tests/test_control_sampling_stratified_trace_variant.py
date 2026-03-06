from __future__ import annotations

import random
import unittest

from phase7.causal_intervention_engine import _order_controls_for_sampling


class ControlSamplingStratifiedTraceVariantTests(unittest.TestCase):
    def test_stratified_policy_prioritizes_trace_level_label_diversity(self) -> None:
        controls = []
        for t in range(4):
            trace_id = f"trace_{t}"
            controls.append({"trace_id": trace_id, "variant": "faithful", "gold_label": "faithful"})
            controls.append({"trace_id": trace_id, "variant": "wrong_intermediate", "gold_label": "unfaithful"})
            controls.append({"trace_id": trace_id, "variant": "reordered_steps", "gold_label": "unfaithful"})

        ordered = _order_controls_for_sampling(
            controls,
            policy="stratified_trace_variant",
            rng=random.Random(13),
        )
        self.assertEqual(len(ordered), len(controls))
        # The first 2 * num_traces controls should include one faithful + one unfaithful per trace.
        head = ordered[: 2 * 4]
        by_trace = {}
        for c in head:
            by_trace.setdefault(str(c["trace_id"]), set()).add(str(c.get("gold_label")))
        self.assertEqual(set(by_trace.keys()), {f"trace_{i}" for i in range(4)})
        for labels in by_trace.values():
            self.assertIn("faithful", labels)
            self.assertIn("unfaithful", labels)

    def test_random_policy_keeps_cardinality(self) -> None:
        controls = [{"trace_id": "t", "variant": f"v{i}", "gold_label": "unfaithful"} for i in range(10)]
        ordered = _order_controls_for_sampling(
            controls,
            policy="random",
            rng=random.Random(7),
        )
        self.assertEqual(len(ordered), len(controls))
        self.assertEqual({c["variant"] for c in ordered}, {c["variant"] for c in controls})


if __name__ == "__main__":
    unittest.main()

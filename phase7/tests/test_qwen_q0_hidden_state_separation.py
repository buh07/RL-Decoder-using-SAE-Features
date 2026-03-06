from __future__ import annotations

import unittest

import torch

try:  # pragma: no cover
    from phase7.qwen_q0_hidden_state_separation import (
        _decide_go_nogo,
        _eligible_trace_ids,
        _group_observed_and_null,
        _sample_trace_ids,
    )
except Exception:  # pragma: no cover
    from qwen_q0_hidden_state_separation import (
        _decide_go_nogo,
        _eligible_trace_ids,
        _group_observed_and_null,
        _sample_trace_ids,
    )


class QwenQ0HelpersTest(unittest.TestCase):
    def test_eligible_trace_ids_requires_both_labels(self) -> None:
        controls = [
            {"trace_id": "t1", "gold_label": "faithful"},
            {"trace_id": "t1", "gold_label": "unfaithful"},
            {"trace_id": "t2", "gold_label": "faithful"},
            {"trace_id": "t3", "gold_label": "unfaithful"},
        ]
        self.assertEqual(_eligible_trace_ids(controls), ["t1"])

    def test_sample_trace_ids_deterministic(self) -> None:
        traces = [f"t{i}" for i in range(10)]
        a = _sample_trace_ids(traces, sample_size=5, seed=7)
        b = _sample_trace_ids(traces, sample_size=5, seed=7)
        c = _sample_trace_ids(traces, sample_size=5, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 5)

    def test_group_permutation_reproducible(self) -> None:
        vectors = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.1, 0.1]),
            torch.tensor([0.9, -0.1]),
        ]
        labels = ["faithful", "unfaithful", "unfaithful", "unfaithful"]
        out_a = _group_observed_and_null(vectors, labels, n_permutations=25, rng_seed=11)
        out_b = _group_observed_and_null(vectors, labels, n_permutations=25, rng_seed=11)
        self.assertEqual(out_a["status"], "ok")
        self.assertAlmostEqual(float(out_a["observed"]), float(out_b["observed"]), places=7)
        self.assertEqual(out_a["null_distribution"], out_b["null_distribution"])

    def test_decide_go_nogo_logic(self) -> None:
        agg_pass = {"margin": 0.02}
        out_pass = _decide_go_nogo(
            agg_pass,
            positive_trace_fraction=0.70,
            min_positive_trace_fraction=0.50,
            min_margin=0.01,
        )
        self.assertTrue(out_pass["pass"])
        self.assertEqual(out_pass["decision"], "go")

        agg_fail = {"margin": -0.01}
        out_fail = _decide_go_nogo(
            agg_fail,
            positive_trace_fraction=0.40,
            min_positive_trace_fraction=0.50,
            min_margin=0.01,
        )
        self.assertFalse(out_fail["pass"])
        self.assertEqual(out_fail["decision"], "no_go")


if __name__ == "__main__":
    unittest.main()

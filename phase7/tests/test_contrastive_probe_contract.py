from __future__ import annotations

import unittest

import torch

from phase7.contrastive_faithfulness_probe import (
    FaithfulnessProbe,
    Sample,
    _parse_csv_list,
    _records_for_source_trace,
    _run_single_source,
    _split_trace_ids,
)


class ContrastiveProbeContractTests(unittest.TestCase):
    def test_probe_output_shape(self) -> None:
        model = FaithfulnessProbe(input_dim=16, hidden_dim=8)
        x = torch.randn(4, 16)
        y = model(x)
        self.assertEqual(tuple(y.shape), (4,))

    def test_variant_split_no_overlap(self) -> None:
        train_variants = set(_parse_csv_list("answer_first_only,wrong_operator,faithful"))
        test_variants = set(_parse_csv_list("wrong_sign,digit_swap,faithful"))
        overlap = train_variants.intersection(test_variants) - {"faithful"}
        self.assertEqual(overlap, set())

    def test_source_trace_mode_reuses_hidden_by_key(self) -> None:
        h_f = torch.tensor([[1.0, 2.0]])
        h_u = torch.tensor([[9.0, 9.0]])
        rows = [
            {
                "trace_id": "t0",
                "step_idx": 0,
                "example_idx": 1,
                "control_variant": "faithful",
                "gold_label": "faithful",
                "raw_hidden": h_f,
            },
            {
                "trace_id": "t0",
                "step_idx": 0,
                "example_idx": 1,
                "control_variant": "wrong_sign",
                "gold_label": "unfaithful",
                "raw_hidden": h_u,
            },
        ]
        out = _records_for_source_trace(rows)
        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0]["raw_hidden"], out[1]["raw_hidden"]))
        self.assertTrue(torch.equal(out[0]["raw_hidden"], h_f))

    def test_trace_split_ids_disjoint(self) -> None:
        train_ids, test_ids = _split_trace_ids(
            [f"t{i}" for i in range(10)],
            test_fraction=0.30,
            seed=7,
        )
        self.assertTrue(bool(train_ids))
        self.assertTrue(bool(test_ids))
        self.assertEqual(train_ids.intersection(test_ids), set())

    def test_run_single_source_blocks_on_class_imbalance(self) -> None:
        samples = [
            Sample(x=torch.randn(4), y=1, trace_id=f"t{i}", variant="wrong_operator")
            for i in range(12)
        ]
        out = _run_single_source(
            samples,
            train_variants={"wrong_operator"},
            test_variants={"wrong_operator"},
            trace_test_fraction=0.30,
            trace_split_seed=11,
            min_class_per_split=1,
            hidden_dim=8,
            lr=1e-3,
            weight_decay=0.01,
            epochs=2,
            device="cpu",
        )
        self.assertEqual(out.get("status"), "blocked_insufficient_class_balance_after_trace_split")

    def test_trace_stratified_random_ignores_variant_filters(self) -> None:
        samples = []
        for i in range(12):
            tid = f"t{i}"
            samples.append(Sample(x=torch.randn(6), y=0, trace_id=tid, variant="faithful"))
            samples.append(Sample(x=torch.randn(6), y=1, trace_id=tid, variant="wrong_operator"))
        out = _run_single_source(
            samples,
            train_variants={"nonexistent_train_variant"},
            test_variants={"nonexistent_test_variant"},
            split_policy="trace_stratified_random",
            trace_test_fraction=0.20,
            trace_split_seed=7,
            min_class_per_split=1,
            hidden_dim=8,
            lr=1e-3,
            weight_decay=0.01,
            epochs=2,
            device="cpu",
        )
        self.assertEqual(out.get("status"), "ok")
        diag = out.get("split_diagnostics") or {}
        self.assertFalse(bool(diag.get("variant_filter_applied")))
        self.assertTrue(bool(diag.get("trace_overlap_check_pass")))


if __name__ == "__main__":
    unittest.main()

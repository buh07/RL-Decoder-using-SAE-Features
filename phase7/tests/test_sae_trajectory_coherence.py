from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from phase7.sae_trajectory_coherence_discrimination import (
    _build_pair_descriptors,
    _magnitude_monotonicity_coherence,
    _select_feature_indices,
    _trajectory_samples,
)


def _row(trace_id: str, variant: str, step_idx: int, label: str, line_index: int) -> dict:
    return {
        "trace_id": trace_id,
        "control_variant": variant,
        "step_idx": step_idx,
        "gold_label": label,
        "line_index": line_index,
        "raw_hidden": torch.zeros((24, 1024), dtype=torch.float32),
    }


class SAETrajectoryCoherenceTests(unittest.TestCase):
    def test_feature_set_selection_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            phase4_path = tdp / "top_features.json"
            divergent_path = tdp / "divergent.json"

            payload = {
                "eq": [[i for i in range(60)] for _ in range(24)],
                "pre_eq": [[1000 + i for i in range(60)] for _ in range(24)],
                "result": [[2000 + i for i in range(60)] for _ in range(24)],
            }
            phase4_path.write_text(json.dumps(payload))
            divergent_payload = {
                "by_layer": {
                    "7": {
                        "feature_divergence": {
                            "top_features_abs_d": [{"feature_idx": 4000 + i} for i in range(70)]
                        }
                    }
                }
            }
            divergent_path.write_text(json.dumps(divergent_payload))

            f_eq, meta_eq = _select_feature_indices(
                feature_set="eq_top50",
                layer=7,
                phase4_top_features_path=phase4_path,
                divergent_source_path=divergent_path,
            )
            self.assertEqual(len(f_eq), 50)
            self.assertEqual(f_eq[0], 0)
            self.assertEqual(meta_eq["feature_set"], "eq_top50")

            f_res, meta_res = _select_feature_indices(
                feature_set="result_top50",
                layer=7,
                phase4_top_features_path=phase4_path,
                divergent_source_path=divergent_path,
            )
            self.assertEqual(len(f_res), 50)
            self.assertEqual(f_res[0], 2000)
            self.assertEqual(meta_res["feature_set"], "result_top50")

            f_all, meta_all = _select_feature_indices(
                feature_set="eq_pre_result_150",
                layer=7,
                phase4_top_features_path=phase4_path,
                divergent_source_path=divergent_path,
            )
            self.assertEqual(len(f_all), 150)
            self.assertEqual(f_all[:3], [0, 1, 2])
            self.assertIn("component_counts", meta_all)

            f_div, meta_div = _select_feature_indices(
                feature_set="divergent_top50",
                layer=7,
                phase4_top_features_path=phase4_path,
                divergent_source_path=divergent_path,
            )
            self.assertEqual(len(f_div), 50)
            self.assertEqual(f_div[0], 4000)
            self.assertEqual(meta_div["feature_set"], "divergent_top50")

    def test_pairing_requires_common_steps_and_uses_step_idx_alignment(self) -> None:
        rows = [
            _row("t1", "faithful", 0, "faithful", 0),
            _row("t1", "faithful", 1, "faithful", 1),
            _row("t1", "faithful", 2, "faithful", 2),
            _row("t1", "faithful", 3, "faithful", 3),
            _row("t1", "order_flip_only", 3, "unfaithful", 0),
            _row("t1", "order_flip_only", 2, "unfaithful", 1),
            _row("t1", "order_flip_only", 1, "unfaithful", 2),
            _row("t1", "order_flip_only", 0, "unfaithful", 3),
        ]
        rows_used, pairs, diag = _build_pair_descriptors(
            rows,
            min_common_steps=3,
            sample_traces=None,
            seed=17,
        )
        self.assertEqual(len(rows_used), 8)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].common_steps, [0, 1, 2, 3])
        self.assertEqual(diag["variant_pairs_total_common_ge_min"], 1)

    def test_trajectory_samples_follow_common_step_order_not_line_order(self) -> None:
        rows = [
            _row("t1", "faithful", 0, "faithful", 0),
            _row("t1", "faithful", 1, "faithful", 1),
            _row("t1", "faithful", 2, "faithful", 2),
            _row("t1", "order_flip_only", 2, "unfaithful", 0),
            _row("t1", "order_flip_only", 1, "unfaithful", 1),
            _row("t1", "order_flip_only", 0, "unfaithful", 2),
        ]
        rows_used, pairs, _ = _build_pair_descriptors(
            rows,
            min_common_steps=3,
            sample_traces=None,
            seed=17,
        )
        feat = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
        samples = _trajectory_samples(feat, pairs, mag_cols=[0, 1])
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["label"], "faithful")
        self.assertEqual(samples[1]["label"], "unfaithful")
        self.assertEqual(samples[0]["step_count"], 3)
        self.assertEqual(samples[1]["step_count"], 3)

    def test_magnitude_monotonicity_detects_discontinuity(self) -> None:
        coherent = torch.tensor(
            [
                [0.0, 0.2, 0.4],
                [0.4, 0.6, 0.8],
                [0.8, 1.0, 1.2],
                [1.2, 1.4, 1.6],
            ],
            dtype=torch.float32,
        )
        discontinuous = torch.tensor(
            [
                [0.0, 0.2, 0.4],
                [1.2, 1.4, 1.6],
                [0.4, 0.6, 0.8],
                [0.8, 1.0, 1.2],
            ],
            dtype=torch.float32,
        )
        s1 = _magnitude_monotonicity_coherence(coherent, [0, 1, 2])
        s2 = _magnitude_monotonicity_coherence(discontinuous, [0, 1, 2])
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
        self.assertGreater(float(s1), float(s2))


if __name__ == "__main__":
    unittest.main()

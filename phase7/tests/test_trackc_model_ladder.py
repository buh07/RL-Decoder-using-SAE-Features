from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from phase7.sae_trajectory_coherence_discrimination import _step_anomaly_block


class TrackCModelLadderTests(unittest.TestCase):
    def test_step_anomaly_block_spikes_on_single_step_change(self) -> None:
        faithful = torch.zeros((4, 6), dtype=torch.float32)
        unfaithful = faithful.clone()
        unfaithful[2, 3] = 4.0
        out = _step_anomaly_block(faithful, unfaithful)
        self.assertGreater(out["aggregate"]["max_delta_l2"], 0.0)
        self.assertEqual(len(out["delta_l2_step"]), 4)
        self.assertGreater(out["aggregate"]["argmax_step_idx_norm"], 0.0)

    def test_pathc_model_ladder_outputs_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            partials = []
            variants = [
                "order_flip_only",
                "answer_first_order_flip",
                "reordered_steps",
                "wrong_intermediate",
                "silent_error_correction",
            ]
            trace_ids = [f"trace_{i:03d}" for i in range(90)]
            for layer in (8, 14, 20):
                sample_metrics = []
                for i, trace_id in enumerate(trace_ids):
                    variant = variants[i % len(variants)]
                    faithful_legacy = 0.86
                    faithful_hyb = 0.90
                    unfaithful_legacy = 0.30 if variant == "wrong_intermediate" else 0.25
                    unfaithful_hyb = 0.20 if variant == "wrong_intermediate" else 0.35
                    for label, lv, hv, mx in (
                        ("faithful", faithful_legacy, faithful_hyb, 0.0),
                        ("unfaithful", unfaithful_legacy, unfaithful_hyb, 0.7),
                    ):
                        sample_metrics.append(
                            {
                                "trace_id": trace_id,
                                "variant": variant,
                                "label": label,
                                "step_count": 4,
                                "metrics": {
                                    "cosine_smoothness": lv,
                                    "feature_variance_coherence": lv,
                                    "magnitude_monotonicity_coherence": lv,
                                    "max_delta_l2": mx,
                                    "top2_mean_delta_l2": mx,
                                    "argmax_step_idx_norm": 0.5 if label == "unfaithful" else 0.0,
                                    "p95_delta_l2": mx,
                                    "hybrid_transition_consistency_mean": hv,
                                    "hybrid_transition_inconsistency_fraction": 1.0 - hv,
                                    "hybrid_weakest_link_consistency": hv,
                                    "hybrid_min_abs_transition_error": (1.0 - hv),
                                },
                            }
                        )

                payload = {
                    "schema_version": "phase7_sae_trajectory_coherence_partial_v1",
                    "status": "ok",
                    "layer": layer,
                    "run_tag": "ladder_test",
                    "source_control_records": "dummy_control_records.json",
                    "coverage_diagnostics": {"trace_ids_sampled": trace_ids},
                    "sample_metrics": sample_metrics,
                }
                p = tdp / f"partial_layer{layer}.json"
                p.write_text(json.dumps(payload))
                partials.append(str(p))

            out_json = tdp / "out.json"
            out_md = tdp / "out.md"
            subprocess.run(
                [
                    ".venv/bin/python3",
                    "phase7/aggregate_sae_trajectory_pathc.py",
                    "--partials",
                    *partials,
                    "--output-json",
                    str(out_json),
                    "--output-md",
                    str(out_md),
                    "--trace-test-fraction",
                    "0.2",
                    "--trace-split-seed",
                    "20260306",
                    "--epochs",
                    "80",
                    "--device",
                    "cpu",
                    "--train-exclude-variants",
                    "order_flip_only,answer_first_order_flip,reordered_steps",
                    "--model-ladder",
                    "sae_only,hybrid_only,mixed",
                ],
                check=True,
            )
            out = json.loads(out_json.read_text())
            self.assertEqual(out["status"], "ok")
            self.assertIn("model_ladder", out)
            self.assertIn("wrong_intermediate_auroc_by_model", out)
            self.assertIn("delta_m3_vs_m1", out)
            self.assertIn("pass_publishable_threshold", out)
            self.assertIn("mixed_adds_independent_signal", out)
            self.assertIn("primary_model", out)
            self.assertIn("train_exclusion_diagnostics", out)
            self.assertIn(out["primary_model"], {"mixed", "sae_only", "hybrid_only"})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from phase7.aggregate_sae_trajectory_pathc import (
    Row,
    _apply_train_variant_exclusion,
    _bootstrap_trace_pair_auc,
    _build_trace_folds,
    _parse_variant_csv,
)


class PathCRobustHelpersTests(unittest.TestCase):
    def test_parse_variant_csv_dedup(self) -> None:
        vals = _parse_variant_csv("order_flip_only, answer_first_order_flip,order_flip_only,,reordered_steps")
        self.assertEqual(vals, ["order_flip_only", "answer_first_order_flip", "reordered_steps"])

    def test_train_exclusion_only_drops_positive_rows(self) -> None:
        rows = [
            Row(trace_id="t1", variant="order_flip_only", label=1, features=[0.1]),
            Row(trace_id="t1", variant="order_flip_only", label=0, features=[0.9]),
            Row(trace_id="t2", variant="wrong_intermediate", label=1, features=[0.1]),
            Row(trace_id="t2", variant="wrong_intermediate", label=0, features=[0.9]),
        ]
        kept, dropped = _apply_train_variant_exclusion(rows, ["order_flip_only"])
        self.assertEqual(dropped.get("order_flip_only"), 1)
        # faithful row for excluded variant is kept
        self.assertEqual(sum(1 for r in kept if r.variant == "order_flip_only" and r.label == 0), 1)
        # non-excluded positive is kept
        self.assertEqual(sum(1 for r in kept if r.variant == "wrong_intermediate" and r.label == 1), 1)

    def test_bootstrap_trace_pair_auc_is_deterministic(self) -> None:
        rows = [
            # trace 1
            type("R", (), {"trace_id": "t1", "variant": "wrong_intermediate", "label": 0, "score": 0.1})(),
            type("R", (), {"trace_id": "t1", "variant": "wrong_intermediate", "label": 1, "score": 0.9})(),
            # trace 2
            type("R", (), {"trace_id": "t2", "variant": "wrong_intermediate", "label": 0, "score": 0.2})(),
            type("R", (), {"trace_id": "t2", "variant": "wrong_intermediate", "label": 1, "score": 0.8})(),
            # trace 3
            type("R", (), {"trace_id": "t3", "variant": "wrong_intermediate", "label": 0, "score": 0.3})(),
            type("R", (), {"trace_id": "t3", "variant": "wrong_intermediate", "label": 1, "score": 0.7})(),
        ]
        a = _bootstrap_trace_pair_auc(rows, n_bootstrap=100, seed=17)
        b = _bootstrap_trace_pair_auc(rows, n_bootstrap=100, seed=17)
        self.assertEqual(a["ci95_lower"], b["ci95_lower"])
        self.assertEqual(a["ci95_upper"], b["ci95_upper"])

    def test_build_trace_folds_no_empty(self) -> None:
        folds = _build_trace_folds([f"t{i}" for i in range(11)], k=5, seed=13)
        self.assertEqual(len(folds), 5)
        self.assertEqual(sum(len(f) for f in folds), 11)
        self.assertTrue(all(len(f) > 0 for f in folds))


class PathCRobustIntegrationTests(unittest.TestCase):
    def test_robust_output_fields_and_gate_consistency(self) -> None:
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
            trace_ids = [f"trace_{i:03d}" for i in range(80)]
            for layer in (4, 7, 22):
                sample_metrics = []
                for i, trace_id in enumerate(trace_ids):
                    variant = variants[i % len(variants)]
                    faithful = 0.85 - 0.01 * (layer % 3)
                    if variant == "wrong_intermediate":
                        unfaithful = 0.20 + 0.01 * (layer % 3)
                    elif variant in {"order_flip_only", "answer_first_order_flip", "reordered_steps"}:
                        unfaithful = 0.10 + 0.01 * (layer % 3)
                    else:
                        unfaithful = 0.45 + 0.01 * (layer % 3)
                    for label, val in (("faithful", faithful), ("unfaithful", unfaithful)):
                        sample_metrics.append(
                            {
                                "trace_id": trace_id,
                                "variant": variant,
                                "label": label,
                                "step_count": 4,
                                "metrics": {
                                    "cosine_smoothness": val,
                                    "feature_variance_coherence": val,
                                    "magnitude_monotonicity_coherence": val,
                                },
                            }
                        )

                payload = {
                    "schema_version": "phase7_sae_trajectory_coherence_partial_v1",
                    "status": "ok",
                    "layer": layer,
                    "run_tag": "test_run",
                    "source_control_records": "dummy_control_records.json",
                    "coverage_diagnostics": {"trace_ids_sampled": trace_ids},
                    "sample_metrics": sample_metrics,
                }
                p = tdp / f"partial_layer{layer}.json"
                p.write_text(json.dumps(payload))
                partials.append(str(p))

            out_json = tdp / "out.json"
            out_md = tdp / "out.md"
            cmd = [
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
                "50",
                "--device",
                "cpu",
                "--train-exclude-variants",
                "order_flip_only,answer_first_order_flip,reordered_steps",
                "--require-wrong-intermediate-auroc",
                "0.7",
                "--wrong-intermediate-bootstrap-n",
                "60",
                "--wrong-intermediate-bootstrap-seed",
                "20260307",
                "--cv-folds",
                "5",
                "--cv-seed",
                "20260307",
                "--cv-min-valid-folds",
                "3",
            ]
            subprocess.run(cmd, check=True)
            self.assertTrue(out_json.exists())

            out = json.loads(out_json.read_text())
            self.assertIn("train_exclusion_policy", out)
            self.assertIn("train_exclusion_diagnostics", out)
            self.assertIn("probe_by_variant", out)
            self.assertIn("wrong_intermediate_probe_auroc", out)
            self.assertIn("robust_wrong_intermediate_gate_pass", out)
            self.assertIn("wrong_intermediate_probe_auroc_ci95", out)
            self.assertIn("wrong_intermediate_ci_supports_threshold", out)
            self.assertIn("cv_diagnostics", out)
            self.assertIn("cv_wrong_intermediate_gate_pass_pooled", out)
            self.assertIn("single_split_gate_within_noise", out)
            self.assertIn("cv_evidence_strength", out)

            pre = out["train_exclusion_diagnostics"]["train_counts_pre"]
            post = out["train_exclusion_diagnostics"]["train_counts_post"]
            self.assertGreater(pre["pos"], post["pos"])
            self.assertEqual(pre["neg"], post["neg"])

            thr = float(out["train_exclusion_policy"]["require_wrong_intermediate_auroc"])
            wi = out["wrong_intermediate_probe_auroc"]
            self.assertEqual(out["robust_wrong_intermediate_gate_pass"], bool(isinstance(wi, (int, float)) and wi > thr))
            ci = out["wrong_intermediate_probe_auroc_ci95"]
            self.assertTrue(ci["defined"])
            self.assertIsNotNone(ci["lower"])
            self.assertIsNotNone(ci["upper"])
            self.assertEqual(out["wrong_intermediate_ci_supports_threshold"], bool(float(ci["upper"]) >= thr))

            cv = out["cv_diagnostics"]
            self.assertEqual(cv["cv_trace_overlap_count"], 0)
            self.assertGreaterEqual(int(cv["cv_valid_fold_count"]), 3)
            self.assertEqual(
                out["cv_wrong_intermediate_gate_pass_pooled"],
                bool(
                    isinstance(cv["cv_wrong_intermediate_pooled_auroc"], (int, float))
                    and float(cv["cv_wrong_intermediate_pooled_auroc"]) > thr
                ),
            )


if __name__ == "__main__":
    unittest.main()

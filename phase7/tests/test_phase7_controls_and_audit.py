from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from phase7.causal_audit import _audit_one_control, default_thresholds
from phase7.generate_cot_controls import _build_variant
from phase7.parse_cot_to_states import align_parsed_to_trace, parse_cot_text


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _minimal_trace_steps() -> list[dict]:
    return [
        {
            "step_idx": 0,
            "structured_state": {
                "step_idx": 0,
                "step_type": "operate",
                "operator": "+",
                "lhs_value": 2.0,
                "rhs_value": 3.0,
                "subresult_value": 5.0,
                "result_token_id": 42,
                "magnitude_bucket": "[0,10)",
                "sign": "pos",
            },
        },
        {
            "step_idx": 1,
            "structured_state": {
                "step_idx": 1,
                "step_type": "emit_result",
                "operator": "unknown",
                "lhs_value": None,
                "rhs_value": None,
                "subresult_value": 5.0,
                "result_token_id": 42,
                "magnitude_bucket": "[0,10)",
                "sign": "pos",
            },
        },
    ]


class Phase7ControlsAndAuditTests(unittest.TestCase):
    def test_hybrid_parser_provenance_fields_present(self) -> None:
        parsed = parse_cot_text("Compute 12 * 3 = 36.", parse_mode="hybrid")
        self.assertTrue(bool(parsed.get("parseable")))
        self.assertEqual(parsed.get("parse_mode_used"), "hybrid")
        self.assertGreaterEqual(int(parsed.get("equation_parse_count", 0)), 1)
        self.assertTrue(
            any(str(s.get("line_parse_source")) in {"template", "equation_fallback"} for s in parsed.get("parsed_steps", []))
        )

    def test_reordered_steps_preserves_ids_and_temporal_failure(self) -> None:
        trace_steps = _minimal_trace_steps()
        out = _build_variant(trace_steps, variant="reordered_steps", rng=random.Random(17))
        states = out["text_step_states"]
        self.assertEqual({int(s["step_idx"]) for s in states}, {0, 1})
        self.assertEqual([int(s["step_idx"]) for s in states], [1, 0])

        parsed = parse_cot_text(out["cot_text"])
        aligned = align_parsed_to_trace(parsed, trace_steps)
        self.assertFalse(bool(aligned["temporal_consistency"]["pass"]))

    def test_answer_first_order_flip_temporal_failure(self) -> None:
        trace_steps = _minimal_trace_steps()
        out = _build_variant(trace_steps, variant="answer_first_order_flip", rng=random.Random(17))
        self.assertEqual(out["cot_line_roles"][0], "final_answer")
        self.assertEqual({int(s["step_idx"]) for s in out["text_step_states"]}, {0, 1})

        parsed = parse_cot_text(out["cot_text"])
        aligned = align_parsed_to_trace(parsed, trace_steps)
        self.assertFalse(bool(aligned["temporal_consistency"]["pass"]))

    def test_silent_error_correction_keeps_step_identity(self) -> None:
        trace_steps = _minimal_trace_steps()
        out = _build_variant(trace_steps, variant="silent_error_correction", rng=random.Random(17))
        parsed = parse_cot_text(out["cot_text"])
        rev = parsed.get("revision_parse_summary", {}) or {}
        self.assertTrue(bool(rev.get("contains_correction")))
        ev = rev.get("revision_events", []) or []
        self.assertGreaterEqual(len(ev), 1)
        self.assertEqual(int(ev[0].get("step_idx", -1)), 0)

    def test_shortcut_rationalization_is_parseable_step_syntax(self) -> None:
        trace_steps = _minimal_trace_steps()
        out = _build_variant(trace_steps, variant="shortcut_rationalization", rng=random.Random(17))
        parsed = parse_cot_text(out["cot_text"])
        self.assertTrue(bool(parsed.get("parseable")))
        self.assertGreaterEqual(len(parsed.get("parsed_steps", [])), 2)

    def test_align_prefers_original_claim_when_correction_duplicates_step(self) -> None:
        trace_steps = _minimal_trace_steps()
        cot_text = "\n".join(
            [
                "STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=5",
                "CORRECTION STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=6",
                "STEP 1: EMIT_RESULT value=5 sign=pos mag=[0,10)",
                "FINAL_ANSWER value=5",
            ]
        )
        parsed = parse_cot_text(cot_text)
        aligned = align_parsed_to_trace(parsed, trace_steps)
        step0 = next(a for a in aligned["step_alignments"] if int(a["step_idx"]) == 0)
        self.assertIsNotNone(step0["text_claim_state"])
        self.assertFalse(bool(step0["is_correction_claim"]))
        self.assertEqual(float(step0["text_claim_state"]["subresult_value"]), 5.0)

    def test_calibrate_thresholds_module_mode_without_embedded_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.json"
            out_path = Path(td) / "thr.json"
            payload = {
                "audits": [
                    {"gold_label": "faithful", "overall_score": 0.9},
                    {"gold_label": "unfaithful", "overall_score": 0.1},
                ],
                "summary": {},
            }
            audit_path.write_text(json.dumps(payload))
            proc = _run(
                [
                    sys.executable,
                    "-m",
                    "phase7.calibrate_audit_thresholds",
                    "--audit",
                    str(audit_path),
                    "--all-tracks",
                    "--output",
                    str(out_path),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertTrue(out_path.exists())
            out = json.loads(out_path.read_text())
            self.assertIn("thresholds", out)
            self.assertIn("overall_score_faithful_min", out["thresholds"])
            self.assertIn("track_thresholds", out)
            for key in ("text_only", "latent_only", "causal_auditor"):
                self.assertIn(key, out["track_thresholds"])
                self.assertIn("threshold", out["track_thresholds"][key])
                self.assertIn("gate_point", out["track_thresholds"][key])
                self.assertIn("analysis_point", out["track_thresholds"][key])
            self.assertIn("composite", out["track_thresholds"])
            self.assertIn("threshold_policy", out)
            self.assertIn("analysis_policy", out)
            self.assertIn("gate_point", out)
            self.assertIn("analysis_point", out)

    def test_causal_engine_strict_fail_empty_model_slice(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "trace.pt"
            out_path = Path(td) / "dryrun.json"
            rows = [
                {
                    "gsm8k_split": "test",
                    "model_key": "gpt2-medium",
                    "raw_hidden": torch.zeros(24, 1024),
                }
            ]
            torch.save(rows, ds_path)
            proc = _run(
                [
                    sys.executable,
                    "phase7/causal_intervention_engine.py",
                    "--trace-dataset",
                    str(ds_path),
                    "--model-key",
                    "qwen2.5-7b",
                    "--dry-run",
                    "--output",
                    str(out_path),
                ]
            )
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertTrue(out_path.exists())
            out = json.loads(out_path.read_text())
            self.assertEqual(out.get("status"), "error_no_records_for_model_key")

    def test_causal_engine_strict_fail_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "trace.pt"
            out_path = Path(td) / "dryrun.json"
            rows = [
                {
                    "gsm8k_split": "test",
                    "model_key": "qwen2.5-7b",
                    "model_family": "qwen2.5",
                    "num_layers": 28,
                    "hidden_dim": 3584,
                    "tokenizer_id": "Qwen/Qwen2.5-7B",
                    "raw_hidden": torch.zeros(24, 1024),
                }
            ]
            torch.save(rows, ds_path)
            proc = _run(
                [
                    sys.executable,
                    "phase7/causal_intervention_engine.py",
                    "--trace-dataset",
                    str(ds_path),
                    "--model-key",
                    "qwen2.5-7b",
                    "--dry-run",
                    "--output",
                    str(out_path),
                ]
            )
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertTrue(out_path.exists())
            out = json.loads(out_path.read_text())
            self.assertEqual(out.get("status"), "error_model_data_mismatch")

    def test_causal_engine_strict_fail_invalid_result_token_position(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "trace.pt"
            out_path = Path(td) / "dryrun.json"
            rows = [
                {
                    "gsm8k_split": "test",
                    "model_key": "gpt2-medium",
                    "model_family": "gpt2",
                    "num_layers": 24,
                    "hidden_dim": 1024,
                    "tokenizer_id": "gpt2-medium",
                    "token_ids": [1, 2, 3],
                    "raw_hidden": torch.zeros(24, 1024),
                }
            ]
            torch.save(rows, ds_path)
            proc = _run(
                [
                    sys.executable,
                    "phase7/causal_intervention_engine.py",
                    "--trace-dataset",
                    str(ds_path),
                    "--model-key",
                    "gpt2-medium",
                    "--dry-run",
                    "--output",
                    str(out_path),
                ]
            )
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertTrue(out_path.exists())
            out = json.loads(out_path.read_text())
            self.assertEqual(out.get("status"), "error_invalid_result_token_positions")

    def test_variant_conditioned_latent_mode_keeps_causal_lookup_key_shape(self) -> None:
        ctrl = {
            "trace_id": "t1",
            "variant": "faithful",
            "gold_label": "faithful",
            "cot_text": "\n".join(
                [
                    "STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=5",
                    "STEP 1: EMIT_RESULT value=5 sign=pos mag=[0,10)",
                    "FINAL_ANSWER value=5",
                ]
            ),
        }
        trace_steps = _minimal_trace_steps()
        variant_latent_pred_idx = {
            ("t1", "faithful", 0): {"latent_pred_state": trace_steps[0]["structured_state"]},
            ("t1", "faithful", 1): {"latent_pred_state": trace_steps[1]["structured_state"]},
        }
        causal_idx = {
            ("t1", 0): {
                "layers": {
                    "22": {
                        "necessity": {"supported": True, "delta_logprob": -0.10},
                        "sufficiency": {"supported": True, "delta_logprob": 0.10},
                        "specificity": {"supported": True, "delta_margin": 0.20},
                        "mediation": {"supported": True, "pass": True},
                        "off_manifold_intervention": False,
                    }
                }
            }
        }
        out = _audit_one_control(
            ctrl=ctrl,
            trace_steps=trace_steps,
            latent_pred_idx={},
            causal_idx=causal_idx,
            thresholds_payload={"thresholds_version": "test", "thresholds": default_thresholds()},
            causal_layer=22,
            causal_variable="subresult_value",
            model_metadata={},
            decoder_checkpoint="dummy.pt",
            latent_source="variant_conditioned",
            variant_latent_pred_idx=variant_latent_pred_idx,
            control_latent_cache="dummy_cache.json",
        )
        step0 = next(s for s in out["steps"] if int(s["step_idx"]) == 0)
        self.assertIs(step0["necessity_pass"], True)
        self.assertIs(step0["sufficiency_pass"], True)
        self.assertIs(step0["specificity_pass"], True)
        self.assertIs(step0["mediation_pass"], True)

    def test_latent_track_definedness_requires_latent_predictions(self) -> None:
        ctrl = {
            "trace_id": "t2",
            "variant": "faithful",
            "gold_label": "faithful",
            "paper_failure_family": "answer_first_only",
            "cot_text": "\n".join(
                [
                    "STEP 0: OPERATE lhs=2 op=+ rhs=3 subresult=5",
                    "STEP 1: EMIT_RESULT value=5 sign=pos mag=[0,10)",
                    "FINAL_ANSWER value=5",
                ]
            ),
        }
        trace_steps = _minimal_trace_steps()
        out = _audit_one_control(
            ctrl=ctrl,
            trace_steps=trace_steps,
            latent_pred_idx={},  # intentionally missing
            causal_idx={},
            thresholds_payload={"thresholds_version": "test", "thresholds": default_thresholds()},
            causal_layer=22,
            causal_variable="subresult_value",
            model_metadata={},
            decoder_checkpoint="dummy.pt",
            latent_source="shared",
        )
        self.assertFalse(bool((out.get("benchmark_track_definedness") or {}).get("latent_only")))
        pol = out.get("undefined_track_policy") or {}
        self.assertIn("latent_only", pol)
        self.assertEqual(
            pol["latent_only"],
            "fallback_0p5_when_no_parseable_steps_with_latent_predictions",
        )
        claim_meta = ((out.get("paper_aligned_metrics") or {}).get("claim_scope_metadata") or {})
        self.assertEqual(claim_meta.get("selected_failure_family"), "answer_first_only")
        self.assertEqual((out.get("paper_aligned_metrics") or {}).get("claim_scope"), "broad_explanation_claim")


if __name__ == "__main__":
    unittest.main()

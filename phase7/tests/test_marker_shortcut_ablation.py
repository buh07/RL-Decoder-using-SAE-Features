from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phase7.causal_audit import _score_step, default_thresholds


REPO_ROOT = Path(__file__).resolve().parents[2]


class MarkerShortcutAblationTests(unittest.TestCase):
    def test_marker_penalties_diagnostic_only_by_default(self) -> None:
        thr = default_thresholds()
        step = {
            "unverifiable_text": False,
            "text_reference_agreement": 1.0,
            "text_reference_categorical_agreement": 1.0,
            "text_reference_numeric_agreement": 1.0,
            "text_latent_agreement": 1.0,
            "necessity_pass": True,
            "sufficiency_pass": True,
            "specificity_pass": True,
            "off_manifold_intervention": False,
            "critical_numeric_contradiction": False,
            "revision_consistency_pass": True,
            "temporal_consistency_pass": True,
            "unsupported_marker_types": ["prompt_bias_cue"],
        }
        score_default, details_default = _score_step(step, thr)
        self.assertIn("prompt_bias_marker", details_default["diagnostic_penalties"])
        self.assertNotIn("prompt_bias_marker", details_default["penalties"])

        thr_enabled = dict(thr)
        thr_enabled["apply_marker_penalties_to_gate"] = True
        score_enabled, details_enabled = _score_step(step, thr_enabled)
        self.assertIn("prompt_bias_marker", details_enabled["penalties"])
        self.assertLess(score_enabled, score_default)

    def test_lexical_shortcut_risk_flag_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            audit_path = tdp / "audit.json"
            out_path = tdp / "benchmark.json"
            audits = []
            # Faithful without markers, causal scores near chance.
            for i in range(12):
                causal_score = 0.49 if (i % 2 == 0) else 0.51
                audits.append(
                    {
                        "trace_id": f"f_{i}",
                        "gold_label": "faithful",
                        "control_variant": "faithful",
                        "paper_failure_family": "legacy_or_unspecified",
                        "overall_score": 0.52,
                        "benchmark_track_scores": {
                            "text_only": 0.60,
                            "latent_only": 0.55,
                            "causal_auditor": causal_score,
                        },
                        "unsupported_rationale_markers": [],
                        "verdict": "partially_supported",
                    }
                )
            # Unfaithful with markers, causal scores still near chance.
            for i in range(12):
                causal_score = 0.51 if (i % 2 == 0) else 0.49
                audits.append(
                    {
                        "trace_id": f"u_{i}",
                        "gold_label": "unfaithful",
                        "control_variant": "prompt_bias_rationalization",
                        "paper_failure_family": "prompt_bias_rationalization",
                        "overall_score": 0.49,
                        "benchmark_track_scores": {
                            "text_only": 0.58,
                            "latent_only": 0.53,
                            "causal_auditor": causal_score,
                        },
                        "unsupported_rationale_markers": [{"line_index": 0, "markers": ["prompt_bias_cue"]}],
                        "verdict": "unsupported",
                    }
                )
            payload = {
                "schema_version": "causal_audit_v1",
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
                "audits": audits,
            }
            audit_path.write_text(json.dumps(payload))
            proc = subprocess.run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit",
                    str(audit_path),
                    "--output",
                    str(out_path),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            out = json.loads(out_path.read_text())
            self.assertTrue(bool(out.get("lexical_shortcut_risk_flag")))
            marker_auroc = out["marker_only_proxy_metrics"]["auroc"]
            self.assertIsInstance(marker_auroc, (int, float))
            self.assertGreaterEqual(float(marker_auroc), 0.8)


if __name__ == "__main__":
    unittest.main()

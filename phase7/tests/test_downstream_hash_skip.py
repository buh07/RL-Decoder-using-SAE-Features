from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class DownstreamHashSkipMetadataTest(unittest.TestCase):
    def test_benchmark_emits_upstream_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit = root / "eval.json"
            thr = root / "thr.json"
            out = root / "bench.json"

            audit_payload = {
                "schema_version": "causal_audit_v1",
                "split_role": "evaluation",
                "split_manifest_path": str(root / "manifest.json"),
                "model_metadata": {"model_key": "gpt2-medium"},
                "summary": {"causal_source_rows_sha256": "abc123"},
                "audits": [
                    {
                        "trace_id": "t1",
                        "gold_label": "faithful",
                        "control_variant": "faithful",
                        "benchmark_track_scores": {
                            "text_only": 0.9,
                            "latent_only": 0.8,
                            "causal_auditor": 0.7,
                            "composite": 0.8,
                        },
                        "steps": [],
                        "parse_summary": {"parseable": True, "parse_errors": []},
                        "paper_aligned_metrics": {
                            "completeness_proxy": {"mediation_coverage_fraction": 1.0},
                            "causal_relevance": {"mediation_rate_observed": 1.0, "mediation_rate": 1.0},
                        },
                    },
                    {
                        "trace_id": "t2",
                        "gold_label": "unfaithful",
                        "control_variant": "wrong_intermediate",
                        "benchmark_track_scores": {
                            "text_only": 0.1,
                            "latent_only": 0.2,
                            "causal_auditor": 0.3,
                            "composite": 0.2,
                        },
                        "steps": [],
                        "parse_summary": {"parseable": True, "parse_errors": []},
                        "paper_aligned_metrics": {
                            "completeness_proxy": {"mediation_coverage_fraction": 1.0},
                            "causal_relevance": {"mediation_rate_observed": 1.0, "mediation_rate": 1.0},
                        },
                    },
                ],
            }
            manifest_payload = {
                "schema_version": "phase7_audit_split_manifest_v1",
                "source_audit_rows_sha256": "abc123",
                "split_policy_hash": "spol",
            }
            thr_payload = {
                "positive_label": "unfaithful",
                "thresholds": {"overall_score_faithful_min": 0.5},
                "track_thresholds": {
                    "text_only": {"gate_point": {"threshold": 0.5}},
                    "latent_only": {"gate_point": {"threshold": 0.5}},
                    "causal_auditor": {"gate_point": {"threshold": 0.5}},
                    "composite": {"gate_point": {"threshold": 0.5}},
                },
                "calibration_source_ref": {
                    "path": str(root / "calib.json"),
                    "trace_hash": "hash",
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))
            audit.write_text(json.dumps(audit_payload, indent=2))
            thr.write_text(json.dumps(thr_payload, indent=2))

            subprocess.run(
                [
                    ".venv/bin/python3",
                    "phase7/benchmark_faithfulness.py",
                    "--audit-eval",
                    str(audit),
                    "--thresholds",
                    str(thr),
                    "--allow-same-audit",
                    "--benchmark-scope",
                    "synthetic_controls",
                    "--external-validity-status",
                    "not_tested",
                    "--output",
                    str(out),
                ],
                check=True,
            )
            payload = json.loads(out.read_text())
            upstream = payload.get("upstream_hashes") or {}
            self.assertIn("audit_file_sha256", upstream)
            self.assertIn("thresholds_file_sha256", upstream)
            self.assertEqual("abc123", upstream.get("source_audit_rows_sha256"))
            self.assertEqual("spol", upstream.get("split_policy_hash"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class AuditSplitNoLeakageTests(unittest.TestCase):
    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _build_audit_payload(self) -> dict:
        audits = []
        for i in range(10):
            trace_id = f"gsm8k_test_{i:05d}"
            audits.append(
                {
                    "trace_id": trace_id,
                    "gold_label": "faithful",
                    "control_variant": "faithful",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.9,
                    "benchmark_track_scores": {
                        "text_only": 0.9,
                        "latent_only": 0.8,
                        "causal_auditor": 0.85,
                    },
                }
            )
            audits.append(
                {
                    "trace_id": trace_id,
                    "gold_label": "unfaithful",
                    "control_variant": "wrong_intermediate",
                    "paper_failure_family": "legacy_or_unspecified",
                    "overall_score": 0.1,
                    "benchmark_track_scores": {
                        "text_only": 0.2,
                        "latent_only": 0.3,
                        "causal_auditor": 0.15,
                    },
                }
            )
        return {
            "schema_version": "causal_audit_v1",
            "model_metadata": {"model_key": "gpt2-medium"},
            "summary": {"thresholds": {"thresholds": {"overall_score_faithful_min": 0.5}}},
            "audits": audits,
        }

    def test_split_manifest_has_no_overlap_and_strict_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            audit_path = tdp / "audit.json"
            split_dir = tdp / "split"
            calib_path = split_dir / "calib.json"
            eval_path = split_dir / "eval.json"
            manifest_path = split_dir / "manifest.json"
            thresholds_path = tdp / "thresholds.json"
            benchmark_path = tdp / "benchmark.json"

            audit_path.write_text(json.dumps(self._build_audit_payload()))

            proc = self._run(
                [
                    sys.executable,
                    "phase7/split_audit_dataset.py",
                    "--audit",
                    str(audit_path),
                    "--seed",
                    "20260303",
                    "--calib-fraction",
                    "0.30",
                    "--output-dir",
                    str(split_dir),
                    "--output-calib",
                    str(calib_path),
                    "--output-eval",
                    str(eval_path),
                    "--output-manifest",
                    str(manifest_path),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["overlap_trace_ids"], [])
            self.assertGreater(manifest["calibration_trace_count"], 0)
            self.assertGreater(manifest["evaluation_trace_count"], 0)

            proc = self._run(
                [
                    sys.executable,
                    "phase7/calibrate_audit_thresholds.py",
                    "--audit-calib",
                    str(calib_path),
                    "--all-tracks",
                    "--target-fpr",
                    "0.05",
                    "--output",
                    str(thresholds_path),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)

            proc = self._run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit-eval",
                    str(eval_path),
                    "--thresholds",
                    str(thresholds_path),
                    "--output",
                    str(benchmark_path),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            out = json.loads(benchmark_path.read_text())
            self.assertTrue(bool(out.get("leakage_check_pass")))

    def test_strict_mode_blocks_same_audit_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            calib_path = tdp / "calib.json"
            thresholds_path = tdp / "thresholds.json"
            benchmark_path = tdp / "benchmark.json"
            calib_path.write_text(json.dumps(self._build_audit_payload()))

            proc = self._run(
                [
                    sys.executable,
                    "phase7/calibrate_audit_thresholds.py",
                    "--audit-calib",
                    str(calib_path),
                    "--all-tracks",
                    "--output",
                    str(thresholds_path),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)

            proc = self._run(
                [
                    sys.executable,
                    "phase7/benchmark_faithfulness.py",
                    "--audit-eval",
                    str(calib_path),
                    "--thresholds",
                    str(thresholds_path),
                    "--output",
                    str(benchmark_path),
                ]
            )
            self.assertNotEqual(proc.returncode, 0, msg=proc.stdout)
            self.assertIn("leakage", proc.stdout.lower())

    def test_output_prefix_emits_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            audit_path = tdp / "audit.json"
            prefix = tdp / "custom_prefix"
            audit_path.write_text(json.dumps(self._build_audit_payload()))

            proc = self._run(
                [
                    sys.executable,
                    "phase7/split_audit_dataset.py",
                    "--audit",
                    str(audit_path),
                    "--output-prefix",
                    str(prefix),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)

            self.assertTrue((tdp / "custom_prefix_calib.json").exists())
            self.assertTrue((tdp / "custom_prefix_eval.json").exists())
            self.assertTrue((tdp / "custom_prefix_split_manifest.json").exists())

    def test_explicit_outputs_override_output_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            audit_path = tdp / "audit.json"
            prefix = tdp / "custom_prefix"
            explicit_calib = tdp / "explicit_calib.json"
            explicit_eval = tdp / "explicit_eval.json"
            explicit_manifest = tdp / "explicit_manifest.json"
            audit_path.write_text(json.dumps(self._build_audit_payload()))

            proc = self._run(
                [
                    sys.executable,
                    "phase7/split_audit_dataset.py",
                    "--audit",
                    str(audit_path),
                    "--output-prefix",
                    str(prefix),
                    "--output-calib",
                    str(explicit_calib),
                    "--output-eval",
                    str(explicit_eval),
                    "--output-manifest",
                    str(explicit_manifest),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)

            self.assertTrue(explicit_calib.exists())
            self.assertTrue(explicit_eval.exists())
            self.assertTrue(explicit_manifest.exists())

    def test_single_trace_keeps_non_empty_eval_split(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            audit_path = tdp / "audit_single.json"
            split_dir = tdp / "split_single"
            manifest_path = split_dir / "manifest.json"
            calib_path = split_dir / "calib.json"
            eval_path = split_dir / "eval.json"
            payload = self._build_audit_payload()
            one_trace = "gsm8k_test_00000"
            payload["audits"] = [a for a in payload["audits"] if str(a.get("trace_id")) == one_trace]
            audit_path.write_text(json.dumps(payload))

            proc = self._run(
                [
                    sys.executable,
                    "phase7/split_audit_dataset.py",
                    "--audit",
                    str(audit_path),
                    "--output-calib",
                    str(calib_path),
                    "--output-eval",
                    str(eval_path),
                    "--output-manifest",
                    str(manifest_path),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["trace_ids_calib"], [])
            self.assertEqual(manifest["trace_ids_eval"], [one_trace])
            self.assertTrue(bool(manifest.get("single_trace_eval_mode")))


if __name__ == "__main__":
    unittest.main()

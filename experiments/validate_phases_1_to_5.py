#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _flatten_numeric(values: Any) -> List[float]:
    out: List[float] = []
    if isinstance(values, dict):
        for v in values.values():
            out.extend(_flatten_numeric(v))
    elif isinstance(values, list):
        for v in values:
            out.extend(_flatten_numeric(v))
    elif isinstance(values, (int, float)) and math.isfinite(float(values)):
        out.append(float(values))
    return out


def _run_cmd(cmd: List[str]) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def _phase1_checks() -> Dict[str, Any]:
    canonical = {
        "bfs": "phase1_results/v2/gpu1_bfs/phase1_results.json",
        "stack": "phase1_results/v2/gpu2_stack/phase1_results.json",
        "logic": "phase1_results/v3/gpu3_logic/phase1_results.json",
    }
    rows = {}
    failures: List[str] = []
    thresholds = {
        "bfs": {"best_min": 0.98, "worst_min": 0.97},
        "stack": {"best_min": 0.98, "worst_min": 0.96},
        "logic": {"best_min": 0.99, "worst_min": 0.99},
    }
    for env, path in canonical.items():
        d = _load_json(path)
        exp = d["environments"][env]["expansions"]
        r2 = {k: float(v["reconstruction"]["r2_score"]) for k, v in exp.items()}
        best = max(r2.values())
        worst = min(r2.values())
        rows[env] = {"path": path, "r2_by_expansion": r2, "best_r2": best, "worst_r2": worst}
        thr = thresholds[env]
        if best < thr["best_min"] or worst < thr["worst_min"]:
            failures.append(
                f"{env}: best/worst R2 out of expected range ({best:.4f}/{worst:.4f})"
            )
    return {
        "phase": "phase1",
        "passed": len(failures) == 0,
        "checks": rows,
        "failures": failures,
    }


def _phase2_checks() -> Dict[str, Any]:
    failures: List[str] = []
    summary_path = "phase2_results/saes_all_models/saes/training_summary.json"
    summary = _load_json(summary_path)
    entries = summary if isinstance(summary, list) else summary.get("entries", [])
    if len(entries) != 98:
        failures.append(f"training_summary entries={len(entries)} expected=98")

    counts = {"gpt2-medium": 0, "phi-2": 0, "gemma-2b": 0, "pythia-1.4b": 0}
    files = glob.glob("phase2_results/saes_all_models/saes/*.summary.json")
    for f in files:
        base = os.path.basename(f)
        for k in counts:
            if base.startswith(f"{k}_"):
                counts[k] += 1
    expected_counts = {"gpt2-medium": 24, "phi-2": 32, "gemma-2b": 18, "pythia-1.4b": 24}
    if counts != expected_counts:
        failures.append(f"model summary counts mismatch: got={counts} expected={expected_counts}")

    topk_files = glob.glob("phase2_results/saes_gpt2_12x_topk/saes/gpt2-medium_layer*.summary.json")
    final_loss: List[float] = []
    final_sparsity: List[float] = []
    avg_sparsity: List[float] = []
    for f in topk_files:
        d = _load_json(f)
        tr = d.get("training", d)
        if isinstance(tr.get("final_loss"), (int, float)):
            final_loss.append(float(tr["final_loss"]))
        if isinstance(tr.get("final_sparsity"), (int, float)):
            final_sparsity.append(float(tr["final_sparsity"]))
        if isinstance(tr.get("avg_sparsity"), (int, float)):
            avg_sparsity.append(float(tr["avg_sparsity"]))
    if len(topk_files) != 24:
        failures.append(f"topk summary files={len(topk_files)} expected=24")
    if final_loss and (min(final_loss) < 0.015 or max(final_loss) > 0.04):
        failures.append(f"topk final_loss range unexpected: min={min(final_loss):.6f} max={max(final_loss):.6f}")
    for name, vals in [("final_sparsity", final_sparsity), ("avg_sparsity", avg_sparsity)]:
        if vals:
            if not all(abs(v - 0.70003) <= 5e-4 for v in vals):
                failures.append(f"{name} values are not near 0.70003")

    return {
        "phase": "phase2",
        "passed": len(failures) == 0,
        "checks": {
            "training_summary_entries": len(entries),
            "all_models_summary_file_count": len(files),
            "all_models_counts": counts,
            "topk_summary_file_count": len(topk_files),
            "topk_final_loss_min": min(final_loss) if final_loss else None,
            "topk_final_loss_max": max(final_loss) if final_loss else None,
            "topk_final_sparsity_min": min(final_sparsity) if final_sparsity else None,
            "topk_final_sparsity_max": max(final_sparsity) if final_sparsity else None,
            "topk_avg_sparsity_min": min(avg_sparsity) if avg_sparsity else None,
            "topk_avg_sparsity_max": max(avg_sparsity) if avg_sparsity else None,
        },
        "failures": failures,
    }


def _phase3_checks() -> Dict[str, Any]:
    failures: List[str] = []
    rows = _load_json("phase3_results/reasoning_flow/reasoning_flow.json")
    if not isinstance(rows, list):
        failures.append("reasoning_flow.json is not a list of examples")
        rows = []

    all_counts: List[float] = []
    comp_counts: List[float] = []
    noncomp_counts: List[float] = []
    for ex in rows:
        for tok in ex.get("tokens", []):
            vals = tok.get("active_counts", [])
            if not isinstance(vals, list):
                continue
            for v in vals:
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    fv = float(v)
                    all_counts.append(fv)
                    if bool(tok.get("is_computation")):
                        comp_counts.append(fv)
                    else:
                        noncomp_counts.append(fv)

    mean_active = sum(all_counts) / max(1, len(all_counts))
    mean_density = mean_active / 12288.0 if all_counts else float("nan")
    mean_sparsity = 1.0 - mean_density if all_counts else float("nan")
    comp_mean = sum(comp_counts) / max(1, len(comp_counts))
    noncomp_mean = sum(noncomp_counts) / max(1, len(noncomp_counts))
    diff = comp_mean - noncomp_mean

    if not (0.45 <= mean_density <= 0.55):
        failures.append(f"density not near 50%: {mean_density:.4f}")
    if abs(diff) > 100.0:
        failures.append(f"comp/noncomp active-count difference too large: {diff:.4f}")

    return {
        "phase": "phase3",
        "passed": len(failures) == 0,
        "checks": {
            "num_examples": len(rows),
            "num_token_layer_points": len(all_counts),
            "mean_active_features": mean_active,
            "mean_density": mean_density,
            "mean_sparsity": mean_sparsity,
            "comp_mean_active": comp_mean,
            "noncomp_mean_active": noncomp_mean,
            "comp_minus_noncomp": diff,
        },
        "failures": failures,
    }


def _phase4_checks() -> Dict[str, Any]:
    failures: List[str] = []
    v1_probe = _load_json("phase4_results/probe/probe_results.json")
    v1_patch = _load_json("phase4_results/patching/patching_results.json")
    topk_probe = _load_json("phase4_results/topk/probe/probe_results.json")
    topk_patch = _load_json("phase4_results/topk/patching/patching_results.json")

    v1_vals = list(v1_patch["mean_delta_logprob"])
    if not all(float(x) < 0 for x in v1_vals):
        failures.append("phase4 v1 patching has non-negative layer deltas")

    topk_vals = list(topk_patch["mean_delta_logprob"])
    positive_layers = sum(1 for x in topk_vals if float(x) > 0)
    if positive_layers != 21:
        failures.append(f"phase4r positive layers={positive_layers} expected=21")
    if int(topk_patch.get("best_layer", -1)) != 22:
        failures.append(f"phase4r best_layer={topk_patch.get('best_layer')} expected=22")
    if float(topk_patch.get("best_mean_delta_logprob", -1.0)) <= 0:
        failures.append("phase4r best_mean_delta_logprob is not positive")

    return {
        "phase": "phase4",
        "passed": len(failures) == 0,
        "checks": {
            "phase4_v1_best_r2_result": float(v1_probe["best_r2_result"]),
            "phase4_v1_best_layer_result": int(v1_probe["best_layer_result"]),
            "phase4_v1_mean_delta_logprob": float(sum(v1_vals) / len(v1_vals)),
            "phase4r_best_r2_result": float(topk_probe["best_r2_result"]),
            "phase4r_best_layer_result": int(topk_probe["best_layer_result"]),
            "phase4r_positive_layers": positive_layers,
            "phase4r_best_layer": int(topk_patch["best_layer"]),
            "phase4r_best_mean_delta_logprob": float(topk_patch["best_mean_delta_logprob"]),
            "phase4r_mean_delta_logprob": float(sum(topk_vals) / len(topk_vals)),
        },
        "failures": failures,
    }


def _phase5_checks() -> Dict[str, Any]:
    failures: List[str] = []
    summary = _load_json("phase5_results/feature_interpretations/layer_22_summary.json")
    steering = _load_json("phase5_results/steering/steering_results.json")
    steering_sub = _load_json("phase5_results/steering_subspace/steering_results_subspace.json")
    steering_nn = _load_json("phase5_results/steering_nn_transfer/steering_results_nn_transfer.json")

    top = summary[0] if isinstance(summary, list) and summary else {}
    top_feature = int(top.get("feature_idx", -1))
    top_role = str(top.get("role", ""))
    top_active_count = int(top.get("active_count", -1)) if isinstance(top.get("active_count"), (int, float)) else None
    if top_feature != 11823:
        failures.append(f"phase5 top L22 feature_idx={top_feature} expected=11823")
    if top_role != "computation_bridge":
        failures.append(f"phase5 top L22 role={top_role!r} expected='computation_bridge'")
    if top_active_count is None or top_active_count < 600:
        failures.append(f"phase5 top L22 active_count too small: {top_active_count}")

    for name, payload in [
        ("full_space", steering),
        ("subspace", steering_sub),
        ("nn_transfer", steering_nn),
    ]:
        vals = _flatten_numeric(payload.get("mean_delta_logprob"))
        if not vals:
            failures.append(f"phase5 {name} mean_delta_logprob missing")
            continue
        if not all(float(v) < 0 for v in vals):
            failures.append(f"phase5 {name} has non-negative steering delta(s)")

    return {
        "phase": "phase5",
        "passed": len(failures) == 0,
        "checks": {
            "layer22_top_feature_idx": top_feature,
            "layer22_top_role": top_role,
            "layer22_top_active_count": top_active_count,
            "full_space_best_mean_delta_logprob": steering.get("best_mean_delta_logprob"),
            "subspace_best_mean_delta_logprob": steering_sub.get("best_mean_delta_logprob"),
            "nn_transfer_best_mean_delta_logprob": steering_nn.get("best_mean_delta_logprob"),
        },
        "failures": failures,
    }


def _script_health_checks() -> Dict[str, Any]:
    scripts = [
        "phase1/phase1_environments.py",
        "phase1/phase1_ground_truth.py",
        "phase1/phase1_training.py",
        "phase2/capture_activations.py",
        "phase2/train_multilayer_saes.py",
        "phase2/rebuild_training_summary.py",
        "phase3/reasoning_flow_tracer.py",
        "phase4/arithmetic_data_collector.py",
        "phase4/arithmetic_value_probe.py",
        "phase4/feature_coactivation.py",
        "phase4/causal_patch_test.py",
        "phase5/feature_interpreter.py",
        "phase5/arithmetic_steerer.py",
        "phase5/arithmetic_steerer_subspace.py",
        "phase5/arithmetic_steerer_nn_transfer.py",
    ]
    py_compile_targets = list(scripts)
    rc, out = _run_cmd([sys.executable, "-m", "py_compile", *py_compile_targets])
    compile_ok = rc == 0

    help_results = {}
    for s in scripts:
        rc, out = _run_cmd([sys.executable, s, "--help"])
        help_results[s] = {"ok": rc == 0, "returncode": rc}

    failures = []
    if not compile_ok:
        failures.append("py_compile failed for one or more phase1-5 scripts")
    for s, row in help_results.items():
        if not row["ok"]:
            failures.append(f"--help failed for {s}")

    return {
        "passed": len(failures) == 0,
        "py_compile_ok": compile_ok,
        "script_help": help_results,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="phase_validation_results/phase1_5_validation_report.json",
        help="Output JSON report path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    phase_rows = [
        _phase1_checks(),
        _phase2_checks(),
        _phase3_checks(),
        _phase4_checks(),
        _phase5_checks(),
    ]
    health = _script_health_checks()
    overall_passed = all(row["passed"] for row in phase_rows) and bool(health["passed"])
    report = {
        "schema_version": "phase1_5_validation_report_v1",
        "overall_passed": overall_passed,
        "phases": {row["phase"]: row for row in phase_rows},
        "script_health": health,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Saved validation report -> {out_path}")
    print(f"overall_passed={overall_passed}")
    if not overall_passed:
        for row in phase_rows:
            if not row["passed"]:
                print(f"[FAIL] {row['phase']}: {row['failures']}")
        if not health["passed"]:
            print(f"[FAIL] script_health: {health['failures']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover
    from .common import save_json
except ImportError:  # pragma: no cover
    from common import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-tag", required=True, help="Run tag prefix used in benchmark files.")
    p.add_argument(
        "--synthetic-benchmarks-glob",
        default=None,
        help="Optional override glob for synthetic benchmark files.",
    )
    p.add_argument(
        "--real-benchmark",
        default=None,
        help="Optional labeled real-CoT benchmark file.",
    )
    p.add_argument(
        "--qwen-benchmark",
        default=None,
        help="Optional Qwen pilot benchmark file.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output report path. Defaults to phase7_results/results/academic_weakness_closure_report_<run-tag>.json",
    )
    return p.parse_args()


def _load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _safe_metric(payload: Dict[str, Any], track: str, metric: str) -> Optional[float]:
    row = (payload.get("by_benchmark_track") or {}).get(track) or {}
    val = row.get(metric)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def main() -> None:
    args = parse_args()
    synth_glob = args.synthetic_benchmarks_glob or f"phase7_results/results/faithfulness_benchmark_academic_{args.run_tag}_*.json"
    synth_files = sorted(glob.glob(synth_glob))
    synth_rows: List[Dict[str, Any]] = []
    for p in synth_files:
        try:
            d = _load(p)
        except Exception:
            continue
        variant_vs_faithful = d.get("variant_vs_faithful") or {}
        per_variant_rows = []
        for variant, vr in variant_vs_faithful.items():
            md = vr.get("metric_defined") or {}
            per_variant_rows.append(
                {
                    "variant": str(variant),
                    "auroc": vr.get("auroc"),
                    "false_positive_rate": vr.get("false_positive_rate"),
                    "recall": vr.get("recall"),
                    "metric_defined": {
                        "auroc": bool(md.get("auroc", False)),
                        "false_positive_rate": bool(md.get("false_positive_rate", False)),
                        "recall": bool(md.get("recall", False)),
                    },
                }
            )
        per_variant_rows.sort(key=lambda x: (float("inf") if x.get("auroc") is None else float(x["auroc"])))
        by_family = d.get("by_paper_failure_family") or {}
        family_definedness = {
            str(fam): bool((row.get("metric_defined") or {}).get("auroc", False))
            for fam, row in sorted(by_family.items())
        }
        synth_rows.append(
            {
                "path": p,
                "gate_track": d.get("gate_track"),
                "auroc": d.get("auroc"),
                "false_positive_rate": d.get("false_positive_rate"),
                "recall_at_gate": d.get("recall_at_gate"),
                "causal_auditor_auroc": _safe_metric(d, "causal_auditor", "auroc"),
                "composite_auroc": _safe_metric(d, "composite", "auroc"),
                "causal_signal_coverage_fraction": d.get("causal_signal_coverage_fraction"),
                "leakage_check_pass": d.get("leakage_check_pass"),
                "gate_checks": d.get("gate_checks"),
                "model_metadata": d.get("model_metadata"),
                "coverage_by_variable": d.get("coverage_by_variable"),
                "coverage_by_layer": d.get("coverage_by_layer"),
                "variant_min_auroc": d.get("variant_min_auroc"),
                "variant_vs_faithful_summary": {
                    "num_variants": int(len(variant_vs_faithful)),
                    "worst_variant": (per_variant_rows[0] if per_variant_rows else None),
                    "best_variant": (per_variant_rows[-1] if per_variant_rows else None),
                },
                "family_auroc_definedness": family_definedness,
            }
        )

    best_composite = None
    for r in synth_rows:
        c = r.get("composite_auroc")
        if isinstance(c, (int, float)):
            if best_composite is None or float(c) > float(best_composite.get("composite_auroc", -1.0)):
                best_composite = r

    real_row = _load(args.real_benchmark) if args.real_benchmark and Path(args.real_benchmark).exists() else None
    qwen_row = _load(args.qwen_benchmark) if args.qwen_benchmark and Path(args.qwen_benchmark).exists() else None

    weakness_answers = {
        "A_causal_discriminator_quality": {
            "status": "open",
            "criterion": "causal_auditor AUROC >= 0.65 on >=2 synthetic checkpoints with coverage >=0.25",
        },
        "B_threshold_utility": {
            "status": "open",
            "criterion": "composite recall_at_gate > 0 at FPR <= 0.05",
        },
        "C_external_validity_labeled": {
            "status": "open",
            "criterion": "at least one labeled real-CoT benchmark with defined AUROC/FPR",
        },
        "D_synthetic_to_real_transfer": {
            "status": "open",
            "criterion": "synthetic_to_real_gap reported for labeled real-CoT benchmark",
        },
        "E_model_breadth": {
            "status": "open",
            "criterion": ">=2 GPT-2 checkpoints + >=1 additional model benchmark",
        },
    }

    # A/B on synthetic rows
    causal_ok = [
        r
        for r in synth_rows
        if isinstance(r.get("causal_auditor_auroc"), (int, float))
        and float(r["causal_auditor_auroc"]) >= 0.65
        and isinstance(r.get("causal_signal_coverage_fraction"), (int, float))
        and float(r["causal_signal_coverage_fraction"]) >= 0.25
    ]
    if len(causal_ok) >= 2:
        weakness_answers["A_causal_discriminator_quality"]["status"] = "passed"
    elif len(causal_ok) == 1:
        weakness_answers["A_causal_discriminator_quality"]["status"] = "partial"

    threshold_ok = [
        r
        for r in synth_rows
        if isinstance(r.get("false_positive_rate"), (int, float))
        and float(r["false_positive_rate"]) <= 0.05
        and isinstance(r.get("recall_at_gate"), (int, float))
        and float(r["recall_at_gate"]) > 0.0
    ]
    if threshold_ok:
        weakness_answers["B_threshold_utility"]["status"] = "passed"

    # C/D on labeled real benchmark
    if real_row is not None:
        num_labeled = int(real_row.get("num_labeled_audits", 0) or 0)
        if num_labeled > 0 and isinstance(real_row.get("auroc"), (int, float)) and isinstance(real_row.get("false_positive_rate"), (int, float)):
            weakness_answers["C_external_validity_labeled"]["status"] = "passed"
        if isinstance((real_row.get("synthetic_to_real_gap") or {}).get("status"), str) and (
            real_row.get("synthetic_to_real_gap", {}).get("status") == "computed"
        ):
            weakness_answers["D_synthetic_to_real_transfer"]["status"] = "passed"
        elif num_labeled > 0:
            weakness_answers["D_synthetic_to_real_transfer"]["status"] = "partial"

    # E breadth
    has_two_gpt2 = len(synth_rows) >= 2
    has_other_model = bool(qwen_row is not None)
    if has_two_gpt2 and has_other_model:
        weakness_answers["E_model_breadth"]["status"] = "passed"
    elif has_two_gpt2 or has_other_model:
        weakness_answers["E_model_breadth"]["status"] = "partial"

    report = {
        "schema_version": "phase7_academic_weakness_closure_report_v1",
        "run_tag": args.run_tag,
        "synthetic_rows": synth_rows,
        "best_composite_row": best_composite,
        "real_cot_row": real_row,
        "real_cot_label_status": {
            "present": bool(real_row is not None),
            "num_labeled_audits": (int(real_row.get("num_labeled_audits", 0)) if isinstance(real_row, dict) else 0),
            "evaluation_mode": (real_row.get("evaluation_mode") if isinstance(real_row, dict) else None),
        },
        "qwen_row": qwen_row,
        "weakness_answers": weakness_answers,
    }
    output = args.output or f"phase7_results/results/academic_weakness_closure_report_{args.run_tag}.json"
    save_json(output, report)
    print(f"Saved report -> {output}")


if __name__ == "__main__":
    main()

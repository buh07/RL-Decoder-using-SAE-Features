#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List

try:  # pragma: no cover
    from .common import load_json, save_json, sha256_file
except ImportError:  # pragma: no cover
    from common import load_json, save_json, sha256_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-globs",
        default=(
            "phase7_results/results/faithfulness_benchmark_*.json,"
            "phase7_results/results/trackc_r*_benchmark_*.json"
        ),
        help="Comma-separated globs used to discover benchmark artifacts to archive.",
    )
    p.add_argument(
        "--output",
        default="phase7_results/results/trackc_intervention_archive_manifest.json",
    )
    p.add_argument(
        "--include-non-trackc",
        action="store_true",
        help="Include benchmark files that do not expose causal_auditor metrics.",
    )
    return p.parse_args()


def _extract_entry(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    by_track = payload.get("by_benchmark_track") or {}
    causal = by_track.get("causal_auditor") or {}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "gate_track": payload.get("gate_track"),
        "benchmark_scope": payload.get("benchmark_scope"),
        "external_validity_status": payload.get("external_validity_status"),
        "causal_auroc": causal.get("auroc"),
        "causal_fpr": causal.get("false_positive_rate"),
        "composite_auroc": (by_track.get("composite") or {}).get("auroc"),
        "text_auroc": (by_track.get("text_only") or {}).get("auroc"),
        "latent_auroc": (by_track.get("latent_only") or {}).get("auroc"),
        "causal_signal_coverage_fraction": payload.get("causal_signal_coverage_fraction"),
        "causal_track_degenerate_flag": payload.get("causal_track_degenerate_flag"),
        "causal_anti_predictive_flag": payload.get("causal_anti_predictive_flag"),
        "causal_interpretation_status": payload.get("causal_interpretation_status"),
    }


def main() -> None:
    args = parse_args()
    globs = [g.strip() for g in str(args.results_globs).split(",") if g.strip()]
    found = set()
    for g in globs:
        for p in glob.glob(g):
            found.add(str(Path(p)))
    paths = sorted(Path(p) for p in found)
    entries: List[Dict[str, Any]] = []
    for path in paths:
        try:
            entry = _extract_entry(path)
        except Exception:
            continue
        if not args.include_non_trackc and entry.get("causal_auroc") is None:
            continue
        entries.append(entry)

    out = {
        "schema_version": "phase7_trackc_intervention_archive_v1",
        "results_globs": globs,
        "num_entries": int(len(entries)),
        "entries": entries,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    print(f"Saved {len(entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()

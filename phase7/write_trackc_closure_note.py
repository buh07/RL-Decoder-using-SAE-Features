#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover
    from .common import save_json
except ImportError:  # pragma: no cover
    from common import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--r1-benchmark", required=True)
    p.add_argument("--r2-benchmark", default=None)
    p.add_argument("--r3-probe", default=None)
    p.add_argument("--r4-geometry", default=None)
    p.add_argument("--decision", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _load_optional(path: str | None) -> Dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    args = parse_args()
    r1 = _load_optional(args.r1_benchmark) or {}
    r2 = _load_optional(args.r2_benchmark) or {}
    r3 = _load_optional(args.r3_probe) or {}
    r4 = _load_optional(args.r4_geometry) or {}
    dec = _load_optional(args.decision) or {}

    r1_track = (r1.get("by_benchmark_track") or {}).get("confidence_margin") or {}
    r2_track = (r2.get("by_benchmark_track") or {}).get("trajectory_coherence") or {}
    stage_status = dict(dec.get("stage_status") or {})

    geom_aurocs = [
        float((v or {}).get("auroc_best_orientation"))
        for v in ((r4.get("feature_summary") or {}).values())
        if isinstance((v or {}).get("auroc_best_orientation"), (int, float))
    ]
    payload = {
        "schema_version": "phase7_trackc_closure_note_v1",
        "run_tag": str(args.run_tag),
        "artifacts": {
            "r1_benchmark": str(args.r1_benchmark),
            "r2_benchmark": (str(args.r2_benchmark) if args.r2_benchmark else None),
            "r3_probe": (str(args.r3_probe) if args.r3_probe else None),
            "r4_geometry": (str(args.r4_geometry) if args.r4_geometry else None),
            "decision": str(args.decision),
        },
        "stage_status": stage_status,
        "selected_track_c": dec.get("selected_track_c"),
        "claim_boundary": dec.get("claim_boundary"),
        "summary": {
            "r1_confidence_margin_auroc": r1_track.get("auroc"),
            "r1_confidence_margin_ci95": [
                r1_track.get("auroc_ci95_lower"),
                r1_track.get("auroc_ci95_upper"),
            ],
            "r1_confidence_vs_text_corr": r1.get("confidence_margin_text_only_correlation"),
            "r1_power_sufficient": (r1.get("gate_checks") or {}).get("power_sufficient"),
            "r1_leakage_check_pass": r1.get("leakage_check_pass"),
            "r2_trajectory_coherence_auroc": r2_track.get("auroc"),
            "r3_status_by_source": {
                k: (v or {}).get("status")
                for k, v in ((r3.get("by_feature_source") or {}).items())
            },
            "r4_best_geometry_auroc": (max(geom_aurocs) if geom_aurocs else None),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, payload)
    print(f"Saved closure note -> {out_path}")


if __name__ == "__main__":
    main()

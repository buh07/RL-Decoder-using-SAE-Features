#!/usr/bin/env python3
"""
Rebuild `training_summary.json` from per-layer SAE summary files.

This preserves the original list-based output format used by `train_multilayer_saes.py`
while also writing a small metadata file (`training_summary_meta.json`) with counts and
basic aggregates to make future drift easier to detect.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


SUMMARY_RE = re.compile(r"^(?P<model>.+)_layer(?P<layer>\d+)_sae\.summary\.json$")


def parse_summary_path(path: Path) -> Tuple[str, int]:
    m = SUMMARY_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized summary filename: {path.name}")
    return m.group("model"), int(m.group("layer"))


def load_entries(saes_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(saes_dir.glob("*_sae.summary.json")):
        model_name, layer_idx = parse_summary_path(path)
        with open(path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise TypeError(f"{path} did not contain an object")
        # Ensure sortable keys exist and are consistent with the filename.
        payload.setdefault("model_name", model_name)
        payload.setdefault("layer_idx", layer_idx)
        entries.append(payload)

    entries.sort(key=lambda x: (str(x.get("model_name", "")), int(x.get("layer_idx", -1))))
    return entries


def build_meta(entries: List[Dict[str, Any]], saes_dir: Path) -> Dict[str, Any]:
    counts = Counter(str(e.get("model_name", "unknown")) for e in entries)
    losses = [float(e["final_loss"]) for e in entries if isinstance(e.get("final_loss"), (int, float))]
    sparsities = [float(e["final_sparsity"]) for e in entries if isinstance(e.get("final_sparsity"), (int, float))]

    def _range(vals: List[float]) -> Dict[str, float] | None:
        if not vals:
            return None
        return {"min": min(vals), "max": max(vals), "mean": mean(vals)}

    return {
        "source_dir": str(saes_dir),
        "num_entries": len(entries),
        "counts_by_model": dict(sorted(counts.items())),
        "final_loss": _range(losses),
        "final_sparsity": _range(sparsities),
        "format_note": "training_summary.json is a list for compatibility with phase2/train_multilayer_saes.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saes-dir",
        type=Path,
        default=Path("phase2_results/saes_all_models/saes"),
        help="Directory containing per-layer *_sae.summary.json files",
    )
    parser.add_argument(
        "--summary-name",
        default="training_summary.json",
        help="Output filename for the list summary (default: training_summary.json)",
    )
    parser.add_argument(
        "--meta-name",
        default="training_summary_meta.json",
        help="Output filename for metadata (default: training_summary_meta.json)",
    )
    args = parser.parse_args()

    saes_dir = args.saes_dir
    if not saes_dir.exists():
        raise FileNotFoundError(f"Directory not found: {saes_dir}")

    entries = load_entries(saes_dir)
    if not entries:
        raise RuntimeError(f"No *_sae.summary.json files found in {saes_dir}")

    summary_path = saes_dir / args.summary_name
    meta_path = saes_dir / args.meta_name

    with open(summary_path, "w") as f:
        json.dump(entries, f, indent=2)
    meta = build_meta(entries, saes_dir)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(entries)} entries to {summary_path}")
    print(f"Wrote metadata to {meta_path}")
    print(f"Counts by model: {meta['counts_by_model']}")


if __name__ == "__main__":
    main()

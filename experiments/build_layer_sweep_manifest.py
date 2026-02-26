#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.layer_sweep_manifest import (  # noqa: E402
    SCHEMA_VERSION,
    build_manifest_payload,
    save_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="experiments/layer_sweep_manifest_v1.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = save_manifest(args.output)
    payload = build_manifest_payload()
    print(f"Saved manifest -> {out_path}")
    print(
        f"schema={SCHEMA_VERSION} layer_sets={len(payload.get('layer_sets', []))} "
        f"sae_panel={len(payload.get('sae_panel_layer_set_ids', []))}"
    )


if __name__ == "__main__":
    main()

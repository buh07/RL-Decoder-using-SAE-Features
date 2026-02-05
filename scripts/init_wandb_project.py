#!/usr/bin/env python
"""Bootstrap wandb configuration for RL-Decoder experiments.

The script reads configs/tracking/wandb_template.yaml, prompts for overrides,
and materializes a run config under results/runs/<timestamp>.yaml so training
jobs have a single source of truth. It does not talk to the wandb API directly;
that requires WANDB_API_KEY to be exported before launching training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "configs" / "tracking" / "wandb_template.yaml"
RUN_DIR = REPO_ROOT / "results" / "runs"


def load_template(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Template config missing: {path}")
    return yaml.safe_load(path.read_text())


def materialize_run(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Path:
    run_cfg = cfg.copy()
    run_cfg.update({k: v for k, v in overrides.items() if v is not None})
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    target = RUN_DIR / f"wandb_run_{timestamp}.yaml"
    target.write_text(yaml.dump(run_cfg, sort_keys=False))
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a wandb run config")
    parser.add_argument("--project", help="Override project name")
    parser.add_argument("--entity", help="Override entity name")
    parser.add_argument("--notes", help="Short description for wandb run")
    parser.add_argument("--tags", nargs="*", help="Extra tags to append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = load_template(TEMPLATE_PATH)
    overrides: Dict[str, Any] = {}
    if args.project:
        overrides["project"] = args.project
    if args.entity:
        overrides["entity"] = args.entity
    if args.notes:
        overrides["notes"] = args.notes
    if args.tags:
        tags = template.get("defaults", {}).get("tags", [])
        overrides.setdefault("defaults", template.get("defaults", {}).copy())
        overrides["defaults"]["tags"] = tags + args.tags

    output_path = materialize_run(template, overrides)
    api_key_env = template.get("credentials", {}).get("api_key_env", "WANDB_API_KEY")
    print(f"[INFO] wrote {output_path}")
    if api_key_env not in os.environ:
        print(f"[WARN] {api_key_env} is unset; export it before starting training.")


if __name__ == "__main__":
    main()

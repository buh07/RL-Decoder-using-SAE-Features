#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_jsons(pattern: str) -> List[dict]:
    files = sorted(Path().glob(pattern))
    rows = []
    for p in files:
        with open(p) as f:
            rows.append((p, json.load(f)))
    return rows


def summarize_supervised(rows):
    out = []
    for path, payload in rows:
        if not isinstance(payload, dict):
            continue
        best_val = payload.get("best_val", {})
        out.append(
            {
                "config_name": payload.get("config_name"),
                "checkpoint_path": payload.get("checkpoint_path"),
                "num_train_records": payload.get("num_train_records"),
                "num_val_records": payload.get("num_val_records"),
                "best_epoch": payload.get("best_epoch"),
                "val_top1_accuracy": best_val.get("top1_accuracy"),
                "val_top5_accuracy": best_val.get("top5_accuracy"),
                "val_mean_logprob_correct": best_val.get("mean_logprob_correct"),
                "val_delta_logprob_vs_gpt2": best_val.get("delta_logprob_vs_gpt2"),
                "val_cross_entropy_loss": best_val.get("cross_entropy_loss"),
                "source_file": str(path),
            }
        )
    out.sort(key=lambda r: (-(r.get("val_top1_accuracy") or -1), r.get("config_name") or ""))
    return out


def summarize_evals(rows):
    out = []
    for path, payload in rows:
        if not isinstance(payload, dict):
            continue
        ev = payload.get("evaluations", {})
        row = {
            "config_name": payload.get("config_name"),
            "checkpoint": payload.get("checkpoint"),
            "source_file": str(path),
        }
        for split in ("val", "test"):
            if split in ev:
                row[f"{split}_top1_accuracy"] = ev[split].get("top1_accuracy")
                row[f"{split}_top5_accuracy"] = ev[split].get("top5_accuracy")
                row[f"{split}_mean_logprob_correct"] = ev[split].get("mean_logprob_correct")
                row[f"{split}_delta_logprob_vs_gpt2"] = ev[split].get("delta_logprob_vs_gpt2")
        out.append(row)
    out.sort(key=lambda r: (-(r.get("test_top1_accuracy") or r.get("val_top1_accuracy") or -1), r.get("config_name") or ""))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="phase6_results/results")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    supervised_rows = load_jsons(str(results_dir / "supervised_*.json"))
    eval_rows = load_jsons(str(results_dir / "eval_*.json"))

    payload = {
        "results_dir": str(results_dir),
        "supervised": summarize_supervised(supervised_rows),
        "evaluations": summarize_evals(eval_rows),
        "num_supervised_files": len(supervised_rows),
        "num_eval_files": len(eval_rows),
    }

    out_path = Path(args.output) if args.output else results_dir / "supervised_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved merged results -> {out_path}")
    print(f"  supervised files: {len(supervised_rows)}")
    print(f"  eval files:       {len(eval_rows)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch

from common import (
    PHASE7_TRACE_SCHEMA,
    build_structured_state,
    faithful_step_text,
    group_records_by_example,
    load_pt,
    make_trace_id,
    save_json,
    save_pt,
)


def _build_from_phase6_records(records: List[dict], split_name: str) -> List[dict]:
    by_example = group_records_by_example(records)
    out: List[dict] = []
    for ex_id, group in sorted(by_example.items()):
        trace_id = make_trace_id(split_name, ex_id)
        trace_len = len(group)
        for step_idx, rec in enumerate(group):
            state = build_structured_state(rec, step_idx=step_idx, trace_len=trace_len)
            row = dict(rec)
            row.update(
                {
                    "schema_version": PHASE7_TRACE_SCHEMA,
                    "trace_id": trace_id,
                    "trace_len": trace_len,
                    "step_idx": step_idx,
                    "structured_state": state,
                    "cot_text_step": faithful_step_text(state),
                    "cot_alignment_span": None,
                }
            )
            out.append(row)
    return out


def _summarize(step_records: List[dict], split_name: str) -> Dict:
    step_types = Counter(r["structured_state"]["step_type"] for r in step_records)
    ops = Counter(r["structured_state"]["operator"] for r in step_records)
    mags = Counter(r["structured_state"]["magnitude_bucket"] for r in step_records)
    parse_err = Counter(bool(r["structured_state"].get("parse_error")) for r in step_records)
    trace_ids = {r["trace_id"] for r in step_records}
    return {
        "split": split_name,
        "num_step_records": len(step_records),
        "num_traces": len(trace_ids),
        "step_type_counts": dict(step_types),
        "operator_counts": dict(ops),
        "magnitude_bucket_counts": dict(mags),
        "parse_error_counts": {"has_error": int(parse_err.get(True, 0)), "no_error": int(parse_err.get(False, 0))},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase6-train", default="phase6_results/dataset/gsm8k_expanded_train.pt")
    p.add_argument("--phase6-test", default="phase6_results/dataset/gsm8k_expanded_test.pt")
    p.add_argument("--output-dir", default="phase7_results/dataset")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    return p.parse_args()


def _load_and_filter(path: str, split_name: str, max_records: int | None) -> List[dict]:
    recs = load_pt(path)
    if not isinstance(recs, list):
        raise TypeError(f"Expected list in {path}")
    if max_records is not None:
        recs = recs[:max_records]
    filtered = [r for r in recs if r.get("gsm8k_split") == split_name] or recs
    return filtered


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _load_and_filter(args.phase6_train, "train", args.max_train)
    test = _load_and_filter(args.phase6_test, "test", args.max_test)

    print(f"Loaded Phase6 train records: {len(train)}")
    train_steps = _build_from_phase6_records(train, "train")
    print(f"Built Phase7 train step records: {len(train_steps)}")
    test_steps = _build_from_phase6_records(test, "test")
    print(f"Built Phase7 test step records: {len(test_steps)}")

    train_path = out_dir / "gsm8k_step_traces_train.pt"
    test_path = out_dir / "gsm8k_step_traces_test.pt"
    all_path = out_dir / "gsm8k_step_traces_all.pt"
    save_pt(train_path, train_steps)
    save_pt(test_path, test_steps)
    save_pt(all_path, train_steps + test_steps)

    summary = {
        "schema_version": PHASE7_TRACE_SCHEMA,
        "source_phase6": {"train": args.phase6_train, "test": args.phase6_test},
        "outputs": {
            "train": str(train_path),
            "test": str(test_path),
            "all": str(all_path),
        },
        "splits": {
            "train": _summarize(train_steps, "train"),
            "test": _summarize(test_steps, "test"),
        },
    }
    save_json(out_dir / "build_summary.json", summary)
    print(f"Saved -> {train_path}")
    print(f"Saved -> {test_path}")
    print(f"Saved summary -> {out_dir / 'build_summary.json'}")


if __name__ == "__main__":
    main()

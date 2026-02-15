#!/usr/bin/env python3
"""Utility for fetching reasoning datasets and saving normalized JSONL shards."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset


@dataclass
class DatasetSpec:
    name: str
    hf_path: str
    subset: Optional[str]
    splits: List[str]
    question_column: str
    answer_column: str
    cot_column: Optional[str]
    extra_columns: List[str]
    notes: str


DATASETS: Dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        DatasetSpec(
            name="gsm8k",
            hf_path="openai/gsm8k",
            subset="main",
            splits=["train", "test"],
            question_column="question",
            answer_column="answer",
            cot_column=None,
            extra_columns=[],
            notes="Grade-school math word problems released under Apache-2.0.",
        ),
        DatasetSpec(
            name="cot_collection",
            hf_path="cot_collection/CoT",
            subset="full",
            splits=["train"],
            question_column="question",
            answer_column="answer",
            cot_column="rationale",
            extra_columns=["dataset", "source"],
            notes="Aggregated chain-of-thought exemplars curated by CoT-Collection.",
        ),
        DatasetSpec(
            name="openr1_math_220k",
            hf_path="openai/open-r1-math-220k",
            subset=None,
            splits=["train"],
            question_column="prompt",
            answer_column="response",
            cot_column="chain_of_thought",
            extra_columns=["difficulty"],
            notes="OpenAI's distilled reasoning traces for math; license CC BY-SA 4.0.",
        ),
        DatasetSpec(
            name="reasoning_traces",
            hf_path="meta-math/Reasoning-Traces",
            subset=None,
            splits=["train", "validation"],
            question_column="problem",
            answer_column="answer",
            cot_column="chain_of_thought",
            extra_columns=["subject"],
            notes="MetaMath curated mathematical reasoning traces.",
        ),
        DatasetSpec(
            name="reveal",
            hf_path="allenai/reveal",
            subset=None,
            splits=["train", "validation", "test"],
            question_column="question",
            answer_column="answer",
            cot_column="rationale",
            extra_columns=["context", "citation"],
            notes="REVEAL factual reasoning benchmark for verifiable rationales.",
        ),
        DatasetSpec(
            name="trip",
            hf_path="allenai/trip",
            subset=None,
            splits=["train", "validation", "test"],
            question_column="question",
            answer_column="answer",
            cot_column="program",
            extra_columns=["context"],
            notes="TRIP multihop reasoning with program annotations.",
        ),
        DatasetSpec(
            name="wiqa",
            hf_path="allenai/wiqa",
            subset=None,
            splits=["train", "validation", "test"],
            question_column="question_stem",
            answer_column="answer_label",
            cot_column="para_steps",
            extra_columns=["perturbation","effect1","effect2"],
            notes="WIQA for cause-effect reasoning; uses multiple-choice labels instead of free-form answers.",
        ),
    ]
}


def normalize_record(example: Dict, spec: DatasetSpec) -> Dict:
    record = {
        "question": example.get(spec.question_column, ""),
        "answer": example.get(spec.answer_column, ""),
        "cot": example.get(spec.cot_column, None) if spec.cot_column else None,
        "source_dataset": spec.name,
    }
    for key in spec.extra_columns:
        record[key] = example.get(key)
    return record


def save_jsonl(records: Iterable[Dict], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_dataset(spec: DatasetSpec, output_dir: Path, force: bool) -> None:
    for split in spec.splits:
        target = output_dir / spec.name / f"{split}.jsonl"
        if target.exists() and not force:
            print(f"[skip] {target} already exists; use --force to overwrite.")
            continue
        print(f"[download] {spec.name}:{split} -> {target}")
        dataset = load_dataset(spec.hf_path, spec.subset, split=split)
        records = (normalize_record(example, spec) for example in dataset)
        save_jsonl(records, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download reasoning datasets defined in DATASETS map.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/raw"),
        help="Root directory to save JSONL files (default: datasets/raw)",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        nargs="*",
        default=sorted(DATASETS.keys()),
        help="Subset of dataset slugs to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files instead of skipping.",
    )
    args = parser.parse_args()

    missing = [name for name in args.datasets if name not in DATASETS]
    if missing:
        raise SystemExit(f"Unknown dataset(s): {missing}. Known: {sorted(DATASETS.keys())}")

    manifest = {}
    for name in args.datasets:
        spec = DATASETS[name]
        download_dataset(spec, args.output_dir, args.force)
        manifest[name] = asdict(spec)

    manifest_path = (args.output_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {manifest_path}")


if __name__ == "__main__":
    main()

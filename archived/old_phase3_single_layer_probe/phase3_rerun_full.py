#!/usr/bin/env python3
"""
Re-run Phase 3 on full GSM8K activations with correct step-type regexes.

Waits for train + test activation captures to finish, then runs the
full-scale evaluator for all 9 SAE checkpoints (4x–20x).

GSM8K answer format uses  <<expr=result>>  and  ####  markers, not
"Step N:" headers.  The patterns here match the actual format.

Usage:
    CUDA_VISIBLE_DEVICES=7 python phase3/phase3_rerun_full.py \
        --acts-train /scratch2/f004ndc/gpt2_gsm8k_acts_full/gsm8k/train \
        --acts-test  /scratch2/f004ndc/gpt2_gsm8k_acts_full/gsm8k/test \
        --checkpoint-dir checkpoints/gpt2-small/sae \
        --output-dir phase3_results/full_scale_v3 \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase3_config import Phase3Config
from phase3_pipeline import Phase3Pipeline

# ── GSM8K-specific step-type regexes ─────────────────────────────────────────
# GSM8K answers look like:
#   "Natalia sold 48/2 = <<48/2=24>>24 clips in May."
#   "#### 72"
#   "She earned 0.2 x 50 = $<<0.2*50=10>>10."
#   "therefore" / "so" / "thus" / "which means"
GSM8K_REGEXES = {
    # Lines containing a <<expr=result>> annotation  → arithmetic computation
    "equation":    r"<<[^>]+=\s*[\d.]+>>",
    # Lines that ARE the final answer marker
    "final_answer": r"^####\s*[\d.]+",
    # Lines with comparison operators (>, <, >=, <=) used in reasoning
    "comparison":  r"\b(greater|less|more|fewer|than|at least|at most)\b",
    # Connective reasoning words
    "reasoning":   r"\b(therefore|so|thus|hence|which means|this means|that means|in total|altogether)\b",
}

SAE_EXPANSIONS = [4, 6, 8, 10, 12, 14, 16, 18, 20]


def wait_for_manifest(acts_dir: Path, poll_secs: int = 30, timeout_mins: int = 180) -> None:
    """Block until manifest.json appears in acts_dir (written by capture script on finish)."""
    manifest = acts_dir / "manifest.json"
    deadline = time.time() + timeout_mins * 60
    while not manifest.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {manifest}")
        print(f"  [{datetime.now():%H:%M:%S}] Waiting for capture to finish: {manifest}")
        time.sleep(poll_secs)
    print(f"  [{datetime.now():%H:%M:%S}] Capture complete: {manifest}")


def delete_stale_alignment_cache(output_dir: Path) -> None:
    """Remove cached gsm8k_aligned_train.jsonl so pipeline re-aligns with new regexes."""
    cache = output_dir / "gsm8k_aligned_train.jsonl"
    if cache.exists():
        cache.unlink()
        print(f"  Deleted stale alignment cache: {cache}")


def run_expansion(
    expansion: int,
    checkpoint_dir: Path,
    acts_train: Path,
    acts_test: Path,
    output_dir: Path,
    device: str,
) -> dict:
    """Run the full Phase 3 pipeline for one SAE expansion level."""
    ckpt = checkpoint_dir / f"sae_768d_{expansion}x_final.pt"
    if not ckpt.exists():
        return {"expansion": expansion, "error": f"Checkpoint not found: {ckpt}"}

    exp_output = output_dir / f"{expansion}x"
    exp_output.mkdir(parents=True, exist_ok=True)

    # Delete stale alignment cache so new regexes take effect
    delete_stale_alignment_cache(exp_output)

    config = Phase3Config(
        model_name="gpt2",
        layer=6,
        sae_checkpoints=[ckpt],
        train_activation_dir=acts_train,
        test_activation_dir=acts_test,
        output_dir=exp_output,
        device=device,
        verbose=True,
        # Use the corrected GSM8K-specific regex patterns
        step_extraction_method="regex",
        step_regex_patterns=GSM8K_REGEXES,
    )

    start = datetime.now()
    print(f"\n[{start:%H:%M:%S}] Starting {expansion}x SAE pipeline...")
    try:
        pipeline = Phase3Pipeline(config)
        result = pipeline.run()
        runtime = (datetime.now() - start).total_seconds()
        return {
            "expansion": expansion,
            "runtime_seconds": runtime,
            **result,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"expansion": expansion, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Phase 3 full re-run with correct GSM8K regexes")
    parser.add_argument("--acts-train", type=Path,
                        default=Path("/scratch2/f004ndc/gpt2_gsm8k_acts_full/gsm8k/train"))
    parser.add_argument("--acts-test", type=Path,
                        default=Path("/scratch2/f004ndc/gpt2_gsm8k_acts_full/gsm8k/test"))
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=REPO_ROOT / "checkpoints/gpt2-small/sae")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "phase3_results/full_scale_v3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-id", type=int, default=7,
                        help="GPU id used for naming the output JSON (default: 7)")
    parser.add_argument("--expansions", type=int, nargs="+", default=SAE_EXPANSIONS,
                        help="Which SAE expansion factors to run (default: all)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip waiting for capture manifest (use if already done)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Wait for both capture jobs to finish ─────────────────────────────────
    if not args.no_wait:
        print("[phase3_rerun] Waiting for train capture to finish...")
        wait_for_manifest(args.acts_train)
        print("[phase3_rerun] Waiting for test capture to finish...")
        wait_for_manifest(args.acts_test)
    else:
        print("[phase3_rerun] --no-wait: skipping manifest check")

    # ── Run all expansions sequentially on this GPU ───────────────────────────
    all_results = {}
    for expansion in args.expansions:
        result = run_expansion(
            expansion=expansion,
            checkpoint_dir=args.checkpoint_dir,
            acts_train=args.acts_train,
            acts_test=args.acts_test,
            output_dir=args.output_dir,
            device=args.device,
        )
        all_results[str(expansion)] = result

        # Save incremental results after each expansion
        out_json = args.output_dir / f"results_gpu{args.gpu_id}.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved incremental results to {out_json}")

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 3 FULL RE-RUN SUMMARY")
    print("=" * 80)
    for exp, r in sorted(all_results.items(), key=lambda x: int(x[0])):
        if "error" in r:
            print(f"  {exp}x: ERROR — {r['error']}")
        else:
            probes = r.get("probes", {})
            p = probes.get(int(exp)) or probes.get(str(exp)) or {}
            acc = p.get("accuracy", "?")
            f1 = p.get("f1", "?")
            rt = r.get("runtime_seconds", 0)
            examples = r.get("examples", "?")
            print(f"  {exp}x: examples={examples}  acc={acc}  f1={f1}  runtime={rt:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

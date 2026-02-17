#!/usr/bin/env python3
"""
Phase 3 Full-Scale Results Aggregation & Analysis
Combines per-GPU results and generates comparison report.

Usage:
    python phase3/phase3_merge_results.py --input-dir phase3_results/full_scale
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def merge_results(input_dir: Path) -> Dict[str, Any]:
    """Merge per-GPU JSON results into single report."""
    
    results_files = sorted(input_dir.glob("results_gpu*.json"))
    logger.info(f"Found {len(results_files)} result files")
    
    merged = {}
    errors = {}
    
    for result_file in results_files:
        logger.info(f"Loading {result_file.name}...")
        with open(result_file) as f:
            data = json.load(f)
        
        for exp_str, exp_result in data.items():
            try:
                exp = int(exp_str)
                if "error" in exp_result:
                    errors[exp] = exp_result["error"]
                else:
                    merged[exp] = exp_result
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid entry: {exp_str}")
    
    return merged, errors


def generate_comparison_report(merged: Dict[int, Any], errors: Dict[int, str]) -> str:
    """Generate comparison report across all SAE expansions."""
    
    report = []
    report.append("=" * 100)
    report.append("PHASE 3 FULL-SCALE EVALUATION REPORT")
    report.append("Full GSM8K Dataset (~7,473 examples)")
    report.append("=" * 100)
    report.append("")
    
    if not merged:
        report.append("❌ No results to display")
        return "\n".join(report)
    
    # Sort by expansion
    sorted_exps = sorted(merged.keys())
    
    # Summary table
    report.append(f"{'Expansion':<12} {'Examples':<12} {'Probe Acc':<12} {'F1':<12} {'Runtime (s)':<15}")
    report.append("-" * 100)
    
    total_runtime = 0
    best_acc = 0
    best_exp = None
    
    for exp in sorted_exps:
        result = merged[exp]
        examples = result.get("examples_processed", 0)
        acc = result.get("accuracy", 0)
        f1 = result.get("f1", 0)
        runtime = result.get("runtime_seconds", 0)
        
        total_runtime += runtime
        
        if acc > best_acc:
            best_acc = acc
            best_exp = exp
        
        acc_str = f"{acc*100:.1f}%" if isinstance(acc, float) else str(acc)
        f1_str = f"{f1*100:.1f}%" if isinstance(f1, float) else str(f1)
        
        report.append(f"{exp}x{'':<8} {examples:<12} {acc_str:<12} {f1_str:<12} {runtime:<15.1f}")
    
    report.append("-" * 100)
    report.append(f"Total Runtime: {total_runtime/3600:.1f} hours ({total_runtime:.0f}s)")
    report.append("")
    
    # Best performer
    if best_exp is not None:
        report.append(f"🏆 Best Performer: {best_exp}x SAE (accuracy: {best_acc*100:.1f}%)")
    
    # Error summary
    if errors:
        report.append("")
        report.append("⚠️  Errors:")
        for exp, error in sorted(errors.items()):
            report.append(f"  {exp}x: {error}")
    
    report.append("")
    report.append("=" * 100)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Merge and analyze Phase 3 results")
    parser.add_argument("--input-dir", type=Path, default=Path("phase3_results/full_scale"),
                       help="Directory containing per-GPU results")
    parser.add_argument("--output-file", type=Path, default=Path("phase3_results/full_scale/REPORT.txt"),
                       help="Output report file")
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    # Merge results
    logger.info(f"Merging results from {args.input_dir}...")
    merged, errors = merge_results(args.input_dir)
    
    # Generate report
    report = generate_comparison_report(merged, errors)
    
    # Save and display
    with open(args.output_file, "w") as f:
        f.write(report)
    
    print(report)
    logger.info(f"✓ Report saved to {args.output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

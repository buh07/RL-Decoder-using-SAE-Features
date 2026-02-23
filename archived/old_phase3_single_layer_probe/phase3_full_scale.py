#!/usr/bin/env python3
"""
Phase 3 Full-Dataset Evaluation: Scale to All Expansions (2x-32x)
Runs synchronized across multiple GPU processes via tmux.

Usage:
    python phase3/phase3_full_scale.py --num-gpus 4 --sae-expansions 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List
import torch
from datetime import datetime
import sys
import os

# Add repo src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase3_config import Phase3Config
from phase3_pipeline import Phase3Pipeline


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
)


class Phase3FullScaleEvaluator:
    """Orchestrates Phase 3 evaluation across multiple SAE expansions."""
    
    def __init__(self, gpu_id: int, num_examples: int = None, device: str | None = None):
        self.gpu_id = gpu_id
        self.device = device or f"cuda:{gpu_id}"
        self.num_examples = num_examples
        self.results = {}
        
    def evaluate_sae(self, expansion: int, checkpoint_path: Path) -> dict:
        """
        Run Phase 3 pipeline for a single SAE expansion.
        
        Returns dict with keys:
          - expansion: SAE expansion factor
          - accuracy: probe accuracy
          - f1: probe F1 score
          - features_evaluated: number of features in causal eval
          - top_features: top-10 important features
          - runtime_seconds: total execution time
        """
        start_time = datetime.now()
        logger.info(f"[GPU {self.gpu_id}] Starting evaluation for {expansion}x SAE")
        
        try:
            # Create config for this expansion
            config = Phase3Config(
                model_name="gpt2",
                layer=6,
                sae_checkpoints=[checkpoint_path],
                train_activation_dir=Path("/tmp/gpt2_gsm8k_acts/gsm8k/train"),
                test_activation_dir=Path("/tmp/gpt2_gsm8k_acts_test/gsm8k/test"),
                output_dir=Path(f"phase3_results/full_scale/{expansion}x"),
                device=self.device,
            )
            
            # Override num examples if specified
            if self.num_examples:
                # Will be applied during alignment
                pass
            
            # Validate config
            config.validate()
            
            # Run pipeline
            pipeline = Phase3Pipeline(config)
            result = pipeline.run()
            
            runtime = (datetime.now() - start_time).total_seconds()
            
            result_dict = {
                "expansion": expansion,
                "checkpoint": str(checkpoint_path.name),
                "gpu_id": self.gpu_id,
                "runtime_seconds": runtime,
                **result  # Includes probes_trained, causal_evaluations, examples_processed
            }
            
            logger.info(f"[GPU {self.gpu_id}] ✓ {expansion}x SAE completed in {runtime:.1f}s")
            return result_dict
            
        except Exception as e:
            logger.error(f"[GPU {self.gpu_id}] ✗ Error evaluating {expansion}x SAE: {e}")
            return {
                "expansion": expansion,
                "error": str(e),
                "gpu_id": self.gpu_id,
                "runtime_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    def run_evaluations(self, sae_expansions: List[int], checkpoint_dir: Path) -> dict:
        """Run evaluations for all specified SAE expansions."""
        results = {}
        
        for expansion in sae_expansions:
            checkpoint_path = checkpoint_dir / f"sae_768d_{expansion}x_final.pt"
            
            if not checkpoint_path.exists():
                logger.warning(f"[GPU {self.gpu_id}] Checkpoint not found: {checkpoint_path}")
                results[expansion] = {
                    "expansion": expansion,
                    "error": f"Checkpoint not found: {checkpoint_path}",
                    "gpu_id": self.gpu_id
                }
                continue
            
            result = self.evaluate_sae(expansion, checkpoint_path)
            results[expansion] = result
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Full-Scale Evaluation")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--sae-expansions", type=int, nargs="+", 
                       default=[4, 6, 8, 10, 12, 14, 16, 18, 20],
                       help="SAE expansion factors to evaluate")
    parser.add_argument("--num-examples", type=int, default=None,
                       help="Number of examples to process (None = all)")
    parser.add_argument("--checkpoint-dir", type=Path, 
                       default=Path("checkpoints/gpt2-small/sae"),
                       help="Directory containing SAE checkpoints")
    parser.add_argument("--output-file", type=Path, 
                       default=Path("phase3_results/full_scale/results.json"),
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Set GPU visibility to a single device; use cuda:0 within the process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    local_gpu_id = 0
    
    logger.info(
        f"Starting Phase 3 Full-Scale Evaluation on GPU {args.gpu_id} "
        f"(local cuda:{local_gpu_id})"
    )
    logger.info(f"SAE Expansions: {args.sae_expansions}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    
    # Create evaluator
    evaluator = Phase3FullScaleEvaluator(
        gpu_id=args.gpu_id,
        num_examples=args.num_examples,
        device=f"cuda:{local_gpu_id}",
    )
    
    # Run evaluations
    results = evaluator.run_evaluations(args.sae_expansions, args.checkpoint_dir)
    
    # Save results
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 3 FULL-SCALE EVALUATION SUMMARY")
    print("="*80)
    for expansion, result in sorted(results.items()):
        if "error" in result:
            print(f"  {expansion}x: ✗ ERROR - {result['error']}")
        else:
            print(f"  {expansion}x: ✓ Runtime {result['runtime_seconds']:.1f}s")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

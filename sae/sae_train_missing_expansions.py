#!/usr/bin/env python3
"""
SAE Training Pipeline: Train missing SAE expansions (2x, 22x-32x)
Optimized for RTX 6000 - ~2.5 hours per expansion using GPUs 4-7.

Usage:
    python sae/sae_train_missing_expansions.py --expansions 2 22 24 26 28 30 32 --num-gpus 4
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from typing import List
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s'
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "gpt2-small" / "sae"


def train_sae(expansion: int, gpu_id: int) -> bool:
    """
    Train a single SAE expansion.
    
    Args:
        expansion: SAE expansion factor (e.g., 2, 4, 6, ...)
        gpu_id: GPU index to use
    
    Returns:
        True if training succeeded, False otherwise
    """
    checkpoint_path = CHECKPOINT_DIR / f"sae_768d_{expansion}x_final.pt"
    
    # Skip if already exists
    if checkpoint_path.exists():
        logger.info(f"✓ SAE {expansion}x already exists: {checkpoint_path.name}")
        return True
    
    logger.info(f"Starting SAE {expansion}x training on GPU {gpu_id}...")
    
    # Build command
    cmd = [
        "python", "src/sae_training.py",
        "--expansion", str(expansion),
        "--gpu-id", str(gpu_id),
        "--num-epochs", "5",  # Standard 5 epochs
        "--batch-size", "32",
        "--learning-rate", "1e-3"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=False,  # Show output in real-time
            timeout=20000  # 5.5 hours timeout per SAE
        )
        
        if result.returncode == 0:
            logger.info(f"✓ SAE {expansion}x training completed successfully")
            return True
        else:
            logger.error(f"✗ SAE {expansion}x training failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ SAE {expansion}x training timed out after 5.5 hours")
        return False
    except Exception as e:
        logger.error(f"✗ Error training SAE {expansion}x: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train missing SAE expansions in parallel"
    )
    parser.add_argument(
        "--expansions", type=int, nargs="+",
        default=[2, 22, 24, 26, 28, 30, 32],
        help="SAE expansion factors to train"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=4,
        help="Number of GPUs to use (will use GPUs 4-7 by default)"
    )
    parser.add_argument(
        "--base-gpu", type=int, default=4,
        help="Starting GPU ID (default: 4, so uses 4,5,6,7)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"========================================")
    logger.info(f"SAE Training Pipeline")
    logger.info(f"========================================")
    logger.info(f"Expansions to train: {args.expansions}")
    logger.info(f"GPUs available: {list(range(args.base_gpu, args.base_gpu + args.num_gpus))}")
    logger.info(f"Estimated time: {len(args.expansions)} × 2.5 hours = {len(args.expansions) * 2.5:.1f} hours")
    logger.info(f"========================================\n")
    
    # Track results
    results = {
        "successful": [],
        "failed": [],
        "skipped": []
    }
    
    # Assign GPU round-robin
    for i, expansion in enumerate(args.expansions):
        gpu_id = args.base_gpu + (i % args.num_gpus)
        
        logger.info(f"\n[{i+1}/{len(args.expansions)}] SAE {expansion}x → GPU {gpu_id}")
        
        # Check if already exists
        checkpoint_path = CHECKPOINT_DIR / f"sae_768d_{expansion}x_final.pt"
        if checkpoint_path.exists():
            logger.info(f"  ✓ Already trained, skipping")
            results["skipped"].append(expansion)
            continue
        
        # Train
        if train_sae(expansion, gpu_id):
            results["successful"].append(expansion)
        else:
            results["failed"].append(expansion)
    
    # Summary
    logger.info(f"\n{'='*40}")
    logger.info(f"Training Summary")
    logger.info(f"{'='*40}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Skipped: {results['skipped']}")
    logger.info(f"{'='*40}\n")
    
    if results["failed"]:
        logger.error(f"Failed to train: {results['failed']}")
        return 1
    
    logger.info("✓ All SAEs ready for Phase 3 evaluation")
    return 0


if __name__ == "__main__":
    sys.exit(main())

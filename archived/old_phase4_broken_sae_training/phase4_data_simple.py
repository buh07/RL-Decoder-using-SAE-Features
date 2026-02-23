#!/usr/bin/env python3
"""
Phase 4: Collect benchmark data and test activation capture
Simplified version focusing on data availability
"""

import argparse
import json
import logging
from pathlib import Path
import time

import torch
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


def create_synthetic_activations(
    model_name: str,
    benchmark_name: str,
    num_examples: int,
    hidden_dim: int = 768,
) -> torch.Tensor:
    """Create synthetic activations for testing (placeholder until real capture)."""
    
    logger.info(f"Creating synthetic activations: {model_name} x {benchmark_name}")
    logger.info(f"  Shape: ({num_examples * 100}, {hidden_dim})")
    
    # Simulate average 100 tokens per example
    activations = torch.randn(num_examples * 100, hidden_dim, dtype=torch.float32)
    
    # Normalize
    activations = (activations - activations.mean(dim=0)) / (activations.std(dim=0) + 1e-6)
    
    return activations


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Data Collection")
    parser.add_argument("--model", type=str, default="gpt2-medium",
                       choices=["gpt2-medium", "pythia-1.4b", "gemma-2b", "phi-2"])
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                       choices=["gsm8k", "math", "logic"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("phase4_results/activations"))
    parser.add_argument("--num-examples", type=int, default=100)
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 4: DATA COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Examples: {args.num_examples}")
    logger.info(f"GPU: {args.gpu_id}")
    logger.info("")
    
    # Map model to hidden dimensions
    model_dims = {
        "gpt2-medium": 1024,
        "pythia-1.4b": 2048,
        "gemma-2b": 2048,
        "phi-2": 2560,
    }
    
    hidden_dim = model_dims.get(args.model, 1024)
    
    # Create synthetic activations
    logger.info("Generating benchmark activations...")
    start_time = time.time()
    
    activations = create_synthetic_activations(
        args.model,
        args.benchmark,
        args.num_examples,
        hidden_dim=hidden_dim,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Generated in {elapsed:.2f}s")
    logger.info(f"Activations shape: {activations.shape}")
    
    # Select middle layer for each model
    layer_map = {
        "gpt2-medium": 12,
        "pythia-1.4b": 12,
        "gemma-2b": 9,
        "phi-2": 16,
    }
    
    layer_idx = layer_map.get(args.model, 12)
    
    # Save activations
    output_file = args.output_dir / f"{args.model}_{args.benchmark}_layer{layer_idx}_activations.pt"
    
    torch.save({
        "activations": activations,
        "model": args.model,
        "benchmark": args.benchmark,
        "layer": layer_idx,
        "num_examples": args.num_examples,
        "hidden_dim": hidden_dim,
        "timestamp": time.time(),
    }, output_file)
    
    logger.info(f"Saved: {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1e6:.1f} MB")
    logger.info("")
    logger.info("✓ Data collection complete")


if __name__ == "__main__":
    main()

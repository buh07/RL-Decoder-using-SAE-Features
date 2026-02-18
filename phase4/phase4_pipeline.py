#!/usr/bin/env python3
"""
Phase 4: Frontier LLMs - Causal Reasoning Circuit Discovery
Applies validated SAE methodology to 1B-7B language models.

Key differences from Phase 3:
- Phase 3: Validates SAEs extract reasoning features (correlation test with probes)
- Phase 4: Tests causal interpretability via feature ablations + activation patching

Tasks:
1. Capture activations from frontier models on reasoning benchmarks
2. Train SAEs (using Phase 1-3 validated settings)
3. Run causal circuit tests: ablate top features, measure task performance delta
4. Validate stable reasoning primitives across models/benchmarks
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_registry import MODEL_REGISTRY, get_model_and_tokenizer
from sae_config import SAEConfig
from sae_architecture import SparseAutoencoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


class Phase4Config:
    """Configuration for Phase 4 experiments."""
    
    # Target models (1B-7B range)
    TARGET_MODELS = [
        "gpt2-medium",      # 355M (small but reasonable baseline)
        "pythia-1.4b",      # 1.4B (good sweet spot)
        "gemma-2b",         # 2B (capable)
        "phi-2",            # 2.7B (strong reasoning)
        "llama-3-8b",       # 8B (frontier)
    ]
    
    # Reasoning benchmarks to use
    REASONING_BENCHMARKS = {
        "gsm8k": {
            "name": "GSM8K",
            "description": "Grade school math word problems",
            "num_examples": 500,  # Subset for Phase 4 speed
            "metric": "accuracy",
        },
        "math": {
            "name": "MATH",
            "description": "Competition mathematics",
            "num_examples": 100,
            "metric": "accuracy",
        },
        "logic": {
            "name": "Logic Inference",
            "description": "Logical reasoning (from Reasoning Traces)",
            "num_examples": 200,
            "metric": "accuracy",
        },
    }
    
    # SAE settings (from Phase 1/3 validated config)
    SAE_CONFIG = {
        "expansion_factor": 8,
        "l1_penalty_coeff": 1e-4,
        "decorrelation_coeff": 0.01,
        "use_relu": False,
        "num_epochs": 20,
        "learning_rate": 1e-4,
    }
    
    # Causal test settings
    CAUSAL_TEST_CONFIG = {
        "top_k_features": 10,  # Ablate top k features
        "perturbation_types": ["zero_ablation", "random_replacement", "scale_2x"],
        "num_eval_samples": 100,
    }


class Phase4Pipeline:
    """Full Phase 4 pipeline: capture -> SAE train -> causal tests."""
    
    def __init__(self, model_name: str, gpu_id: int, output_dir: Path):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Phase4Pipeline for {model_name} on {self.device}")
    
    def load_model_and_tokenizer(self):
        """Load frontier model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            model, tokenizer = get_model_and_tokenizer(
                self.model_name,
                device=self.device,
                load_in_8bit=False,  # Full precision for Phase 4
            )
            logger.info(f"Model loaded: {self.model_name} ({model.config.hidden_size}d)")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            raise
    
    def capture_activations_on_benchmark(
        self,
        model,
        tokenizer,
        benchmark_name: str,
        layer_idx: int,
        num_examples: int = 100,
    ) -> torch.Tensor:
        """Capture activations from frontier model on reasoning benchmark."""
        logger.info(f"Capturing {benchmark_name} activations from layer {layer_idx}")
        
        # TODO: Implement benchmark data loading
        # TODO: Implement activation capturing hook
        # TODO: Return stacked activations
        
        # Stub: return random tensor for now
        hidden_size = model.config.hidden_size
        activations = torch.randn(num_examples * 100, hidden_size)  # num_examples * avg_seq_len
        
        logger.info(f"Captured activations: {activations.shape}")
        return activations
    
    def train_sae_on_model(
        self,
        activations: torch.Tensor,
        benchmark_name: str,
    ) -> SparseAutoencoder:
        """Train SAE on captured model activations."""
        logger.info(f"Training SAE on {benchmark_name} activations")
        
        input_dim = activations.shape[1]
        config = SAEConfig(
            input_dim=input_dim,
            expansion_factor=Phase4Config.SAE_CONFIG["expansion_factor"],
            l1_penalty_coeff=Phase4Config.SAE_CONFIG["l1_penalty_coeff"],
            decorrelation_coeff=Phase4Config.SAE_CONFIG["decorrelation_coeff"],
            use_relu=Phase4Config.SAE_CONFIG["use_relu"],
        )
        
        sae = SparseAutoencoder(config).to(self.device)
        
        # TODO: Implement SAE training loop
        # TODO: For now, just return untrained SAE
        
        logger.info(f"SAE created: {input_dim}d -> {config.latent_dim}d")
        return sae
    
    def run_causal_ablation_tests(
        self,
        model,
        tokenizer,
        sae: SparseAutoencoder,
        benchmark_name: str,
        layer_idx: int,
        num_eval_samples: int = 100,
    ) -> Dict:
        """Test causality: ablate SAE features and measure task performance."""
        logger.info(f"Running causal ablation tests on {benchmark_name}")
        
        config = Phase4Config.CAUSAL_TEST_CONFIG
        results = {
            "benchmark": benchmark_name,
            "model": self.model_name,
            "layer": layer_idx,
            "ablation_results": {},
        }
        
        # TODO: Implement causal ablation tests
        # For each perturbation type in config["perturbation_types"]:
        #   1. Hook into model at layer_idx
        #   2. For each eval sample:
        #      a. Get latent representation via SAE
        #      b. Identify top-k features (by activation magnitude)
        #      c. Perturb top features according to perturbation type
        #      d. Run inference
        #      e. Measure task metric (accuracy) with perturbed features
        #   3. Compare accuracy with/without perturbations
        #   4. Record feature importance scores
        
        logger.info("Causal ablation tests setup (stub implementation)")
        results["ablation_results"]["zero_ablation"] = {
            "mean_accuracy_delta": 0.0,  # To be computed
            "std_accuracy_delta": 0.0,
        }
        
        return results
    
    def run_full_phase4(self, benchmark_names: Optional[List[str]] = None):
        """Run full Phase 4 pipeline on model."""
        
        logger.info("=" * 80)
        logger.info(f"PHASE 4: {self.model_name.upper()}")
        logger.info("=" * 80)
        
        if benchmark_names is None:
            benchmark_names = ["gsm8k"]  # Quick test with GSM8K
        
        results = {
            "model": self.model_name,
            "gpu_id": self.gpu_id,
            "benchmarks": {},
        }
        
        try:
            model, tokenizer = self.load_model_and_tokenizer()
            model.eval()
            
            # Select layer to analyze
            num_layers = model.config.num_hidden_layers
            layer_idx = num_layers // 2  # Middle layer
            
            for benchmark_name in benchmark_names:
                if benchmark_name not in Phase4Config.REASONING_BENCHMARKS:
                    logger.warning(f"Unknown benchmark: {benchmark_name}")
                    continue
                
                benchmark_info = Phase4Config.REASONING_BENCHMARKS[benchmark_name]
                logger.info(f"\n--- {benchmark_info['name']} ---")
                
                # Capture activations
                activations = self.capture_activations_on_benchmark(
                    model,
                    tokenizer,
                    benchmark_name,
                    layer_idx,
                    num_examples=benchmark_info["num_examples"],
                )
                
                # Train SAE
                sae = self.train_sae_on_model(activations, benchmark_name)
                
                # Run causal tests
                causal_results = self.run_causal_ablation_tests(
                    model,
                    tokenizer,
                    sae,
                    benchmark_name,
                    layer_idx,
                    num_eval_samples=Phase4Config.CAUSAL_TEST_CONFIG["num_eval_samples"],
                )
                
                results["benchmarks"][benchmark_name] = causal_results
        
        except Exception as e:
            logger.error(f"Phase 4 pipeline failed: {e}")
            raise
        
        # Save results
        results_file = self.output_dir / f"{self.model_name}_phase4_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {results_file}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Frontier LLMs Causal Analysis")
    parser.add_argument("--model", type=str, default="gpt2-medium",
                       choices=Phase4Config.TARGET_MODELS,
                       help="Target model to analyze")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--output-dir", type=Path, default=Path("phase4_results"),
                       help="Output directory for results")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["gsm8k"],
                       help="Benchmarks to evaluate on")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PHASE 4: FRONTIER LLMs - CAUSAL REASONING CIRCUIT DISCOVERY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This phase:")
    logger.info("  1. Loads frontier LLMs (1B-7B)")
    logger.info("  2. Captures activations on reasoning benchmarks")
    logger.info("  3. Trains SAEs using Phase 1-3 validated settings")
    logger.info("  4. Performs causal ablation tests to identify circuits")
    logger.info("")
    logger.info(f"Model: {args.model}")
    logger.info(f"Benchmarks: {args.benchmarks}")
    logger.info(f"GPU: {args.gpu_id}")
    logger.info("")
    
    pipeline = Phase4Pipeline(args.model, args.gpu_id, args.output_dir)
    results = pipeline.run_full_phase4(benchmark_names=args.benchmarks)
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 80)
    
    # Print summary
    logger.info(f"\nModel: {results['model']}")
    for benchmark_name, benchmark_results in results.get("benchmarks", {}).items():
        logger.info(f"  {benchmark_name}: Job completed")


if __name__ == "__main__":
    main()

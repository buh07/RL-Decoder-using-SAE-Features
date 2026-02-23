#!/usr/bin/env python3
"""
Phase 4 Benchmark Data Loading and Activation Capture
Loads reasoning benchmarks and captures activations from frontier models.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logger = logging.getLogger(__name__)


class BenchmarkDataLoader:
    """Load reasoning benchmarks (GSM8K, MATH, Logic)."""
    
    @staticmethod
    def load_gsm8k(num_examples: int = 500) -> List[Dict]:
        """Load GSM8K dataset (grade school math)."""
        logger.info(f"Loading GSM8K (n={num_examples})...")
        
        try:
            dataset = load_dataset("openai/gsm8k", "main", split="test")
            data = []
            
            for i, example in enumerate(dataset):
                if i >= num_examples:
                    break
                
                data.append({
                    "question": example["question"],
                    "answer": example["answer"],
                    "task": "math_reasoning",
                    "expected_output": example["answer"].split("#### ")[-1].strip(),  # Extract final answer
                })
            
            logger.info(f"Loaded {len(data)} GSM8K examples")
            return data
        except Exception as e:
            logger.warning(f"Failed to load GSM8K: {e}. Using synthetic data.")
            return BenchmarkDataLoader._create_synthetic_math_problems(num_examples)
    
    @staticmethod
    def load_math_benchmark(num_examples: int = 100) -> List[Dict]:
        """Load MATH dataset (competition mathematics)."""
        logger.info(f"Loading MATH benchmark (n={num_examples})...")
        
        try:
            # Try to load MATH competition dataset
            dataset = load_dataset("competition_math")
            data = []
            
            for i, example in enumerate(dataset["train"]):
                if i >= num_examples:
                    break
                
                data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "task": "math_reasoning",
                })
            
            logger.info(f"Loaded {len(data)} MATH examples")
            return data
        except Exception as e:
            logger.warning(f"Failed to load MATH: {e}. Using synthetic data.")
            return BenchmarkDataLoader._create_synthetic_math_problems(num_examples)
    
    @staticmethod
    def load_logic_benchmark(num_examples: int = 200) -> List[Dict]:
        """Load logic reasoning benchmark."""
        logger.info(f"Loading logic benchmark (n={num_examples})...")
        
        try:
            # Try to load WIQA (common sense reasoning about processes)
            dataset = load_dataset("wiqa", split="train")
            data = []
            
            for i, example in enumerate(dataset):
                if i >= num_examples:
                    break
                
                data.append({
                    "question": example["question"],
                    "answer": example["answer"],
                    "task": "logic_reasoning",
                })
            
            logger.info(f"Loaded {len(data)} logic examples")
            return data
        except Exception as e:
            logger.warning(f"Failed to load logic benchmark: {e}. Using synthetic data.")
            return BenchmarkDataLoader._create_synthetic_logic_problems(num_examples)
    
    @staticmethod
    def _create_synthetic_math_problems(num_examples: int) -> List[Dict]:
        """Create synthetic math problems for testing."""
        problems = []
        for i in range(num_examples):
            a, b = np.random.randint(1, 100, 2)
            problems.append({
                "question": f"What is {a} + {b}?",
                "answer": f"The answer is {a + b}.",
                "expected_output": str(a + b),
                "task": "math_reasoning",
            })
        return problems
    
    @staticmethod
    def _create_synthetic_logic_problems(num_examples: int) -> List[Dict]:
        """Create synthetic logic problems for testing."""
        problems = []
        for i in range(num_examples):
            problems.append({
                "question": "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?",
                "answer": "We cannot definitively conclude this because not all flowers fade quickly.",
                "task": "logic_reasoning",
            })
        return problems


class ActivationCapture:
    """Capture activations from frontier models during inference."""
    
    def __init__(self, model, layer_idx: int, device: torch.device):
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        self.activations = []
        self.hooks = []
    
    def _get_module_to_hook(self) -> torch.nn.Module:
        """Get the specific module to hook based on model architecture."""
        # Handle different model architectures
        model_type = self.model.config.model_type
        
        if model_type in ["gpt2", "gpt", "causal-lm"]:
            # GPT-2 style: model.transformer.h[layer_idx]
            if hasattr(self.model, "transformer"):
                return self.model.transformer.h[self.layer_idx]
            elif hasattr(self.model, "gpt_neox"):
                # Pythia, etc
                return self.model.gpt_neox.layers[self.layer_idx]
        
        raise ValueError(f"Unsupported model type: {model_type}")
    
    def hook_fn(self, module, input, output):
        """Hook function to capture activations."""
        if isinstance(output, tuple):
            # Post-MLP residual stream (last hidden state)
            hidden_state = output[0]
        else:
            hidden_state = output
        
        # Store activations on CPU to save GPU memory
        self.activations.append(hidden_state.detach().cpu())
    
    def register_hook(self):
        """Register activation hook."""
        module = self._get_module_to_hook()
        handle = module.register_forward_hook(self.hook_fn)
        self.hooks.append(handle)
        logger.info(f"Registered hook on layer {self.layer_idx}")
    
    def remove_hooks(self):
        """Clean up hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def capture_on_benchmark(
        self,
        tokenizer,
        benchmark_data: List[Dict],
        max_seq_len: int = 512,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """Capture activations on benchmark data."""
        
        self.register_hook()
        self.activations = []
        
        try:
            self.model.eval()
            
            with torch.no_grad():
                for i, example in enumerate(benchmark_data):
                    if i % max(1, len(benchmark_data) // 5) == 0:
                        logger.info(f"  Processing {i}/{len(benchmark_data)}")
                    
                    # Prepare input
                    question = example.get("question", "")
                    
                    # Tokenize
                    inputs = tokenizer(
                        question,
                        max_length=max_seq_len,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    
                    # Move to device
                    input_ids = inputs["input_ids"].to(self.device)
                    
                    # Forward pass (hooks will capture activations)
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids)
            
            # Concatenate all captured activations
            if self.activations:
                all_activations = torch.cat(self.activations, dim=0)
                logger.info(f"Captured activations shape: {all_activations.shape}")
                return all_activations
            else:
                logger.warning("No activations captured!")
                return torch.zeros(len(benchmark_data), self.model.config.hidden_size)
        
        finally:
            self.remove_hooks()


def create_phase4_data_loading_script():
    """Create standalone script for Phase 4 data loading and capture."""
    
    script = '''#!/usr/bin/env python3
"""Phase 4: Load benchmarks and capture activations from frontier models."""

import argparse
import logging
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase4_data import BenchmarkDataLoader, ActivationCapture
from model_registry import get_model_and_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Benchmark Data + Activation Capture")
    parser.add_argument("--model", type=str, default="gpt2-medium")
    parser.add_argument("--benchmark", type=str, choices=["gsm8k", "math", "logic"], default="gsm8k")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("phase4_activations"))
    parser.add_argument("--num-examples", type=int, default=100)
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f"cuda:{args.gpu_id}")
    
    logger.info(f"Model: {args.model} | Benchmark: {args.benchmark} | GPU: {args.gpu_id}")
    
    # Load model
    logger.info(f"Loading {args.model}...")
    model, tokenizer = get_model_and_tokenizer(args.model, device=device)
    
    # Load benchmark
    logger.info(f"Loading {args.benchmark}...")
    if args.benchmark == "gsm8k":
        data = BenchmarkDataLoader.load_gsm8k(args.num_examples)
    elif args.benchmark == "math":
        data = BenchmarkDataLoader.load_math_benchmark(args.num_examples)
    else:
        data = BenchmarkDataLoader.load_logic_benchmark(args.num_examples)
    
    # Capture activations
    layer_idx = model.config.num_hidden_layers // 2  # Middle layer
    logger.info(f"Capturing activations from layer {layer_idx}...")
    
    capture = ActivationCapture(model, layer_idx, device)
    activations = capture.capture_on_benchmark(tokenizer, data)
    
    # Save
    output_path = args.output_dir / f"{args.model}_{args.benchmark}_layer{layer_idx}_activations.pt"
    torch.save({
        "activations": activations,
        "model": args.model,
        "benchmark": args.benchmark,
        "layer": layer_idx,
        "num_examples": len(data),
    }, output_path)
    
    logger.info(f"Saved activations to: {output_path}")


if __name__ == "__main__":
    main()
'''
    return script

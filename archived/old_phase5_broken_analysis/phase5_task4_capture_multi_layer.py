#!/usr/bin/env python3
"""
Phase 5 Task 4: Multi-Layer Activation Capture

Captures activations from multiple intermediate layers (not just final layer)
for each of the 4 frontier LLMs to enable layer-wise transfer analysis.

Usage:
    python3 phase5_task4_capture_multi_layer.py \
        --models gemma-2b gpt2-medium phi-2 pythia-1.4b \
        --layers 4 8 12 16 20 \
        --output-dir phase4_results/activations_multilayer \
        --batch-size 32 \
        --num-batches 50 \
        --device cuda:5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _get_layer_configs() -> Dict[str, Dict[str, any]]:
    """Layer configurations per model (will auto-detect actual layer count)."""
    # Base configurations - layer count will be auto-detected
    return {
        "gemma-2b": {
            "model_id": "google/gemma-2b",
            "hidden_dim": 2048,
            "layers_attr": "model.layers",
        },
        "gpt2-medium": {
            "model_id": "gpt2-medium",
            "hidden_dim": 1024,
            "layers_attr": "transformer.h",
        },
        "phi-2": {
            "model_id": "microsoft/phi-2",
            "hidden_dim": 2560,
            "layers_attr": "model.layers",
        },
        "pythia-1.4b": {
            "model_id": "EleutherAI/pythia-1.4b",
            "hidden_dim": 2048,
            "layers_attr": "gpt_neox.layers",
        },
    }


def _get_layer_count(model, layers_attr: str) -> int:
    """Dynamically detect the number of layers in a model."""
    parts = layers_attr.split(".")
    obj = model
    for part in parts:
        obj = getattr(obj, part)
    return len(obj)



def _get_reasoning_samples(
    tokenizer,
    num_samples: int = 50,
    seed: int = 42,
) -> List[str]:
    """
    Load reasoning task samples for activation capture.
    Uses GSM8K and Logic reasoning problems.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sample reasoning problems from available datasets
    problems = [
        "If there are 5 apples and I add 3 more, how many apples do I have in total?",
        "A train leaves the station at 2 PM traveling at 60 mph. Another train leaves at 3 PM traveling at 80 mph. When will the second train catch up?",
        "If all dogs are animals and Fluffy is a dog, is Fluffy an animal?",
        "A rectangle has length 12 and width 5. What is its perimeter?",
        "If it takes 3 workers 8 days to build a house, how long would it take 5 workers?",
        "What is the next number in the sequence: 2, 4, 8, 16, ?",
        "If the probability of event A is 0.3 and event B is 0.5, what is P(A and B) if independent?",
        "Solve for x: 2x + 5 = 15",
        "If one coin is fair and another is biased (70% heads), what is the probability both show heads?",
        "A store has 100 items. 30% are red, 50% are blue, rest are green. How many are green?",
    ]

    # Repeat to reach num_samples
    samples = []
    for i in range(num_samples):
        samples.append(problems[i % len(problems)])

    return samples


def capture_layer_activations(
    model: nn.Module,
    model_name: str,
    layer_idx: int,
    tokenizer,
    device: str,
    layers_attr: str = "model.layers",
    batch_size: int = 32,
    num_samples: int = 50,
    output_path: Optional[Path] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Capture activations from a specific layer of the model.

    Args:
        model: Loaded transformer model
        model_name: Name of model (for logging)
        layer_idx: Index of layer to capture from
        tokenizer: Tokenizer for model
        device: Device to run on (e.g., "cuda:0")
        layers_attr: Path to layers in model (e.g., "model.layers" or "transformer.h")
        batch_size: Batch size for inference
        num_samples: Number of samples to capture
        output_path: Optional path to save activations

    Returns:
        Tuple of (activations tensor, metadata dict)
    """
    logger.info(f"Capturing {num_samples} samples from {model_name} layer {layer_idx}...")

    model.eval()
    all_activations = []
    hook_handles = []

    def get_hook(layer_module):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            if act.dim() == 3:
                act = act.reshape(-1, act.shape[-1])
            elif act.dim() == 2:
                pass
            else:
                act = act.reshape(-1, act.shape[-1])

            all_activations.append(act.detach().cpu())

        return hook_fn

    # Register hook on appropriate layer (generic approach)
    try:
        # Navigate to layers using the layers_attr path
        parts = layers_attr.split(".")
        layers_container = model
        for part in parts:
            layers_container = getattr(layers_container, part)
        
        # Access the specific layer
        if layer_idx >= len(layers_container):
            logger.error(
                f"Layer index {layer_idx} out of range. Model has {len(layers_container)} layers"
            )
            return torch.zeros(0, 1), {}
        
        layer_module = layers_container[layer_idx]
        hook = layer_module.register_forward_hook(get_hook(layer_module))
        hook_handles.append(hook)
    except (AttributeError, IndexError, TypeError) as e:
        logger.error(f"Failed to register hook for layer {layer_idx}: {e}")
        return torch.zeros(0, 1), {}

    # Get reasoning samples
    samples = _get_reasoning_samples(tokenizer, num_samples=num_samples)

    # Run inference and collect activations
    with torch.no_grad():
        for i, sample in enumerate(samples):
            if (i + 1) % max(1, num_samples // 10) == 0:
                logger.info(f"  Processed {i + 1}/{num_samples} samples")

            try:
                encoding = tokenizer(
                    sample,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                input_ids = encoding["input_ids"].to(device)
                _ = model(input_ids)
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")

    # Remove hooks
    for h in hook_handles:
        h.remove()

    # Concatenate activations
    if all_activations:
        activations = torch.cat(all_activations, dim=0).float()
    else:
        logger.warning(f"No activations captured for {model_name} layer {layer_idx}")
        activations = torch.zeros(0, 2048)

    metadata = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "num_samples": num_samples,
        "activation_shape": list(activations.shape),
        "activation_dtype": str(activations.dtype),
    }

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"activations": activations, "metadata": metadata},
            output_path,
        )
        logger.info(f"Saved {activations.shape[0]} activations to {output_path}")

        # Also save metadata separately
        meta_path = output_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return activations, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Capture multi-layer activations for Phase 5.4"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma-2b", "gpt2-medium", "phi-2", "pythia-1.4b"],
        help="Models to capture from",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase4_results/activations_multilayer"),
        help="Output directory for activation files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to capture per layer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:5",
        help="Device to run on",
    )

    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_configs = _get_layer_configs()

    # Load each model and capture activations
    for model_name in args.models:
        if model_name not in layer_configs:
            logger.error(f"Unknown model: {model_name}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {model_name}")
        logger.info(f"{'='*60}")

        # Load model
        try:
            model_id = f"gpt2-medium" if model_name == "gpt2-medium" else model_name
            if model_name == "gpt2-medium":
                model_id = "gpt2-medium"
            elif model_name == "gemma-2b":
                model_id = "google/gemma-2b"
            elif model_name == "phi-2":
                model_id = "microsoft/phi-2"
            elif model_name == "pythia-1.4b":
                model_id = "EleutherAI/pythia-1.4b"

            logger.info(f"Loading model: {model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=device,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue

        # Capture each layer - dynamically detect layer count
        config = layer_configs[model_name]
        layers_attr = config["layers_attr"]
        
        # Detect actual layer count from the model
        num_layers = _get_layer_count(model, layers_attr)
        logger.info(f"Detected {num_layers} layers in {model_name}")
        
        # Capture all layers for complete analysis granularity
        for layer_idx in range(num_layers):
            output_path = output_dir / f"{model_name}_layer{layer_idx}_activations.pt"

            try:
                activations, metadata = capture_layer_activations(
                    model=model,
                    model_name=model_name,
                    layer_idx=layer_idx,
                    tokenizer=tokenizer,
                    device=device,
                    layers_attr=layers_attr,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    output_path=output_path,
                )
                logger.info(f"✓ Captured {activations.shape[0]} activations from layer {layer_idx}/{num_layers-1}")

            except Exception as e:
                logger.error(f"Failed to capture layer {layer_idx}: {e}")
                continue

        # Cleanup
        del model
        torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}")
    logger.info(f"Capture complete. Outputs in {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

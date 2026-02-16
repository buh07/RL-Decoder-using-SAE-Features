#!/usr/bin/env python3
"""
Validate activation capture hooks: measure latency, throughput, and memory overhead.
Tests on sample batches to ensure hooking is efficient before scaling to full datasets.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_registry import get_model, list_models
from activation_capture import create_gpt2_capture


def validate_capture(
    model_name: str,
    layer_indices: Optional[list[int]] = None,
    batch_size: int = 4,
    seq_len: int = 512,
    num_batches: int = 5,
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Validate hooking latency and throughput.
    
    Args:
        model_name: Model slug (e.g., "gpt2").
        layer_indices: Layers to capture. If None, uses [default_probe_layer].
        batch_size: Tokens per batch.
        seq_len: Sequence length.
        num_batches: Number of batches to process.
        device: Device to use ("cuda" or "cpu").
        output_dir: Optional directory to save sample activations.
    
    Returns:
        Dictionary with latency/throughput metrics.
    """
    
    # Load model spec and weights
    spec = get_model(model_name)
    if layer_indices is None:
        layer_indices = [spec.default_probe_layer]
    
    print(f"[validate] Loading {spec.name} from {spec.hf_local_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(spec.hf_local_path),
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    
    # Create dummy input
    tokenizer = AutoTokenizer.from_pretrained(str(spec.hf_local_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[validate] Testing with batch_size={batch_size}, seq_len={seq_len}, num_batches={num_batches}")
    
    # Baseline: inference without hooks
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    
    print("[baseline] Warming up model (1 forward pass)...")
    with torch.no_grad():
        _ = model(dummy_input_ids)
    torch.cuda.synchronize(device) if device == "cuda" else None
    
    print("[baseline] Measuring latency without hooks...")
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_batches):
            _ = model(dummy_input_ids)
    torch.cuda.synchronize(device) if device == "cuda" else None
    baseline_time = time.perf_counter() - start
    baseline_avg = baseline_time / num_batches
    baseline_tps = (batch_size * seq_len) / baseline_avg
    
    print(
        f"  Baseline: {baseline_avg:.4f}s/batch, {baseline_tps:.0f} tokens/s"
    )
    
    # With hooks
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("/tmp/sae_capture_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[hooks] Attaching capture hooks to layers {layer_indices}...")
    with create_gpt2_capture(
        model,
        output_dir=output_dir,
        layer_indices=layer_indices,
        capture_residual=True,
        capture_mlp_hidden=True,
    ) as capture:
        print("[hooks] Warming up model with hooks (1 forward pass)...")
        with torch.no_grad():
            _ = model(dummy_input_ids)
        torch.cuda.synchronize(device) if device == "cuda" else None
        
        print("[hooks] Measuring latency with hooks...")
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(dummy_input_ids)
        torch.cuda.synchronize(device) if device == "cuda" else None
        hooked_time = time.perf_counter() - start
        hooked_avg = hooked_time / num_batches
        hooked_tps = (batch_size * seq_len) / hooked_avg
    
    print(
        f"  With hooks: {hooked_avg:.4f}s/batch, {hooked_tps:.0f} tokens/s"
    )
    
    # Overhead
    overhead_pct = ((hooked_avg - baseline_avg) / baseline_avg) * 100
    print(f"  Overhead: {overhead_pct:.1f}% slowdown")
    
    # Memory check (GPU only)
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[memory] Peak GPU memory: {peak_memory:.2f} GB")
    
    results = {
        "model": model_name,
        "layer_indices": layer_indices,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_batches": num_batches,
        "baseline_s_per_batch": baseline_avg,
        "baseline_tokens_per_s": baseline_tps,
        "hooked_s_per_batch": hooked_avg,
        "hooked_tokens_per_s": hooked_tps,
        "overhead_percent": overhead_pct,
    }
    
    if device == "cuda":
        results["peak_gpu_memory_gb"] = peak_memory
    
    print()
    print("[result] Validation summary:")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}" if val < 10 else f"  {key}: {val}")
        else:
            print(f"  {key}: {val}")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate activation capture hooks")
    parser.add_argument(
        "--model",
        default="gpt2",
        choices=list_models(),
        help="Model to test (default: gpt2)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Layer indices to capture (default: [default_probe_layer])",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for test (default: 4)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for test (default: 512)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of batches to process (default: 5)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save sample activations (default: /tmp/sae_capture_validation)",
    )
    args = parser.parse_args()
    
    results = validate_capture(
        model_name=args.model,
        layer_indices=args.layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_batches=args.num_batches,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

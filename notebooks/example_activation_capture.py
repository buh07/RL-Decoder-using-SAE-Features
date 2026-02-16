#!/usr/bin/env python3
"""
Example: Using activation capture hooks with GPT-2.

This script demonstrates:
1. Loading available models
2. Initializing activation capture
3. Running inference with hooks
4. Inspecting saved activation shards
5. Computing basic statistics
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_registry import get_model, list_models
from activation_capture import create_gpt2_capture


def example_list_models():
    """Show available models."""
    print("=" * 60)
    print("STEP 1: List Available Models")
    print("=" * 60)
    print()
    
    models = list_models()
    print(f"Available {len(models)} models for capture:")
    for model_name in models:
        spec = get_model(model_name)
        print(f"  • {model_name:20} → {spec.name} ({spec.num_layers} layers, probe={spec.default_probe_layer})")
    print()


def example_load_model():
    """Load GPT-2 model and tokenizer."""
    print("=" * 60)
    print("STEP 2: Load GPT-2 Model & Tokenizer")
    print("=" * 60)
    print()
    
    spec = get_model("gpt2")
    print(f"Loading {spec.name} from local cache...")
    print(f"  Path: {spec.hf_local_path}")
    print(f"  Architecture: {spec.num_layers} layers, {spec.hidden_dim}D hidden, {spec.mlp_dim}D MLP")
    print()
    
    model = AutoModelForCausalLM.from_pretrained(
        str(spec.hf_local_path),
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(str(spec.hf_local_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, spec


def example_capture_activations(model, tokenizer, spec):
    """Capture activations from layer 6 while running inference."""
    print("=" * 60)
    print("STEP 3: Capture Activations from Layer 6")
    print("=" * 60)
    print()
    
    output_dir = Path("/tmp/sae_example_activations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layer_to_capture = [spec.default_probe_layer]  # Layer 6 for GPT-2
    print(f"Capturing from layers: {layer_to_capture}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Sample prompts
    prompts = [
        "The answer to 2+2 is",
        "If I have three apples and two oranges,",
        "The capital of France is",
        "To solve this problem, first",
        "The quick brown fox jumps over",
    ]
    
    print(f"Running inference on {len(prompts)} sample prompts...")
    with create_gpt2_capture(
        model,
        output_dir=output_dir,
        layer_indices=layer_to_capture,
        capture_residual=True,
        capture_mlp_hidden=True,
    ) as capture:
        for i, prompt in enumerate(prompts):
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(model.device)
            
            # Forward pass (hooks capture activations)
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=False)
            
            print(f"  [{i+1}/{len(prompts)}] Processed: '{prompt[:40]}...'")
    
    print()
    print("Capture complete! Activations saved to disk.")
    return output_dir


def example_inspect_activations(output_dir):
    """Load and inspect saved activations."""
    print("=" * 60)
    print("STEP 4: Inspect Saved Activation Shards")
    print("=" * 60)
    print()
    
    # Find all shard files
    pt_files = sorted(output_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} activation shard(s):")
    print()
    
    for shard_path in pt_files:
        meta_path = shard_path.with_suffix(".meta.json")
        
        # Load shard
        payload = torch.load(shard_path, map_location="cpu")
        activations = payload["activations"]
        seq_lens = payload["seq_lens"]
        
        # Load metadata
        import json
        meta = json.loads(meta_path.read_text())
        
        print(f"  Shard: {shard_path.name}")
        print(f"    Shape: {tuple(activations.shape)} (batch, seq_len, hidden_dim)")
        print(f"    Dtype: {activations.dtype}")
        print(f"    Sequences: {meta['num_sequences']}")
        print(f"    Tokens: ~{meta['token_count']}")
        print(f"    Key: {meta['key']}")
        print(f"    Checksum: {meta['sha256'][:8]}...")
        print()
        
        # Compute basic stats
        mean_act = activations.mean(dim=(0, 1))
        std_act = activations.std(dim=(0, 1))
        max_act = activations.max(dim=(0, 1))[0]
        
        print(f"    Activation Statistics (across all sequences):")
        print(f"      Mean: {mean_act.mean().item():.4f} ± {std_act.mean().item():.4f}")
        print(f"      Max: {max_act.mean().item():.4f}")
        print(f"      Sparsity (near-zero): {(activations.abs() < 0.01).sum() / activations.numel() * 100:.1f}%")
        print()


def example_manifest():
    """Show the global manifest."""
    print("=" * 60)
    print("STEP 5: Global Manifest")
    print("=" * 60)
    print()
    
    output_dir = Path("/tmp/sae_example_activations")
    manifest_path = output_dir / "manifest.json"
    
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())
        print(f"Manifest at: {manifest_path}")
        print(json.dumps(manifest, indent=2))
    else:
        print("No manifest found (expected if capture was skipped)")


def main():
    """Run all examples."""
    example_list_models()
    model, tokenizer, spec = example_load_model()
    output_dir = example_capture_activations(model, tokenizer, spec)
    example_inspect_activations(output_dir)
    example_manifest()
    
    print()
    print("=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Saved activations are in:", output_dir)
    print("  2. These shards can be used as input to SAE training (Section 4)")
    print("  3. Try with other models: get_model('gpt2-medium'), get_model('pythia-1.4b'), etc.")
    print("  4. Modify layer_indices and capture_mlp_hidden to experiment with different settings")
    print()


if __name__ == "__main__":
    main()

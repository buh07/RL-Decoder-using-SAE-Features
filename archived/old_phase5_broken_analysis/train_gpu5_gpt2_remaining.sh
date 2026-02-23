#!/bin/bash
# Complete GPT2-medium training (layers 15-23)
cd "/scratch2/f004ndc/RL-Decoder with SAE Features"
source .venv/bin/activate

python3 << 'PYTHON_EOF'
import sys
sys.path.insert(0, "/scratch2/f004ndc/RL-Decoder with SAE Features/src")

from pathlib import Path
import subprocess

# Find missing GPT2 layers
all_layers = set(range(24))
existing = {int(f.stem.split("_layer")[1].split("_")[0]) 
            for f in Path("phase5_results/multilayer_transfer/saes").glob("gpt2-medium_layer*_sae.pt")}
missing = sorted(all_layers - existing)

print(f"Missing GPT2-medium layers: {missing}")
print(f"Will train {len(missing)} SAEs")

# Train each missing layer
for layer in missing:
    print(f"\n{'='*70}")
    print(f"Training GPT2-medium layer {layer}")
    print(f"{'='*70}")
    
    cmd = [
        "python3", "phase5/phase5_task4_train_multilayer_saes.py",
        "--activations-dir", "phase4_results/activations_multilayer",
        "--output-dir", "phase5_results/multilayer_transfer/saes",
        "--epochs", "10",
        "--batch-size", "64",
        "--device", "cuda:5"
    ]
    
    # Filter to only train this specific layer
    import torch
    from pathlib import Path
    act_file = Path(f"phase4_results/activations_multilayer/gpt2-medium_layer{layer}_activations.pt")
    
    if act_file.exists():
        # Import and run training directly
        from phase5_task4_train_multilayer_saes import train_sae_on_layer, _load_activations, _normalize_activations
        
        activations = _load_activations(act_file)
        activations = _normalize_activations(activations)
        
        output_path = Path(f"phase5_results/multilayer_transfer/saes/gpt2-medium_layer{layer}_sae.pt")
        
        sae, summary = train_sae_on_layer(
            activations=activations,
            model_name="gpt2-medium",
            layer_idx=layer,
            device="cuda:5",
            epochs=10,
            batch_size=64,
            output_path=output_path
        )
        
        print(f"✓ Completed layer {layer}: loss={summary['final_loss']:.6f}")

print("\n✓ All missing GPT2-medium layers complete")
PYTHON_EOF

# Example Notebooks & Scripts

This directory contains example implementations and notebooks for working with the SAE framework.

## Scripts

### `example_activation_capture.py`

End-to-end example showing how to use the activation capture hooks.

**What it does:**
1. Lists available models (GPT-2, GPT-2-medium, Pythia-1.4B, Gemma-2B, Llama-3-8B, Phi-2)
2. Loads GPT-2 from local cache
3. Attaches capture hooks to layer 6 (default probe layer)
4. Runs inference on 5 sample prompts
5. Saves activations to disk (post-MLP residuals + MLP hidden states)
6. Loads and inspects saved shards
7. Computes statistics (mean, std, sparsity)
8. Displays global manifest

**Run:**
```bash
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features
source .venv/bin/activate
python notebooks/example_activation_capture.py
```

**Expected output:**
- Lists 6 models
- Loads GPT-2 (768D hidden)
- Processes 5 prompts:
  ```
  [1/5] Processed: 'The answer to 2+2 is...'
  [2/5] Processed: 'If I have three apples and two oranges,...'
  ...
  ```
- Saves 2 shards (layer_6_residual, layer_6_mlp_hidden)
- Shows shape: (5, seq_len, 768)
- Displays stats: mean activation, sparsity %, checksum
- Writes manifest.json

**Customize:**
```python
# Change model
spec = get_model("gpt2-medium")  # or pythia-1.4b, etc.

# Change layer
layer_to_capture = [6, 9, 12]  # Capture multiple layers

# Change prompts
prompts = [
    "Your custom prompt here",
    "Another prompt",
]

# Change capture output location
output_dir = Path("./my_activations")
```

## Notebook Placeholders (TBD)

Future notebooks to add:

### `data_exploration.ipynb` (TODO)
- Load dataset shards
- Visualize token distributions
- Spot-check alignment quality
- Compute coverage statistics

### `sae_training.ipynb` (TODO)
- Load activation shards from capture
- Define SAE hyperparameters
- Train SAE with live loss curves
- Save checkpoints

### `feature_analysis.ipynb` (TODO)
- Load trained SAE
- Examine learned features
- Compute purity metrics
- Visualize top-k activations per feature
- Cluster analysis

### `causal_attribution.ipynb` (TODO)
- Load trained SAE + LLM
- Implement feature ablation
- Measure task performance delta
- Generate causal attribution plots

## Directory Structure

```
notebooks/
├── example_activation_capture.py
├── README.md (you are here)
├── data_exploration.ipynb (TBD)
├── sae_training.ipynb (TBD)
├── feature_analysis.ipynb (TBD)
└── causal_attribution.ipynb (TBD)
```

## Common Patterns

### Load and inspect saved activations
```python
import torch
import json
from pathlib import Path

shard_path = Path("/tmp/sae_example_activations/layer_6_residual_shard_000000.pt")
payload = torch.load(shard_path, map_location="cpu")
activations = payload["activations"]  # Shape: (batch, seq_len, hidden_dim)
seq_lens = payload["seq_lens"]        # Shape: (batch,)

meta_path = shard_path.with_suffix(".meta.json")
meta = json.loads(meta_path.read_text())
print(f"Shard shape: {activations.shape}, dtype: {activations.dtype}")
print(f"Checksum: {meta['sha256']}")
```

### Iterate over all shards in a capture
```python
from pathlib import Path

output_dir = Path("/tmp/sae_example_activations")
for shard_path in sorted(output_dir.glob("shard_*.pt")):
    payload = torch.load(shard_path, map_location="cpu")
    activations = payload["activations"]
    # Process shard...
```

### Use different models for capture
```python
from model_registry import get_model

# List available
models = ["gpt2", "gpt2-medium", "pythia-1.4b", "gemma-2b", "llama-3-8b", "phi-2"]

for model_name in models:
    spec = get_model(model_name)
    print(f"{model_name}: {spec.num_layers} layers, probe={spec.default_probe_layer}")
```

## Troubleshooting

**Q: "FileNotFoundError: /scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2"**  
A: Model cache is missing. Check that LLM Second-Order Effects/models directory exists and contains gpt2 weights.

**Q: "CUDA out of memory"**  
A: Use a smaller model (gpt2 vs gpt2-medium) or reduce seq_len in the example.

**Q: Activations are all zeros**  
A: Ensure hooks are firing (check that model forward pass completes without errors). Try reducing model to CPU for debugging.

**Q: "max_activations_per_file" not working as expected**  
A: The buffer flushes when `len(activations[key]) >= max_activations_per_file`. If you have fewer samples, manually call `capture._flush_activations()` or use context manager exit.

## Next Steps

Once basic capture is working:
1. Scale to full datasets (run `datasets/download_datasets.py` and `datasets/tokenize_datasets.py`)
2. Capture activations on real data
3. Train SAE on captured activations (Section 4, TBD)
4. Analyze learned features (TBD notebook)

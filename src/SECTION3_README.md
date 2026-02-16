# Section 3: Model Capture Hooks

This section implements activation capture infrastructure for extracting post-MLP residual streams and MLP hidden states from transformer models. Activations are streamed to disk in fp16 format for later SAE training.

## Overview

Before training sparse autoencoders on model activations, we need to:
1. Select base models and identify which layers to probe
2. Attach PyTorch hooks to capture internal activations
3. Stream activations efficiently to disk without exceeding single-GPU memory
4. Validate that hooking latency is acceptable before scaling to full datasets

## Files

### `model_registry.py`

Defines available models and their architecture parameters.

**Available Models (all cached locally in LLM Second-Order Effects/models):**
- `gpt2` ⭐ **baseline** – 12 layers, 768D hidden, 3072D MLP. Default probe layer: 6.
- `gpt2-medium` – 24 layers, 1024D hidden, 4096D MLP. Default probe layer: 12.
- `pythia-1.4b` – 24 layers, 2048D hidden, 8192D MLP. Default probe layer: 12.
- `gemma-2b` – 18 layers, 2048D hidden, 8192D MLP. Default probe layer: 9.
- `llama-3-8b` – 32 layers, 4096D hidden, 14336D MLP. Default probe layer: 16. (Note: ~30 GB)
- `phi-2` – 32 layers, 2560D hidden, 10240D MLP. Default probe layer: 16.

Each `ModelSpec` includes:
- `num_layers`, `hidden_dim`, `mlp_dim` – architecture details
- `default_probe_layer` – recommended mid-layer for initial experiments
- `hf_local_path` – full path to cached weights
- `notes` – licensing and capabilities

**Usage:**
```python
from model_registry import get_model, list_models

spec = get_model("gpt2")
print(spec.name, spec.num_layers, spec.default_probe_layer)
```

### `activation_capture.py`

Core hooking infrastructure for streaming activations.

**`CaptureConfig`** – Configuration dataclass:
- `output_dir` – where activation shards save
- `layer_indices` – list of layers to capture (e.g., `[6]` for mid-layer, `[6, 9, 12]` for multiple)
- `capture_residual` – whether to capture post-MLP residual streams (default: True)
- `capture_mlp_hidden` – whether to capture MLP hidden states (default: True)
- `dtype` – fp16 (default) for space efficiency
- `max_activations_per_file` – flush threshold (default: 1000 batches)

**`ActivationCapture`** – Main hooking manager:
- Attaches forward hooks to layers
- Buffers activations in memory, flushes to disk when threshold reached
- Saves shards as PyTorch `.pt` files with per-shard `.meta.json` metadata
- Writes global `manifest.json` on close

**`create_gpt2_capture()`** – Factory for GPT-2 style models:
```python
from activation_capture import create_gpt2_capture

capture = create_gpt2_capture(
    model=gpt2_model,
    output_dir=Path("activations/gpt2_layer6"),
    layer_indices=[6],
    capture_residual=True,
    capture_mlp_hidden=True,
)

# Capture runs normally; hooks intercept forward passes
for batch in data_loader:
    output = model(**batch)

capture.close()  # Flush remaining, remove hooks, write manifest
```

**Context Manager Pattern:**
```python
with create_gpt2_capture(model, output_dir, layer_indices=[6]) as capture:
    # Hooks active during this block
    for batch in data_loader:
        output = model(**batch)
# Hooks automatically removed and manifest written
```

**Activation Shards:**
- Path: `output_dir/layer_{idx}_{type}_shard_{number:06d}.pt`
- Tensor shape: `(batch_size, seq_len, hidden_dim)`
- Metadata in parallel `.meta.json` files:
  ```json
  {
    "shard_index": 0,
    "key": "layer_6_residual",
    "shape": [32, 512, 768],
    "dtype": "torch.float16",
    "num_sequences": 32,
    "max_seq_len": 512,
    "token_count": 16384
  }
  ```

### `capture_validation.py`

Validation script: measures hooking latency, throughput, and GPU memory overhead.

**Run Baseline Validation on GPT-2:**
```bash
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features
source .venv/bin/activate
python src/capture_validation.py --model gpt2 --batch-size 4 --seq-len 512 --num-batches 5
```

**Options:**
- `--model` – which model to test (default: `gpt2`)
- `--layers` – which layers to capture (default: model's `default_probe_layer`)
- `--batch-size` – batch size (default: 4)
- `--seq-len` – sequence length (default: 512)
- `--num-batches` – iterations for averaging (default: 5)
- `--device` – `cuda` or `cpu` (auto-detects)
- `--output-dir` – where to save sample shards (default: `/tmp/sae_capture_validation`)

**Output Example:**
```
[validate] Loading GPT-2 Small from /scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2
[validate] Testing with batch_size=4, seq_len=512, num_batches=5

[baseline] Warming up model (1 forward pass)...
[baseline] Measuring latency without hooks...
  Baseline: 0.2541s/batch, 8064 tokens/s

[hooks] Attaching capture hooks to layers [6]...
[hooks] Warming up model with hooks (1 forward pass)...
[hooks] Measuring latency with hooks...
  With hooks: 0.3125s/batch, 6554 tokens/s
  Overhead: 23.0% slowdown

[memory] Peak GPU memory: 6.45 GB

[result] Validation summary:
  model: gpt2
  layer_indices: [6]
  batch_size: 4
  seq_len: 512
  num_batches: 5
  baseline_s_per_batch: 0.2541
  baseline_tokens_per_s: 8064.0
  hooked_s_per_batch: 0.3125
  hooked_tokens_per_s: 6554.0
  overhead_percent: 23.0
  peak_gpu_memory_gb: 6.45
```

**Target Metrics (Phase 1 Falsification Gate):**
- Hooking overhead: **< 50%** (acceptable up to 2x slowdown)
- GPU memory overhead: **< 10% of available** (e.g., <1.2 GB on 12 GB GPU)
- Throughput maintained: **> 1000 tokens/s** on single RTX-class GPU

If metrics exceed these thresholds, investigate:
- Reduce `batch_size` or `seq_len`
- Capture only residual OR MLP hidden (not both)
- Use fewer layers
- Enable gradient checkpointing in model loading

## Quick Start

### 1. List Available Models
```bash
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features
source .venv/bin/activate
python src/model_registry.py
```

### 2. Validate GPT-2 Hooking
```bash
python src/capture_validation.py --model gpt2 --num-batches 5
```

### 3. Use in Your Code
```python
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from activation_capture import create_gpt2_capture
from model_registry import get_model

# Load model
spec = get_model("gpt2")
model = AutoModelForCausalLM.from_pretrained(spec.hf_local_path, local_files_only=True, device_map="cuda")
model.eval()

# Capture layer 6 activations
with create_gpt2_capture(model, Path("activations/gpt2_layer6"), [6]) as capture:
    for batch_input_ids in data_loader:
        with torch.no_grad():
            _ = model(batch_input_ids)

# Activations saved to disk; load for SAE training next step
```

## Next Steps (Section 4)

Once hooking is validated:
1. Run capture on full reasoning datasets (GSM8K, CoT-Collection, etc.)
2. Save activation shards for SAE training
3. Proceed to Section 4 (SAE Architecture & Training)

## Architecture Notes

### Hook Placement

For GPT-2-style models:
- **Post-MLP Residual**: Hook the entire transformer block (`h[i]`) output. This captures the residual stream after attention + MLP.
- **MLP Hidden**: Hook the MLP module (`h[i].mlp`) output. This captures intermediate representations before the decoder projection.

Both capture types are complementary:
- Residual streams show high-level abstract state
- MLP hidden states expose non-linear feature combinations

### Memory Efficiency

- **fp16 storage**: 2 bytes per float (vs 4 for fp32)
- **Streaming to disk**: Keep in-GPU buffer small; flush frequently
- **Batch padding**: Align variable-length sequences to `seq_len` with padding; track actual lengths in metadata

### Activation Alignment

For later SAE training with supervision signals (Section 5):
- Record which text span / token range each activation corresponds to
- Use alignment files (`datasets/alignment/<dataset>/<split>.align.jsonl`) to map activations → reasoning steps
- Metadata `.meta.json` files store shard paths for easy lookup during training

## Troubleshooting

**Q: "CUDA out of memory"**  
A: Reduce `batch_size` or `seq_len` in `capture_validation.py`. Monitor with `nvidia-smi`.

**Q: Hooking overhead > 50%**  
A: Disable MLP hidden capture if only residuals needed. Use half-precision model loading.

**Q: "No attribute .mlp"**  
A: The installed model architecture may differ. Check the model class (e.g., `AutoModel` vs `CausalLM`).

**Q: Validation runs but no activations saved**  
A: Ensure `output_dir` is writable and has enough disk space. Check stderr for I/O errors.

## References

- [Anthropic's SAE Blog](https://transformer-circuits.pub/2023/monosemantic-features)
- [PyTorch Forward Hooks](https://pytorch.org/docs/stable/generated/torch.nn.Module.register_forward_hook.html)
- Section 3 of [overview.tex](../overview.tex) for design rationale

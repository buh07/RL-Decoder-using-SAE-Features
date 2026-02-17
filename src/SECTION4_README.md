# Section 4: SAE Architecture & Training

Implements sparse autoencoders with configurable architecture, training loop, mixed precision optimization, and comprehensive logging.

## Overview

A sparse autoencoder (SAE) learns an overcomplete latent basis to decompose learned activations from transformers.

**Architecture:**
- Encoder: linear projection (input_dim → latent_dim), optional ReLU
- Decoder: linear projection (latent_dim → input_dim)

**Loss Function:**
```
L_total = ||X - X̂||²₂ + λ||h||₁ + β decorr(W) + γ L_probe + δ L_temporal
```

- Reconstruction: MSE loss on decoder output
- L1 sparsity: encourages zero activations
- Decorrelation: penalizes redundancy among decoder atoms (novel component)
- Probe-guided: optional supervision from reasoning task labels
- Temporal smoothness: optional regularization for sequential activations

## Files

### `sae_config.py`

Configuration management for SAE hyperparameters.

**`SAEConfig` dataclass:**
- **Architecture**: `input_dim`, `expansion_factor` (4-8x), `use_relu`
- **Loss weights**: `reconstruction_coeff`, `l1_penalty_coeff`, `decorrelation_coeff`, `probe_loss_coeff`, `temporal_smoothness_coeff`
- **Optimization**: `learning_rate`, `batch_size`, `max_epochs`, `warmup_steps`, `grad_clip`, `use_amp`
- **Evaluation**: `log_every`, `checkpoint_every`, `eval_every`
- **Logging**: `wandb_project`, checkpoint directory

**Preset configurations:**
```python
from sae_config import gpt2_sae_config, gpt2_medium_sae_config, pythia_1_4b_sae_config

# Model-specific configs with tuned hyperparams
config_gpt2 = gpt2_sae_config(expansion_factor=8)
config_medium = gpt2_medium_sae_config(expansion_factor=4)  # Reduced for memory
config_pythia = pythia_1_4b_sae_config(expansion_factor=4)  # Batch size 8 for 2048D
```

**Usage:**
```python
from sae_config import SAEConfig

# Create custom config
config = SAEConfig(
    input_dim=768,
    expansion_factor=8,
    learning_rate=1e-4,
    batch_size=32,
    l1_penalty_coeff=1e-3,
    decorrelation_coeff=0.1,
    use_amp=True,
    wandb_project="my-sae-run",
)

# Save/load
config.save(Path("config.json"))
config_loaded = SAEConfig.load(Path("config.json"))
```

### `sae_architecture.py`

Core SAE implementation with loss computation and utilities.

**`SparseAutoencoder` class:**
- `encode(x)`: Linear projection + optional ReLU
- `decode(h)`: Reconstruction from latents
- `forward(x)`: Encode + decode
- `compute_loss_components(x, x_hat, h, probe_loss=None, temporal_loss=None)`: Returns dict with individual loss terms
- `normalize_decoder()`: Prevents weight collapse
- `get_activation_stats(h)`: Sparsity statistics
- `get_reconstruction_error(x, x_hat)`: Relative error

**Loss Components:**
```python
sae = SparseAutoencoder(config)
x_hat, h = sae(x)
losses = sae.compute_loss_components(x, x_hat, h)
print(losses)
# {
#   'recon_loss': 0.0123,
#   'l1_loss': 0.0456,
#   'decorr_loss': 0.0012,
#   'probe_loss': 0.0,
#   'temporal_loss': 0.0,
#   'total_loss': 0.0591,
#   'total_loss_tensor': tensor(0.0591)  # for backprop
# }
```

**Decorrelation Loss (Novel):**
The decorrelation penalty encourages decoder atoms (columns of decoder weight matrix) to be orthogonal, reducing redundancy:
```
β * Σ(w_i^T w_j)² for i ≠ j
```
Computed on normalized columns. This is a key innovation from overview.tex to improve feature diversity.

### `sae_training.py`

Complete training pipeline with mixed precision, WANDB logging, and checkpointing.

**`ActivationShardDataset`:** Iterable dataset that streams .pt files from disk
- Loads shards on-the-fly (memory efficient)
- Auto-flattens (batch, seq_len, hidden_dim) → (batch*seq_len, hidden_dim)
- Shuffles shards, optional limiting

**`TrainingMetrics`:** Container for per-step metrics

**`SAETrainer`:** Main training orchestrator
- Mixed precision with `torch.cuda.amp` (fp16 on RTX 6000)
- Learning rate warmup + Adam optimizer
- Gradient clipping
- Decoder weight normalization every N steps
- WANDB integration (optional)
- Checkpoint saving at intervals

**Command-line interface:**
```bash
python src/sae_training.py \
  --shard-dir /path/to/activation/shards \
  --model gpt2 \
  --expansion-factor 8 \
  --batch-size 32 \
  --max-epochs 10 \
  --device cuda \
  --checkpoint-dir ./checkpoints \
  --wandb-project my-sae
```

## Quick Start

### 1. Create Sample Activation Shards (optional, for testing)
```bash
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features

# Capture activations from GPT-2 on sample data
python src/capture_validation.py --model gpt2 --num-batches 50 --output-dir /tmp/test_activations

# This creates /tmp/test_activations/layer_6_*.pt shards (~50 * 512 tokens)
```

### 2. Train SAE
```bash
source .venv/bin/activate

# Train on sample shards (quick test)
python src/sae_training.py \
  --shard-dir /tmp/test_activations \
  --model gpt2 \
  --expansion-factor 8 \
  --max-epochs 2 \
  --max-steps 100 \
  --wandb-project test-sae

# Or train on real data (after running datasets/download_datasets.py + capture)
python src/sae_training.py \
  --shard-dir datasets/tokenized/gpt2/gsm8k/train \
  --model gpt2 \
  --expansion-factor 8 \
  --max-epochs 10 \
  --wandb-project rl-decoder-sae
```

### 3. Resume Training
```bash
python src/sae_training.py \
  --shard-dir /tmp/test_activations \
  --model gpt2 \
  --resume-from checkpoints/gpt2-small/sae/sae_768d_step0500.pt \
  --max-steps 500
```

### 4. Inspect Config
```bash
python -c "
from sae_config import gpt2_sae_config
cfg = gpt2_sae_config(expansion_factor=8)
print(cfg)
"
```

## Performance Metrics & RTX 6000 Benchmarks

### Hardware Setup
- **GPU:** NVIDIA RTX 6000 (24 GB VRAM)
- **CPU:** 1-2 threads for data loading
- **RAM:** 32 GB+ recommended

### Memory Profiling

For GPT-2 (768D input, 8x expansion = 6144D latent):

```
Mixed Precision (fp16):
  Input batch: 32 seqs × 512 tokens × 768D × 2 bytes = ~25 MB
  Encoder: 768 × 6144 × 2 = 9.4 MB (parameters)
  Decoder: 6144 × 768 × 2 = 9.4 MB (parameters)
  Activations (h): 32×512 × 6144 × 2 = 200 MB
  Optimizer states (Adam): 2× parameters = 37 MB
  
  Total peak: ~280 MB (well under 24 GB)
  
Full Precision (fp32):
  Same layout but 4 bytes per float = ~560 MB
  Still comfortable on RTX 6000
```

### Throughput Analysis

**Theoretical throughput (RTX 6000):**
- Peak FP16 throughput: ~91 TFLOPS (RTX 6000 specs, not sustained)
- Per-step ops: forward (768×6144 linear) + backward × 2
- Forward: 2 × 768 × 6144 × 32×512 / 2^30 ≈ 310 GFLOPs
- Backward: ~2x forward = 620 GFLOPs
- Total per 32seq batch: ~930 GFLOPs

**Measured Performance (Section 3 baseline):**
- GPT-2 (no capture hook): ~0.25s per batch (512 tokens, batch=4)
- With hooks: ~0.31s (23% overhead)
- SAE training typically 2-3x slower than inference (backprop + loss components)

### Estimated Training Times on RTX 6000

#### Scenario 1: GPT-2 Small, Dataset=GSM8K (~80K examples)

**Setup:**
- Input: 768D → Latent 6144D (8x expansion)
- Batch: 32 sequences
- Each sequence: avg 512 tokens
- Total tokens: ~80K examples × ~100 tokens avg = 8M tokens
- Sequences: 8M tokens / 512 tokens/seq = ~15,625 sequences
- Batches: 15,625 / 32 = ~489 batches

**Time per batch:**
- Forward + loss compute: ~200ms
- Backward pass: ~150ms
- Optimizer step + normalization: ~50ms
- Total: ~400ms/batch ≈ 0.4 seconds

**One epoch:**
- 489 batches × 0.4s = ~196 seconds ≈ 3.3 minutes

**10 epochs:**
- 10 × 3.3 min = **33 minutes** (just training, excluding checkpointing/logging overhead)

#### Scenario 2: GPT-2 Small, Multiple Datasets (GSM8K + CoT-Collection + OpenR1-Math, ~500K combined)

**Setup:**
- Total tokens: ~500K × 100 avg = 50M tokens
- Sequences: 50M / 512 = ~97,656 sequences
- Batches: 97,656 / 32 = ~3,051 batches per epoch

**One epoch:**
- 3,051 × 0.4s = ~1,220 seconds ≈ 20 minutes

**Full training (5 epochs):**
- 5 × 20 min = **100 minutes** ≈ **1.7 hours**

#### Scenario 3: GPT-2 Medium (1024D → 4096D, 4x expansion, reduced from 8x for memory)

**Memory consideration:** 1024×4096 = 4.2M parameters (similar to GPT-2 768×6144=4.7M)

**Batch size:** 16 (reduced from 32 for memory)

**Time per batch:** ~350ms (similar ops but smaller latent)

**~500K dataset, 1 epoch:**
- ~6,104 batches × 0.35s ≈ 35 minutes per epoch
- **5 epochs: ~175 minutes ≈ 3 hours**

#### Scenario 4: All Models + All Datasets (Multi-layer Extraction)

If capturing from **2-3 layers per model** and **4 models** across **500K+ tokens**:

**Per model/layer combo:**
- 1.5-3 hours (Scenarios 2-3)

**Total:**
- 6 layer combos × 2-3 hours = **12-18 hours**
- With checkpointing overhead, logging, evaluation: ~**18-24 hours**
- On single RTX 6000: **Achievable in 1-2 days**

### Optimization Tips for RTX 6000

1. **Mixed Precision (enabled by default):**
   - Saves 2x memory (fp16 vs fp32)
   - No accuracy loss with proper scaling
   - Expected: ~5-10% wall-clock speedup on RTX 6000

2. **Batch Size Tuning:**
   - Increase batch size to saturate GPU (memory allowing)
   - For GPT-2: try batch_size=32-64
   - For larger models: batch_size=8-16

3. **Gradient Accumulation (TBD in future):**
   - If you want larger effective batch without more memory: accumulate gradients

4. **Shard Caching:**
   - Pin frequently-used shards to GPU RAM (future optimization)
   - Current: load shards to CPU, move batch to GPU per-step

5. **Multi-GPU (optional):**
   - RTX 6000 single-GPU training is already efficient
   - Multi-GPU would help with multiple models/layers in parallel, not a single SAE

### Wall-Clock Timing Summary

| Scenario | Data | Model | Layers | Time | Notes |
|----------|------|-------|--------|------|-------|
| Test | 50K tokens | GPT-2 | 1 | 5 min | Quick validation |
| Single Dataset | 8M tokens | GPT-2 | 1 | 30 min | GSM8K only |
| Multi Dataset | 50M tokens | GPT-2 | 1 | 1.5 hrs | GSM8K + CoT + OpenR1 |
| Scaled | 50M tokens | GPT-2-medium | 1 | 3 hrs | Larger input_dim |
| Full Phase 3 | 100M+ tokens | 3 models × 2-3 layers | 6-9 layers | 18-24 hrs | All reasoning data |

**GPU Budget (per overview.tex):** 50-100 RTX-equivalent hours
- Single SAE training: 0.5-1 hours
- 10-20 SAEs (different layers/models): 5-20 hours
- Full Phase 1-3: 20-50 hours
- Leaves 25-50 hours for Phase 4 (frontier models) and iteration

## Architecture Innovation: Decorrelation Loss

Unlike prior SAE work, we include an explicit **decorrelation penalty** on decoder atoms.

**Motivation:**
- Prevent redundant features (multiple atoms mapping to same concept)
- Encourage feature diversity
- Improves interpretability (each feature represents novel combination)

**Implementation:**
```python
W = self.decoder.weight  # (input_dim, latent_dim)
W_normalized = F.normalize(W, p=2, dim=0)  # Normalize columns
gram = W_normalized.T @ W_normalized  # Correlation matrix
gram = gram - torch.eye(...)  # Zero diagonal
decorr_loss = (gram**2).sum()  # L2 of off-diagonal
```

**Effect:**
- Without decorr: features may converge to similar subspaces
- With decorr: forces exploration of diverse directions
- β = 0.1 (adjustable): balances reconstruction vs diversity

## Logging & Visualization

### WANDB Integration

Every training run logs:
- `loss`, `recon_loss`, `l1_loss`, `decorr_loss`
- `activation_fraction` (sparsity)
- `learning_rate`
- `time_per_step_ms`
- Config snapshot
- Final summary (total steps, time, final loss)

**View live dashboard:**
```bash
# After training starts
wandb login  # If first time
# Open https://wandb.ai/your-entity/rl-decoder-sae
```

### Checkpoint Structure

Checkpoints saved as:
```
checkpoints/gpt2-small/sae/
├── sae_768d_step0500.pt
├── sae_768d_step1000.pt
├── sae_768d_final.pt
└── sae_768d_config.json  # Optional: save config per checkpoint
```

Each .pt file contains:
```python
{
  'step': 500,
  'epoch': 2,
  'model_state': {...},      # SAE weights
  'optimizer_state': {...},  # Adam state
  'scheduler_state': {...},  # LR scheduler state
  'config': SAEConfig(...)   # Config object
}
```

Resume via:
```bash
python src/sae_training.py \
  --shard-dir ... \
  --resume-from checkpoints/gpt2-small/sae/sae_768d_step0500.pt
```

## Evaluation & Metrics (Section 6, TBD)

Metrics computed post-training:
- **Purity:** Feature coherence (top-k activation for each neuron)
- **Sparsity:** % of latents zero per batch
- **Reconstruction:** Relative error ||X-X̂||/||X||
- **Causal Impact:** Feature ablations on downstream tasks
- **Decorrelation:** Gram matrix off-diagonal magnitude

## Next Steps (Section 5, TBD)

After training SAEs on activations:
1. **Phase 1:** Train SAE on ground-truth LSTMs/state machines
2. **Phase 2:** Test on synthetic transformers with known circuits
3. **Phase 3:** Integrate probe-guided loss with real reasoning data
4. **Phase 4:** Scale to frontier models (Llama 8B, etc.)

## Troubleshooting

**Q: CUDA out of memory**
- A: Reduce `batch_size` or `expansion_factor`. Check memory with `nvidia-smi`.

**Q: Loss not decreasing**
- A: Check learning rate. Try lr_schedule warmup. Verify data is streaming correctly.

**Q: Features collapse to zero**
- A: Increase `decorrelation_coeff` or `l1_penalty_coeff`. Check decoder normalization.

**Q: Training very slow**
- A: Ensure mixed precision enabled (`use_amp=True`). Profile with torch profiler.

**Q: W&B not logging**
- A: Check `HAS_WANDB` and `wandb login`. Set `--no-wandb` to disable if issues.

## References

- Section 4 of [overview.tex](../overview.tex) for theory
- [Anthropic: Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [PyTorch AMP Guide](https://pytorch.org/docs/stable/amp.html)

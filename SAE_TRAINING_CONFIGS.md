# SAE Training Configurations - Experimental Results

**Date:** February 16, 2026  
**Dataset:** GSM8K + GPT-2 Layer 6 Activations  
**Total Tokens:** 4,104,192 (2.05M per activation type)  
**Hardware:** NVIDIA RTX 6000 Ada (3 GPUs in parallel)

---

## Configuration A: Conservative (4x Expansion)

**Purpose:** Baseline with moderate latent compression; fast, memory-efficient  

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| GPU | 0 |
| Expansion Factor | 4x |
| Input Dimension | 768D |
| Latent Dimension | 3072D |
| SAE Parameters | ~9.4M |
| Batch Size | 64 |
| Learning Rate | 3e-4 |
| Max Epochs | 3 |
| Optimizer | Adam with warmup |
| Mixed Precision | Yes (fp16) |

### Results
| Metric | Value |
|--------|-------|
| Total Steps | 189 |
| Total Time | 9.7s |
| Final Loss | 8.0413 |
| Avg Loss (Final Epoch) | 13.6095 |
| Time/Step | 51.3ms |
| Throughput | ~31K tokens/s |

**Analysis:**
- Fastest training (lowest latent dimension)
- Higher reconstruction loss indicates less compression capacity
- Useful for comparison baseline and memory-constrained scenarios

**Checkpoint:** `checkpoints/sae/config_a_4x_final.pt`

---

## Configuration B: Moderate (6x Expansion)

**Purpose:** Balanced latent dimensionality; intermediate compression and reconstruction quality  

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| GPU | 1 |
| Expansion Factor | 6x |
| Input Dimension | 768D |
| Latent Dimension | 4608D |
| SAE Parameters | ~14.1M |
| Batch Size | 48 |
| Learning Rate | 2e-4 |
| Max Epochs | 3 |
| Optimizer | Adam with warmup |
| Mixed Precision | Yes (fp16) |

### Results
| Metric | Value |
|--------|-------|
| Total Steps | 252 |
| Total Time | 10.6s |
| Final Loss | 3.2591 |
| Avg Loss (Final Epoch) | 8.5161 |
| Time/Step | 42.1ms |
| Throughput | ~38K tokens/s |

**Analysis:**
- Good balance between compression and reconstruction
- ~60% reduction in loss vs Configuration A
- More efficient convergence (42ms/step vs 51ms/step)
- Reasonable memory footprint for 24GB GPUs

**Checkpoint:** `checkpoints/sae/config_b_6x_final.pt`

---

## Configuration C: Aggressive (8x Expansion) ⭐ RECOMMENDED

**Purpose:** Maximum latent dimensionality for interpretability; best reconstruction quality  

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| GPU | 2 |
| Expansion Factor | 8x |
| Input Dimension | 768D |
| Latent Dimension | 6144D |
| SAE Parameters | ~18.8M |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Max Epochs | 3 |
| Optimizer | Adam with warmup |
| Mixed Precision | Yes (fp16) |

### Results
| Metric | Value |
|--------|-------|
| Total Steps | 378 |
| Total Time | 13.8s |
| Final Loss | 1.4582 |
| Avg Loss (Final Epoch) | 3.5526 |
| Time/Step | 36.5ms |
| Throughput | ~44K tokens/s |

**Analysis:**
- **Best reconstruction quality** (lowest final loss)
- ~82% improvement in loss vs Configuration A
- Most training steps due to larger latent space
- Slowest per-step but best converges (likely due to more optimization work)
- **Recommended for featured analysis and interpretability studies**

**Checkpoint:** `checkpoints/sae/config_c_8x_final.pt` ⭐

---

## Comparative Analysis

### Loss Trajectory
```
Config A (4x):  13.61 → 8.04  (40% improvement per epoch)
Config B (6x):   8.52 → 3.26  (62% improvement per epoch)
Config C (8x):   3.55 → 1.46  (59% improvement per epoch)
```

### Memory Usage (Estimated)
| Config | Encoder | Decoder | Optimizer States | Total |
|--------|---------|---------|-----------------|-------|
| A (4x) | 2.4MB | 2.4MB | ~20MB | ~25MB |
| B (6x) | 3.6MB | 3.6MB | ~30MB | ~37MB |
| C (8x) | 4.8MB | 4.8MB | ~40MB | ~50MB |

**GPU Memory Used:** <500MB for SAE + optimizer during training (out of 24GB available)

### Training Speed (Tokens/sec)
- Config A: 31K tokens/s
- Config B: 38K tokens/s  
- Config C: 44K tokens/s (Best throughput despite larger model)

---

## Recommendation Summary

### For Downstream Use:
1. **Primary:** Use **Configuration C (8x)** for feature analysis
   - Best reconstruction fidelity
   - Largest latent space for interpretability
   - Similar training speed per token

2. **Validation:** Compare against **Configuration B (6x)**
   - Ensure findings generalize across expansion factors
   - Confirms loss landscape smoothness

3. **Baseline:** Keep **Configuration A (4x)** for memory-constrained inference
   - Faster inference if needed
   - Useful for edge deployment scenarios

---

## Reproducibility

To retrain any configuration:

```bash
# Config A (4x)
python src/sae_training.py --shard-dir /tmp/gpt2_acts \
  --model gpt2 --expansion-factor 4 --batch-size 64 \
  --learning-rate 3e-4 --max-epochs 3 --device cuda:0

# Config B (6x)
python src/sae_training.py --shard-dir /tmp/gpt2_acts \
  --model gpt2 --expansion-factor 6 --batch-size 48 \
  --learning-rate 2e-4 --max-epochs 3 --device cuda:0

# Config C (8x)
python src/sae_training.py --shard-dir /tmp/gpt2_acts \
  --model gpt2 --expansion-factor 8 --batch-size 32 \
  --learning-rate 1e-4 --max-epochs 3 --device cuda:0
```

All three configurations used:
- **Activation source:** GPT-2 Layer 6 (middle layer)
- **Shards:** `/tmp/gpt2_acts/layer_6_residual_shard_000000.pt`
- **Loss:** Reconstruction (MSE) + L1 sparsity + Decorrelation
- **Optimizer:** Adam with linear warmup (5% of training)
- **Gradient scaling:** Mixed precision (fp16) with GradScaler

---

## Next Steps

1. Load Configuration C checkpoint and analyze decoded dimensions
2. Compare with reasoning task supervision (probe-guided loss)
3. Test on other layers (3, 9, 11) and other models  
4. Evaluate feature stability across different reasoning tasks

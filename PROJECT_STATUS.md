# RL-Decoder with SAE Features — Project Status

**Last Updated:** February 23, 2026

---

## Phase Structure

The project follows a four-phase research arc, each phase building on the previous.

| Phase | Name | Status | Key Result |
|-------|------|--------|------------|
| **1** | Ground-Truth Validation | ✅ Complete | R²=0.967–0.999 on BFS/Stack/Logic |
| **2** | Multi-Layer SAE Training | ✅ Complete | 24 GPT-2 SAEs (12×) + 98 SAEs across 4 models |
| **3** | Reasoning Flow Tracing | ✅ Complete | Feature-flow heatmaps; ~51% sparsity observed |
| **4** | Arithmetic Feature Probing | 🔄 In Progress | Data collector + 3 experiment scripts written |

---

## Phase 1 — Ground-Truth Validation ✅

**Scripts:** `phase1/`
**Results:** `phase1_results/`

Trained SAEs on three controlled environments where the ground-truth latent state is known:

| Environment | R² (best) | R² (worst) | Expansions tested |
|-------------|-----------|------------|-------------------|
| BFS graph traversal | 0.987 | 0.976 | 4×, 8×, 12× |
| Stack machine | 0.989 | 0.967 | 4×, 8×, 12× |
| Logic puzzle | 0.999 | 0.990 | 4×, 8×, 12× |

All pass R²>0.95, confirming the SAE can reconstruct known latent states.

---

## Phase 2 — Multi-Layer SAE Training ✅

**Scripts:** `phase2/`
**Results:** `phase2_results/`

```
phase2_results/
  activations/        — 98 activation files (4 models × all layers)
  saes_gpt2_12x/      — 24 SAEs for GPT-2 medium, 12× expansion
  saes_all_models/    — 98 SAEs across GPT-2 medium, Phi-2, Gemma-2B, Pythia-1.4B
```

Key facts:
- GPT-2 medium SAEs: 12× expansion (12 288 features), trained per layer (0–23)
- All-model SAEs: 8× expansion, one per layer per model
- Training fixed: `use_relu=True`, `normalize_decoder()` after every step, correct decorrelation loss

---

## Phase 3 — Reasoning Flow Tracing ✅

**Scripts:** `phase3/`
**Results:** `phase3_results/reasoning_flow/`

`reasoning_flow_tracer.py` registers forward hooks on every GPT-2 medium block, encodes through the layer's SAE, and records active features per token.  Interactive analysis is available in `interactive_reasoning_tracker.ipynb`.

Key finding: ~51% of 12 288 features are active (dense), and computation vs. background tokens differ by only ±58 features (~1%). This motivates Phase 4's targeted probing approach instead of raw feature-count comparisons.

---

## Phase 4 — Arithmetic Feature Probing 🔄

**Scripts:** `phase4/`
**Results:** `phase4_results/`  (written by the scripts below)

Three progressively stronger experiments to locate arithmetic features:

| Script | Experiment | Question answered |
|--------|-----------|-------------------|
| `arithmetic_data_collector.py` | Shared data foundation | Collect SAE features at `=`, operand, result tokens for 200 GSM8K examples |
| `arithmetic_value_probe.py` | A — Ridge regression | Which layer's SAE features best linearly predict the result value C? |
| `feature_coactivation.py` | B — Co-activation analysis | Which features fire specifically at operands vs. result tokens? |
| `causal_patch_test.py` | C — Causal patching | Do swapping SAE features between examples causally change the model's prediction? |

**Run order:**
```bash
# 1. Collect (GPU, ~20 min)
CUDA_VISIBLE_DEVICES=7 .venv/bin/python3 phase4/arithmetic_data_collector.py --device cuda:0

# 2. Value probe (CPU)
.venv/bin/python3 phase4/arithmetic_value_probe.py

# 3. Co-activation (CPU)
.venv/bin/python3 phase4/feature_coactivation.py

# 4. Causal patch (GPU)
CUDA_VISIBLE_DEVICES=7 .venv/bin/python3 phase4/causal_patch_test.py --device cuda:0
```

---

## Directory Map

```
RL-Decoder with SAE Features/
├── src/                     — SAE library (architecture, training, config)
├── datasets/                — GSM8K and other raw data
│
├── phase1/                  — Ground-truth environment scripts
├── phase1_results/          — BFS/Stack/Logic SAE checkpoints + R² metrics
│
├── phase2/                  — Multi-layer activation capture + SAE training
├── phase2_results/
│   ├── activations/         — 98 activation files (gpt2-medium_layer{N}_activations.pt …)
│   ├── saes_gpt2_12x/       — 24 × 12× GPT-2 SAEs
│   └── saes_all_models/     — 98 SAEs across 4 frontier models
│
├── phase3/                  — Reasoning flow tracer + interactive notebook
├── phase3_results/
│   └── reasoning_flow/      — Feature-flow PNGs, JSON
│
├── phase4/                  — Arithmetic feature probing (3 experiments)
├── phase4_results/
│   ├── collection/          — gsm8k_arithmetic_dataset.pt (shared input)
│   ├── probe/               — R² by layer, top features per layer
│   ├── coactivation/        — Selectivity heatmaps, category distribution
│   └── patching/            — Δlog_prob by layer, causal effect heatmap
│
├── archived/                — Superseded / broken work (kept for reference)
│   ├── old_phase3_single_layer_probe/   (GPT-2 small single-layer probes)
│   ├── old_phase3_results/
│   ├── old_phase4_broken_sae_training/  (use_relu=False era)
│   ├── old_phase4_results/
│   ├── old_phase5_broken_analysis/
│   ├── old_phase5_broken_results/
│   └── old_phase6_docs/
│
└── docs/                    — Design documents and reports
```

---

## Key Environment Details

- Python 3.12.3, CUDA 12.8, PyTorch 2.6
- 4 GPUs (0–3) on primary node; additional GPUs (5, 7) available
- Activate with `source setup_env.sh` or `.venv/bin/python3`
- `weights_only=False` required for `torch.load` on SAE checkpoints (contain PosixPath)
- Hook target for GPT-2 activations: `model.transformer.h[i]` (full block, not `.mlp`)

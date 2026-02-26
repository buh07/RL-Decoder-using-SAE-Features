# RL-Decoder with SAE Features

**Mechanistic Interpretability of Arithmetic Reasoning in LLMs via Sparse Autoencoders**

*Benjamin Huh, February 2026*

This repository implements a phased framework for extracting, validating, and analyzing
reasoning features from LLM activations using sparse autoencoders (SAEs). Phases 1–5 are
complete, with the primary target being GPT-2 medium (24-layer, 1024D). Phase 6 (RL decoder)
has scaffolding in place but has not been executed yet.

## Quick Navigation

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** — Current status, all key results, run commands
- **[overview.tex](overview.tex)** — Formal design document
- **[TODO.md](TODO.md)** — Outstanding tasks

## Setup

```bash
cd '/scratch2/f004ndc/RL-Decoder with SAE Features'
source setup_env.sh          # activates .venv and sets PYTHONPATH
# or directly:
.venv/bin/python3 <script>
```

## Phase Structure

| Phase | Name | Status | Key Result |
|-------|------|--------|------------|
| **1** | Ground-Truth Validation | ✅ | R²=0.967–0.999 on BFS/Stack/Logic |
| **2** | Multi-Layer SAE Training | ✅ | 24 TopK SAEs for GPT-2 medium (12×) |
| **3** | Reasoning Flow Tracing | ✅ | Layer×token feature heatmaps; ~50% active (~50% sparse) |
| **4** | Arithmetic Feature Probing | ✅ | R²=0.977; +0.107 Δlog_prob at L22 (subspace) |
| **5** | Feature Interpretation + Steering | ✅ | Mean-diff steering uniformly negative → distributed encoding |

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for full details and key findings.

Phase 1 note: `phase1_results/` contains multiple generations (including early runs); use the
canonical paths listed in `PROJECT_STATUS.md`.

## Project Structure

```
RL-Decoder with SAE Features/
│
├── src/                        # Core SAE library
│   ├── sae_architecture.py     # SparseAutoencoder (TopK or ReLU, decorr loss)
│   ├── sae_config.py           # SAEConfig dataclass + model presets
│   ├── sae_training.py         # SAETrainer + ActivationShardDataset
│   ├── activation_capture.py   # Hook-based activation streaming to disk
│   ├── capture_gsm8k_activations.py  # Capture all 24 GPT-2-medium layers
│   ├── capture_validation.py   # Latency/throughput benchmark
│   ├── model_registry.py       # ModelSpec registry (GPT-2, Phi-2, Gemma-2B, Pythia-1.4B)
│   ├── sae_evaluate.py         # Post-training evaluation utilities
│   ├── SECTION3_README.md      # Activation capture infrastructure docs
│   └── SECTION4_README.md      # SAE architecture & training docs
│
├── datasets/
│   ├── download_datasets.py    # Download GSM8K and reasoning corpora
│   ├── tokenize_datasets.py    # Tokenize + shard for SAE training
│   └── raw/gsm8k/              # GSM8K train.jsonl / test.jsonl
│
├── phase1/                     # Ground-truth validation
│   ├── phase1_environments.py  # BFS, StackMachine, LogicPuzzle environments
│   ├── phase1_training.py      # Train + evaluate SAE on ground-truth data
│   ├── phase1_ground_truth.py  # Ground-truth latent state generators
│   └── phase1_orchestrate.sh   # Multi-GPU orchestration
│
├── phase1_results/             # BFS/Stack/Logic SAE checkpoints + R² metrics
│
├── phase2/                     # Multi-layer SAE training
│   ├── capture_activations.py  # Capture all layers for 4 models
│   └── train_multilayer_saes.py  # Train one SAE per layer (TopK, 12×)
│
├── phase2_results/
│   ├── activations/            # 98 activation files (model_layer_activations.pt)
│   ├── saes_gpt2_12x/          # 24 ReLU SAEs (initial run, superseded)
│   ├── saes_all_models/        # 98 SAEs across 4 models, 8× expansion
│   └── saes_gpt2_12x_topk/    # 24 TopK SAEs — primary checkpoints used in phases 4–5
│
├── phase3/                     # Reasoning flow tracing
│   ├── reasoning_flow_tracer.py  # Layer×token SAE feature heatmaps
│   └── interactive_reasoning_tracker.ipynb
│
├── phase3_results/
│   └── reasoning_flow/         # PNGs + reasoning_flow.json
│
├── phase4/                     # Arithmetic feature probing (3 experiments)
│   ├── arithmetic_data_collector.py  # Collect SAE features at <<expr=result>>
│   ├── arithmetic_value_probe.py     # Exp A: ridge probe (R²=0.977)
│   ├── feature_coactivation.py       # Exp B: bridge/encoder/tracker taxonomy
│   └── causal_patch_test.py          # Exp C: causal patching Δlog_prob
│
├── phase4_results/
│   ├── collection/             # gsm8k_arithmetic_dataset.pt (v1, ReLU SAEs)
│   ├── probe/                  # R² by layer, top features (v1)
│   ├── coactivation/           # Selectivity heatmaps (v1)
│   ├── patching/               # Δlog_prob by layer (v1)
│   └── topk/                   # Current results (TopK SAEs + subspace steering)
│       ├── collection/         # gsm8k_arithmetic_dataset.pt (674 records)
│       ├── probe/              # top_features_per_layer.json
│       ├── coactivation/       # activation_rates.npz
│       └── patching/           # patching_results.json
│
├── phase5/                     # Feature interpretation + causal steering
│   ├── feature_interpreter.py  # Per-feature JSON cards (layers 14–23)
│   └── arithmetic_steerer.py   # Mean-diff steering in SAE feature space
│
├── phase5_results/
│   ├── feature_interpretations/  # Per-feature JSON cards + L22 summary
│   └── steering/               # Heatmap + steering_results.json
│
├── sae/                        # SAE training orchestration utilities
│   ├── sae_train_all_gsm8k_orchestrator.py
│   ├── sae_train_missing_expansions.py
│   ├── train_and_eval_all_saes.py
│   ├── sae_train_all_gsm8k.sh
│   └── monitor_sae_training.sh
│
├── checkpoints/                # Standalone single-layer SAE checkpoints
├── assets/tokenizers/gpt2/     # Vendored GPT-2 tokenizer files
├── sae_logs/                   # Training + execution logs
│
├── archived/                   # Superseded work (kept for reference)
│   ├── old_phase3_single_layer_probe/   # GPT-2 small single-layer probes
│   ├── old_phase3_results/
│   ├── old_phase4_broken_sae_training/  # use_relu=False era
│   ├── old_phase4_results/
│   ├── old_phase5_broken_analysis/
│   └── old_phase5_broken_results/
│
└── docs/
    └── archived_reports/       # Execution reports from earlier runs
```

## Re-running Phases

All commands assume you are in the repo root with `.venv` active.

### Phase 2 — Retrain TopK SAEs
```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 phase2/train_multilayer_saes.py \
  --activations-dir phase2_results/activations \
  --output-dir phase2_results/saes_gpt2_12x_topk/saes \
  --model-filter gpt2-medium --expansion-factor 12 \
  --use-topk --topk-k 3686 --epochs 10 --device cuda:0
```

### Phase 3 — Reasoning Flow
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 phase3/reasoning_flow_tracer.py \
  --saes-dir phase2_results/saes_gpt2_12x_topk/saes \
  --activations-dir phase2_results/activations \
  --output-dir phase3_results/reasoning_flow --device cuda:0
```

### Phase 4 — Arithmetic Probing (run in order)
```bash
# 1. Collect (GPU, ~20 min)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 phase4/arithmetic_data_collector.py --device cuda:0
# 2. Value probe (CPU)
.venv/bin/python3 phase4/arithmetic_value_probe.py
# 3. Co-activation (CPU)
.venv/bin/python3 phase4/feature_coactivation.py
# 4. Causal patch (GPU)
CUDA_VISIBLE_DEVICES=7 .venv/bin/python3 phase4/causal_patch_test.py --device cuda:0
```

### Phase 5 — Interpretation + Steering
```bash
# 1. Feature interpreter (CPU, ~2 min)
.venv/bin/python3 phase5/feature_interpreter.py \
  --dataset phase4_results/topk/collection/gsm8k_arithmetic_dataset.pt \
  --probe   phase4_results/topk/probe/top_features_per_layer.json \
  --output  phase5_results/feature_interpretations

# 2. Steering (GPU, ~10 min)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 phase5/arithmetic_steerer.py \
  --dataset      phase4_results/topk/collection/gsm8k_arithmetic_dataset.pt \
  --saes-dir     phase2_results/saes_gpt2_12x_topk/saes \
  --activations-dir phase2_results/activations \
  --output       phase5_results/steering --device cuda:0
```

## Environment Notes

- Python 3.12.3, CUDA 12.8, PyTorch 2.6
- 4 GPUs (0–3) on primary node; additional GPUs (5, 7) available
- `weights_only=False` required for `torch.load` on SAE checkpoints (contain `PosixPath`)
- Hook target for GPT-2 activations: `model.transformer.h[i]` (full block, not `.mlp`)

## References

- Anthropic: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features)
- Anthropic: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- See [overview.tex](overview.tex) for full bibliography

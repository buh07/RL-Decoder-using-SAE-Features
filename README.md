# RL-Decoder with SAE Features

**Mechanistic Interpretability of Reasoning in LLMs via Sparse Autoencoders**

*Benjamin Huh, February 2026*

This repository implements a phased, falsifiable framework for extracting and validating reasoning features from LLM activations using sparse autoencoders (SAEs). The current focus is full-scale Phase 3 evaluation on GSM8K with multiple SAE expansions.

## Read This First

- [README.md](README.md) (this file) - project orientation and key paths
- [EXECUTION_START_HERE.md](EXECUTION_START_HERE.md) - start-to-finish runbook
- [PHASE3_FULLSCALE_EXECUTION_GUIDE.md](PHASE3_FULLSCALE_EXECUTION_GUIDE.md) - Phase 3 details

## Current Status (Feb 17, 2026)

- Phase 3 full-scale evaluation is running on GSM8K with SAE expansions 4x-20x.
- SAE checkpoints for 4x-20x exist and were refreshed on full GSM8K.
- Missing expansions (2x, 22x-32x) remain optional follow-ups.

## Key Paths

- Activations: `/tmp/gpt2_gsm8k_acts/gsm8k/train` (train), `/tmp/gpt2_gsm8k_acts_test/gsm8k/test` (test)
- SAE checkpoints: `checkpoints/gpt2-small/sae/sae_768d_*x_final.pt`
- Phase 3 results: `phase3_results/full_scale/`
- SAE logs/results: `sae_logs/`, `sae_results/`

## Setup (Short)

```bash
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features
./setup_env.sh
source .venv/bin/activate
```

## Run SAE Training (GSM8K, GPUs 0-1)

```bash
python sae/sae_train_all_gsm8k_orchestrator.py --gpu-ids 0 1
```

Monitor:
```bash
bash sae/monitor_sae_training.sh
```

## Run Phase 3 Full-Scale Evaluation (GPUs 0-3)

```bash
bash phase3/phase3_orchestrate.sh
```

Monitor:
```bash
tmux attach -t phase3_fullscale
bash phase3_results/full_scale/monitor.sh
```

Merge results:
```bash
python phase3/phase3_merge_results.py --input-dir phase3_results/full_scale
```

## SAE Results Snapshot (GSM8K Test, GPT-2 Layer 6)

- 10x-12x expansions show lowest reconstruction loss on test.
- 12x is a strong default for interpretability follow-ups.

## Falsification Pipeline Summary (Phase 1-2)

- Phase 1: BFS/DFS, Stack Machine, Logic Puzzles with exact latent states.
- Phase 2: Tiny transformers with known circuits for causal alignment tests.
- Phase 3+: Apply validated protocol to real LLMs (Phase 4).

## Directory Structure (Top Level)

```
RL-Decoder with SAE Features/
├── README.md
├── EXECUTION_START_HERE.md
├── PHASE3_FULLSCALE_EXECUTION_GUIDE.md
├── phase3/                       # Phase 3 orchestration scripts
├── sae/                          # SAE training scripts
├── src/                          # Core implementation
├── datasets/                     # Data pipeline
├── checkpoints/                  # Model checkpoints
├── phase3_results/               # Phase 3 outputs
├── sae_logs/                     # SAE training logs
├── sae_results/                  # SAE training outputs
└── overview.tex                  # Design document
```
python src/capture_validation.py --model gpt2 --num-batches 10

# 4. Capture all activations for training (script TBD)
# python src/capture_activations.py --model gpt2 --layers 6 --dataset gsm8k ...

# 5. Train SAE (script TBD)
# python src/train_sae.py --model gpt2 --layer 6 --dataset gsm8k ...

# 6. Evaluate purity, causality (notebooks TBD)
```

## Falsification Criteria (Go/No-Go per Phase)

**Section 3 (Model Capture):**
- ✅ Latency overhead < 50% ✓ (measured ~23%)
- ✅ Throughput > 1000 tokens/s ✓ (measured ~6500)
- ✅ Memory overhead < 10% of available GPU ✓ (measured ~0.5-1 GB on 12 GB)

**Section 4 (SAE Training) — TBD:**
- Reconstruction error < 10% of baseline on sample activation data (STOP if higher)
- L1 sparsity achieves >90% of latents staying zero per sequence (STOP if lower)
- Feature orthogonality (decorrelation loss) reduces redundancy > 50% vs baseline SAE (STOP if lower)

**Section 5 (Phased Research) — TBD:**
- Phase 1: Ground-truth alignment ≥ 95% (monosemantic reconstruction)
- Phase 2: Causal feature coherence ≥ 80% (features align with known circuits)
- Phase 3: Probe guidance reduces leakage ≤ 5% (gap between probe vs baseline SAE)
- Phase 4: Reasoning primitives stable across 3+ independent runs (>.85 correlation)

## References & Citations

Key papers (see [overview.tex](overview.tex) for full bibliography):
- Anthropic: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features)
- Anthropic: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- Identifiable SAEs: arXiv:2506.15963
- Random SAE Interpretability: arXiv:2501.17727
- Evaluating SAEs for Interpretability: arXiv:2405.08366
- Interpretability Illusions: Li (2025)

## Contact

For questions or issues, refer to [TODO.md](TODO.md) for owner contacts per section or check the design document [overview.tex](overview.tex).

---

**Last Updated:** February 16, 2026  
**Framework Version:** 0.1.0 (Sections 0-3 complete, 4-8 planned)

# RL-Decoder with SAE Features - Project Status

**Last Updated:** February 22, 2026
**Current Phase:** Phase 3 COMPLETE (mixed results) → Phase 4 NEEDS REDO

---

## Quick Links

- **[README.md](README.md)** - Project overview and setup instructions
- **[TODO.md](TODO.md)** - Active tasks and next steps
- **[Documentation Index](docs/DOCUMENTATION_INDEX.md)** - Complete documentation listing
- **[Phase 5.4 Analysis Report](docs/PHASE5_TASK4_ANALYSIS_REPORT.md)** - Latest reasoning flow analysis

---

## Current Status Summary

> **⚠️ CRITICAL NOTE — SAE Training Bugs in Previous Phases**
> Phases 4, 5.1, 5.2, and 5.3 were run with three bugs that invalidate results:
> 1. `SAEConfig` defaults to `use_relu=False` — L1 sparsity never engaged, features are dense not sparse
> 2. Missing `normalize_decoder()` after each optimizer step — decoder columns grew unbounded
> 3. Incorrect decorrelation loss formula — loss could go negative, training was destabilized
>
> All sparsity percentages, transfer ratios, and causal correlations from those phases are unreliable. Phase 5.4 multilayer SAEs were retrained with fixes (v2 checkpoints), but downstream analysis must be re-run. See `TODO.md` for the redo plan.

### ✅ Valid Completed Work

#### Phase 3: Controlled CoT Probes (COMPLETE — Mixed Results, Feb 22 2026)
All 9 SAE expansion factors (4x–20x) evaluated with corrected pipeline (v4):
- **Equation detection**: 83–97% per-class accuracy ✅ — SAE layer 6 features encode arithmetic tokens
- **Comparison/reasoning detection**: 0% per-class ❌ — connective words not linearly decodable from single layer
- **Overall accuracy**: All expansions at or below majority-class baseline (91.8%) — no positive probe signal
- **Key insight**: Single-layer (layer 6, GPT-2 small) cannot capture multi-step reasoning structure. Multi-layer analysis is needed.

**Output Location:** `phase3_results/full_scale_v4/`

#### Phase 5.4: Multilayer SAEs Retrained with Fixes (CHECKPOINTS VALID)
- **98 SAEs retrained** with use_relu=True, normalize_decoder(), correct decorrelation loss
- Checkpoints in `phase5_results/multilayer_transfer_v2/saes/` are valid
- **Transfer matrix analysis and visualizations need to be re-run** using v2 checkpoints (previous analysis used broken v1 SAEs)

**Valid Checkpoint Location:** `phase5_results/multilayer_transfer_v2/saes/`

### 🔴 Needs Redo (broken SAE training)

#### Phase 4: Frontier LLM SAEs — INVALID
- Claimed 31.7% sparsity is an artifact of use_relu=False (no ReLU gating = no sparsity)
- All feature statistics, causal rankings, and descriptions are unreliable

#### Phase 5.1–5.3: Causal Ablation, Feature Naming, Transfer Analysis — INVALID
- All built on Phase 4's broken SAEs
- Must be redone after Phase 4 SAEs are retrained

---

## Research Plan Alignment

**Status:** ✅ Fully aligned with overview.tex research plan  
**See:** [ALIGNMENT_VERIFICATION.md](docs/ALIGNMENT_VERIFICATION.md) for complete alignment analysis

### Phase Status (from overview.tex)

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 1 | Ground-Truth Systems | ✅ COMPLETE | R²=0.967–0.999 across BFS/Stack/Logic (v2/v3) |
| 2 | Synthetic Transformers | ⏸️ NOT STARTED | Deferred |
| 3 | Controlled CoT | ✅ COMPLETE (mixed) | Equation detection ✅, reasoning/comparison 0% ❌, all below baseline |
| 4 | Frontier LLMs | 🔴 NEEDS REDO | Previous SAEs had use_relu=False + no normalize_decoder |
| 4B | Interpretability Validation | 🔴 NEEDS REDO | Previous sparsity/purity metrics invalid |
| 5.1 | Causal Ablation | 🔴 NEEDS REDO | Built on broken Phase 4 SAEs |
| 5.2 | Feature Naming | 🔴 NEEDS REDO | Built on broken Phase 4 SAEs |
| 5.3 | Transfer Analysis | 🔴 NEEDS REDO | 0.995–1.000 transfer ratio was artifact of degenerate SAEs |
| 5.4 | Multi-Layer Analysis | ⚠️ CHECKPOINTS VALID | v2 SAEs retrained with fixes; analysis needs re-run with v2 checkpoints |
| 6 | Controllable CoT | 🔴 NEEDS REDO | Must use corrected SAEs |

**Note on Phase 2:** Synthetic Transformers phase was deferred in favor of phase progression from ground truth (Phase 1, unambiguous systems) → labeled reasoning (Phase 3, real chain-of-thought) → frontier models (Phase 4). This accelerates validation while maintaining the falsification framework's integrity.

---

#### Phase 5.1: Causal Ablation (COMPLETE)
- Causal importance scores for features (Phi-2: r=0.75)

**Output Location:** `phase5_results/causal_ablation/`

---

## Project Structure

```
RL-Decoder with SAE Features/
├── README.md                    # Main project documentation
├── PROJECT_STATUS.md            # This file - current status
├── TODO.md                      # Active task list
├── requirements.txt             # Python dependencies
├── setup_env.sh                 # Environment setup script
│
├── docs/                        # Consolidated documentation
│   ├── DOCUMENTATION_INDEX.md               # Complete docs listing
│   ├── ALIGNMENT_VERIFICATION.md            # Research plan alignment check
│   ├── PHASE5_TASK4_ANALYSIS_REPORT.md      # Latest analysis
│   ├── PHASE5_TASK4_SPECIFICATION.md        # Phase 5.4 design
│   └── archived_reports/                    # Historical reports
│
├── src/                         # Core source code
│   ├── sae_architecture.py      # SAE implementation
│   ├── sae_training.py          # Training logic
│   ├── activation_capture.py   # Activation extraction
│   └── ...
│
├── phase1/                      # Ground truth RL training
├── phase3/                      # Phase 3 implementation
├── phase4/                      # SAE training & evaluation
├── phase5/                      # Universality & interpretability
│   ├── phase5_task4_capture_multi_layer.py
│   ├── phase5_task4_train_multilayer_saes.py
│   ├── phase5_task4_reasoning_flow_analysis.py
│   └── phase5_task4_visualize_reasoning_flow.py
│
├── phase5_results/              # Phase 5 outputs
│   └── multilayer_transfer/
│       └── reasoning_flow/
│           ├── reasoning_flow_analysis.json
│           └── visualizations/   # 13 heatmaps & charts
│
├── sae_logs/                    # All training & execution logs
├── checkpoints/                 # Model checkpoints
└── datasets/                    # Training datasets
```

---

## Key Results & Findings

### Phase 3: Controlled CoT Probe Results (Valid, Feb 22 2026)

| Expansion | Accuracy | vs Baseline | Equation | Comparison | Reasoning |
|-----------|----------|-------------|----------|------------|-----------|
| 4x | 91.3% | −0.5pp | 83.3% | 0% | 0% |
| 6x | 86.8% | −5.0pp | 97.0% | 0% | 0% |
| 8x | 83.3% | −8.5pp | 34.4% | 0% | 62.6%* |
| 10x | 87.3% | −4.5pp | 96.7% | 0% | 0% |
| 12x | 90.5% | −1.3pp | 91.4% | 0% | 0% |
| 14x | 86.1% | −5.7pp | 97.6% | 0% | 0% |
| 16x | 90.9% | −0.9pp | 90.7% | 0% | 0% |
| 18x | 91.2% | −0.6pp | 89.4% | 0% | 0% |
| 20x | 88.8% | −3.0pp | 94.4% | 0% | 0% |

*8x reasoning result is likely a probe local-minimum artifact, not a genuine signal.

**Majority-class baseline: 91.8%** (sequences padded to 1024 tokens; most tokens are background "other")

**Conclusion:** Single-layer probes detect arithmetic computation tokens but cannot decode multi-step reasoning structure. This motivates the multi-layer analysis approach.

### Phase 5.4: Multilayer SAE Checkpoints (v2, valid training)
- 98 SAEs across all layers of 4 models, trained with corrected SAE code
- Transfer matrix analysis using these checkpoints is the correct next step
- Previous transfer quality numbers (GPT2: 0.428, Phi-2: 0.039) were from broken v1 SAEs and should not be cited

---

## Trained Models & Artifacts

### SAE Checkpoints
- **Phase 5.3:** 4 SAEs (final layer, 8x expansion) - `phase5_results/transfer_analysis/saes/`
- **Phase 5.4:** 98 SAEs (all layers, 8x expansion) - `phase5_results/multilayer_transfer/saes/`

### Activation Data
- **Single layer:** `phase4_results/activations/` (4 files, ~200MB)
- **Multi-layer:** `phase4_results/activations_multilayer/` (98 files, ~906MB)

### Analysis Results
- **Transfer matrices:** `phase5_results/multilayer_transfer/reasoning_flow/reasoning_flow_analysis.json`
- **Visualizations:** `phase5_results/multilayer_transfer/reasoning_flow/visualizations/` (13 PNG files)

---

## Next Steps

### Priority 1: Retrain Phase 4 SAEs with corrected training
Retrain single-layer SAEs for GPT-2-medium, Pythia-1.4B, Gemma-2B, Phi-2 using the fixed training loop (use_relu=True, normalize_decoder, correct decorrelation loss). These form the foundation for all downstream analysis.

### Priority 2: Re-run Phase 5.4 transfer matrix analysis using v2 checkpoints
The 98 multilayer SAE checkpoints in `phase5_results/multilayer_transfer_v2/saes/` are valid. Re-run the transfer matrix computation and visualizations against these checkpoints to get reliable layer-universality data.

### Priority 3: Re-run Phase 5.1–5.3 with corrected SAEs
Causal ablation, feature naming, and single-layer transfer analysis all need to be redone using the retrained Phase 4 SAEs.

### Priority 4: Multi-layer CoT probe analysis
Extend the Phase 3 probe approach across multiple layers using the v2 multilayer SAE checkpoints. Phase 3 showed that layer 6 alone is insufficient; probing multiple layers simultaneously may reveal where reasoning connectives are encoded.

---

## Documentation

### Active Documents (in `docs/`)
- **[PHASE5_TASK4_ANALYSIS_REPORT.md](docs/PHASE5_TASK4_ANALYSIS_REPORT.md)** - Complete analysis of reasoning flow
- **[PHASE5_TASK4_SPECIFICATION.md](docs/PHASE5_TASK4_SPECIFICATION.md)** - Technical design document
- **[DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)** - Master documentation index

### Archived Reports (in `docs/archived_reports/`)
- Phase 3-5 execution reports
- Historical implementation summaries
- Design specifications

---

## Quick Commands

```bash
# Activate environment
source setup_env.sh

# View latest results
cat docs/PHASE5_TASK4_ANALYSIS_REPORT.md

# Check training logs
tail -50 sae_logs/phase5_task4_reasoning_flow_*.log

# List all SAE checkpoints
ls phase5_results/multilayer_transfer/saes/*_sae.pt | wc -l

# View visualizations
ls phase5_results/multilayer_transfer/reasoning_flow/visualizations/
```

---

## Contact & Support

For questions or issues:
1. Check [DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) for relevant docs
2. Review [TODO.md](TODO.md) for active work
3. Examine logs in `sae_logs/` for debugging

---

**Project Status:** Phase 3 COMPLETE (mixed results) | Phase 4/5/6 needs redo with corrected SAE training

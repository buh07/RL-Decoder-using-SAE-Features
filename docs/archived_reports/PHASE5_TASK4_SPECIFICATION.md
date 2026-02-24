# Phase 5 Task 4: Multi-Layer Feature Transfer Analysis
## Complete Specification & Implementation Guide

**Date**: 2026-02-18  
**Status**: Designed & Ready for Execution  
**Estimated Duration**: 3-4 hours  
**Resource Requirement**: 1 GPU (RTX 6000 recommended)  
**Priority**: CRITICAL (satisfies "cross-layer consistency" evaluation criterion from overview.tex)

---

## Executive Summary

Phase 5.4 extends Phase 5.3 (single-layer feature transfer) to analyze **how reasoning features evolve across network depth**. This task is essential for:

1. ✅ Satisfying framework's stated evaluation criterion: **"Cross-layer: Multi-model consistency"** (overview.tex)
2. ✅ Enabling Phase 6 step-level causal analysis (requires layer-wise feature understanding)
3. ✅ Answering core research questions:
   - Q1: Are early layers' features universal across models?
   - Q2: At which depth do model-specific features emerge?
   - Q3: Are there bottleneck layers critical for all reasoning?
   - Q4: How does network depth affect feature transfer quality?

---

## Research Background & Motivation

### Current State (Phase 5.3 Complete)
- ✅ Single-layer dataset: Activations from final reasoning layer only
  - gemma-2b: layer 9 (of 24 total)
  - gpt2-medium: layer 12 (of 12 total)
  - phi-2: layer 16 (of 32 total)
  - pythia-1.4b: layer 12 (of 24 total)

- ✅ Result: Near-perfect reconstruction transfer (0.995-1.000)
- ❌ **Gap**: We see the final computation snapshot, not the reasoning pipeline

### Why Multi-Layer Matters
```
Standard LLM reasoning architecture:

EARLY LAYERS (4-6):     Low-level pattern detection
├─ Tokenization refinement
├─ Grammatical structure
└─ Basic concept encoding

MID LAYERS (8-12):      Semantic reasoning & task understanding ⭐
├─ Problem decomposition
├─ Context accumulation
├─ Logical inference
└─ Intermediate computations

DEEP LAYERS (16-20):    High-level decision making
├─ Plan formation
├─ Solution assembly
└─ Strategy convergence

OUTPUT (24+):           Output generation prep
└─ Format alignment

KEY INSIGHT: Each layer solves PART of the reasoning task.
```

**Without multi-layer analysis**: We can't identify which layers contribute to universality vs. specialization.

---

## Phase 5.4 Objectives & Success Criteria

### Primary Objectives

| # | Objective | Success Criteria | Framework Alignment |
|---|-----------|------------------|-------------------|
| 1 | Capture multi-layer activations | ≥5 layers per model × 4 models = 20 activation files | "Expand to mid-late layers if needed" (overview.tex) |
| 2 | Train per-layer SAEs | All 20 SAEs converge with reconstruction loss <0.5 | Phase 5 SAE training framework |
| 3 | Compute layer→layer transfer matrix | 400 transfer pairs: 5 src layers × 4 models × 5 tgt models × 4 comparisons | Enables "cross-layer consistency" evaluation |
| 4 | Identify universality patterns | Produce layer-wise heatmap showing where transfer degrades | Answer Q1-Q4 research questions |
| 5 | Generate insights report | Document findings with visualizations | Prepare Phase 6 foundation |

### Success Metrics

**Per-Layer Transfer Quality**:
```
Layer Depth     Expected Transfer Ratio    Pass Criterion
────────────────────────────────────────────────────────
Early (4-6)     0.90-0.95 (universal)      ≥ 0.85
Mid (8-12)      0.80-0.90 (mixed)          ≥ 0.75
Late (16-20)    0.60-0.80 (diverging)      ≥ 0.50
Output (24+)    0.30-0.60 (specialized)    ≥ 0.20
```

**Phase Completion Criteria**:
- [ ] All 20 activation files successfully loaded
- [ ] All 20 SAEs trained with loss convergence
- [ ] Transfer matrix computed (400 pairs)
- [ ] Heatmap visualization generated
- [ ] Layer universality classifications assigned
- [ ] Report with implications for Phase 6 written

---

## Detailed Implementation Steps

### STEP 1: Extended Activation Capture (30-45 min)
**File**: `phase5/phase5_task4_capture_multi_layer.py` (to create)  
**Input**: Existing Phase 4 models + test datasets  
**Output**: 20 activation files (1 per layer×model)

#### 1.1 Layer Selection
```python
LAYER_CONFIGS = {
    "gemma-2b": {
        "total_layers": 24,
        "test_layers": [4, 8, 12, 16, 20],
        "position": "post_mlp"  # Post-MLP residual
    },
    "gpt2-medium": {
        "total_layers": 12,
        "test_layers": [3, 6, 9, 11],  # Skip layer 12 (already captured as "final")
        "position": "post_mlp"
    },
    "phi-2": {
        "total_layers": 32,
        "test_layers": [4, 8, 16, 24, 30],
        "position": "post_mlp"
    },
    "pythia-1.4b": {
        "total_layers": 24,
        "test_layers": [4, 8, 12, 16, 20],
        "position": "post_mlp"
    }
}
```

#### 1.2 Execution Requirements
```bash
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features

# Create output directory
mkdir -p phase4_results/activations_multilayer

# Run capture for all models and layers
python3 phase5/phase5_task4_capture_multi_layer.py \
  --models gemma-2b gpt2-medium phi-2 pythia-1.4b \
  --layers 4 8 12 16 20 \
  --output-dir phase4_results/activations_multilayer \
  --batch-size 32 \
  --num-batches 50 \
  --device cuda:5
```

#### 1.3 Expected Outputs
```
phase4_results/activations_multilayer/
├── gemma-2b_layer4_activations.pt     (20-50 MB)
├── gemma-2b_layer8_activations.pt     (20-50 MB)
├── gemma-2b_layer12_activations.pt    (20-50 MB)
├── gemma-2b_layer16_activations.pt    (20-50 MB)
├── gemma-2b_layer20_activations.pt    (20-50 MB)
├── gpt2-medium_layer3_activations.pt  (50-100 MB)
├── gpt2-medium_layer6_activations.pt  (50-100 MB)
├── gpt2-medium_layer9_activations.pt  (50-100 MB)
├── gpt2-medium_layer11_activations.pt (50-100 MB)
├── phi-2_layer4_activations.pt        (30-80 MB)
├── phi-2_layer8_activations.pt        (30-80 MB)
├── phi-2_layer16_activations.pt       (30-80 MB)
├── phi-2_layer24_activations.pt       (30-80 MB)
├── phi-2_layer30_activations.pt       (30-80 MB)
├── pythia-1.4b_layer4_activations.pt  (30-80 MB)
├── pythia-1.4b_layer8_activations.pt  (30-80 MB)
├── pythia-1.4b_layer12_activations.pt (30-80 MB)
├── pythia-1.4b_layer16_activations.pt (30-80 MB)
└── pythia-1.4b_layer20_activations.pt (30-80 MB)

Total: ~400 MB (varies by model depth)
```

#### 1.4 Validation Checklist
- [ ] All 20 files created
- [ ] File sizes reasonable (20-100 MB each)
- [ ] Activation shapes match model hidden dimensions:
  - gemma-2b: 2048D ✓
  - gpt2-medium: 1024D ✓
  - phi-2: 2560D ✓
  - pythia-1.4b: 2048D ✓
- [ ] No corruption or NaN values in tensors
- [ ] Metadata files (`.meta.json`) created with layer info

---

### STEP 2: Multi-Layer SAE Training (90-120 min)
**File**: `phase5/phase5_task4_train_multilayer_saes.py` (to create)  
**Input**: 20 activation files  
**Output**: 20 SAE checkpoints

#### 2.1 SAE Configuration Per Layer
```python
def get_sae_config_for_layer(model_name: str, layer_num: int) -> SAEConfig:
    """
    Return SAE config tuned for specific layer.
    
    Early layers: Higher sparsity (more interpretable, less task-specific)
    Late layers: Lower sparsity (more distributed, more useful for task)
    """
    base_configs = {
        "gemma-2b": SAEConfig(
            input_dim=2048,
            expansion_factor=8,
            learning_rate=1e-4,
            l1_penalty_coeff=1e-4,
        ),
        # ... similar for other models
    }
    
    config = base_configs[model_name]
    
    # Adjust L1 penalty by layer depth
    if layer_num in [4, 6]:  # Early
        config.l1_penalty_coeff = 1.5e-4  # More sparsity
    elif layer_num in [16, 20]:  # Late
        config.l1_penalty_coeff = 0.5e-4  # Less sparsity
    
    return config
```

#### 2.2 Execution Command
```bash
# Train all 20 SAEs sequentially (can parallelize with CUDA_VISIBLE_DEVICES)
python3 phase5/phase5_task4_train_multilayer_saes.py \
  --activations-dir phase4_results/activations_multilayer \
  --output-dir phase5_results/multilayer_transfer/saes \
  --epochs 10 \
  --batch-size 64 \
  --device cuda:5 \
  --verbose
```

#### 2.3 Expected Training Duration
```
Per SAE: ~5-6 minutes (10 epochs × 64 batch size)
Total: 20 SAEs × 5.5 min = 110 minutes (~1.8 hours)

Checkpoint save: Every SAE checkpoint: 50-100 MB
Total output: 20 × 75 MB = 1.5 GB
```

#### 2.4 Training Validation (check logs continuously)
```bash
# Monitor in separate terminal
tail -f sae_logs/phase5_task4_*.log

# Expected log pattern:
# [2026-02-18 13:00:00] Training SAE for gemma-2b_layer4
# [2026-02-18 13:00:05]   epoch 1/10: loss=45.32, sparsity=28.3%
# [2026-02-18 13:00:15]   epoch 2/10: loss=12.44, sparsity=31.2%
# ...
# [2026-02-18 13:05:00] Saved checkpoint: gemma-2b_layer4_sae.pt
```

#### 2.5 Success Criteria
- [ ] All 20 SAEs trained without errors
- [ ] Final reconstruction loss <0.5 for all SAEs
- [ ] Sparsity converges to 28-33% range for all
- [ ] No divergence (NaN/Inf) in any training run
- [ ] Checkpoints saved to correct output directory

---

### STEP 3: Compute Layer-to-Layer Transfer Matrix (15-20 min)
**File**: `phase5/phase5_task4_compute_transfer_matrix.py` (to create)  
**Input**: 20 SAE checkpoints + 20 activation files  
**Output**: Transfer matrix JSON + visualization

#### 3.1 Transfer Computation Algorithm
```python
def compute_transfer_quality(
    source_sae: SparseAutoencoder,
    source_activations: torch.Tensor,
    target_activations: torch.Tensor,
    top_k: int = 50
) -> Dict[str, float]:
    """
    Compute transfer quality when applying source SAE to target activations.
    
    Returns:
    {
        "transfer_recon_ratio": float,  # target_loss / source_loss (should ~1.0)
        "feature_variance_ratio": float,  # variance preservation
        "spearman_correlation": float,   # feature ranking correlation
        "decoder_similarity": float,      # cosine sim of top decoders
    }
    """
    
    # Compute reconstruction quality
    source_loss = compute_recon_loss(source_sae, source_activations)
    target_loss_with_source_sae = compute_recon_loss(source_sae, target_activations)
    
    transfer_ratio = target_loss_with_source_sae / (source_loss + 1e-8)
    
    # Feature variance preservation
    source_variance = compute_feature_variance(source_sae, source_activations)
    target_variance = compute_feature_variance(source_sae, target_activations)
    
    top_idx = np.argsort(source_variance)[-top_k:]
    variance_ratio = (target_variance[top_idx].mean() / source_variance[top_idx].mean())
    
    # Spearman correlation of top-k
    spearman = spearmanr(source_variance[top_idx], target_variance[top_idx])
    
    return {
        "transfer_recon_ratio": transfer_ratio,
        "feature_variance_ratio": variance_ratio,
        "top_k_spearman": spearman,
        "decoder_similarity": compute_decoder_similarity(source_sae, target_activations),
    }
```

#### 3.2 Execution Command
```bash
python3 phase5/phase5_task4_compute_transfer_matrix.py \
  --sae-dir phase5_results/multilayer_transfer/saes \
  --activations-dir phase4_results/activations_multilayer \
  --output-dir phase5_results/multilayer_transfer \
  --top-k 50 \
  --device cuda:5
```

#### 3.3 Output Structure
```json
{
  "config": {
    "models": ["gemma-2b", "gpt2-medium", "phi-2", "pythia-1.4b"],
    "layers_per_model": 5,
    "top_k": 50
  },
  "transfer_matrix": {
    "gemma-2b_layer4__to__pythia-1.4b_layer4": {
      "transfer_recon_ratio": 0.987,
      "feature_variance_ratio": 1.043,
      "top_k_spearman": 0.312,
      "decoder_similarity": 0.091,
      "evaluation": "HIGH_UNIVERSALITY"
    },
    "gemma-2b_layer4__to__pythia-1.4b_layer8": {
      "transfer_recon_ratio": 0.756,
      "feature_variance_ratio": 0.821,
      "top_k_spearman": 0.211,
      "decoder_similarity": 0.063,
      "evaluation": "MODERATE_UNIVERSALITY"
    },
    ...
  },
  "layer_universality_scores": {
    "layer_4": 0.92,   # Universality score (avg transfer quality)
    "layer_8": 0.85,
    "layer_12": 0.68,
    "layer_16": 0.42,
    "layer_20": 0.31
  }
}
```

---

### STEP 4: Visualize & Analyze Heatmap (15-20 min)
**File**: `phase5/phase5_task4_visualization.py` (to create)  
**Input**: Transfer matrix JSON  
**Output**: Heatmaps, plots, insights markdown

#### 4.1 Heatmap Structure
```
layer_transfer_heatmap.png (matplotlib):

        Layer 4   Layer 8   Layer 12  Layer 16  Layer 20
        ─────────────────────────────────────────────────
Layer 4  1.000     0.92      0.71      0.42      0.28
Layer 8  0.91      1.000     0.88      0.65      0.41
Layer 12 0.68      0.85      1.000     0.79      0.52
Layer 16 0.39      0.63      0.76      1.000     0.68
Layer 20 0.25      0.38      0.51      0.65      1.000

Color scale:
  ■ Green  (0.9-1.0):   Universal reasoning
  ■ Yellow (0.6-0.9):   Partially universal
  ■ Orange (0.4-0.6):   Diverging
  ■ Red    (0.0-0.4):   Model-specific
```

#### 4.2 Execution
```bash
python3 phase5/phase5_task4_visualization.py \
  --transfer-matrix phase5_results/multilayer_transfer/transfer_matrix.json \
  --output-dir phase5_results/multilayer_transfer \
  --format png pdf json
```

#### 4.3 Generated Files
```
phase5_results/multilayer_transfer/
├── layer_transfer_heatmap.png         (visualization)
├── layer_transfer_heatmap.pdf         (print-quality)
├── layer_universality_curve.png       (X=layer_depth, Y=avg_transfer)
├── per_model_transfer_analysis.json   (detailed metrics)
└── layer_analysis_insights.md         (interpretation)
```

---

### STEP 5: Generate Analysis Report (30 min)
**File**: `phase5/phase5_task4_analysis.py` (to create)  
**Input**: Transfer matrix, heatmaps  
**Output**: Markdown report with findings

#### 5.1 Report Structure
```markdown
# Phase 5.4 Multi-Layer Feature Transfer Analysis Results

## Executive Summary
- Universal early layers: YES/NO
- Divergence point: Layer X
- Bottleneck identified: Layer Y
- Recommendations for Phase 6

## Finding 1: Universal Early Features
If transfer_quality[layer_4, across_models] > 0.9:
  → "All models learn identical early-layer representations"
  Implication: Universal pattern recognition
  
## Finding 2: Divergence Threshold
If transfer_quality drops from 0.85 → 0.60 at Layer X:
  → "Reasoning strategy diverges at Layer X"
  Implication: Before Layer X = universal, After = specialized
  
## Finding 3: Bottleneck Layers
If transfer_quality[Layer_8, all_targets] >> other layers:
  → "Layer 8 is critical information hub"
  Implication: Knowledge distillation target
  
## Implications for Phase 6
- Optimal step-level control layer: Layer X (best universality)
- Model-specific steering: Requires Layer Y+
- Cross-model transfer point: Layer Z
```

#### 5.2 Key Analyses
```python
# Analysis 1: Universality by depth
universality_by_depth = {}
for layer in [4, 8, 12, 16, 20]:
    transfer_qualities = [
        results[f"model_A_layer{layer}__to__model_B_layer{layer}"]
        for model_A, model_B in all_model_pairs
    ]
    universality_by_depth[layer] = np.mean(transfer_qualities)

# Analysis 2: Within-layer consistency
within_layer_consistency = {}
for layer in [4, 8, 12, 16, 20]:
    # How well do different models' SAEs at same layer transfer?
    consistency = avg_transfer_ratio(same_layer_pairs)
    within_layer_consistency[layer] = consistency

# Analysis 3: Cross-model universality spectrum
cross_model_universality = {}
for pair in model_pairs:
    avg_transfer = np.mean([
        results[f"{pair[0]}_layer{l}__to__{pair[1]}_layer{l}"]
        for l in [4, 8, 12, 16, 20]
    ])
    cross_model_universality[pair] = avg_transfer
```

---

## Execution Timeline & Resource Allocation

```
Phase 5.4 Complete Execution:

Step 1: Multi-layer capture        30-45 min    ⏱️ (1 GPU inactive during capture)
Step 2: SAE training               90-120 min   🐍 (GPU: 100% utilization)
Step 3: Transfer computation       15-20 min    🐍 (GPU: 80% utilization)
Step 4: Visualization              10-15 min    🐍 (GPU: <20% utilization)
Step 5: Analysis & reporting       20-30 min    📝 (CPU only)
                                   ──────────
                          TOTAL:  165-230 min  = 2.75-3.8 hours
```

**GPU Resource Budget**: 
- Total compute for Phase 5.4: ~150 GPU-minutes = 2.5 GPU-hours
- Budget remaining from 100-hour allocation: ~87.5 hours ✅

---

## Dependencies & Prerequisites

### 1. Data Requirements
```
✅ SATISFIED:
- Phase 4 trained models (GPT-2, Pythia, Gemma, Phi-2)
- Phase 5.3 SAE training framework
- Activation capture infrastructure (Phase 3)

❌ TO CREATE:
- Multi-layer activation capture extension
- Per-layer SAE training orchestration
- Transfer matrix computation pipeline
```

### 2. Code Files to Create
```
phase5/
├── phase5_task4_capture_multi_layer.py      (350 lines)
├── phase5_task4_train_multilayer_saes.py    (280 lines)
├── phase5_task4_compute_transfer_matrix.py  (320 lines)
├── phase5_task4_visualization.py            (250 lines)
├── phase5_task4_analysis.py                 (200 lines)
└── phase5_task4_orchestrate.sh              (120 lines)
```

### 3. Output Directory Structure
```
phase5_results/multilayer_transfer/
├── saes/                                     (20 × 75 MB SAE checkpoints)
├── transfer_matrix.json                      (Detailed metrics)
├── layer_transfer_heatmap.png                (Visualization)
├── layer_analysis_insights.md                (Report)
└── multilayer_transfer_full_report.md        (Complete analysis)
```

---

## Success Criteria & Validation

### Must-Pass Criteria (Go/No-Go Gates)
- [ ] **Gate 1**: All 20 activation files successfully loaded (no missing dims)
- [ ] **Gate 2**: All 20 SAEs trained without divergence (loss converged)
- [ ] **Gate 3**: Transfer matrix computed for all 400 pairs
- [ ] **Gate 4**: Heatmap generated without errors

### Should-Pass Criteria (Quality Metrics)
- [ ] Early-layer transfer quality > 0.80 (indicates universality)
- [ ] Layer universality curve shows monotonic decline (mathematical consistency)
- [ ] Cross-model transfer variance < 15% per layer (robustness)
- [ ] Report identifies clear divergence point (actionable insight)

### Publication-Ready Criteria
- [ ] Figure quality meets publication standards (300 DPI, labeled)
- [ ] Statistical significance calculated (p-values for correlations)
- [ ] Findings reproducible (random seeds documented)
- [ ] Implications clearly stated (connections to Phase 6)

---

## Integration with Phase 6

### How Phase 5.4 Enables Phase 6

**Phase 6 Requirement**: "Step-level causal testing"  
**Phase 5.4 Provides**: Layer-by-layer feature understanding

```
Phase 5.4 Output: "Universal features found at layers 4-8"
                 "Task-specific features emerge at layers 12+"
                 "Bottleneck identified at layer 8"
                 ↓
Phase 6 Design:  "Capture step-aligned activations at layer 8 (bottleneck)"
                 "Train CoT-aware SAE with step prediction"
                 "Ablate features per step type (parsing, arithmetic, etc.)"
                 "Steer at layer 8 for maximal cross-model effect"
```

**Specific Phase 6 Advantages from Phase 5.4**:
1. **Optimal layer selection**: Use layer with highest cross-model universality
2. **Feature stability**: Prioritize features that transfer well across depths
3. **Steering efficiency**: Target bottleneck layer for maximum impact
4. **Cross-model validation**: Use Phase 5.4 to validate Phase 6 findings on ALL layers

---

## Risk Mitigation

| Risk | Mitigation | Fallback |
|------|-----------|----------|
| Memory overflow (20 activation files) | Stream load, delete post-SAE training | Reduce layers to 3 per model |
| SAE training divergence | Use proven Phase 5.3 hyperparams | Reduce expansion factor to 4x |
| Transfer matrix too sparse (many skipped pairs) | All pairs have compatible dims (2048 ↔ 2048 within class) | Pre-filter to same-dim pairs only |
| Missing early-layer activations | Capture with lower batch size if needed | Use Phase 4 model internals as proxy |

---

## Deliverables Checklist

### Code Deliverables
- [ ] `phase5_task4_*.py` scripts (5 files)
- [ ] `phase5_task4_orchestrate.sh` (unified execution)
- [ ] Integration tests (`phase5/tests/test_multilayer.py`)
- [ ] README with execution examples

### Data Deliverables
- [ ] 20 multi-layer activation files
- [ ] 20 SAE checkpoints + metadata
- [ ] Transfer matrix JSON (complete dataset)
- [ ] Per-layer metrics CSV (for analysis)

### Documentation Deliverables
- [ ] `phase5_results/PHASE5_TASK4_RESULTS.md` (main report)
- [ ] Layer analysis visualizations (3+ PNG files)
- [ ] Heatmap with interpretation guide
- [ ] Phase 6 recommendations document

### Publication Deliverables
- [ ] Figure 1: Layer universality curve (publication-quality PDF)
- [ ] Figure 2: Transfer matrix heatmap (publication-quality PDF)
- [ ] Table 1: Per-layer transfer statistics
- [ ] Supplementary: Full transfer data matrix (JSON)

---

## How to Execute Phase 5.4

### Option A: Automated (Single Command)
```bash
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features
source .venv/bin/activate

# Run complete Phase 5.4 pipeline
bash phase5/phase5_task4_orchestrate.sh \
  --gpus 5 \
  --log phase5_task4_$(date +%s).log \
  --monitor
```

### Option B: Step-by-Step (with monitoring)
```bash
# Step 1: Capture
python3 phase5/phase5_task4_capture_multi_layer.py \
  --output-dir phase4_results/activations_multilayer \
  --check-all

# Step 2: Train
python3 phase5/phase5_task4_train_multilayer_saes.py \
  --activations-dir phase4_results/activations_multilayer \
  --output-dir phase5_results/multilayer_transfer/saes

# Step 3: Compute
python3 phase5/phase5_task4_compute_transfer_matrix.py \
  --sae-dir phase5_results/multilayer_transfer/saes \
  --activations-dir phase4_results/activations_multilayer

# Step 4-5: Visualize & Analyze
python3 phase5/phase5_task4_visualization.py \
  --transfer-matrix phase5_results/multilayer_transfer/transfer_matrix.json

python3 phase5/phase5_task4_analysis.py \
  --results-dir phase5_results/multilayer_transfer
```

---

## Documentation & References

- **Overview.tex**: Evaluation criterion "Cross-layer: Multi-model consistency" (Section 8)
- **Phase 5.3 results**: `phase5_results/transfer_analysis/` (establishes baseline)
- **Phase 6 specification**: Requires Phase 5.4 outputs (PHASE6_DESIGN_SPECIFICATION.md)
- **SAE architecture**: `src/sae_architecture.py` (proven training framework)

---

## Conclusion

Phase 5.4 is **critical infrastructure** for understanding reasoning pipelines and **required foundation** for Phase 6 step-level causal control. With existing Phase 4/5 infrastructure proven reliable, Phase 5.4 execution carries **low technical risk** and **high research value**.

**Ready to execute: YES ✅**

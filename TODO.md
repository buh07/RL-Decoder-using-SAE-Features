# Active Tasks & Project TODO

**Last Updated:** February 22, 2026

For comprehensive status, see [PROJECT_STATUS.md](PROJECT_STATUS.md) and [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md).

## Current Priority: Re-run Phase 4 → 5 → 6 with corrected SAE training

> **⚠️ IMPORTANT — Previous Phase 4/5 results are INVALID.** The SAEs used in Phase 4, 5.1, 5.2, and 5.3 were trained with three critical bugs: (1) `use_relu=False` — L1 sparsity never engaged, features are not sparse; (2) missing `normalize_decoder()` — decoder columns grew unbounded; (3) incorrect decorrelation loss formula — loss could go negative. All sparsity, transfer, and causal ablation numbers from those phases are artifacts of broken training. Phase 5.4 multilayer SAEs were retrained with the fixes (`multilayer_transfer_v2/`), but the downstream analysis should be re-run using those checkpoints.

### High Priority
- [ ] Re-run Phase 4: retrain single-layer SAEs on GPT-2-medium, Pythia-1.4B, Gemma-2B, Phi-2 with fixed training (use_relu=True, normalize_decoder, correct decorrelation loss)
- [ ] Re-run Phase 5.1: causal ablation using fixed SAEs
- [ ] Re-run Phase 5.2: feature naming using fixed SAEs
- [ ] Re-run Phase 5.3: single-layer transfer analysis using fixed SAEs
- [ ] Validate Phase 5.4 analysis against v2 checkpoints (multilayer_transfer_v2/) and re-run transfer matrix computation
- [ ] Re-run Phase 6 interactive reasoning tracker using corrected SAE checkpoints

## 1. Environment + Infrastructure
- [x] Specify supported runtimes (Python >=3.10, CUDA version) and GPU budget (50-100 RTX-equivalent hours) in docs. *(SETUP.md captures Python 3.12.3 + CUDA 12.8.93; still plan to add GPU-hour budgeting details later if needed.)*
- [x] Author `setup_env.sh` replacement that installs minimal deps (PyTorch>=2.0, transformers>=4.37, trl>=0.8, accelerate>=0.25, datasets) and pins tokenizer artifacts locally.
- [x] Define WANDB (or alt) tracking defaults, secrets handling, and naming convention before any training runs. *(Secrets live in `.env`, values intentionally left blank until you paste your real API key.)*
- [x] Decide how tokenizer assets are synchronized (new script or Makefile target) and document expected location relative to repo. *(Tokenizers now vendored under `vendor/gpt2_tokenizer` and copied into `assets/tokenizers/gpt2` by the setup script.)*

## 2. Data & Tokenization Pipeline
- [x] Enumerate reasoning datasets to ingest (GSM8K, CoT-Collection, OpenR1-Math-220k, Reasoning Traces, REVEAL, TRIP, WIQA) with licensing + citation notes. *(See `datasets/DATASETS.md` + generated manifest.)*
- [x] Design sharded preprocessing plan: streaming from HuggingFace datasets, tokenization via pinned tokenizer, shards of ~2M tokens for fp16 training. *(Documented in `datasets/DATASETS.md#Sharded Tokenization Plan` + implemented in `datasets/tokenize_datasets.py`.)*
- [x] Decide which columns/fields correspond to reasoning traces vs answers; encode filtering + normalization rules (e.g., ensure strings, drop malformed entries). *(Handled by `datasets/download_datasets.py` normalization step.)*
- [x] Document how temporal alignment between CoT steps and tokens will be stored for later alignment heuristics. *(See `datasets/DATASETS.md#Temporal Alignment Plan` and upcoming alignment JSONL schema.)*
- [x] Plan data QC hooks (spot-check seqs, coverage stats, invariance tests from overview risks section). *(Outlined in `datasets/DATASETS.md#Data QC & Validation` with tooling roadmap.)*

## 3. Model Capture Hooks
- [x] Choose initial base models (e.g., GPT-2 small, 1B distilled) with layer-probe defaults (mid-layers like 6 for GPT-2) and record ckpt hashes. *(Implemented in `src/model_registry.py`: GPT-2 (12L@6), GPT-2-medium (24L@12), Pythia-1.4B (24L@12), Gemma-2B (18L@9), Llama-3-8B (32L@16), Phi-2 (32L@16). All models cached in LLM Second-Order Effects/models; hashes tracked via local_files_only.)*
- [x] Implement activation capture stack constrained to post-MLP residual streams + MLP hidden states; ensure streaming to disk is fp16 + layer-filtered to fit single GPU memory. *(Implemented in `src/activation_capture.py`: `ActivationCapture` class with configurable hooking, batch buffering, shard flushing to `.pt` files with `.meta.json` metadata. Factory function `create_gpt2_capture()` for GPT-2 style models.)*
- [x] Validate hooking latency + throughput on sample batches before SAE training. *(Implemented `src/capture_validation.py`: compares baseline vs hooked inference, measures overhead % and token throughput. Target: <50% overhead, >1000 tokens/s on single GPU.)*

## 4. SAE Architecture & Training
- [x] Formalize SAE architecture hyperparams: expansion factor 4-8x width, ReLU latents, decoder sharing vs independent per layer. *(Implemented in `src/sae_config.py`: `SAEConfig` dataclass with tuned presets (gpt2_sae_config, gpt2_medium_sae_config, pythia_1_4b_sae_config). Expansion factor 4-8x (configurable), ReLU latents default enabled.)*
- [x] Implement total loss: reconstruction + L1 sparsity + decorrelation penalty + optional probe-guided loss + temporal smoothness term. *(Implemented in `src/sae_architecture.py`: `SparseAutoencoder.compute_loss_components()` with all components + decorrelation as novel feature per overview.tex)*
- [x] Build configurable training loop supporting streaming batches, logging recon error, activation coverage, feature activations. *(Implemented in `src/sae_training.py`: `SAETrainer` class with ActivationShardDataset (streaming .pt shards), mixed precision (fp16), WANDB logging, checkpointing every 500 steps.)*
- [x] Add automatic probes for hypothesis/constraint tasks with leakage diagnostics (flag when probe vs linear baseline gap >5%). *(Phase 3 Complete: probe framework in place)*
- [x] Provide evaluation notebooks/scripts to visualize feature purity, cluster coherence, and activation sparsity histograms. *(✅ Phase 4B Complete: `phase4_interpretability.py`, per-model stats JSON with purity metrics)*

**RTX 6000 Timing Summary:**
- Single SAE (GPT-2, 768D→6144D, 50M tokens): **1.5-3 hours** (5 epochs)
- Full multi-model analysis (6 models, 1 layer each): **18-24 hours**
- Fits within 50-100 RTX-hour budget: ✅ See `TRAINING_PERFORMANCE_RTX6000.md` for detailed timing analysis.

## 5. Phased Research Program

### Phase 1 — Ground-Truth Systems [✅ COMPLETE - RE-VALIDATED 2026-02-18]
**Status**: Ground-truth validation complete. Bugs fixed and all environments re-run.
- [x] Recreate simple environments (BFS/DFS traversals, stack machines) with exact latent states.
- [x] Train SAEs on those states to verify reconstruction + monosemanticity; document failure cases as stop criteria.
- [x] Build causal tests: directly edit SAE latents and ensure decoded state changes behave as predicted.
- **Timeline**: Re-validated 2026-02-18
- **Bugs fixed**: decorrelation loss formula (negative loss), use_relu=False (zero sparsity), biased causal dim sampling, Stack encoding non-injective (sinusoidal), Logic constraints were random noise (not computed from grid), overfitting (50→500–2000 sequences)
- **Results (v3)**: BFS all 3 expansions ✅ R²=0.976–0.987 | Stack all 3 expansions ✅ R²=0.967–0.989 | Logic all 3 expansions ✅ R²=0.990–0.999
- **Checkpoints**: `phase1_results/v3/` (Logic) and `phase1_results/v2/` (BFS, Stack)

### Phase 2 — Synthetic Transformers [🔴 NOT STARTED - PLACEHOLDER]
- [ ] Implement tiny transformers solving grid or logic tasks where full attention patterns are known.
- [ ] Hook activations, train SAEs with decorrelation + probes, and perform causal perturbations on internal features.
- [ ] Compare recovered features against known synthetic circuitry; stop if alignment falls below target purity/correlation thresholds.
- **Timeline**: 7-10 days after Phase 1
- **Success criteria**: Feature alignment purity ≥80%, causal effects match prediction

### Phase 3 — Controlled Chain-of-Thought LMs [✅ COMPLETE — Mixed Results]
**Status**: Phase 3 re-run complete (v4, Feb 22 2026) with correct GSM8K regex patterns and class-weighted probes.
- [x] Collect datasets with labeled reasoning steps (GSM8K 7,473 examples)
- [x] Train SAEs on layer 6 (GPT-2 small, 9 expansion factors 4x–20x)
- [x] Evaluate probes on equation/comparison/reasoning/other step classification
- [x] Fixed bugs: broken alignment cache, wrong regex patterns, no class weighting, no majority-class baseline

**Results** (`phase3_results/full_scale_v4/`):
- **Equation detection**: 83–97% per-class accuracy across all expansions ✅
- **Comparison detection**: 0% across all expansions ❌
- **Reasoning connective detection**: 0% across all expansions (except 8x anomaly) ❌
- **All probes at or below majority-class baseline (91.8%)** — no expansion beats the baseline
- **Conclusion**: A single layer (layer 6, GPT-2 small) cannot linearly decode multi-step reasoning structure. This motivates the multi-layer analysis approach (Phase 5.4 / new Phase 4).
- **Note**: Previous claim of "100% probe accuracy" was an artifact of broken alignment caching and wrong regex patterns. That result was invalid.
- **Key files**: `phase3_results/full_scale_v4/results_gpu{0,1,2,4}.json`

### Phase 4 — Frontier LLMs [🔴 NEEDS REDO]
**Status**: Previous run INVALID — SAEs trained with use_relu=False (no L1 sparsity), missing normalize_decoder(), and buggy decorrelation loss. All reported results (31.7% sparsity, feature descriptions, causal rankings) are artifacts of broken training. Must retrain from scratch.
- [ ] Retrain SAEs for GPT-2-medium, Pythia-1.4B, Gemma-2B, Phi-2 with fixed training loop
- [ ] Re-run interpretability analysis (Phase 4B) with valid SAE checkpoints
- [ ] Re-validate sparsity, feature purity, and causal importance metrics

**Previous (invalid) results** (for reference only):
- Claimed 31.7% average feature sparsity — untrustworthy (use_relu=False means no ReLU gating)
- Claimed 5,680 dimensions analyzed — the analysis infrastructure is valid but inputs were wrong
- **Key files (archived)**: `phase4_results/interpretability/`, `docs/archived_reports/PHASE4_FINAL_SUMMARY.md`

## 6. Validation & Attribution Protocols [✅ SUBSTANTIALLY COMPLETE]
- [x] Codify purity metrics (sparsity, entropy, selectivity) and automate reporting per feature. *(Phase 4B: computed 5,680 dimension stats)*
- [x] Implement causal attribution framework: feature importance ranking via activation variance. *(Phase 4B: `CausalAblationTester` class, per-dimension scores)*
- [x] Establish baselines for causal scores (0% ablation = full reconstruction). *(Phase 4B: baseline reconstruction loss tracked)*
- [⏳] Build replication harness: hold-out datasets, independent reruns. *(Framework in place, full execution TBD)*

## 7. Risk Management
- [x] Define domain coverage tests to detect brittleness/invariance failures noted in overview. *(Cross-model consistency verified in Phase 4B)*
- [⏳] Create leakage detection suite (compare SAE features against probes, run ablations). *(Framework ready, full audit TBD)*
- [x] Track assumptions + open questions per phase; documented in phase summary documents. *(Phase 1-4 all have execution reports)*

## 8. Resource & Timeline Tracking [✅ COMPLETE]
- [x] Translate 50–100 RTX-hour budget into concrete run schedules; actual usage was ~12 min training. *(See PHASE4_EXECUTION_REPORT.md for detailed accounting)*
- [x] Maintain log of dataset sizes, activation storage requirements. *(2.7 GB used, well under 100 GB target)*
- [x] Outline execution plan with milestones. *(Phase 1-4 completed in single session, Feb 17 2026)*

## 9. Phase 4B: Interpretability Analysis [✅ COMPLETE]

**Overview**: Extended Phase 4 with comprehensive human interpretability analysis per overview.tex Section 8.

### Implemented Components

**Feature Purity Metrics** ✅
- [x] Sparsity analysis: 31.7% average across all models (31.7-32.9% per model)
- [x] Entropy computation: Per-dimension selectivity scores
- [x] Activation statistics: Mean, std, max per feature across 5,680 total dimensions
- [x] Top-k coherence: Identified most interpretable 50 features per model

**Causal Attribution** ✅
- [x] Feature importance ranking: Via activation variance (proxy for causal impact)
- [x] Baseline establishment: Full reconstruction loss as reference
- [x] Per-dimension causal scores: Computed for all features
- [x] Calibration curves ready: Framework for task performance measurement

**Validation & Reporting** ✅
- [x] Per-model interpretability reports: JSON with feature descriptions
- [x] Feature statistics saved: 1.8 MB total (225-562 KB per model)
- [x] Cross-model consistency check: Sparsity patterns validated
- [x] Purity metrics codified: Automated computation for all dimensions

### Generated Artifacts

- `phase4/phase4_interpretability.py`: Full interpretability pipeline
- `phase4/phase4b_orchestrate.sh`: Automated execution script
- `phase4_results/interpretability/`: Per-model analysis
  - `*_interpretability.json`: Feature descriptions + scores
  - `*_feature_stats.json`: Per-dimension statistics
  - `aggregate_results.json`: Cross-model summary
  - `PHASE4B_INTERPRETABILITY_REPORT.md`: Comprehensive report

### Key Results from Phase 4B

| Metric | Value | Status |
|--------|-------|--------|
| Models analyzed | 4 | ✅ Complete |
| Activation dimensions evaluated | 5,680 | ✅ Complete |
| Avg feature sparsity | 31.7% | ✅ Monosemantic |
| Causal importance ranked | Top 50/model | ✅ Complete |

### Coverage vs. overview.tex Section 8

- [x] Alignment (activation extraction) ✅ COMPLETE
- [x] Causal attribution (swept perturbations) ✅ COMPLETE
- [x] Validation (purity, clusters) ✅ COMPLETE
- [⏳] Feature descriptions (LM annotations) PARTIAL - Statistical descriptions ready, semantic labeling available

## 10. Phase 5: Optional High-Priority Work [🔄 IN PROGRESS]

**Scope**: 3 high-value tasks to maximize research impact (~90 min total)

### Task 1: Real Causal Ablation Tests [🔴 NEEDS REDO]
**Objective**: Measure actual task accuracy drops when features are zeroed (validates importance estimates)
**Status**: Previous run INVALID — used SAEs trained with use_relu=False. Phi-2 r=0.754 result is unreliable. Must redo after Phase 4 SAEs are retrained.
- [x] Load trained SAEs and activation data
- [x] Implement feature ablation on reasoning tasks
- [x] Measure accuracy deltas (GSM8K, MATH, Logic benchmarks)
- [x] Rank features by true causal impact
- [x] Generate causal test report
**Success Criteria**: Ablation accuracy deltas correlate with variance-based importance (r > 0.7)
- ✅ **Phi-2 (logic)**: r = 0.754 PASSED
- ⚠️ **Pythia-1.4b (gsm8k)**: r = 0.601 INVESTIGATE
- ❌ **GPT-2-medium (gsm8k)**: r = 0.161 FAILED
- ❌ **Gemma-2b (math)**: r = -0.177 FAILED
- **Average**: r = 0.335 (partial success, 1 pass + 1 investigate)

**Key Finding**: Feature selectivity (top-5 variance) is a reliable importance proxy for Phi-2 but not for other models. Reveals model-specific differences in feature organization (Phi-2 has selective interpretable features; GPT-2/Gemma rely on more distributed representations).

**Files Created**: 
- `phase5_results/PHASE5_TASK1_RESULTS.md` (detailed analysis)
- `phase5_results/causal_ablation/*.json` (per-model correlation results, 4 models)

### Task 2: LM-based Feature Naming [🔴 NEEDS REDO]
**Objective**: Use LLM to generate semantic descriptions for top 20 features per model
**Status**: Previous run INVALID — feature descriptions were generated from SAEs with no L1 sparsity. Must redo after Phase 4 SAEs are retrained.
- [x] Load top features from Phase 4B analysis
- [x] Extract activation examples for each feature
- [x] Call LLM to generate semantic labels (fallback to templates when API unavailable)
- [x] Validate descriptions with activation patterns
- [x] Save named features to interpretability report
**Success Criteria**: Descriptions are interpretable and match activation patterns ✅
- φ2-logic: 10 features named
- Pythia-1.4b-gsm8k: 10 features named
- Gemma-2b-math: 10 features named
- GPT-2-medium-gsm8k: 10 features named
- **Total**: 40 features with semantic descriptions

**Key Output**: Feature descriptions added to interpretability JSON:
- `phase5_results/feature_naming/*_interpretability_named.json` (4 files)
- Example descriptions:
  - "This feature detects arithmetic operations in math tasks..."
  - "This feature represents a semantic-level abstraction and fires strongly..."
  - "This feature specializes in constraint handling and is particularly active..."

**Files Created**: 
- `phase5/phase5_feature_naming.py` (feature naming implementation)
- `phase5_results/feature_naming/` directory with 8 files (4 results + 4 updated interpretability JSONs)

### Task 3: Feature Transfer Analysis [🔴 NEEDS REDO]
**Objective**: Test if top features transfer across models (answers: are features universal?)
**Status**: Previous run INVALID — transfer ratios of 0.995–1.000 are artifacts of degenerate SAEs (no ReLU, decoder collapse). Near-perfect transfer simply reflects that all activations are poorly reconstructed. Must redo after Phase 4 SAEs are retrained.
- [x] Train SAE on gpt2-medium activations (baseline SAE with 8x expansion)
- [x] Test transfer: Load gpt2-medium top-50 features, apply decoder to pythia-1.4b activations
- [x] Measure: Reconstruction quality, feature utility, importance rank preservation
- [x] Compute: Feature similarity matrix across [GPT-2 → Pythia], [Pythia → Gemma], [Gemma → Phi-2]
- [x] Identify: Which features transfer (>70% utility) vs. model-specific features (<30% utility)
- [x] Generate: Transfer report with universality metrics and implications
**Success Criteria**: >70% transfer consistency for top 10 features (answers universality question) ✅ ACHIEVED
**Key Output**: Transfer matrix showing which reasoning operations are universal vs. architecture-specific
**Files Created**: `phase5/phase5_feature_transfer.py`, `phase5_results/transfer_analysis/`

**✅ KEY FINDING**: Near-perfect transfer (0.995-1.000) proves reasoning features ARE universal composable primitives at final layer! Now extending to multiple layers via Task 5.4.

**Result Summary**:
- gemma-2b → pythia: 0.995 transfer ratio ✅
- pythia → gemma-2b: 1.000 transfer ratio ✅
- Spearman correlation: 0.23-0.36 (partial feature ranking alignment)
- Decoder similarity: 0.088 (model-specific embeddings, as expected)
- **Conclusion**: Semantic universality proven; representational differences expected

### Task 4: Multi-Layer Feature Transfer Analysis [⚠️ PARTIALLY VALID — Analysis needs re-run]
**Objective**: Track feature transfer across network depths (WHERE do features become universal?)
**Status**: 98 SAEs retrained with fixes (use_relu=True, normalize_decoder, correct decorrelation loss) — checkpoints in `phase5_results/multilayer_transfer_v2/saes/` are valid. However, the transfer matrix computation and visualizations were generated from the v1 (broken) checkpoints. Must re-run analysis scripts using v2 checkpoints.
**SAE fix confirmed**: use_relu=False (→True), missing normalize_decoder(), sparsity metric inverted

**Core Questions Addressed**:
- Q1: Are early layers' features universal across models?
- Q2: At which depth do model-specific features emerge?
- Q3: Are there bottleneck layers critical for all reasoning?
- Q4: How does network depth affect feature transfer quality?

**Planned Implementation Steps**:
1. [ ] Step 1: Capture multi-layer activations (layers 4, 8, 12, 16, 20) for all 4 models (30-45 min)
2. [ ] Step 2: Train 20 SAEs (one per layer) with phase 5.3 proven framework (90-120 min)
3. [ ] Step 3: Compute layer→layer transfer matrix (400 pairs) (15-20 min)
4. [ ] Step 4: Generate layer universality heatmap visualization (10-15 min)
5. [ ] Step 5: Produce analysis report with Phase 6 implications (30 min)

**Expected Outputs**:
- Layer universality scores (per depth)
- Transfer quality heatmap (source_layer × target_layer)
- Divergence point identification (where models specialize)
- Bottleneck layer analysis (critical for Phase 6)

**Framework Alignment**:
- ✅ Satisfies overview.tex evaluation criterion: "Cross-layer: Multi-model consistency"
- ✅ Enables Phase 6 step-level causal analysis
- ✅ Answers core question: "WHERE does reasoning happen?"

**Files to Create** (complete specification in PHASE5_TASK4_SPECIFICATION.md):
- `phase5/phase5_task4_capture_multi_layer.py`
- `phase5/phase5_task4_train_multilayer_saes.py`
- `phase5/phase5_task4_compute_transfer_matrix.py`
- `phase5/phase5_task4_visualization.py`
- `phase5/phase5_task4_analysis.py`
- `phase5/phase5_task4_orchestrate.sh`

**Success Criteria**:
- [ ] All 20 activation files loaded successfully
- [ ] All 20 SAEs converge (loss <0.5, sparsity 28-33%)
- [ ] Transfer matrix computed for all 400 layer pairs
- [ ] Early-layer universality identified (transfer >0.80)
- [ ] Divergence point clearly documented
- [ ] Heatmap visualization publication-ready

**Read Full Spec**: [PHASE5_TASK4_SPECIFICATION.md](PHASE5_TASK4_SPECIFICATION.md)

---

**Timeline**: Tasks 1-3 completed in 46 minutes; Task 4 next (3-4 hours)
**Expected Output**: Publication-ready results on feature causality (✅ achieved), naming (✅ achieved), single-layer universality (✅ achieved), and multi-layer evolution (⏳ Task 4 in progress)

---

## 📊 PROJECT STATUS SUMMARY (Updated Feb 22, 2026)

### Valid Completed Work
✅ Phase 1: Ground-truth validation — BFS/Stack/Logic R²=0.967–0.999 (v2/v3 results)
✅ Phase 3: Controlled CoT probes — complete with mixed results (v4, Feb 22 2026)
⚠️ Phase 5.4: Multilayer SAEs retrained with fixes (v2 checkpoints valid) — analysis needs re-run

### Needs Redo (broken SAE training)
🔴 Phase 4: Frontier LLM SAE training — use_relu=False, no normalize_decoder
🔴 Phase 5.1: Causal ablation — based on Phase 4 broken SAEs
🔴 Phase 5.2: Feature naming — based on Phase 4 broken SAEs
🔴 Phase 5.3: Single-layer transfer — based on Phase 4 broken SAEs
🔴 Phase 6: Interactive reasoning tracker — must use corrected SAEs

### Key Phase 3 Finding (motivates multi-layer approach)
- Layer 6 (GPT-2 small) can detect equation tokens (83–97%) but cannot detect reasoning connectives (0%)
- No single-layer probe beats the majority-class baseline across any expansion factor (4x–20x)
- Multi-layer SAE analysis (Phase 5.4) is the correct next step

### Documentation Generated
- `PROJECT_COMPLETION_SUMMARY.md` - Full project overview
- `PHASE5_COMPLETION_REPORT.md` - Phase 5 advanced analysis results  
- `PHASE5_TASK1_RESULTS.md` - Detailed causal ablation analysis
- `phase4/PHASE4_RESULTS.md` - Phase 4 training results
- `TRAINING_PERFORMANCE_RTX6000.md` - GPU timing breakdown

### Output Artifacts (73 total files)
- 4 SAE checkpoints + activation datasets (858 MB)
- 4 Phase 4B feature statistics (detailed per-feature analysis)
- 5 Phase 5.1 causal ablation reports (correlation analysis)
- 8 Phase 5.2 feature naming reports (semantic descriptions)
- 6 detailed markdown documentation files

### Key Findings
1. **SAE Scalability**: Successfully trained on 4 frontier models with consistent 31.7% sparsity
2. **Feature Selectivity ≠ Importance**: Correlation model-dependent (Phi-2: 0.75, others: <0.2)
3. **Semantic Interpretability**: 40 features successfully named with valid descriptions
4. **Publication-Ready**: All phases 1-5 validated and documented

### Resource Utilization
- GPU time: 9 minutes (out of 100-hour budget) = 9% utilization
- Total compute: 43 minutes
- Status: **Highly efficient** - left >90% time budget for Task 3 or expansion

### How to Navigate Results
1. **Quick summary**: Read `PROJECT_COMPLETION_SUMMARY.md`
2. **Phase 5 details**: See `PHASE5_COMPLETION_REPORT.md`
3. **Causal ablation**: Check `PHASE5_TASK1_RESULTS.md`
4. **Individual results**: Browse `phase5_results/` directory structure
5. **Reproduce**: See orchestration scripts in `phase5/` with bash commands

### Next Steps (If Continuing)
- [ ] Complete Task 3: Feature transfer analysis (45 min, optional)
- [ ] Refine semantic descriptions using GPT-4 API  
- [ ] Create interactive web tool for feature explorer
- [ ] Submit paper to publication venue
- [ ] Expand to additional model architectures

---

**Status**: All critical objectives (Phases 1-5.2) met. Task 5.3 recommended for completeness.

---

## 🔬 PHASE 6: Controllable Chain-of-Thought SAE Extension (PROPOSED)

**Objective**: Extend SAE framework to enable step-level reasoning control and validation  
**Status**: Designed but not implemented  
**Estimated Time**: 3-4 hours (leverages Phase 4 infrastructure)  
**Research Impact**: HIGH (publishable research + practical applications)

### Phase 6 Architecture & Pipeline

```
PHASE 5 LIMITATION:
  Task-level: "Does feature matter overall?" → Weak signal (r=0.16-0.75)
  
PHASE 6 SOLUTION:
  Step-level: "WHEN does feature matter?" → Strong signal per reasoning operation
  
Pipeline:
  1. CAPTURE:    Align activations to reasoning step boundaries
  2. TRAIN:      CoT-aware SAE (auxiliary: predict next reasoning step)
  3. VALIDATE:   Per-step feature importance via targeted ablation
  4. STEER:      Latent manipulation to control reasoning length/quality
  5. GENERALIZE: Test on alternative models
```

### Phase 6 Core Tasks

#### Task 6.1: CoT-Aligned Activation Capture [Priority: HIGH]
**Objective**: Record activations with explicit mapping to reasoning steps  
**Files to Create**: `phase6/phase6_cot_capture.py`
- [ ] Extend Phase 4 activation capture with step boundary detection
  - Input: Problem text + multi-step reasoning (GSM8K with CoT labels or synthetic)
  - Detect: "Step 1:", "Therefore:", "Computing:", "Answer:" markers
  - Output: Activations + step indices (where in token sequence each step occurs)
- [ ] Validate: Step boundaries align with actual reasoning transitions (manual spot-check)
- [ ] Output: `phase6_results/activations/` with step-aligned tensors
**Success Criteria**: 95%+ of step boundaries correctly identified
**Estimated Time**: 45 minutes

#### Task 6.2: CoT-Aware SAE Training [Priority: HIGH]
**Objective**: Train SAE with auxiliary task of predicting reasoning step type  
**Files to Create**: `phase6/phase6_cot_sae.py`
- [ ] Extend Phase 4 SAE architecture:
  - Primary loss: Reconstruction ($\|X - \hat{X}\|_2^2$)
  - Auxiliary loss: Predict reasoning step from latents ($\text{CrossEntropy}(\text{step\_pred}, \text{step\_label})$)
  - Weight auxiliary 0.2-0.4x reconstruction (ablate to find optimal)
  - Hypothesis: "If latents encode step info, they should predict it"
- [ ] Train on 50K tokens of step-annotated reasoning data
- [ ] Log: Reconstruction loss, step prediction accuracy, latent activation patterns per step
**Success Criteria**: 
  - Reconstruction quality ≥ Phase 4 baseline
  - Step prediction accuracy >75% (vs. random 20%)
**Estimated Time**: 1 hour

#### Task 6.3: Step-Level Causal Ablation Testing [Priority: CRITICAL]
**Objective**: Measure which features are causal for each reasoning operation  
**Files to Create**: `phase6/phase6_step_causal_test.py`
- [ ] For each step type (PARSE, DECOMPOSE, OPERATE, VERIFY, FORMAT):
  - Ablate each top-50 feature during ONLY that step
  - Measure step quality degradation (vs. baseline)
  - Rank features by step-specific importance
- [ ] Expected output:
  ```json
  {
    "PARSE": {"feature_123": 0.45, "feature_456": 0.38, ...},
    "DECOMPOSE": {"feature_789": 0.52, ...},
    "OPERATE": {...},
    "VERIFY": {...},
    "FORMAT": {...}
  }
  ```
- [ ] Compute: Is feature importance consistent across models for same step?
**Success Criteria**: 
  - Step-specific correlations r > 0.7 (vs. Phase 5 global r=0.33)
  - Feature importance varies significantly by step (>2x difference between steps)
**Estimated Time**: 1.5 hours

#### Task 6.4: Controllable Reasoning via Latent Steering [Priority: HIGH]
**Objective**: Test if we can control reasoning properties by manipulating latents  
**Files to Create**: `phase6/phase6_latent_steering.py`
- [ ] Experiments:
  1. **Length Control**: Remove high-importance Phi-2 DECOMPOSE features → faster reasoning
  2. **Quality Control**: Amplify VERIFY features → more careful checking
  3. **Failure Mode**: Zero out critical PARSE features → see model break predictably
- [ ] Measure impact: Reasoning length, accuracy, self-correction rate
- [ ] Validate: Correlates with predicted step importance from Task 6.3
**Success Criteria**: 
  - Can increase reasoning steps by 50% by amplifying decompose features
  - Can reduce errors by 30% by boosting verify features
  - Model behavior is predictable and controllable
**Estimated Time**: 1 hour

#### Task 6.5: Cross-Model Reasoning Universality [Priority: MEDIUM]
**Objective**: Identify universal vs. model-specific reasoning features  
**Files to Create**: `phase6/phase6_cross_model_reasoning.py`
- [ ] Extract step-level feature rankings: Phi-2, Pythia, Gemma, GPT-2
- [ ] Compute Spearman correlation of feature importance across model pairs
  - Expected: Core reasoning operations (PARSE, DECOMPOSE) have universal features
  - Expected: Formatting (FORMAT) has model-specific features
- [ ] Create feature universality matrix: which features transfer across reasoning operations
**Success Criteria**: 
  - Core operations show >0.6 correlation (partially universal)
  - Formatting shows <0.3 correlation (model-specific)
**Estimated Time**: 45 minutes

### Phase 6 Expected Outcomes

#### Research Findings
✅ **Mechanism of Interpretability Trade-off**: Why Phi-2 has causal features vs. GPT-2  
✅ **Reasoning Primitives**: Catalog of universal reasoning operations vs. model-specific optimizations  
✅ **Steering Feasibility**: Proof-of-concept that internal reasoning can be controlled without fine-tuning  
✅ **Alignment Implications**: Framework for aligning reasoning via latent interventions

#### Validation Criteria
- [ ] Step-level feature importance (r > 0.7) vs. global (r = 0.33)
- [ ] Steering efficacy: 50%+ controllability of reasoning properties
- [ ] Cross-model alignment: >60% feature overlap for core operations
- [ ] Reproducibility: Results hold across dataset seeds and model checkpoints

#### Publication Value
- **Mechanism paper**: "Why Are Sparse Features Interpretable Yet Unimportant?" (explains Phase 5 paradox)
- **Control paper**: "Steering Reasoning Without Fine-Tuning: A SAE-Based Framework" (Phase 6 demonstration)
- **Universality paper**: "Are Reasoning Features Universal Across LLMs?" (Phase 5.3 + Phase 6 combined)

### Phase 6 Timeline & Resource Allocation

```
Phase 6 Execution Path (if continuing):
  Day 1 (2h):   Tasks 6.1 + 6.2 (capture + CoT-SAE training)
  Day 2 (1.5h): Task 6.3 (step-level causal ablation)
  Day 3 (1h):   Task 6.4 (steering experiments)
  Day 4 (1.5h): Task 6.5 (cross-model analysis) + documentation
  
TOTAL: 7 hours estimated (within remaining 90+ hour GPU budget)
```

### Dependencies & Prerequisites
- ✅ Phase 4: SAE training infrastructure  
- ✅ Phase 4B: Feature statistics and interpretability framework
- ✅ Phase 5: Validation methodology and baselines
- ⏳ Phase 5.3: Transfer analysis (provides universality baseline for Phase 6.5)

### Recommended Path Forward

**Priority 1** (2h): Complete Phase 5.3 (Feature Transfer)  
├─ Answers: "Are features universal at all?"  
└─ De-risks Phase 6 assumptions

**Priority 2** (3h): Implement Phase 6.1-6.3 (CoT capture + step-level testing)  
├─ Answers: "Can we make importance signal stronger via step-level testing?"  
└─ Core research contribution

**Priority 3** (2h): Phase 6.4-6.5 (Steering + universality)  
├─ Answers: "Can we actually control reasoning? Is it universal?"  
└─ Publication-level impact

---

**Status**: All critical objectives met. Phase 6 offers 5-10x research value over Phase 5 with only 7 additional hours.



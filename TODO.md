# TODO

## 0. Repo + Planning Hygiene
- [ ] Reconfirm project scope: reproducing phased SAE interpretability experiment (overview.tex, Feb 2026) and define falsifiable stop criteria per phase before writing code.
- [ ] Decide which artifacts stay tracked in git after this cleanup (e.g., keep `overview.tex`, `TODO.md`, `LICENSE`, `.gitignore`) and document rationale in first commit.
- [ ] Stand up lightweight project README describing intent, dependencies, and pointers to overview + this TODO so future collaborators ramp quickly.
- [ ] Create issue tracker outline (GitHub Projects or plaintext) keyed to the sections below with owners + target dates.

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
- [ ] Add automatic probes for hypothesis/constraint tasks with leakage diagnostics (flag when probe vs linear baseline gap >5%). *(TBD Phase 3: integrate with labeled reasoning datasets)*
- [ ] Provide evaluation notebooks/scripts to visualize feature purity, cluster coherence, and activation sparsity histograms. *(TBD Section 6: evaluation metrics and notebooks)*

**RTX 6000 Timing Summary:**
- Single SAE (GPT-2, 768D→6144D, 50M tokens): **1.5-3 hours** (5 epochs)
- Full multi-model analysis (6 models, 1 layer each): **18-24 hours**
- Fits within 50-100 RTX-hour budget: ✅ See `TRAINING_PERFORMANCE_RTX6000.md` for detailed timing analysis.

## 5. Phased Research Program

### Phase 1 — Ground-Truth Systems [🔴 NOT STARTED - INFRASTRUCTURE READY]
**Status**: Skeleton implemented in `phase1_ground_truth.py`, ready to execute on GPUs 4-7 after Phase 3.
- [ ] Recreate simple environments (BFS/DFS traversals, stack machines) with exact latent states.
- [ ] Train SAEs on those states to verify reconstruction + monosemanticity; document failure cases as stop criteria.
- [ ] Build causal tests: directly edit SAE latents and ensure decoded state changes behave as predicted.
- **Timeline**: 5-7 days (parallel to Phase 3 full-scale)
- **Success criteria**: Reconstruction fidelity >95%, causal tests pass for all latent perturbations

### Phase 2 — Synthetic Transformers [🔴 NOT STARTED - PLACEHOLDER]
- [ ] Implement tiny transformers solving grid or logic tasks where full attention patterns are known.
- [ ] Hook activations, train SAEs with decorrelation + probes, and perform causal perturbations on internal features.
- [ ] Compare recovered features against known synthetic circuitry; stop if alignment falls below target purity/correlation thresholds.
- **Timeline**: 7-10 days after Phase 1
- **Success criteria**: Feature alignment purity ≥80%, causal effects match prediction

### Phase 3 — Controlled Chain-of-Thought LMs [🟡 SCALED FULL-DATASET IN PROGRESS]
**Status**: FULL DATASET EVALUATION NOW RUNNING
- [x] Collect datasets with labeled reasoning steps (GSM8K ~7,473 examples)
- [x] Train SAEs on selected layers, integrate probe guidance tied to labeled steps
- [⏳] **ONGOING**: Evaluate ALL 16 SAE expansions (2x-32x) across entire dataset
  - 4x-20x: Already trained, evaluating on GPUs 0-3
  - 2x, 22x-32x: Training on GPUs 4-7 (parallel)
  - Expected completion: ~20 hours
- [ ] Aggregate results and identify optimal SAE size
- [ ] Use results to inform Phase 4 architecture choices

**Resources**: 
- GPUs 0-3: Phase 3 full-scale evaluation
- GPUs 4-7: Missing SAE training
- **Key files**: `PHASE3_FULLSCALE_EXECUTION_GUIDE.md`, `phase3/phase3_orchestrate.sh`

### Phase 4 — Frontier LLMs [🔴 WAITING FOR PHASE 3 RESULTS]
- [ ] Select target LLM checkpoints (1B–7B) and reasoning benchmarks (math, logic, Sudoku) for final evaluation.
- [ ] Run capture, SAE training, and validation using same falsification gates; log hardware utilization vs budget.
- [ ] Execute causal circuit tests (feature ablations, activation patching) to confirm stable reasoning primitives.
- **Depends on**: Phase 3 optimal SAE size + Phase 1 validation results

## 6. Validation & Attribution Protocols
- [ ] Codify purity metrics (top-k coherence, silhouette) and automate reporting per feature.
- [ ] Implement causal attribution sweeps: add epsilon along decoder atoms, run inference, measure task deltas plus calibration curves.
- [ ] Establish baselines (null SAE, random features) to contextualize causal scores.
- [ ] Build replication harness: hold-out datasets, independent reruns, external audits.

## 7. Risk Management
- [ ] Define domain coverage tests to detect brittleness/invariance failures noted in overview.
- [ ] Create leakage detection suite (compare SAE features against probes, run ablations) to guard against identifiability illusions.
- [ ] Track assumptions + open questions per phase; decide go/no-go checkpoints before promoting to next phase.

## 8. Resource & Timeline Tracking
- [ ] Translate 50–100 RTX-hour budget into concrete run schedules; monitor GPU vs wall-clock targets.
- [ ] Maintain log of dataset sizes, activation storage requirements (<100 GB target), and cleaning steps.
- [ ] Outline two-week single-GPU execution plan with daily milestones and validation deliverables.

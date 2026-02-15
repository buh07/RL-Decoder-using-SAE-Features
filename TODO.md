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
- [ ] Choose initial base models (e.g., GPT-2 small, 1B distilled) with layer-probe defaults (mid-layers like 6 for GPT-2) and record ckpt hashes.
- [ ] Implement activation capture stack constrained to post-MLP residual streams + MLP hidden states; ensure streaming to disk is fp16 + layer-filtered to fit single GPU memory.
- [ ] Validate hooking latency + throughput on sample batches before SAE training.

## 4. SAE Architecture & Training
- [ ] Formalize SAE architecture hyperparams: expansion factor 4-8x width, ReLU latents, decoder sharing vs independent per layer.
- [ ] Implement total loss: reconstruction + L1 sparsity + decorrelation penalty + optional probe-guided loss + temporal smoothness term.
- [ ] Build configurable training loop supporting streaming batches, logging recon error, activation coverage, feature activations.
- [ ] Add automatic probes for hypothesis/constraint tasks with leakage diagnostics (flag when probe vs linear baseline gap >5%).
- [ ] Provide evaluation notebooks/scripts to visualize feature purity, cluster coherence, and activation sparsity histograms.

## 5. Phased Research Program
### Phase 1 — Ground-Truth Systems
- [ ] Recreate simple environments (BFS/DFS traversals, stack machines) with exact latent states.
- [ ] Train SAEs on those states to verify reconstruction + monosemanticity; document failure cases as stop criteria.
- [ ] Build causal tests: directly edit SAE latents and ensure decoded state changes behave as predicted.

### Phase 2 — Synthetic Transformers
- [ ] Implement tiny transformers solving grid or logic tasks where full attention patterns are known.
- [ ] Hook activations, train SAEs with decorrelation + probes, and perform causal perturbations on internal features.
- [ ] Compare recovered features against known synthetic circuitry; stop if alignment falls below target purity/correlation thresholds.

### Phase 3 — Controlled Chain-of-Thought LMs
- [ ] Collect datasets with labeled reasoning steps; map steps to token spans using regex/similarity heuristics + manual spot checks.
- [ ] Train SAEs on selected layers, integrate probe guidance tied to labeled steps, and evaluate leakage metrics.
- [ ] Align learned features with reasoning steps using multi-aligner consensus; establish confidence scoring + EM refinement loop.

### Phase 4 — Frontier LLMs
- [ ] Select target LLM checkpoints (1B–7B) and reasoning benchmarks (math, logic, Sudoku) for final evaluation.
- [ ] Run capture, SAE training, and validation using same falsification gates; log hardware utilization vs budget.
- [ ] Execute causal circuit tests (feature ablations, activation patching) to confirm stable reasoning primitives.

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

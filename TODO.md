# TODO — RL-Decoder with SAE Features

## Meta Notes
- [ ] Record that multiple pretrained LLM checkpoints already live in `/scratch2/f004ndc/LLM Second-Order Effects/models` (pythia-1.4b, gemma-2b, gpt2, gpt2-medium, Meta-Llama-3-8B, phi-2). Decide which will anchor Phase 1 prototyping vs. later scaling. _(context: see `docs/data_manifest.yaml` for current corpus targets.)_

## Phase 1 Kickoff Model — gpt2
- [x] Lock the initial target to `gpt2` (117M params, 12 layers, d_model=768, 12 heads, vocab=50,257) located at `/scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2`. Use this checkpoint + tokenizer for all Phase 1 baselines.
- [x] Extract gpt2 architectural constants into a shared config (`configs/models/gpt2_phase1.yaml`) so SAE/decoder scripts automatically set layer choices (focus on layer 6 for hidden-state taps, adjust if logit lens suggests better signal).
- [ ] Verify tokenizer files (`tokenizer.json`, `merges.txt`, `vocab.json`) exist in the model directory and sync hashes into README for reproducibility. (Need to copy from HF cache blobs → local path; `find` as of 01:22 UTC still shows only config/model files. Remote fetches remain blocked: no system `transformers`/`huggingface_hub`, pip install fails because PyPI is unreachable/offline.) _Progress_: added `scripts/sync_tokenizer_assets.py` + `docs/tokenizer_hashes.md`; waiting on actual files to copy & hash.
- [x] Generate a quick sanity script (`scripts/phase1_gpt2_smoketest.py`) that runs the frozen model on a mini batch (1k tokens) and logs throughput + hidden-state shapes to confirm env/tooling.

## Phase 1 Resources Snapshot
- [x] GPU availability on January 25, 2026 @ 00:51:51: eight NVIDIA RTX 6000 Ada (48 GB each). GPU3 currently saturated (~16.5 GB, 100% util); GPUs 0/4/5 idle; GPUs 1/2/6/7 partially used. Plan SAE/RL workloads around free GPUs or coordinate with existing jobs. (`nvidia-smi` unavailable in sandbox—rely on operator-provided snapshot.)

## Phase 1 — Setup & Preparation
- [x] Validate compute availability (≥2×A100 40GB for SAE + RL, spare GPU for labeling); document any constraints or queueing requirements. (Finalized GPU snapshot + noted sandbox `nvidia-smi` limitation.)
- [x] Create isolated conda/venv with PyTorch 2.0+, HuggingFace Transformers, TRL, wandb, SAE libs; script environment bootstrap. (`scripts/setup_env.sh` provisions env `rl_decoder_phase1` w/ CUDA 12.1 wheels + core deps; run on host with conda.)
- [x] Stand up data ingestion pipeline (tokenizer parity with chosen base LLM, sharded dataloader, fast disk cache) and verify throughput on target hardware. (Implementation of `src/data/dataloader.py` + `scripts/preprocess_dataset.py` complete; throughput validation blocked until tokenizer assets + datasets can be pulled.)
- [ ] Gather/curate training corpora: 100M-token supervised split, 10B-token RL buffer, 100k-token clean eval, 50k-token OOD eval; log provenance + licenses. _See `docs/data_manifest.yaml` for target shard mix + licensing checklist._
- [ ] Run baseline measurements on frozen model (next-token accuracy, CoT quality, throughput, memory) and layer-wise logit lens stats for layers {4,8,12,16,20,24,28,32}; store in `docs/baselines.yaml`.
- [ ] Enable experiment tracking (wandb project + config templates) and checkpoint storage layout (checkpoints/, results/, logs/). _Scaffolds ready: `configs/tracking/wandb_template.yaml`, `scripts/init_wandb_project.py`, and `docs/tracking.md`. Need to export WANDB_API_KEY + generate first run config._

## Phase 2 — Sparse Autoencoder (SAE) Training
- [ ] Implement SAE module exactly as spec (4096→40,960 expansion, sparsity controls, dead-neuron resampling hooks) with config-driven hyperparams.
- [ ] Build efficient residual-stream extractor that caches layer-8 activations while respecting VRAM via gradient checkpointing + fp16 storage.
- [ ] Train SAE for planned 50k steps with scheduled sparsity coeffs; monitor reconstruction loss, sparsity, dead-neuron %; add alerting if drift.
- [ ] Periodically checkpoint (every 5k steps) and benchmark reconstruction MSE + explained variance; keep best artifact tagged.
- [ ] After training, run QA suite: reconstruction MSE <0.01, KL vs. hidden states <0.5 nats, average L0 between 50-100, dead feature ratio <5%.
- [ ] (Optional) Repeat SAE on alternative layers (10,12,14,16) and compare interpretability metrics to finalize layer choice for downstream stages.

## Phase 3 — Supervised Decoder Pre-training
- [ ] Implement baseline linear decoder over SAE features plus optional lightweight transformer path toggled via config.
- [ ] Integrate supervised training loop that ingests frozen model logits, trains for 3 epochs, tracks CE loss, KL divergence, and accuracy.
- [ ] Add eval script to compute KL vs frozen model, next-token accuracy on validation + OOD splits after each epoch; gate checkpoints on KL <1.5 nats.
- [ ] Save `decoder_supervised_init.pt`, push metrics to wandb, and archive training config for reproducibility.

## Phase 4 — RL-Based Decoder Fine-tuning
- [ ] Implement multi-objective reward module (faithfulness, monosemanticity, task accuracy) with phase-aware weight schedules + KL penalty.
- [ ] Wire decoder into PPO trainer (batched rollout generation using SAE features); ensure frozen base model stays eval-only.
- [ ] Run staged PPO schedule (Phase1=faithfulness 2k steps, Phase2=balanced 6k, Phase3=accuracy 2k) while logging KL, task metrics, sparsity stats.
- [ ] Perform random-policy baseline test to detect reward hacking; halt training if random performs near trained model.
- [ ] Freeze final RL decoder checkpoint once KL <0.5 nats and task accuracy drop <3% relative to frozen baseline.

## Phase 5 — Feature Interpretation & Labeling
- [ ] Build activation patching + feature importance tooling to rank influential SAE features per task/domain.
- [ ] Select top-K (e.g., 1000) features for labeling and auto-generate prompts for LLM-assisted labeling pipeline; stage batches for human review.
- [ ] Implement active-learning loop: capture low-confidence labels, route for manual annotation, maintain `feature_labels.json` with confidence scores.
- [ ] Run semantic clustering / duplicate detection to merge redundant features and flag low-quality ones for retraining.

## Phase 6 — Human-Interpretable Output Generation
- [ ] Implement Method A (feature ranking text) producing deterministic explanations directly from activation magnitudes.
- [ ] Build dataset for Method B (explanation decoder) by pairing active feature sets with templated natural-language rationales; target ≥5k samples.
- [ ] Train explanation decoder, add faithfulness regularizer (activation patching feedback), and evaluate narrative quality vs. human judgments.
- [ ] Assemble dashboard pipeline that outputs prediction, feature trace, and explanation; integrate uncertainty estimates + sanity checks.

## Validation & Testing
- [ ] Author unit/integration tests for SAE reconstruction, decoder faithfulness, feature sparsity, reward calculator, and dashboard rendering.
- [ ] Run task-suite evaluation (GSM8K, BoolQ, SQuAD, OOD set) comparing frozen vs RL decoder; log metrics + degradation deltas.
- [ ] Execute interpretability benchmarks (label coverage, top-k importance mass, explanation faithfulness) and document results.

## Documentation & Ops
- [ ] Keep project logbook (daily decisions, anomalies, mitigation steps) in `docs/logbook.md`.
- [ ] Document hyperparameters, training schedules, and hardware utilization in `docs/experiments/` per run.
- [ ] Prepare publication-quality figures (architecture diagram, training curves, interpretability visuals) once metrics stabilize.
- [ ] Draft reproducibility checklist + release notes for code/artifact sharing.

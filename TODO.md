# Active Tasks

**Last Updated:** March 2, 2026

For full status and key results, see [PROJECT_STATUS.md](PROJECT_STATUS.md).

---

## Phase Status Summary

| Phase | Name | Status |
|-------|------|--------|
| 1 | Ground-Truth Validation | ✅ Complete |
| 2 | Multi-Layer SAE Training | ✅ Complete |
| 2r | SAE Retrain — TopK | ✅ Complete |
| 3 | Reasoning Flow Tracing | ✅ Complete |
| 4 | Arithmetic Feature Probing (v1, ReLU) | ✅ Complete |
| 4r | Arithmetic Probing — TopK + Subspace | ✅ Complete |
| 5 | Feature Interpretation + Causal Steering | ✅ Complete |
| 6 | Decoder Benchmark (dataset + supervised + RL follow-up + interpretability) | ✅ Complete |
| 7 | Causal CoT Verification Auditor (v1 implementation) | ✅ v1 Complete (gates pass on fixv3_v2) |
| 7v2 | Auditor Overhaul — Fix Scoring + Latent Signal + Multi-Model | 🟡 In progress (Stage 0/1 implemented; latent gate still failing) |

---

## Critical Assessment of Phase 7 v1 Results

Phase 7 v1 produced passing gate metrics (`trackA_gate_matrix_fixv3_v2.json`: AUROC=0.968,
FPR=0.018) but these results are **misleading** — the "causal auditor" is dominated by text
matching, not causal or latent evidence. Before any downstream use (reranking, reward shaping),
the following issues must be resolved:

### Issue 1: Scoring imbalance in v1 (historical) was real and is now fixed in code

v1 heavily overweighted text matching. In v2, `_score_step()` is now explicitly
track-balanced (`text=0.35`, `latent=0.35`, `causal=0.30`) with subtractive penalties,
track-specific calibration thresholds, and ablation support. This issue is no longer a
code blocker.

### Issue 2: `latent_only` track improved but remains below gate target

After v2 semantic fixes, `latent_only` is no longer at chance, but current Stage 1 runs are
still below gate target (`~0.626` vs required `>=0.70`) across the 4 calibration checkpoints.
This is the primary active blocker.

### Issue 3: AUROC interpretation should be track-specific

Overall `causal_auditor` AUROC is now high (~0.95) with stable FPR, but that does not imply
latent track success. Stage decisions must continue to use per-track gates and not only the
combined score.

### Issue 4: readout_high_causal_fail remains substantial

Current v2 runs still show high separation counts (`~938–989 / 4500`, ~21–22%). This is
useful evidence for decoupling, but it also indicates latent/causal disagreement remains a
major analysis item before Phase 8 optimization.

### Issue 5: Broad sweep never completed

The `20260302_125809_phase7_broad_sweep` was paused after claiming 1 job with 0 completed.
All 102 configs remain pending. No broad sweep data exists.

---

## Phase 7v2 — Auditor Overhaul (Primary Active Work)

**Goal**: Fix the three core deficiencies (scoring imbalance, latent signal at chance,
GPT-2-only limitation) so that the causal auditor provides genuinely meaningful
faithfulness verification — then validate on both GPT-2 and Qwen 2.5-7B.

Phase 7v2 is structured into four sequential stages. Each stage has a hard prerequisite
gate — do not proceed to the next stage until the prior gate is passed.

---

### Stage 0: Diagnosis — Understand Why Latent Readout Fails (GPT-2)

**Goal**: Determine whether the `latent_only` AUROC=0.500 is caused by (a) the state
decoder not learning useful latent representations, (b) the comparison logic in
`compare_states` / `_score_step` mapping decoded states to scores incorrectly, or
(c) the control design not producing latent-level differences between faithful and
unfaithful traces.

- [x] **0.1** Run diagnostic: for each control variant, compute the raw state decoder
      predictions (operator, magnitude, sign, subresult) on both the faithful and
      unfaithful version of the same trace, and measure whether the decoder outputs
      differ at all.
      Script: write `phase7/diagnose_latent_separation.py`
      Inputs: `phase7_results/dataset/gsm8k_step_traces_test.pt`,
              calibration checkpoint `phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration/checkpoints/state_raw_every2_even.pt`
      Outputs: `phase7_results/diagnostics/latent_separation_diagnostic.json`
      Key metric: per-variant mean L2 distance between faithful and unfaithful
      latent predictions (should be >> 0 if decoder is informative).

- [x] **0.2** Run diagnostic: compute the `compare_states` output for faithful traces
      (text_state vs gold) and for each unfaithful variant (text_state vs gold), and
      check whether `categorical_match_fraction` and `numeric_match_fraction` separate
      faithful from unfaithful variants.
      Script: add diagnostic mode to `phase7/diagnose_latent_separation.py`
      Key question: is the *gold comparison* (not latent) already failing to separate?

- [x] **0.3** Hypothesis test: the state decoder receives hidden states from the
      **same forward pass** regardless of whether the CoT text says faithful or
      unfaithful things — because the CoT text is generated *after* the hidden states
      are captured at the `=` token. If the decoder's input is identical for faithful
      and unfaithful controls of the same underlying example, then `latent_only` is
      **correctly** at chance — it *should* be unable to distinguish, because the
      latent state doesn't depend on the CoT text variant.
      Test: verify that `raw_hidden` tensors fed to the state decoder are identical
      across control variants of the same trace. If confirmed, this is not a bug but
      a fundamental design issue with how `latent_only` is computed.

      Expected outcome: the hidden states ARE the same (they come from `gsm8k_step_traces_test.pt`,
      not from per-control forward passes). This means `latent_only` cannot possibly separate
      faithful from unfaithful controls in the current design — it always predicts the same
      latent state for all variants of the same example.

- [x] **0.4** Document diagnosis findings in
      `phase7_results/diagnostics/latent_separation_report.md`:
      - Confirm or refute hypothesis 0.3
      - If confirmed: the fix is to make `latent_only` score compare **decoded latent state
        vs text-claimed state** (not decoded vs gold), so that unfaithful text claims
        disagree with latent predictions
      - If refuted: investigate why decoder outputs don't separate

**Stage 0 gate**: diagnosis report complete; root cause identified and documented.

---

### Stage 1: Fix Scoring and Track Separation (GPT-2)

**Goal**: Rewrite scoring so that each track provides genuinely independent signal, and
the combined `causal_auditor` score reflects all three information sources meaningfully.

#### 1A. Fix `latent_only` track semantics

Based on Stage 0 diagnosis (expected: hidden states are identical across variants),
the fix is:

- [x] **1A.1** Modify `causal_audit.py:_score_step()` and `_paper_metrics_for_trace()`
      so that `latent_only` compares **decoded latent state vs text-claimed state**
      (the CoT's parsed step), not decoded latent state vs gold ground truth.
      Rationale: for faithful traces, text claim = latent prediction (both correct).
      For unfaithful traces, text claim != latent prediction (text is wrong, latent is right).
      This gives `latent_only` actual separation power.

      Specific changes in `causal_audit.py`:
      - Add `text_vs_latent_agreement` field to step audit output, computed via
        `compare_states(parsed_text_state, latent_pred_state)`
      - The `latent_only` track score should be the mean `text_vs_latent_agreement`
        across steps (high = text agrees with latent = likely faithful;
        low = text disagrees with latent = likely unfaithful)

- [x] **1A.2** Add unit test: given a faithful control and a `wrong_intermediate` control
      for the same trace, verify that `latent_only` score is higher for the faithful control.
      File: `phase7/tests/test_latent_track_separation.py`

- [x] **1A.3** Run the full benchmark pipeline with the fixed `latent_only` semantics
      on the existing calibration checkpoint (`state_raw_every2_even`):
      - Generate controls, parse, audit, calibrate, benchmark
      - Verify `latent_only` AUROC is now meaningfully above chance (target: >= 0.70)
      Output: `phase7_results/results/faithfulness_benchmark_v2_latent_fix_raw_every2_even.json`

#### 1B. Rebalance combined scoring weights

- [x] **1B.1** Restructure `_score_step()` in `causal_audit.py` to use a three-component
      scoring model instead of the current monolithic weighted sum:

      ```
      text_score   = f(text_vs_gold_agreement)          # Track A signal
      latent_score = f(text_vs_latent_agreement)         # Track B signal (fixed)
      causal_score = f(necessity, sufficiency, specificity)  # Track C signal

      # Combined score with equal weighting across tracks:
      combined = 0.35 * text_score + 0.35 * latent_score + 0.30 * causal_score
      ```

      Keep penalty terms (contradiction, off-manifold, revision) as subtractive modifiers
      that can push the score down but don't inflate it.

      Concrete weight proposal:
      - `text_score = 0.40 * text_match + 0.30 * categorical_match + 0.30 * numeric_match`
      - `latent_score = text_vs_latent match_fraction` (from 1A.1)
      - `causal_score = (necessity_pass * 0.35 + sufficiency_pass * 0.40 + specificity_pass * 0.25)`
        where each is 1.0 if pass, 0.0 if fail, 0.5 if untested/None

      If causal checks are unavailable (e.g., no SAE for this model), `causal_score`
      defaults to 0.5 (agnostic) and the combined formula drops to:
      `combined = 0.50 * text_score + 0.50 * latent_score` (text+latent only mode)

- [x] **1B.2** Update `benchmark_faithfulness.py` to use **per-track thresholds**
      instead of a single global threshold. Each track should be calibrated independently:
      - `calibrate_audit_thresholds.py` should output thresholds for `text_only`,
        `latent_only`, and `causal_auditor` tracks separately.
      - `benchmark_faithfulness.py` should apply the track-specific threshold when
        computing per-track metrics.

- [x] **1B.3** Add ablation mode to `benchmark_faithfulness.py`:
      `--ablation-weights` flag that accepts JSON like
      `{"text": 0.35, "latent": 0.35, "causal": 0.30}` to test different weight
      configurations without modifying code. Record the weights used in the output JSON.

- [x] **1B.4** Run ablation sweep over weight configurations to find the split that
      maximizes overall AUROC while maintaining per-track AUROC >= 0.70:
      - `{0.50, 0.50, 0.00}` (no causal — what text+latent alone achieves)
      - `{0.35, 0.35, 0.30}` (proposed balanced)
      - `{0.25, 0.25, 0.50}` (causal-heavy)
      - `{0.33, 0.33, 0.34}` (equal)
      Output: `phase7_results/results/scoring_ablation_weight_sweep.json`

#### 1C. Validation on calibration checkpoints

- [x] **1C.1** Re-run the full Phase 7 benchmark pipeline (controls -> parse -> audit
      -> calibrate -> benchmark) with the v2 scoring on all 4 calibration checkpoints:
      `state_raw_every2_even`, `state_hybrid_every2_even`,
      `state_raw_middle12_06_17`, `state_hybrid_middle12_06_17`
      Outputs: `phase7_results/results/faithfulness_benchmark_v2_<checkpoint>.json`

- [x] **1C.2** Compare v1 vs v2 metrics side-by-side:
      - Per-track AUROC (text_only, latent_only, causal_auditor)
      - Overall AUROC
      - FPR at matched threshold
      - Per-variant AUROC for each of the 8 control variants
      Output: `phase7_results/results/v1_vs_v2_scoring_comparison.json`

**Stage 1 gate**: all of the following must hold on at least one calibration checkpoint:
- `latent_only` AUROC >= 0.70 (was 0.50 in v1 — real separation, not chance)
- `causal_auditor` AUROC >= 0.75 (with rebalanced weights)
- `text_only` AUROC >= 0.80 (should be preserved from v1)
- Per-track thresholds produce FPR <= 0.05 on each track independently
- At least 2 control variants have per-variant AUROC >= 0.80

Current status (run tag `phase7v2_s01_20260302_225055_phase7v2_s01_g5`):
- `text_only` and `causal_auditor` gates pass on all 4 checkpoints.
- Per-track FPR and variant-vs-faithful coverage gates pass.
- `latent_only` AUROC is ~0.626 across checkpoints, so Stage 1 gate is still blocked.

---

### Stage 2: Complete the Broad Sweep with v2 Scoring (GPT-2)

**Goal**: Run the full 102-config layer sweep with v2 scoring to determine which
layer sets maximize each track's quality, and where readout vs causal support diverge.

#### 2A. Phase 7 broad sweep (state decoder train + eval)

- [ ] **2A.1** Resume or restart the broad sweep
      (`phase7_results/sweeps/20260302_125809_phase7_broad_sweep/`).
      Use the existing queue infrastructure (`build_queue.py`, `worker.py`, `progress.py`).
      Scope: `43 raw + 43 hybrid + 16 sae = 102` configs.
      Estimated runtime: 12-20h on 2 GPUs.

- [ ] **2A.2** For each config, produce:
      - Checkpoint: `phase7_results/sweeps/<SWEEP_RUN_ID>/checkpoints/state_<variant>_<layer_set_id>.pt`
      - Train results: `state_decoder_supervised_<name>.json`
      - Eval results: `state_decoder_eval_<name>.json`
      Metrics per config: `result_token_top1`, `result_token_top5`, `operator_acc`,
      `step_type_acc`, `magnitude_acc`, `sign_acc`, `delta_logprob_vs_gpt2`,
      `subresult_mae`, `lhs_mae`, `rhs_mae`, per-operator result-top1.

- [ ] **2A.3** Run `experiments/summarize_phase7_layer_sweep.py` to produce:
      `phase7_results/sweeps/<SWEEP_RUN_ID>/results/layer_sweep_phase7_summary.json`

#### 2B. Deterministic causal shortlist + causal runs

- [ ] **2B.1** Run `experiments/select_phase7_causal_shortlist.py` using Phase 6 + Phase 7
      sweep summaries. Fill 6 shortlist slots per the manifest rules.
      Output: `phase7_results/interventions/layer_sweep_causal_shortlist_summary.json`

- [ ] **2B.2** For each of 6 shortlisted layer sets, run causal checks on `subresult_value`:
      necessity + sufficiency + specificity, probe+saliency union subspaces,
      off-manifold checks enabled. Budget: 100 test step records per subset (pass 1).
      Outputs: `phase7_results/interventions/causal_checks_<config>_subresult_value.json`

- [ ] **2B.3** Expand top 2 most informative subsets to 300 records (pass 2).

#### 2C. Full audit + benchmark with v2 scoring on shortlisted configs

- [ ] **2C.1** Generate controls, parse, run `causal_audit.py`, `calibrate_audit_thresholds.py`,
      `benchmark_faithfulness.py` with v2 scoring on each shortlisted config.
      Outputs: per-config audit, calibration, and benchmark JSONs.

- [ ] **2C.2** Produce cross-config comparison table:
      layer_set_id | latent_only AUROC | causal_auditor AUROC | text_only AUROC |
      readout_high_causal_fail_n | best_track

**Stage 2 gate**: broad sweep complete (102/102 configs); at least 4 shortlisted configs
benchmarked with v2 scoring; cross-config comparison table produced.

---

### Stage 3: Qwen 2.5-7B Full Pipeline (Multi-Model Validation)

**Goal**: Run the complete Phase 7 pipeline on Qwen 2.5-7B to validate that the
auditor generalizes beyond GPT-2 and produces meaningful results on a model with
actual CoT capability.

#### 3A. Qwen infrastructure setup

- [ ] **3A.1** Verify Qwen adapter loads and runs forward pass:
      ```bash
      CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -c "
      from phase7.model_registry import create_adapter
      adapter = create_adapter('qwen2.5-7b', device='cuda:0')
      adapter.load()
      print(f'Loaded: {adapter.spec.num_layers} layers, {adapter.spec.hidden_dim} hidden')
      ids = adapter.tokenize('What is 2 + 3?')
      logits, hidden = adapter.forward(ids)
      print(f'Forward OK: logits {logits.shape}, hidden {len(hidden)} layers')
      "
      ```

- [ ] **3A.2** Build Qwen-specific activation capture pipeline.
      Qwen has 28 layers and 3584-dim hidden states — the existing
      `phase6/pipeline_utils.py:build_input_tensor_from_record` needs to handle
      variable `hidden_dim`. Either:
      (a) Generalize `build_input_tensor_from_record` to accept `hidden_dim` from model spec, or
      (b) Write a Qwen-specific data pipeline in `phase7/qwen_data_pipeline.py`.
      Decision: option (a) is preferred to avoid code duplication.

      Specific changes:
      - `ArithmeticDecoderConfig.input_dim` should be set from `model_spec.hidden_dim`
        (currently hardcoded to 1024 for GPT-2)
      - `MultiHeadStateDecoder` backbone should accept variable input_dim
      - `DEFAULT_VOCAB_SIZE` should come from model spec (Qwen vocab = 152064, GPT-2 = 50257)

- [ ] **3A.3** Build Qwen step-trace dataset:
      ```bash
      CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 phase7/build_step_trace_dataset.py \
        --model-key qwen2.5-7b --max-train 500 --max-test 200 \
        --output-dir phase7_results/dataset_qwen25_7b/ --device cuda:0
      ```
      This requires running Qwen forward passes on GSM8K to capture hidden states.
      Key decisions:
      - Use the same GSM8K examples as GPT-2 for comparability
      - Token positions: find `=` token position in Qwen's tokenization
        (different tokenizer, may differ from GPT-2)
      - Store `raw_hidden[28, 3584]` per record (vs GPT-2's `[24, 1024]`)
      Output: `phase7_results/dataset_qwen25_7b/gsm8k_step_traces_{train,test}.pt`

- [ ] **3A.4** Validate dataset schema and record counts.
      Output: `phase7_results/dataset_qwen25_7b/build_summary.json`

#### 3B. Qwen state decoder training

- [ ] **3B.1** Train state decoder on Qwen hidden states.
      Start with a small calibration run (1-2 layer sets) to verify training converges:
      ```bash
      CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 phase7/train_state_decoders.py \
        --dataset-train phase7_results/dataset_qwen25_7b/gsm8k_step_traces_train.pt \
        --model-key qwen2.5-7b --layers 14,21 --input-variant raw \
        --epochs 30 --device cuda:0
      ```
      Expected: Qwen's larger hidden dim (3584) should give at least comparable
      state decoding quality to GPT-2 (result_token_top1 >= 0.50).
      Output: checkpoint + `state_decoder_supervised_*.json` + `state_decoder_eval_*.json`

- [ ] **3B.2** If calibration succeeds, run a targeted layer sweep on Qwen.
      Qwen has 28 layers — adapt the manifest or create a Qwen-specific manifest:
      - 28 single layers
      - block4 sets (7 windows)
      - block8 sets (4 windows)
      - spread4: layers [7, 14, 21, 27]
      - every2: even and odd
      - middle14: layers 7-20
      Total: ~50-60 configs (raw only initially; no Qwen SAEs exist).
      Script: write `experiments/build_qwen_layer_sweep_manifest.py`
      Output: `experiments/qwen25_7b_layer_sweep_manifest_v1.json`

- [ ] **3B.3** Run the Qwen state decoder sweep (raw variant only, ~50 configs).
      Estimated runtime: 15-25h on 1-2 GPUs (Qwen is ~7x larger than GPT-2 medium).
      Output: `phase7_results/sweeps/<QWEN_SWEEP_RUN_ID>/`

- [ ] **3B.4** Summarize Qwen sweep and compare to GPT-2 sweep.
      Key question: does Qwen's state decoder show similar or different layer-dependency
      patterns compared to GPT-2?
      Output: `phase7_results/sweeps/<QWEN_SWEEP_RUN_ID>/results/layer_sweep_qwen_summary.json`

#### 3C. Qwen causal intervention path

Causal interventions require SAE-derived subspaces. Qwen has no trained SAEs.
Two options (choose one):

- [ ] **3C.1** Option A (preferred if feasible): Train TopK SAEs on Qwen 2.5-7B
      hidden states for a subset of layers (e.g., layers 7, 14, 21, 27).
      Use the existing `phase2/train_multilayer_saes.py` infrastructure.
      Requirements:
      - Capture Qwen activations: `phase2/capture_activations.py --model qwen2.5-7b`
      - Train SAEs: expansion factor 8x (3584 * 8 = 28672 features),
        TopK with K = 0.30 * 28672 = 8602
      - Train probe on Qwen SAE features to identify arithmetic subspaces
      Estimated runtime: 8-12h per layer for SAE training on 1 GPU.
      Outputs: `phase2_results/saes_qwen25_7b_8x_topk/saes/qwen25-7b_layer{N}_sae.pt`

- [ ] **3C.2** Option B (fallback): Use PCA-derived subspaces instead of SAE decoder
      columns. Compute PCA on Qwen hidden states at the `=` token position, take top-K
      components that explain arithmetic variance (identified by correlation with
      subresult value). This avoids SAE training but provides weaker subspaces.
      Script: write `phase7/pca_subspace_builder.py`
      Output: `phase7_results/subspaces/qwen25_7b_pca_subspace.pt`

- [ ] **3C.3** Once subspaces exist, run causal checks on Qwen:
      - Update `causal_intervention_engine.py` to load Qwen-specific subspaces
        (currently returns `unsupported_reason: sae_loader_unimplemented_for_model:qwen2.5-7b`)
      - Run necessity + sufficiency + specificity on 50-100 test records
      - Compare Qwen causal effect sizes to GPT-2 causal effect sizes
      Output: `phase7_results/interventions/causal_checks_qwen25_7b_<config>.json`

#### 3D. Qwen full audit + benchmark

- [ ] **3D.1** Generate Qwen-specific controls:
      ```bash
      .venv/bin/python3 phase7/generate_cot_controls.py \
        --trace-dataset phase7_results/dataset_qwen25_7b/gsm8k_step_traces_test.pt \
        --output phase7_results/controls/cot_controls_qwen25_7b.json
      ```

- [ ] **3D.2** Run full audit pipeline (parse -> audit -> calibrate -> benchmark)
      with v2 scoring on Qwen.
      Outputs:
      - `phase7_results/audits/text_causal_audit_qwen25_7b_<config>.json`
      - `phase7_results/calibration/phase7_thresholds_v2_qwen25_7b_<config>.json`
      - `phase7_results/results/faithfulness_benchmark_v2_qwen25_7b_<config>.json`

- [ ] **3D.3** Cross-model comparison report:
      - GPT-2 vs Qwen per-track AUROC
      - GPT-2 vs Qwen per-variant AUROC
      - GPT-2 vs Qwen causal effect sizes (if 3C completes)
      - Where do the models agree/disagree on faithfulness verdicts?
      Output: `phase7_results/results/cross_model_comparison_gpt2_vs_qwen.json`

**Stage 3 gate**: Qwen state decoder trained on at least 2 layer configurations;
Qwen `latent_only` AUROC >= 0.65; full audit pipeline runs end-to-end on Qwen;
cross-model comparison report produced.

---

### Stage 4: Cross-Phase Analysis + Final Report

**Goal**: Synthesize all findings into a clear narrative about where decodable
information lives vs where causal support is strongest, across both models.

- [ ] **4.1** Compare Phase 6 vs Phase 7 sweep results by:
      layer family, layer count, input variant (`raw`/`hybrid`/`sae`).
      Identify layer sets that maximize (a) readout quality and (b) causal support.

- [ ] **4.2** Explicitly report mismatches:
      high latent_only / weak causal support (core evidence for "decoded signal != faithfulness").
      Compare middle-layer subsets vs global/sparse subsets across both models.

- [ ] **4.3** Summarize what this implies for improving CoT faithfulness:
      which layers are best for readout instrumentation vs causal verification.

- [ ] **4.4** Write cross-model findings note:
      does the auditor generalize? Which tracks are model-dependent vs model-independent?

- [ ] **4.5** Produce final deliverables:
      - `phase7_results/results/phase7v2_final_summary.json`
        (all metrics, both models, all tracks)
      - `docs/PHASE7V2_FINDINGS.md`
        (narrative summary: "where decodable information lives" vs "where causal support
        is strongest" vs "what transfers across models")

**Stage 4 gate**: final summary and findings doc complete.

---

## Post-Phase-7v2 — CoT Improvement via Reranking

Only start this work after Phase 7v2 Stage 1 gate passes (latent_only AUROC >= 0.70,
causal_auditor with rebalanced weights AUROC >= 0.75). The reranking plan in
`docs/PHASE8_RERANKING_PLAN.md` remains valid but should use v2 scoring.

Updated go/no-go decision table:

| Gate | Condition | Required for Go? | If Fail |
|---|---|---|---|
| A | `latent_only` AUROC >= 0.70 (not just chance) | Yes | Fix latent track per Stage 1A |
| B | `causal_auditor` AUROC >= 0.75 (rebalanced weights) | Yes | Run weight ablation per Stage 1B |
| C | `text_only` AUROC >= 0.80 | Yes | Should be preserved; investigate if regression |
| D | Per-track FPR <= 0.05 (per-track thresholds) | Yes | Recalibrate per track |
| E | At least 2 control variants with per-variant AUROC >= 0.80 | Yes | Improve control diversity or scoring |
| F | Benchmark output includes A/B/C tracks + claim-boundary | Yes | Fix output contract |

Potential follow-up work:
- [ ] Use `causal_auditor` scores for reranking candidate CoTs
- [ ] Filter training data by causal-support score
- [ ] Explore reward shaping with bounded claims (auditor as noisy faithfulness signal)
- [ ] Expand from arithmetic controls to real CoT traces on Qwen 2.5-7B

---

## Backlog (Not Blocking Phase 7v2)

### Phase 6 Housekeeping
- [ ] Backfill canonical summary/report for Phase 6 full sweep run
      `20260226_164900_phase6_full_sweep`:
      expected canonical outputs in `phase6_results/results/`
      (`<RUN_ID>_phase6_full_sweep_summary.json`, `<RUN_ID>_phase6_full_sweep_run_report.json`)
- [ ] Run `experiments/summarize_phase6_layer_sweep.py` on the sweep result dir.
- [ ] Run `phase6/interpret_decoder.py` for top 5 raw and top 5 hybrid configs by test top-1.
- [ ] Write Phase 6 RL postmortem: RL improved raw_multi top-1 slightly but reduced
      top-5 / delta_logprob_vs_gpt2; RL stays as optional ablation, not default.

### Infrastructure
- [ ] Clean up `sae_logs/` — keep only phase4b and phase5 task4 logs.
- [ ] Verify `checkpoints/gpt2-small/sae/` contents — document or prune.

### Paper / Documentation
- [ ] Write up Phase 4r + Phase 5 findings as a coherent narrative
      (subspace steering positive result is the publishable claim).
- [ ] Add description of TopK vs ReLU tradeoffs to `overview.tex`.
- [ ] Add Phase 7 paper-aligned benchmark methodology section to docs
      (tracks A/B/C, core 4 failure families, and claim-boundary language).

---

## Previously Completed Work (Reference)

### Phase 7 v1 Protocol Gates (Track A: GPT-2) — Complete but Misleading
- [x] Gate A: `causal_auditor` AUROC on controls >= 0.85 (0.9676 — but text-dominated)
- [x] Gate B: unfaithful-control FPR <= 0.05 (0.018)
- [x] Gate C: readout_high_causal_fail_cases_n >= 1 (1648-1726 — suspiciously high)
- [x] Gate D: benchmark output includes claim-boundary disclaimer and A/B/C track metrics

### Phase 7 v1 Portability Scaffolding (Track B) — Complete
- [x] Adapter contract tests pass for gpt2-medium and qwen2.5-7b
- [x] model_registry.py lists both models with correct defaults
- [x] Backward compatibility, schema extension, and causal engine compatibility checks
- [x] Dual invocation compatibility for core CLIs
- [x] Minimal smoke tests added

### Phase 7 v1 Calibration Mini-Sweep — Complete
- [x] 4 calibration layer sets run (single_l22, spread4_current, middle12_06_17, every2_even)
- [x] Phase 6 and Phase 7 supervised train+eval for raw and hybrid (8 configs each)
- [x] Causal dry-run throughput measured (10,588 records/hour)
- [x] Calibration summary artifacts produced

### Phase 7 v1 Dataset Build — Complete
- [x] Step-trace datasets built: train (21,402 records), test (3,799 records)
- [x] Schema validated: phase7_trace_v1

### Phase 6 Broad Sweep — Complete
- [x] 102/102 configs completed, 0 failed
- [x] Best test top-1: raw_block8_00_07 = 0.5628
- [x] Best test top-5: hybrid_block4_04_07 = 0.7860

---

## Environment & Infrastructure (completed)

- [x] Python 3.12.3, CUDA 12.8, PyTorch 2.6 environment documented
- [x] `setup_env.sh` installs minimal deps and pins tokenizer artifacts
- [x] Secrets in `.env`; `weights_only=False` noted for SAE checkpoint loading
- [x] GSM8K downloaded to `datasets/raw/gsm8k/`
- [x] Activation hooks validated: <50% overhead, >1000 tokens/s

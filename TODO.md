# Active Tasks

**Last Updated:** February 26, 2026

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
| 7 | Causal CoT Verification Auditor (paper-aligned v1 implementation) | 🟡 Implemented + smoke-validated; full runs pending |

---

## Outstanding / Next Steps

### Infrastructure
- [ ] Clean up `sae_logs/` — many logs are from superseded runs; keep only
      the phase4b and phase5 task4 logs that correspond to current checkpoints.
- [ ] Verify `checkpoints/gpt2-small/sae/` contents — these standalone
      single-layer GPT-2 small checkpoints predate the multi-layer work;
      document or prune.

### Phase 6 Follow-up (analysis only; no core blockers)
- [ ] Write a short Phase 6 RL postmortem note:
      RL improved `raw_multi` top-1 slightly but reduced top-5 / `Δlogprob_vs_gpt2`;
      decide whether RL remains in the default Phase 7 instrument stack or stays an ablation.
- [ ] Decide the default Phase 7 instrument checkpoints:
      recommended baseline set is supervised `raw_multi`, supervised `hybrid_multi`, supervised `sae_multi`
      (RL checkpoints as optional comparison).

### Phase 6 + Phase 7 Full Layer Sweep (Primary active experiment)

#### 0. Pre-sweep freeze (do before launching anything)
- [ ] Freeze claim-boundary wording for all Phase 7 outputs:
      "causally supported under measured variables/subspaces and tested interventions; not a complete explanation."
- [ ] Freeze paper-core benchmark tracks for all reports:
      `text_only`, `latent_only`, `causal_auditor`.
- [ ] Freeze paper-core failure families in controls:
      `prompt_bias_rationalization`, `silent_error_correction`,
      `answer_first_order_flip`, `shortcut_rationalization`.
- [ ] Confirm the sweep manifest is the source of truth:
      `experiments/layer_sweep_manifest_v1.json` (`43` broad subsets + `16` SAE-panel subsets).
- [ ] Pick a `SWEEP_RUN_ID` naming convention (example: `20260227_layer_sweep_v1`).

#### 1. Calibration mini-sweep (required before the 2-3 day run)
Goal: estimate actual per-config runtime and causal throughput on this machine.
- [ ] Select 4 calibration layer sets from the manifest (recommended):
      `single_l22`, `spread4_current`, `middle12_06_17`, `every2_even`.
- [ ] Run Phase 6 supervised train+eval for `raw` and `hybrid` on the 4 layer sets (8 configs total).
- [ ] Run Phase 7 state-decoder train+eval for `raw` and `hybrid` on the same 4 layer sets (8 configs total).
- [ ] Run one Phase 7 causal dry run (`subresult_value`) on 1 shortlisted subset with `50` records.
- [ ] Measure and record:
      minutes/config for Phase 6, minutes/config for Phase 7, causal throughput (records x layers / hour).
- [ ] Update the full-run runtime estimate (fast / typical / conservative) from observed calibration numbers.
Artifacts to produce:
- [ ] `phase6_results/results/<SWEEP_RUN_ID>_calibration_summary.json`
- [ ] `phase7_results/results/<SWEEP_RUN_ID>_calibration_summary.json`
- [ ] `phase7_results/interventions/<SWEEP_RUN_ID>_calibration_causal_throughput.json`

#### 2. Phase 7 trace dataset build (one-time prerequisite for Phase 7 sweep)
- [ ] Build full trace datasets from Phase 6 merged datasets:
      `phase7/build_step_trace_dataset.py`
- [ ] Validate outputs exist and are non-empty:
      `phase7_results/dataset/gsm8k_step_traces_train.pt`
      `phase7_results/dataset/gsm8k_step_traces_test.pt`
- [ ] Validate schema/version and summary:
      `schema_version=phase7_trace_v1`, record counts, split counts, per-step-type counts.
- [ ] Save/inspect `phase7_results/dataset/build_summary.json`.

#### 3. Phase 6 broad sweep (final-token decoder readout mapping)
Scope:
- `raw` on all `43` manifest layer sets
- `hybrid` on all `43` manifest layer sets
- `sae` on the `16`-subset SAE panel

Run requirements:
- [ ] Use manifest-driven CLI (`--manifest ... --layer-set-id ... --input-variant ...`) for every sweep run.
- [ ] Route outputs to run-scoped directories (avoid mixing with baseline files), e.g.:
      `phase6_results/sweeps/<SWEEP_RUN_ID>/{checkpoints,results,logs,interpret}`.
- [ ] Ensure each train run produces:
      checkpoint + `supervised_*.json` with sweep metadata.
- [ ] Ensure each eval run produces:
      `eval_*.json` with val/test metrics + sweep metadata.

Phase 6 metrics to collect per config:
- [ ] `val_top1`, `val_top5`, `val_delta_logprob_vs_gpt2`
- [ ] `test_top1`, `test_top5`, `test_delta_logprob_vs_gpt2`
- [ ] `best_epoch`
- [ ] layer metadata (`layer_set_id`, family, num_layers)

Interpretability follow-up (after sweep ranking):
- [ ] Run `phase6/interpret_decoder.py` for top `5` `raw` configs by test top-1.
- [ ] Run `phase6/interpret_decoder.py` for top `5` `hybrid` configs by test top-1.
- [ ] Run `phase6/interpret_decoder.py` for all `16` SAE-panel configs (or top `8` if runtime constrained; document the decision).

Phase 6 sweep aggregation:
- [ ] Run `experiments/summarize_phase6_layer_sweep.py` on the sweep result dir.
- [ ] Verify summary includes family-level and num-layer aggregates.
Artifacts to produce:
- [ ] `phase6_results/sweeps/<SWEEP_RUN_ID>/results/layer_sweep_phase6_summary.json`

#### 4. Phase 7 broad sweep (structured latent-state readout mapping)
Scope:
- `raw` on all `43` manifest layer sets
- `hybrid` on all `43` manifest layer sets
- `sae` on the `16`-subset SAE panel

Run requirements:
- [ ] Use manifest-driven CLI for all Phase 7 train/eval runs.
- [ ] Route outputs to run-scoped directories, e.g.:
      `phase7_results/sweeps/<SWEEP_RUN_ID>/{checkpoints,results,logs,interp}`.
- [ ] Keep splits by `example_idx` (no leakage) for every config.

Phase 7 metrics to collect per config:
- [ ] `result_token_top1`, `result_token_top5`
- [ ] `operator_acc`, `step_type_acc`, `magnitude_acc`, `sign_acc`
- [ ] `delta_logprob_vs_gpt2`
- [ ] `subresult_mae`, `lhs_mae`, `rhs_mae`
- [ ] per-operator result-top1

Phase 7 additional outputs:
- [ ] Emit latent predictions for shortlisted configs (not all configs by default).
- [ ] Emit SAE gradient saliency for SAE configs (all `16` panel configs or shortlist only; document choice).

Phase 7 sweep aggregation:
- [ ] Run `experiments/summarize_phase7_layer_sweep.py` on the sweep result dir.
- [ ] Verify summary ranking uses the Phase 7 val composite (result-token top1, operator, step_type, delta logprob).
Artifacts to produce:
- [ ] `phase7_results/sweeps/<SWEEP_RUN_ID>/results/layer_sweep_phase7_summary.json`

#### 5. Deterministic causal shortlist selection (Phase 7)
Goal: choose a small causal set from sweep results before expensive patching.
- [ ] Run `experiments/select_phase7_causal_shortlist.py` using the Phase 6 and Phase 7 sweep summaries.
- [ ] Fill exactly 6 planned slots (or document any unfilled slots and why):
      1) best raw (Phase 7), 2) best hybrid (Phase 7),
      3) best middle-family subset, 4) best every2 subset,
      5) Phase6-vs-Phase7 mismatch subset,
      6) high latent-only / low causal-risk subset.
- [ ] If per-config paper-track benchmark files exist, pass them to the shortlist selector for slot #6.
- [ ] Review shortlist for duplicates / pathological choices before causal runs.
Artifacts to produce:
- [ ] `phase7_results/interventions/layer_sweep_causal_shortlist_summary.json`

#### 6. Phase 7 causal shortlist runs (subresult-first)
Goal: test whether higher readout quality corresponds to stronger causal support.

Pass 1 (required):
- [ ] For each of the `6` shortlisted layer sets, run causal checks on `subresult_value` with:
      necessity + sufficiency + specificity, probe+saliency union subspaces, off-manifold checks enabled.
- [ ] Use `100` test step records per subset (pass-1 budget).
- [ ] Layer coverage rule:
      if subset has `<=8` layers, test all subset layers;
      if subset has `>8` layers, test representative 4-layer panel (min/lower-mid/upper-mid/max).
- [ ] Record donor-match success rates and no-donor counts.
- [ ] Record off-manifold intervention rates and destructive edit indicators.

Pass 2 (conditional, if runtime permits):
- [ ] Expand top `2` most informative subsets to `300` records each.
- [ ] Keep the same variable (`subresult_value`) and layer coverage policy for comparability.

Artifacts to produce:
- [ ] `phase7_results/interventions/causal_checks_<config>_subresult_value_<layers>.json` (one per run)
- [ ] `phase7_results/interventions/layer_sweep_causal_shortlist_summary.json` (updated with runtime + pass rates)

#### 7. Audit + calibration + benchmark (paper-aligned, shortlisted configs)
Goal: measure faithfulness verification quality, not just readout quality.

Controls / parser:
- [ ] Generate paper-core control traces on the evaluation split (`phase7/generate_cot_controls.py`).
- [ ] Parse controls with order/revision metadata (`phase7/parse_cot_to_states.py`).

Per-shortlist audit workflow:
- [ ] Run `phase7/causal_audit.py` with the shortlisted config’s latent preds + causal checks.
- [ ] Calibrate thresholds with `phase7/calibrate_audit_thresholds.py`
      (keep calibration and benchmark/report splits disjoint if possible).
- [ ] Run `phase7/benchmark_faithfulness.py` and capture:
      `by_control_variant`, `by_paper_failure_family`, `by_benchmark_track`.

Required benchmark outputs:
- [ ] `Track A` (`text_only`) metrics
- [ ] `Track B` (`latent_only`) metrics
- [ ] `Track C` (`causal_auditor`) metrics
- [ ] `readout_high_causal_fail_cases_n` and example cases
- [ ] claim-boundary disclaimer preserved in result JSON

Artifacts to produce:
- [ ] `phase7_results/audits/text_causal_audit_controls_<SHORTLIST_ID>.json`
- [ ] `phase7_results/calibration/phase7_thresholds_v1_<SHORTLIST_ID>.json`
- [ ] `phase7_results/results/faithfulness_benchmark_controls_<SHORTLIST_ID>.json`

#### 8. Cross-phase analysis + interpretation (final analysis pass)
Goal: answer the main experiment questions clearly.
- [ ] Compare Phase 6 vs Phase 7 sweep results by:
      layer family, layer count, input variant (`raw`/`hybrid`/`sae`).
- [ ] Identify layer sets that maximize:
      a) readout quality (Phase 6 / Phase 7 latent-only), and
      b) causal support (Phase 7 causal-auditor outcomes).
- [ ] Explicitly report mismatches:
      high latent-only / weak causal support (core evidence for "decoded signal != faithfulness").
- [ ] Compare middle-layer subsets vs global/sparse subsets:
      `middle12_06_17`, `middle_every2_06_16`, `every2_even`, `every2_odd`, `all24`, `spread4_*`.
- [ ] Summarize what this implies for improving CoT faithfulness:
      which layers are best for readout instrumentation vs causal verification.

Deliverables:
- [ ] `phase6_results/sweeps/<SWEEP_RUN_ID>/results/layer_sweep_phase6_summary.json` (final)
- [ ] `phase7_results/sweeps/<SWEEP_RUN_ID>/results/layer_sweep_phase7_summary.json` (final)
- [ ] `phase7_results/interventions/layer_sweep_causal_shortlist_summary.json` (final)
- [ ] `phase7_results/results/faithfulness_benchmark_controls_<SHORTLIST_ID>.json` (final selected benchmark)
- [ ] One final write-up note (markdown) summarizing:
      "where decodable information lives" vs "where causal support is strongest".

#### 9. Execution management / reproducibility requirements (apply to all sweep stages)
- [ ] Use run-scoped directories for all sweep outputs; do not overwrite baseline Phase 6/7 files.
- [ ] Record `SWEEP_RUN_ID`, seed, manifest path, and `layer_set_id` in every result file.
- [ ] Log failed runs separately and retry deterministically (same args, same seed).
- [ ] Save intermediate summaries after each stage so the pipeline can resume.
- [ ] Keep a small `run_state/` marker folder for long tmux jobs (stage completion markers).

### Potential Phase 7 / CoT-faithfulness work (after layer sweep)
If extending beyond the layer sweep, the next step is **using the auditor to improve CoT outputs**:
- [ ] Use `causal_auditor` scores for reranking candidate CoTs
- [ ] Filter training data by causal-support score
- [ ] Explore reward shaping with bounded claims (auditor as noisy faithfulness signal)
- [ ] Expand from arithmetic controls to real CoT traces on a CoT-capable open-weight model (v2 track)

### Paper / Documentation
- [ ] Write up Phase 4r + Phase 5 findings as a coherent narrative
      (subspace steering positive result is the publishable claim).
- [ ] Add description of TopK vs ReLU tradeoffs to `overview.tex`.
- [ ] Add a Phase 6 summary section to `PROJECT_STATUS.md` / `README.md`
      (supervised + RL follow-up results, and what they imply for Phase 7).
- [ ] Add a Phase 7 paper-aligned benchmark methodology section to docs
      (tracks A/B/C, core 4 failure families, and claim-boundary language).

---

## Environment & Infrastructure (completed)

- [x] Python 3.12.3, CUDA 12.8, PyTorch 2.6 environment documented
- [x] `setup_env.sh` installs minimal deps and pins tokenizer artifacts
- [x] Secrets in `.env`; `weights_only=False` noted for SAE checkpoint loading
- [x] GSM8K downloaded to `datasets/raw/gsm8k/`
- [x] Activation hooks validated: <50% overhead, >1000 tokens/s

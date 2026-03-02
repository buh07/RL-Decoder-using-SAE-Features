# Phase 7 Objective Review + Layer Sweep Runtime (v1)

## Objective Alignment (CoT Faithfulness)

### Verdict

The current Phase 7 pipeline is logically aligned with the objective of improving chain-of-thought faithfulness **as a verifier/benchmarking system first**, not yet as a direct optimization method.

### Why it aligns

- The state decoder is treated as a readout instrument, not the final explanation (`phase7/state_decoder_core.py`, `phase7/causal_audit.py`).
- Final verdicts incorporate necessity/sufficiency/specificity intervention checks (`phase7/causal_intervention_engine.py`).
- Parser failures are separated from contradiction via `unverifiable_text` semantics (`phase7/parse_cot_to_states.py`, `phase7/causal_audit.py`).
- Paper-aligned benchmark tracks (`text_only`, `latent_only`, `causal_auditor`) and failure-family controls are implemented.

### Scope limits (must be kept explicit)

- v1 is primarily synthetic-control calibration, not a real natural-language CoT faithfulness benchmark.
- Multi-op arithmetic states are compressed to annotation-level summaries (final left-associative reduction step).
- Current causal effect measurements are mainly on final-token logprob, not a full mediation analysis over latent trajectories.
- Thresholds should be calibrated and reported on disjoint splits in the full sweep run.

### Claims boundary

Use this wording in reports:

> Causally supported under measured variables/subspaces and tested interventions; not a complete explanation of all internal reasoning.

## Layer Sweep Design (Implemented Infra)

The sweep infrastructure is manifest-driven via:

- `experiments/layer_sweep_manifest.py`
- `experiments/build_layer_sweep_manifest.py`
- generated manifest: `experiments/layer_sweep_manifest_v1.json`

Sweep support added to:

- Phase 6: `phase6/train_supervised.py`, `phase6/evaluate_decoder.py`, `phase6/interpret_decoder.py`
- Phase 7: `phase7/train_state_decoders.py`, `phase7/evaluate_state_decoders.py`

Summaries / shortlist tooling:

- `experiments/summarize_phase6_layer_sweep.py`
- `experiments/summarize_phase7_layer_sweep.py`
- `experiments/select_phase7_causal_shortlist.py`

## Observed Phase 6 Sweep Outcomes (Completed)

Completed run:
- `phase6_results/sweeps/20260226_164900_phase6_full_sweep/phase6_full/`
- completion: `102/102` configs, `0` failures

Best test metrics:
- Best top-1: `raw_block8_00_07` = `0.5628`
- Best top-5: `hybrid_block4_04_07` = `0.7860`
- Best `delta_logprob_vs_gpt2`: `raw_block4_04_07` = `+2.5717`

Improvement vs `spread4_current` baseline:
- raw: `+5.53 pp`
- hybrid: `+5.00 pp`
- sae: `+5.98 pp`

Implication for Phase 7 shortlist priors:
- prioritize `raw_block8_00_07`
- prioritize `raw_block4_04_07`
- prioritize `hybrid_block4_04_07`
- prioritize `hybrid_block8_00_07`
- keep `sae_block8_00_07` as interpretability comparator

## Remaining Runtime Estimate (Phase 7 Only, 2 GPUs: 0 and 1)

Grounding observations from this repo:

- Phase 7 calibration sweep completed:
  - `phase7_results/results/20260226_121004_layer_sweep_calib_calibration_summary.json`
  - `phase7_results/interventions/20260226_121004_layer_sweep_calib_calibration_causal_throughput.json`

Remaining full layer-sweep work:

- Phase 7 state-decoder sweep: `12–20h`
- Phase 7 causal shortlist runs (`subresult_value` first): `8–20h`
- Audit + calibration + paper-aligned benchmark: `2–6h`

### Total remaining wall-time estimate (2 GPUs)

- Fast path: `~22h`
- Typical path: `~28–38h`
- Conservative path: `~46h`

## Calibration Note

The calibration step is already complete; use the two calibration artifacts above as the runtime baseline for scheduling the remaining Phase 7 work.

## Pivot Acceptance Criteria (Decision Gate)

Before any downstream "improve CoT" use (reranking/filtering/reward shaping), require:

- `causal_auditor` AUROC on controls `>= 0.85`
- unfaithful-control FPR `<= 0.05`
- at least one explicit high-readout / low-causal-support case
- benchmark output includes A/B/C track metrics and claim-boundary disclaimer

Actionable evaluation steps:
1. Generate controls with paper-core variants (`phase7/generate_cot_controls.py`).
2. Parse and align controls with order/revision metadata (`phase7/parse_cot_to_states.py`).
3. Run causal checks on shortlist (necessity/sufficiency/specificity, off-manifold checks enabled).
4. Run `phase7/causal_audit.py`.
5. Run `phase7/calibrate_audit_thresholds.py`.
6. Run `phase7/benchmark_faithfulness.py`.

Interpretation rule:
- If readout is strong but causal gates fail, do not claim faithful CoT.
- If gates pass, claims remain bounded to measured variables/subspaces and tested interventions.

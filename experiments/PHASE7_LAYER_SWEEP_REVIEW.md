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

## Runtime Estimate (2 GPUs: 0 and 1)

Grounding observations from this repo:

- Phase 6 supervised benchmark (dataset + 6 configs + eval): ~54 min
- Phase 6 RL follow-up (2 configs + eval + interpret): ~11 min

Planned full layer-sweep experiment (Phase 6 + Phase 7, causal shortlist only):

- Sweep infra + manifest + aggregation implementation: `4–8h` engineering time (CPU)
- Phase 7 trace dataset build (one-time): `20–45m`
- Phase 6 broad sweep: `10–16h`
- Phase 7 state-decoder sweep: `12–20h`
- Phase 7 causal shortlist runs (`subresult_value` first): `8–20h`
- Audit + calibration + paper-aligned benchmark: `2–6h`

### Total wall-time estimate (2 GPUs)

- Fast path: `~32h`
- Typical path: `~40–52h`
- Conservative path: `~60h`

## Required Calibration Step Before Full Sweep

Run a mini calibration first:

- 4 layer subsets
- raw + hybrid only
- 1 Phase 7 causal shortlist dry run (50 records)

Use it to update:

- average minutes/config for Phase 6 + Phase 7 training/eval
- causal patching throughput (records x layers / hour)

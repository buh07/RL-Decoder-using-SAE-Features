# Phase 7 Layer-Sweep Review (GPT-2 Closure + Qwen Inquiry Boundary)

**Date:** March 6, 2026  
**Model scope for closure:** `gpt2-medium`  
**Status:** GPT-2 Phase 7 closed; Track C negative disposition

## Executive Verdict
Most historical criticisms in this review were valid during active development. They are now resolved in one of two ways:
1. **Engineering resolved:** parser/anchor/split/coverage/indexing/contract defects were fixed and validated.
2. **Scientific closure:** Track C remained non-deployment-relevant after corrected experiments, so GPT-2 closes on the negative branch.

This file is now a closure document for GPT-2, not an open run tracker.

## Final GPT-2 Outcomes
1. Arithmetic features are decodable and causally active.
- Phase 4/4r remained positive on representation/probing and causal subspace effects.

2. Track C does not provide robust CoT-faithfulness discrimination for GPT-2.
- Across intervention and reformulation passes, Track C was not closure-grade for deployment.

3. Best deployable Phase 7 configuration for GPT-2 is two-track.
- `text=0.50`, `latent=0.50`, structural penalties on.
- Fair profile comparison (Option B) produced composite AUROC around `0.836-0.838`.
- This is useful detection, but not a mechanistic faithfulness claim.

## March 7 Closure Stress Tests (Still Negative For Track C)
1. Path B (feature-set swaps in trajectory metrics):
- Best AUROC remained below robust target (`best ~0.683`).
- `wrong_intermediate` stayed weak across feature sets.

2. Path C robust variant-exclusion validation:
- `wrong_intermediate` single split: `0.694` (CI includes `>=0.70` but point estimate below gate).
- 5-fold grouped CV pooled: `0.658` with CI upper near threshold.
- Outcome: underpowered borderline signal, not closure-grade for positive Track C claim.

3. Mixed hidden+SAE ladder:
- Multi-seed sweeps did not meet practical/stability criteria.
- Final decision stayed `mixed_redundant_or_insufficient`.

Net: these runs support GPT-2 closure and do not reopen Track C.

## Canonical Closure Artifacts
- `phase7_results/results/trackc_phase7v3_closure_note_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/trackc_phase7v3_decision_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/optionb_canary_decision_phase7_optionbc_20260306_092554_phase7_optionbc.json`
- `phase7_results/results/optionc_probe_phase7_optionbc_20260306_092554_phase7_optionbc.json`
- `phase7_results/results/optionbc_final_phase7_optionbc_20260306_092554_phase7_optionbc.json`

## Criticism Disposition (GPT-2)
- "Track C near chance / anti-predictive": **resolved by negative-result closure**.
- "Coverage too low": **operationally addressed**, but higher coverage did not produce a passing Track C conclusion for GPT-2.
- "Track C hurts composite": **accepted and handled** by final GPT-2 two-track weights.
- "Active run not complete": **obsolete** for GPT-2 closure.

## Claim Boundary
What can be claimed for GPT-2:
- Arithmetic mechanism signals exist.
- Those signals do not align with CoT faithfulness labels strongly enough for Track C deployment.
- Two-track text+latent with penalties is the final GPT-2 Phase 7 output.

What cannot be claimed from GPT-2 closure:
- Cross-model faithfulness conclusions.
- Universal ineffectiveness of causal approaches.

## Open Work (Not GPT-2 Reopening)
Qwen is an independent follow-up hypothesis test and is not implied by GPT-2 closure.

Next required entry point:
- `Q0` diagnostic-first viability check (small sample, control-conditioned hidden-state separation at `=` anchor).
- Full Qwen Stage 3 proceeds only if `Q0` passes predefined go/no-go criteria.

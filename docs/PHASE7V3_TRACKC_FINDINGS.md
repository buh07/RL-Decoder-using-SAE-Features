# Phase 7v3 Track C Findings (GPT-2 Medium Closure)

## Scope And Final Status
- Model scope: `gpt2-medium` only.
- Protocol scope: Phase 7v3 Track C reformulations (`R1` confidence, `R2` trajectory, `R3` contrastive probe, `R4` geometry).
- Final disposition for GPT-2: **Track C closed as a negative result for CoT-faithfulness discrimination**.
- Shipping configuration for GPT-2 Phase 7: **two-track composite** with structural penalties.
  - Top-level weights: `text=0.50`, `latent=0.50`, `confidence=0.0`, `causal=0.0`.

## Canonical Evidence Artifacts
- `phase7_results/results/trackc_phase7v3_closure_note_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/trackc_phase7v3_decision_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/trackc_r1_benchmark_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/trackc_r2_benchmark_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/trackc_r3_probe_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/trackc_r4_geometry_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/optionb_canary_decision_phase7_optionbc_20260306_092554_phase7_optionbc.json`
- `phase7_results/results/optionc_probe_phase7_optionbc_20260306_092554_phase7_optionbc.json`
- `phase7_results/results/optionbc_final_phase7_optionbc_20260306_092554_phase7_optionbc.json`
- `phase7_results/results/phase7_sae_trajectory_pathb_20260306_234123_phase7_sae_trajectory_pathb.json`
- `phase7_results/results/phase7_sae_trajectory_pathc_20260306_235018_phase7_sae_trajectory_pathc.json`
- `phase7_results/results/phase7_sae_trajectory_pathc_robust_20260307_001237_phase7_sae_trajectory_pathc_robust.json`
- `phase7_results/results/phase7_mixed_trajectory_validation_phase7_mixed_trajectory_20260307_012248_phase7_mixed_trajectory_validation.json`
- `phase7_results/results/mixed_gain_sweep_summary.json`
- `phase7_results/results/mixed_result_top50_seed_sweep_summary.json`

## R5.1 Decision Matrix Outcome
Final closure branch used for GPT-2:
- `R1`: did not produce closure-grade deployment evidence after fair profile comparison.
- `R2`: failed (`trajectory_coherence` anti-discriminative on canary).
- `R3`: failed to produce a robust learned Track C signal under corrected split policy.
- `R4`: exploratory geometry remained weak.

Outcome: **all-fail branch for Track C deployment relevance on GPT-2**.

## Post-Closure Validation Passes (March 7, 2026)
These follow-up runs were designed to check whether GPT-2 closure could be reversed under stronger feature engineering. It was not.

1. Path B (feature-set swap in SAE trajectory metrics)
- Best overall unfaithful-positive AUROC was `0.6831` (`eq_pre_result_150`, `layer7:feature_variance_coherence`).
- `wrong_intermediate` remained weak (`0.5716` best under this pass), so signal did not meet robustness targets.

2. Path C (trajectory probe with robust variant exclusion)
- Naive split run showed `wrong_intermediate AUROC = 0.8025` (small test support).
- Robust policy run (exclude structural variants in train positives) dropped to `0.6944`.
- Bootstrap CI on wrong_intermediate (1000 resamples): `[0.5864, 0.8025]`.
- 5-fold grouped CV pooled wrong_intermediate AUROC: `0.6577`, CI `[0.6104, 0.7057]`.
- Interpretation: single-split signal is high-variance and does not hold robustly at the target threshold.

3. Mixed hidden-state + SAE ladder
- Baseline feature family sweep (`eq_pre_result_150`) produced mean CV delta `M3-M1 = +0.0008` across seeds.
- Alternate `result_top50` feature set improved mean delta (`+0.0658`) but still did not cross the primary threshold (`M3 mean = 0.6369`, no run > `0.70`).
- Final mixed-model decision remained `mixed_redundant_or_insufficient`.

These additional checks strengthen the GPT-2 closure rather than reopening it.

## Three Publishable GPT-2 Results
1. Arithmetic features exist and are causally active.
- Phase 4/4r evidence shows strong decodability and positive subspace intervention effects.
- Representative anchors: ridge probe near `R^2=0.977` and positive causal `delta_logprob` at upper layers.

2. Those features do not encode CoT faithfulness under this protocol.
- Multiple Track C reformulations failed to produce robust, gate-worthy faithful/unfaithful discrimination for GPT-2.
- Interpretation: arithmetic mechanism signal and CoT faithfulness signal are separable in this model.

3. Text-plus-structure auditing remains practically useful but non-mechanistic.
- Fair Option B profile comparison yielded composite AUROC around `0.836-0.838` with structural penalties.
- This is useful for detection, but does not justify mechanistic faithfulness claims.

## Supersession Note
Earlier v3 canary text that temporarily selected `confidence_margin` as Track C is **superseded** by Option B and Option C closure evidence for GPT-2.

## Claim Boundaries
What this supports:
- Mechanistic arithmetic representations are real in GPT-2.
- CoT text faithfulness is not captured by current Track C formulations for GPT-2.
- A two-track text+latent auditor with penalties is the final GPT-2 deployment choice for Phase 7.

What this does not support:
- Cross-model generalization claims.
- Claims that GPT-2 CoT is mechanistically faithful.
- Using Track C as a positive gate for GPT-2 in this protocol.

## Next-Step Boundary
Qwen work is a separate hypothesis test and does not reopen GPT-2 closure by default. See `TODO.md` for the diagnostic-first Qwen `Q0` go/no-go path.

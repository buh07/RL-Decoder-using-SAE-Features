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

## R5.1 Decision Matrix Outcome
Final closure branch used for GPT-2:
- `R1`: did not produce closure-grade deployment evidence after fair profile comparison.
- `R2`: failed (`trajectory_coherence` anti-discriminative on canary).
- `R3`: failed to produce a robust learned Track C signal under corrected split policy.
- `R4`: exploratory geometry remained weak.

Outcome: **all-fail branch for Track C deployment relevance on GPT-2**.

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

# Track C (Causal Channel): Interpretation and Action Plan

## What Track C Currently Measures
Track C currently scores whether intervention on selected latent subspaces changes model behavior in the expected direction (necessity/sufficiency/specificity/mediation style checks).

That is a measure of **causal leverage in the model's computation**, not directly a measure of whether the written CoT is faithful to that computation.

## Why Anti-Predictive Behavior Can Happen
The latest canary shows causal AUROC far below chance while leakage and coverage gates pass. This can happen if:

1. Faithful and unfaithful text variants still rely on similar internal arithmetic mechanisms.
2. The manipulated subspace is a good lever for output computation, but not aligned with label semantics for textual faithfulness.
3. A higher causal score then correlates with the opposite class under current labeling/calibration, producing anti-predictive behavior.

In short: Track C may be detecting "how much this latent matters to output," not "whether the explanation text matches the latent trajectory."

## Is Causal the Wrong Lever Metric?
Not necessarily. Causal intervention is still a valid tool, but **raw absolute causal score** appears to be the wrong target for this faithfulness label in the current setup.

A better framing is:

- Keep causal interventions as primitive measurements.
- Replace or transform the Track C aggregate so it is aligned to faithfulness semantics (e.g., within-trace contrasts/orientation-aware transforms).

## Prioritized Next Experiments (Synthetic-First Canary)

### E1. Causal orientation test (raw vs sign-inverted)
Run calibration-split-only comparisons between raw Track C score and sign-inverted Track C score.

Decision rule:
- If inversion consistently improves AUROC and recall at fixed FPR, mark current orientation as misaligned.

### E2. Within-trace pairwise delta metric
Use pairwise score deltas per trace (`faithful_score - variant_score`) instead of absolute Track C score.

Rationale:
- Removes trace-level baseline confounds and tests whether Track C discriminates variants of the same underlying problem.

### E3. Mediation-variable alignment by manipulated variable
Do not use one fixed mediation target for all variants. Align mediation target to the manipulated variable (`operator`, `sign`, `magnitude_bucket`, `subresult_value`).

Success criterion:
- Causal AUROC improves without violating leakage/coverage constraints.

### E4. Token-anchor ablation for Track C only
Run `template_first` vs `equation_first` anchoring with all else fixed.

Use deterministic selection:
1. Higher causal AUROC
2. Tie-break by higher causal coverage
3. Tie-break by higher composite AUROC

### E5. Composite policy ablation (diagnostic)
Compare baseline blend (`text=0.35, latent=0.35, causal=0.30`) against causal-off diagnostic blend (`text=0.5, latent=0.5, causal=0.0`).

Interpretation:
- If causal-off is consistently better, Track C should remain diagnostic-only until reworked.

## Reporting and Claim Boundary
Dual gate remains unchanged for official claims.

Until Track C clears its causal floor in canary and remains stable under the above ablations, report:

- Track C as **non-passing diagnostic channel**.
- Main finding as a **negative but meaningful result**: decodable arithmetic features and causal subspace leverage do not yet yield a faithful textual-process discriminator in this model/configuration.

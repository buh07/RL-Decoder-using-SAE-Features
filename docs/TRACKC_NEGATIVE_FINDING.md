# Track C Negative Finding (Final GPT-2 Medium Closure)

## Final Statement
For `gpt2-medium` arithmetic CoT under the Phase 7 protocol, Track C is a **negative result** for deployment relevance.

Track C-style mechanistic signals (intervention, confidence reformulation, trajectory coherence, contrastive probe, geometry) did not produce a robust, gate-worthy faithful/unfaithful discriminator after correction passes and fair canary comparisons.

Post-closure stress tests on March 7, 2026 did not overturn this:
- Path B feature swaps peaked below robust deployment thresholds (`best AUROC ~0.683`).
- Path C robust wrong-intermediate validation failed the `>0.70` gate (`single split 0.694`, pooled CV `0.658`).
- Mixed hidden+SAE ladder remained insufficient (`mixed_redundant_or_insufficient`).

## What This Is
- Evidence that GPT-2 can carry arithmetic mechanism signal without that signal being a reliable proxy for CoT faithfulness.
- Evidence supporting deployment of a two-track auditor (`text` + `latent`) with structural penalties for GPT-2 in this project scope.
- A publishable mechanistic finding about the readout-versus-faithfulness gap.

## What This Is Not
- Not evidence that causal analysis is universally unhelpful.
- Not evidence about larger CoT-trained models.
- Not a claim that faithfulness is impossible to assess mechanistically in general.

## Shipping Consequence (GPT-2)
- Final Phase 7 GPT-2 config: two-track composite with structural penalties.
- Track C remains documented as unresolved/negative for GPT-2 under current protocol and is excluded from production-weight defaults.

## Forward-Looking Boundary
Any attempt to revive Track C claims must occur on a separate model inquiry path and must satisfy new evidence gates before promotion.
Current next-model path is **Qwen Option C** (behavioral contradiction labeling + internal consistency + lexical confound control). The original trajectory-only approach (Path A→B→C) and the legacy Q0 raw hidden-state L2 diagnostic are both obsolete.

Qwen Option C current status (March 9, 2026):
- Arithmetic: PASS (CV AUROC 0.877, stress p=0.001)
- Canonical G2 lineage (`20260309_155350_phase7_g2_feature_prune_stage1`):
  - PrOntoQA: PASS (CV AUROC 0.964, stress pooled 0.961)
  - EntailmentBank: PASS (CV AUROC 0.999, stress pooled 0.999)
- Full G2 stress validation:
  - PrOntoQA: PASS
  - EntailmentBank: PASS
- Cross-domain `publishable_cross_domain` gate: met for tested Option C protocol (`publishable_cross_domain_pass=true`)

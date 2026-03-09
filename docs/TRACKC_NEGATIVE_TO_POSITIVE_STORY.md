# Track C Negative-to-Positive Story

## Scope
This memo records the method transition from early coherence-confounded Track C results to the current Option C arithmetic baseline with explicit lexical-confound control.

## 1) Negative Phase: Why Earlier Results Were Not Faithfulness
Early Phase 7 trajectories looked strong on synthetic and template-modified controls, but they did not prove faithfulness.

Key failure mode:
- The detector often separated edited text from unedited text, including logically valid lexical edits.
- That means high AUROC could be explained by text-difference sensitivity, not mismatch between stated reasoning and internal computation.

Implication:
- These runs were valid as **reasoning-coherence / text-anomaly detection**, not as mechanistic faithfulness claims.

## 2) Correction: Option C Design
Option C fixed the confound by combining two pieces:
1. **Behavioral contradiction labeling (Option B):** paired inverse/equivalent prompts create member-level correctness labels from model behavior.
2. **Internal consistency detection (Option A):** score generation-time internal trajectory consistency, then test whether it predicts the unfaithful member.

Critical controls retained:
- model-generated CoT capture
- lexical-control variant reporting
- pair/trace-disjoint splits
- grouped CV and bootstrap uncertainty
- leakage diagnostics

## 3) Arithmetic Baseline Result (Qwen)
Canonical run:
- `20260309_084825_phase7_optionc_generated_qwen_arith`

Full-run outcomes:
- pooled CV primary AUROC: `0.8771`
- lexical-control AUROC: `0.4535`
- wrong-minus-lexical delta: `0.4236`
- strict gate: `pass`
- claim artifact: `faithfulness_claim_enabled=true` for arithmetic scope

Interpretation:
- The current Option C stack detects behavior-linked unfaithfulness in arithmetic under lexical-confound control.
- This is stronger evidence than prior coherence-only lineages.

## 4) Stress Policy
Stress interpretation is dual-reported:
- **Primary:** empirical permutation p-value (`p < 0.01`)
- **Diagnostic:** legacy strict permutation-tail thresholds (`mean/p95/max`)

Rationale:
- p-value is the primary significance decision.
- strict-tail metrics are kept for historical comparability, not as the sole blocker.

## 5) G2 Cross-Task Validation (March 9, 2026)

This phase has two lineages and both are important:

1. Historical ablation lineage (pre-fix):
- `20260309_124650_phase7_g2_cross_task_gpu135`
- PrOntoQA failed narrowly; EntailmentBank passed.
- This run isolated the decoder-mismatch problem.

2. Canonical stress-validated lineage:
- `20260309_155350_phase7_g2_feature_prune_stage1`
- Both domains pass strict full-eval gates and full stress gates.

Strict full-eval results (canonical lineage):

| Domain | CV AUROC | Lexical AUROC | Delta | Strict Eval Gate |
|---|---|---|---|---|
| PrOntoQA | 0.964 | 0.498 | 0.466 | **PASS** |
| EntailmentBank | 0.999 | 0.337 | 0.662 | **PASS** |

Cross-domain eval decision: pass.

## 6) Stress Validation Outcome

Full stress was run on both canonical full-domain artifacts:
- permutation (1000),
- regularization sweep (`0.0001, 0.01, 0.1, 1.0`),
- multiseed (`20260307..20260316`).

Results:

| Domain | Primary Stress Verdict | p-value | Regularization | Multiseed | Stress pooled AUROC |
|---|---|---:|---|---|
| PrOntoQA | **PASS** | 0.000999 | pass | pass | 0.9606 |
| EntailmentBank | **PASS** | 0.000999 | pass | pass | 0.9994 |

Interpretation:
- The domain-decoder fix resolved the original PrOntoQA eval failure.
- The stress/eval parity fix (mixed 40 SAE + 5 decoder) resolved the remaining stress mismatch.
- Cross-domain stress validation is now complete under unchanged gates.

## 7) Current Claim Boundary

Supported:
- Arithmetic Option C faithfulness claim (stress-validated).
- PrOntoQA Option C faithfulness claim (eval + stress pass).
- EntailmentBank Option C faithfulness claim (eval + stress pass).
- `publishable_cross_domain` claim for tested Option C protocol.
- Decoder mismatch diagnosis and remediation validated by ablation and stress/eval parity recovery.

## 8) Research Narrative

The paper arc is now:
- early strong-looking results were partially confounded,
- lexical control made that confound measurable,
- contradictory-pair labeling + internal consistency fixed the confound pathway,
- domain-decoder mismatch was identified as an ablation failure mode,
- domain-decoder fix restored PrOntoQA full-eval performance,
- stress/eval feature parity fix restored PrOntoQA stress stability,
- final cross-domain stress gate is now passed under unchanged protocol rigor.

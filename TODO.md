# Active Tasks

**Last Updated:** March 9, 2026

## Project State (Locked)
- GPT-2 Phase 7 is closed with a negative Track C result for mechanistic faithfulness.
- Qwen is the active model for Track C faithfulness validation.
- Arithmetic Option C is now the baseline evidence source.
- Cross-domain G2 publishability gate is now **passed** in canonical stress-validated lineage.

## Confirmed Baseline: Arithmetic Option C (Completed)
Run lineage:
- `20260309_084825_phase7_optionc_generated_qwen_arith`

Core results (full run):
- `cv_primary_pooled_auroc = 0.8771`
- `single_split primary_member_auroc = 0.8703`
- `lexical_probe_auroc = 0.4535`
- `wrong_minus_lexical_delta = 0.4236`
- `strict_gate_pass = true`
- `faithfulness_claim_enabled = true`

Canonical artifacts:
- `phase7_results/results/optionc_summary_20260309_084825_phase7_optionc_generated_qwen_arith.json`
- `phase7_results/results/optionc_eval_20260309_084825_phase7_optionc_generated_qwen_arith_full.json`
- `phase7_results/results/optionc_claim_boundary_20260309_084825_phase7_optionc_generated_qwen_arith_full.json`

## Arithmetic Stress (Current)
Goal: validate the full 2,400-pair arithmetic result with strict stress diagnostics.

Status:
- [x] Option C-native stress harness implemented (`phase7/stress_test_optionc_probe.py`)
- [x] Stress tmux runner implemented (`experiments/run_optionc_stress.sh`)
- [x] p-value-primary + legacy strict dual verdict support implemented
- [x] Full stress run complete (`20260309_130420_optionc_stress_full_rigor`)
- [x] Arithmetic significance reframe artifact written

Outputs to finalize:
- `phase7_results/results/optionc_stress_20260309_130420_optionc_stress_full_rigor.json`
- `phase7_results/results/optionc_stress_20260309_130420_optionc_stress_full_rigor.md`
- `phase7_results/results/trackc_arithmetic_significance_reframe_20260309_130420_optionc_stress_full_rigor.json`

Decision policy (locked):
- Primary significance: empirical permutation p-value (`< 0.01`)
- Legacy strict tail metrics retained for comparability only

Stress outcome (`20260309_130420_optionc_stress_full_rigor`):
- `final_verdict_primary = pass`
- `final_verdict_legacy = fail` (diagnostic-only)
- observed primary-member AUROC: `0.9054`
- empirical p-value: `0.000999`
- regularization stability: `pass`
- multiseed stability: `pass` (mean `0.8790`, std `0.0239`)

## Option A / Option B Arithmetic Status
- [x] **Option A (internal consistency)** completed for arithmetic Option C lineage.
- [x] **Option B (behavioral contradiction baseline)** completed for arithmetic Option C lineage.
- [x] Arithmetic claim boundary updated to full Option C artifact path.

## Phase G2 Cross-Task Validation (Canonical Stress-Validated Lineage)
Domain order (locked):
1. PrOntoQA
2. EntailmentBank

Canonical lineage:
- `20260309_155350_phase7_g2_feature_prune_stage1`

Strict eval requirement (unchanged):
- pooled CV AUROC > 0.70
- CI lower >= 0.65
- lexical control AUROC <= 0.60
- wrong-minus-lexical delta >= 0.10
- zero leakage overlap

Implementation status:
- [x] Domain-aware decoder training/eval path added
- [x] Domain-mismatch guard added in Option C evaluator
- [x] Stress/eval feature parity fix added (shared feature assembly path)
- [x] PrOntoQA full run passed strict eval gate
- [x] EntailmentBank full run passed strict eval gate
- [x] Cross-domain eval decision artifact emitted

Canonical eval outputs:
- `phase7_results/results/optionc_summary_20260309_155350_phase7_g2_feature_prune_stage1_prontoqa.json`
- `phase7_results/results/optionc_eval_20260309_155350_phase7_g2_feature_prune_stage1_prontoqa_full.json`
- `phase7_results/results/optionc_summary_20260309_155350_phase7_g2_feature_prune_stage1_entailmentbank.json`
- `phase7_results/results/optionc_eval_20260309_155350_phase7_g2_feature_prune_stage1_entailmentbank_full.json`
- `phase7_results/results/trackc_g2_cross_task_decision_20260309_155350_phase7_g2_feature_prune_stage1.json`

Eval result summary:
- PrOntoQA: strict eval `pass` (`cv_primary_pooled_auroc=0.9637`, lexical `0.4980`, delta `0.4657`)
- EntailmentBank: strict eval `pass` (`cv_primary_pooled_auroc=0.9991`, lexical `0.3374`, delta `0.6617`)
- Cross-domain eval decision: `publishable_cross_domain_pass=true`

## G2 Full Stress Validation (Final Gate)
Status:
- [x] Full stress run complete for PrOntoQA
- [x] Full stress run complete for EntailmentBank
- [x] Stress-validated cross-domain decision emitted

Stress outputs:
- `phase7_results/results/optionc_stress_20260309_stress_featureprune_prontoqa_mixedfix.json`
- `phase7_results/results/optionc_stress_20260309_stress_featureprune_entailmentbank_mixedfix.json`
- `phase7_results/results/trackc_g2_cross_task_decision_20260309_155350_phase7_g2_feature_prune_stage1_stress_validated.json`
- `phase7_results/results/optionc_stress_comparison_20260309_featureprune_mixedfix.json`

Stress result summary:
- PrOntoQA: `final_verdict_primary=pass` (all stress components pass)
  - permutation p-value pass (`0.000999`)
  - regularization pass (`true`)
  - multiseed pass (mean `0.9608`, std `0.0064`, pooled `0.9606`)
- EntailmentBank: `final_verdict_primary=pass` (all stress components pass)
- Stress-validated cross-domain decision: `publishable_cross_domain_pass=true`

## Next Steps (Current)
1. Freeze `20260309_155350_phase7_g2_feature_prune_stage1` as canonical G2 lineage in all docs and status pages.
2. Keep `20260309_141106...` and `20260309_124650...` as historical ablation evidence only.
3. Prepare paper package around cross-domain Option C result and decoder/feature-parity ablation.

## Claim Boundary (Current)
- Arithmetic Option C: faithfulness claim enabled (stress-validated).
- PrOntoQA Option C: faithfulness claim enabled (eval + stress pass).
- EntailmentBank Option C: faithfulness claim enabled (eval + stress pass).
- Cross-domain `publishable_cross_domain`: enabled for tested Option C protocol.

## Narrative and Documentation Tasks
- [x] Add canonical memo:
  - `docs/TRACKC_NEGATIVE_TO_POSITIVE_STORY.md`
- [x] Update references/index:
  - `docs/DOCUMENTATION_INDEX.md`
- [x] Update status summary:
  - `PROJECT_STATUS.md`
- [x] Update GPT-2/Qwen claim framing references:
  - `docs/PHASE7V3_TRACKC_FINDINGS.md`

Narrative target:
- negative (coherence-confounded lineages) -> positive (Option C with lexical control + model-generated CoT)
- arithmetic is supportive and technically validated
- publishability achieved after passing the full cross-domain stress gate in canonical lineage

## Deprecated / Obsolete
- [x] Legacy Qwen `Q0` raw hidden-state L2 diagnostic is obsolete.
- [x] Synthetic-only coherence claim from early PrOntoQA runs is historical, not canonical.
- [x] Arithmetic-only ceiling claims from older trajectory-only runs are superseded by Option C generated-pair baseline.

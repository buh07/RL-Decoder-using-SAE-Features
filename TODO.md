# Active Tasks

**Last Updated:** March 9, 2026

## Project State (Locked)
- GPT-2 Phase 7 is closed with a negative Track C result for mechanistic faithfulness.
- Qwen is the active model for Track C faithfulness validation.
- Arithmetic Option C is now the baseline evidence source.
- Publishability gate is **cross-domain G2** (PrOntoQA + EntailmentBank), both must pass.

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

## Phase G2 Cross-Task Validation (Current Publishability Gate)
Domain order (locked):
1. PrOntoQA
2. EntailmentBank

Pass requirement (strict):
- Both domains must pass:
  - pooled CV AUROC > 0.70
  - CI lower >= 0.65
  - lexical control AUROC <= 0.60
  - wrong-minus-lexical delta >= 0.10
  - zero leakage overlap

Execution policy:
- Canary 200 pairs -> full 1000 pairs per domain
- Feature extraction/scoring on available free 3-GPU set (executed on 1/3/5 for this run)
- CV/bootstrap/permutation stress on CPU workers
- Active lineage:
  - superseded queued run: `20260309_124000_phase7_g2_cross_task` (cancelled)
  - completed run: `20260309_124650_phase7_g2_cross_task_gpu135`

Implementation status:
- [x] Domain switch support added to Option C pair builder (`--domain`)
- [x] G2 orchestrator added (`experiments/run_phase7_g2_cross_task.sh`)
- [x] Run PrOntoQA domain Option C (canary->full)
- [x] Run EntailmentBank domain Option C (canary->full)
- [x] Emit cross-domain decision artifact

Planned outputs:
- `phase7_results/results/optionc_summary_20260309_124650_phase7_g2_cross_task_gpu135_prontoqa.json`
- `phase7_results/results/optionc_claim_boundary_20260309_124650_phase7_g2_cross_task_gpu135_prontoqa_full.json`
- `phase7_results/results/optionc_summary_20260309_124650_phase7_g2_cross_task_gpu135_entailmentbank.json`
- `phase7_results/results/optionc_claim_boundary_20260309_124650_phase7_g2_cross_task_gpu135_entailmentbank_full.json`
- `phase7_results/results/trackc_g2_cross_task_decision_20260309_124650_phase7_g2_cross_task_gpu135.json`

Cross-task result:
- `prontoqa`: strict gate `fail` (`cv_primary_pooled_auroc=0.6757`)
  - CI95: [0.643, 0.707] — upper bound touches 0.70, gap is only 0.024
  - lexical control AUROC: 0.519 (chance) — lexical gate passes cleanly
  - wrong-minus-lexical delta: 0.157 — passes delta >= 0.10 gate
  - behavioral contradiction rate: 60.2% (Qwen is weak on syllogistic reasoning)
  - behavioral pair AUROC: 0.809
  - 5-fold CV range: 0.655–0.702 (fold 3 crosses threshold)
  - **Diagnosis**: decoder mismatch — arithmetic decoder features are noise for syllogisms
- `entailmentbank`: strict gate `pass` (`cv_primary_pooled_auroc=0.9820`)
  - lexical control AUROC: 0.489 (chance)
  - wrong-minus-lexical delta: 0.493
- final publishability: `fail` (both-domains requirement not met)

## Next Steps: PrOntoQA Domain-Specific Decoder (Active)

### Diagnosis
PrOntoQA failed G2 by only 0.024 AUROC. Root cause: the 5 decoder transition-consistency
features were computed using the **arithmetic** decoder checkpoint, which predicts magnitude,
sign, operator, and numeric subresult — all meaningless for syllogistic reasoning. These
garbage features add noise that hurts the marginal PrOntoQA probe.

EntailmentBank passed despite the same mismatch because its signal is overwhelming (0.982).
PrOntoQA is marginal (0.676), so decoder noise matters.

### Plan
1. **Build PrOntoQA structured state labels** — parse generated CoT text to extract:
   - `inference_type`: universal_instantiation, modus_ponens, class_subsumption (3-4 way)
   - `conclusion_class`: class concluded at this step (14-way: mammal, vertebrate, etc.)
   - `premise_class`: input class at this step (14-way)
   - `chain_depth`: step position in chain (5-way)
   - `truth_value`: whether step conclusion is valid given premises (3-way: True/False/Uncertain)
   - `target_entity`: entity being tracked (10-way: Ava, Ben, Cara, etc.)

2. **Train PrOntoQA-specific decoder** — same architecture as arithmetic decoder
   (`MultiHeadStateDecoder`) but with syllogistic heads instead of arithmetic heads.
   Train on faithful-member rows from PrOntoQA paired dataset (~1,679 members, ~3,800 rows).

3. **Adapt transition consistency features** — replace arithmetic decoder features with:
   - `decoder_chain_coherence`: does conclusion_class[step_i] == premise_class[step_i+1]?
   - `decoder_truth_consistency`: is truth_value stable across the chain?
   - `decoder_answer_alignment`: does final truth_value match the model's claimed answer?
   - `decoder_weakest_link`: min consistency across all transitions
   - `decoder_p95_transition_error`: p95 of prediction confidence gaps across transitions

4. **Rerun G2 PrOntoQA** with domain-specific decoder checkpoint:
   - Regenerate features with PrOntoQA decoder
   - Same CV/bootstrap/permutation stress pipeline
   - Target: push 0.676 → >0.70

### Implementation files to modify
- `phase7/contradictory_pair_prepare.py` — add PrOntoQA structured state parsing
- `phase6/decoder_model.py` or new `phase7/prontoqa_decoder_model.py` — syllogistic heads
- `phase7/train_state_decoders.py` — config for PrOntoQA decoder training
- `phase7/evaluate_optionc.py` — domain-aware decoder feature computation

### Risk assessment
Gap is only 0.024. Even moderate decoder accuracy on truth_value and conclusion_class
should close it. Main risk: Qwen may not distinctly encode syllogistic class identity
internally (60% contradiction rate suggests weak syllogistic competence).

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
- publishability requires cross-domain pass in G2

## Claim Boundary (Current)
- Arithmetic Option C supports a faithfulness-enabled claim **within arithmetic**.
- EntailmentBank Option C supports a faithfulness-enabled claim **within entailment**.
- Cross-domain publishability is currently **not met** (PrOntoQA fail, EntailmentBank pass).
- PrOntoQA failure diagnosed as **decoder mismatch**, not method failure — remediation active.
- GPT-2 closure remains unchanged and is not reopened by Qwen results.

## Deprecated / Obsolete
- [x] Legacy Qwen `Q0` raw hidden-state L2 diagnostic is obsolete.
- [x] Synthetic-only coherence claim from early PrOntoQA runs is historical, not canonical.
- [x] Arithmetic-only ceiling claims from older trajectory-only runs are superseded by Option C generated-pair baseline.

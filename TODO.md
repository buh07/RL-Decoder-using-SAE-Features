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
3. Execute the paper-readiness roadmap below.

---

## Paper-Readiness Roadmap

The cross-domain detection result is validated but insufficient alone for a top venue.
The roadmap below transforms this from a probing study into a mechanistic + practical contribution.
Items are ordered by acceptance-impact, not difficulty.

### Tier 1 — Must-Have (blocks submission)

#### 1. Causal Intervention (~2-3 weeks)
**Goal:** Show that ablating the features the probe identifies actually *changes* whether the model is faithful. Transforms the paper from "we found a correlation" to "we found a mechanism."

Phase 4r's subspace patching (+0.107 Δlog_prob at L22) already demonstrated the mechanism in arithmetic. Extend to Option C's feature set and domains.

- [ ] Design causal experiment: identify top-K probe features per domain, ablate/amplify them during inference, measure faithfulness change (behavioral contradiction rate before vs after).
- [ ] Implement causal intervention script (extend `phase4/causal_patch_test.py` to Option C feature set).
- [ ] Run on arithmetic domain first (existing infrastructure, fastest iteration).
- [ ] Run on PrOntoQA and EntailmentBank.
- [ ] Report: Δ(contradiction_rate) and Δ(log_prob) per domain with CI.
- [ ] Key success criterion: ablating probe-identified features measurably increases unfaithfulness (or amplifying them decreases it).

#### 2. Multi-Model Replication (~2-3 weeks, largest single acceptance boost)
**Goal:** Show the pipeline generalizes beyond Qwen 2.5-7B. Even one additional model with a positive result fundamentally changes the paper's scope.

- [~] Select target model (Llama 3 8B preferred — open weights, well-studied, different architecture family).
  - In progress: `phi-2` external-model canary lineage started as immediate replication bootstrapping with local SAE assets; if stable, promote to larger-model replication.
- [ ] Train SAEs on target model (can reuse phase2 training infrastructure with architecture adapters).
- [ ] Run full Option C pipeline: pair generation → SAE trace → decoder training → eval → stress.
- [ ] Keep exact rigor controls: lexical control, exclusions, CV/bootstrap/permutation.
- [ ] Run on arithmetic first (fastest; if it works, extend to one G2 domain).
- [ ] Promote cross-model claim only if external model reproduces stress-validated pass on ≥1 domain.
- [ ] If negative: report honestly as a scope boundary (still publishable with honest framing).

#### 3. Lexical AUROC Discrepancy Resolution (~2-3 days)
**Goal:** Eliminate a reviewer rejection vector. The stress test reports lexical_variant_auroc=1.0 while eval reports 0.498. Must be explained and resolved before submission.

- [x] Root cause confirmed: stress `lexical_variant_auroc` used lexical-variant rows with original labels; eval lexical metric is computed on `label==0` rows against `lexical_control`.
- [x] Verify split, labeling, and metric orientation for lexical rows in eval vs stress paths.
- [x] Patch stress outputs to include eval-aligned lexical metric (`lexical_control_probe_auroc`).
- [x] Rerun affected stress artifacts with unchanged protocol gates and regenerate canonical stress summaries.
  - PrOntoQA rerun: `20260309_1835_optionc_stress_prontoqa_lexicalfix`
  - EntailmentBank rerun: `20260309_1843_optionc_stress_entailmentbank_lexicalfix`
  - Regenerated cross-domain decision:
    `phase7_results/results/trackc_g2_cross_task_decision_20260309_155350_phase7_g2_feature_prune_stage1_stress_validated_lexicalfix.json`
- [x] Add explicit lexical-metric provenance fields to stress outputs to prevent ambiguity.

### Tier 2 — Strong Differentiators (significantly improve acceptance odds)

#### 4. Practical Faithfulness Improvement (~2 weeks)
**Goal:** Use the detector to steer or filter at inference time and show measurable downstream improvement. "We detect unfaithfulness AND here's what to do about it" is a much stronger pitch than detection alone.

- [ ] Design inference-time intervention: either (a) reject-and-resample CoTs flagged as unfaithful, or (b) steer activations toward faithful subspace during generation.
- [ ] Implement rejection-sampling baseline: generate N CoTs per question, score each with the probe, select the highest-faithfulness one.
- [ ] Measure: accuracy improvement on held-out test set (does selecting "faithful" CoTs improve task performance?).
- [ ] Measure: faithfulness improvement (does the selected CoT have lower behavioral contradiction rate?).
- [ ] If steering: extend Phase 4r subspace patching to inject faithful-direction activations during generation.
- [ ] Report results with CI across all three domains.

#### 5. Decoder Dominance Attribution (~1 week)
**Goal:** Honestly characterize what the probe is detecting. The SAE-only vs mixed AUROC gap (0.66 → 0.96 for PrOntoQA) needs to be explained, not hidden.

- [ ] Run strict ablations on identical folds: `sae_only`, `decoder_only`, `mixed` across all 3 domains.
- [ ] Report AUROC deltas + CI and interaction effect per domain.
- [ ] Characterize per-domain: which feature class dominates and why.
- [ ] Reframe method language: "decoder transition consistency + SAE trajectory coherence" (not SAE-first).
- [ ] Positive spin: showing that *different* internal feature types matter for *different* reasoning domains is itself an insight.

#### 6. Baseline Comparisons (~1 week)
**Goal:** Show Option C adds something over simpler approaches.

- [ ] Implement baselines: (a) attention-pattern entropy, (b) logit-lens/tuned-lens probes, (c) early-answering detection (Lanham et al.), (d) CoT text perplexity.
- [ ] Evaluate all baselines on same split policy, exclusions, and test sets.
- [ ] Report incremental gain of Option C over strongest baseline per domain.
- [ ] If a baseline is competitive: explain what Option C adds (e.g., lexical confound resistance).

### Tier 3 — Ceiling Raisers (pushes toward 70%+ acceptance)

#### 7. Scaling Analysis (~3-4 weeks, high effort but compelling narrative)
**Goal:** Show how faithfulness detection changes across model sizes. If the signal strengthens with scale, this supports an "emergent faithfulness" narrative that is highly publishable.

- [ ] Select model family with multiple sizes (e.g., Qwen 2.5 0.5B / 1.5B / 7B / 14B, or Llama 3 1B / 8B / 70B).
- [ ] Run Option C pipeline on ≥3 sizes (arithmetic domain minimum).
- [ ] Plot: probe AUROC vs model size, decoder feature importance vs model size.
- [ ] Report: does faithfulness signal emerge at a particular scale? Does decoder vs SAE balance shift?
- [ ] This pairs naturally with multi-model (#2) — can share infrastructure.

#### 8. Mechanistic Feature Analysis (~1-2 weeks)
**Goal:** Provide interpretive depth on *what* the probe features represent, not just that they predict.

- [ ] For top-5 probe features per domain: trace back to SAE decoder directions, interpret via max-activating examples.
- [ ] Characterize decoder transition features: which transition consistency checks (step_i→step_i+1) carry signal?
- [ ] Visualize: layer-by-layer feature importance heatmap across domains.
- [ ] Connect to existing Phase 3 reasoning flow tracer results if applicable.

---

## Estimated Timeline to Submission

| Milestone | Items | Cumulative Time |
|---|---|---|
| Tier 1 complete (submittable) | #1, #2, #3 | ~4-5 weeks |
| Tier 2 complete (competitive) | +#4, #5, #6 | ~7-8 weeks |
| Tier 3 complete (strong) | +#7, #8 | ~11-12 weeks |

**Recommended target:** Complete Tier 1 + items #5 and #6 from Tier 2 (~6 weeks). This gives a solid submission with causal evidence, multi-model validation, honest feature attribution, and baselines. Add #4 (practical improvement) if time permits — it's the single highest-impact Tier 2 item.

**Venue targets (in order of fit):**
- EMNLP 2026 / ACL 2027 (NLP venues, most receptive to probing+faithfulness)
- NeurIPS 2026 / ICLR 2027 (ML venues, need Tier 2 minimum)
- ICML MechInterp Workshop (lower bar, good for early visibility)

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

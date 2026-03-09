# Active Tasks

**Last Updated:** March 8, 2026

## Project State (Decision Locked)
- GPT-2 Phase 7 is **closed**.
- Canonical GPT-2 policy is **robust-only closure evidence**.
- Final GPT-2 Phase 7 deployment config remains two-track:
  - `text=0.50`, `latent=0.50`, `confidence=0.0`, `causal=0.0`
  - structural penalties enabled.
- Qwen is a **separate next-model inquiry**.

Canonical GPT-2 closure references:
- `docs/PHASE7V3_TRACKC_FINDINGS.md`
- `docs/TRACKC_NEGATIVE_FINDING.md`
- `phase7_results/results/trackc_phase7v3_closure_note_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/trackc_phase7v3_decision_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json`
- `phase7_results/results/optionbc_final_phase7_optionbc_20260306_092554_phase7_optionbc.json`
- `phase7_results/results/phase7_sae_trajectory_pathc_robust_20260307_001237_phase7_sae_trajectory_pathc_robust.json`

---

## Completed (GPT-2)
- [x] Phase 7v3 Track C closure decision matrix completed.
- [x] Option B fair-profile comparison completed.
- [x] Option C split-corrected probe completed.
- [x] SAE trajectory ladder completed on GPT-2:
  - Path A baseline
  - Path B feature-set swap
  - Path C + robust variant exclusion + bootstrap + grouped CV
- [x] Mixed hidden+SAE analysis completed and classified as non-gating exploratory for closure decisions.

---

## Qwen Active Workstream (Phase 7 SAE Trajectory)

### Scope
- Synthetic-only Track C continuation on `qwen2.5-7b`.
- Full ladder: **feature discrimination → Path A → Path B → Path C**.
- Execution topology: tmux on GPUs **5/6/7**.

### Required Precompute
- [ ] Collect Qwen expanded dataset (`train` + `test`) with:
  - `--capture-extra-anchors`
  - SAE features enabled.
- [ ] Build Qwen step-trace datasets from expanded records.
- [ ] Train Qwen D1 decoder checkpoints (4 parity configs):
  - `raw_every2_even`
  - `hybrid_every2_even`
  - `raw_middle14_07_20`
  - `hybrid_middle14_07_20`
- [ ] Derive Qwen `top_features_per_layer` from expanded train split.
- [ ] Build Qwen variable subspaces for:
  - `subresult_value`, `operator`, `magnitude_bucket`, `sign`.
- [ ] Generate Qwen controls + control-conditioned records artifact.

### Canary Then Full Ladder
- [ ] Canary run (`sample_traces=80`) for:
  - SAE feature discrimination (Qwen divergent source)
  - Path A
  - Path B (`result_top50`, `eq_pre_result_150`, `divergent_top50`)
  - Path C robust
- [ ] Promotion condition for full run:
  - all canary artifacts exist and parse
  - trace-overlap checks pass (`0` overlap)
  - no fatal runtime errors.
- [ ] Full run (`sample_traces=0/all eligible`) for same ladder.

### Primary Qwen Success Criterion
- [x] Robust Path C criterion:
  - `wrong_intermediate` robust-CV AUROC `> 0.70`. **FAIL** (0.5727 CV pooled).
  - Full wrong_intermediate AUROC = 0.6701, 95% CI [0.562, 0.778].
  - Classified as not yet publishable for mechanistic faithfulness claim.
  - Result: `phase7_results/results/phase7_qwen_sae_trajectory_ladder_20260308_013903_phase7_qwen_sae_trajectory.json`

### Qwen Arithmetic Conclusion
- SAE trajectory coherence detects **structural** unfaithfulness well (order_flip ~0.98)
  but converges to the same ~0.67 ceiling on **content** unfaithfulness (wrong_intermediate)
  as GPT-2 Medium. This ceiling appears method-inherent, not model-specific.
- Path B divergent_top50 AUROC = 0.831 (inflated by structural variants).
- The SAE trajectory pipeline does NOT use the state decoder — decoder accuracy is not a factor.

---

## General CoT Faithfulness Generalization

### Goal
Generalize the SAE trajectory coherence method from arithmetic to general
chain-of-thought reasoning. The core hypothesis: if a model faithfully reasons,
its internal SAE features should evolve coherently across CoT steps, and
unfaithful reasoning should show detectable inconsistencies.

### Key Architectural Changes Needed
1. **Step segmentation**: Replace arithmetic operation parsing with general
   sentence/clause boundary detection.
2. **Anchor tokens**: Replace `=` sign anchor with step-conclusion final tokens.
3. **Control variants**: Replace arithmetic-specific variants (wrong number)
   with domain-appropriate variants (wrong conclusion, reordered steps, etc.).
4. **Feature selection**: Replace `eq_top50` with `conclusion_top50` (features
   active at reasoning conclusion positions) and `divergent_top50` (unchanged).
5. **Per-step anomaly detection**: Replace trajectory-level summary metrics
   with step-level max-anomaly scoring to catch single-step content corruption.

### Phase G1 — PrOntoQA Pilot (Start Here)

**Why PrOntoQA**: Synthetic syllogistic reasoning with perfect ground truth at
every step, mechanically generated variants, controllable chain length (2–5 hops),
clean one-deduction-per-sentence structure. If the method can't detect
wrong_intermediate here, it won't work on messier datasets.

- Source: https://github.com/asaparov/prontoqa

#### G1.A — Data Pipeline
- [ ] Download/generate PrOntoQA dataset (5,000+ examples, chain lengths 3–5).
- [ ] Prompt Qwen to generate CoT solutions (or use synthetic CoT from dataset).
- [ ] Generate control variants mechanically:
  - `wrong_intermediate` — swap one intermediate conclusion for a wrong entity class.
  - `order_flip` — reorder two non-adjacent deduction steps.
  - `skipped_step` — remove one deduction from the chain.
  - `wrong_premise` — substitute an incorrect rule.
  - `irrelevant_insertion` — insert a true-but-irrelevant statement.
- [ ] Capture Qwen hidden states at step-conclusion tokens (period at end of each
  deduction) for faithful + all variants.
- [ ] Run SAE encoding on captured activations (all 28 layers).

#### G1.B — Direct Transfer Test
- [ ] Run existing SAE trajectory coherence pipeline on PrOntoQA data with
  minimal changes (swap anchor positions and step segmentation only).
- [ ] Evaluate per-variant AUROC breakdown (especially wrong_intermediate).
- [ ] Compare to arithmetic baseline (~0.67 ceiling).

#### G1.C — Architecture Improvements
- [ ] Implement per-step anomaly detection:
  - For each step, compute `delta_i = SAE(unfaithful, step_i) - SAE(faithful, step_i)`.
  - Aggregate via `max(anomaly_score_i)` instead of trajectory-level mean.
- [ ] Test all 28 Qwen layers (not just 3).
- [ ] Run full Path A→B→C ladder on PrOntoQA.

#### G1.D — Success Criterion
- [ ] `wrong_intermediate` robust-CV AUROC > 0.70 on PrOntoQA.
- [ ] If pass: proceed to Phase G2 (cross-task validation).
- [ ] If fail: investigate per-step anomaly detection before concluding.

### Phase G2 — Cross-Task Validation

- [ ] Repeat pipeline on **EntailmentBank** (multi-step textual entailment trees).
  - Source: https://allenai.org/data/entailmentbank
  - ~1,800 problems with gold reasoning trees.
  - Control variants: corrupt one intermediate entailment node.
- [ ] Repeat on **FOLIO** (first-order logic NLI with FOL annotations).
  - Source: https://github.com/Yale-LILY/FOLIO
  - ~1,400 examples.
- [ ] If wrong_intermediate AUROC > 0.70 on robust CV across 2+ datasets:
  - **Publishable cross-domain mechanistic faithfulness result.**

### Phase G3 — Semi-Structured Reasoning (Stretch)

- [ ] BIG-Bench Hard (logical deduction / tracking shuffled objects).
- [ ] CLUTRR (compositional relational reasoning).
- [ ] HotpotQA (multi-hop factual QA — hardest, requires fact retrieval).

---

## Policy And Claim Boundary
- Qwen arithmetic results do **not** reopen GPT-2 closure.
- General CoT generalization (Phase G) is a new research direction, not a retry.
- Cross-domain claims require robust evidence on 2+ non-arithmetic datasets.

---

## Deprecated / Obsolete
- [x] Legacy Qwen `Q0` raw hidden-state L2 diagnostic is obsolete for active planning.
- [x] Qwen entrypoint is now SAE trajectory ladder (Path A→B→C), not Q0 go/no-go.
- [x] Arithmetic-only wrong_intermediate ceiling (~0.67) confirmed across GPT-2 + Qwen.

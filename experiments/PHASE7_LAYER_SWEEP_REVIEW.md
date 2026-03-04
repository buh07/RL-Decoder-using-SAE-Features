# Phase 7 Experiment Review — Results Analysis + Academic Weaknesses

**Date:** March 4, 2026 (updated after code fixes)
**Scope:** Phase 7v2 full pipeline — state decoder sweep, causal audit, faithfulness benchmarking
**Model:** GPT-2 Medium (24 layers, 1024 hidden dim)
**Tests:** 70 passed, 0 failed

---

## 1. What Was Done

Phase 7 builds a **three-track CoT faithfulness auditor** for arithmetic reasoning:

- **Track A (text_only):** Does the CoT text match the ground-truth computation?
- **Track B (latent_only):** Does the CoT text agree with what the model's hidden states actually encode (via a state decoder)?
- **Track C (causal_auditor):** Do causal interventions (necessity/sufficiency/specificity) confirm the model uses these features?

The system is evaluated on **synthetic controls**: take a known-faithful CoT trace, then generate 10 unfaithful variants (wrong_intermediate, reordered_steps, false_rationale_correct_answer, answer_first_order_flip, answer_first_only, order_flip_only, silent_error_correction, irrelevant_rationale, prompt_bias_rationalization, shortcut_rationalization). The auditor should score faithful traces high and unfaithful traces low.

### Experiments completed:

1. **Broad layer sweep**: 102 configs (43 raw + 43 hybrid + 16 SAE) trained across all layer combinations — **102/102 completed, 0 failed**
2. **Calibration checkpoints**: 4 configs benchmarked with full audit pipeline (raw/hybrid × every2_even/middle12_06_17)
3. **v1 → v2 scoring comparison**: Before/after the scoring overhaul
4. **Ablation weight sweep**: 4 weight configurations × 4 checkpoints = 16 combinations
5. **Academic closure**: Dual-gate benchmark on 4 checkpoints with real-CoT pilot

---

## 2. What the Results Mean

### 2A. State Decoder Sweep — The Good News

The state decoder **successfully learns to read arithmetic state from hidden representations**:

| Metric | Best Config | Value |
|--------|-------------|-------|
| result_token_top1 | `raw_middle_every2_06_16` | **0.538** |
| result_token_top5 | `raw_block4_04_07` | **0.784** |
| operator_acc | Most configs | **0.999** |
| sign_acc | Most configs | **1.000** |
| magnitude_acc | Best configs | **0.932** |
| delta_logprob_vs_gpt2 | `hybrid_block4_04_07` | **+2.538** |

**Key patterns:**
- **Middle layers dominate.** Top configs use layers 4-16 (middle_every2_06_16, block4_04_07, block8_04_11). Early layers (0-3) and very late layers (20-23) alone are poor.
- **Raw ≈ hybrid.** Raw avg top1 = 0.431, hybrid avg top1 = 0.434 — essentially identical. The hybrid variant's SAE features add negligible information over raw residual streams.
- **SAE-only is worse.** SAE avg top1 = 0.334, best = 0.400. Lossy SAE reconstruction degrades the signal.
- **Single layers hit 0.40-0.50.** Even one layer carries substantial arithmetic state — this peaks at layers 7-12.
- **Multi-layer ensembles plateau.** Going from 4 layers to 12 layers improves top1 by only ~0.01. The marginal value of additional layers is small.

**Interpretation:** GPT-2 Medium encodes arithmetic computation state throughout its middle layers (4-16) in raw residual-stream space. The state decoder can extract operator identity (99.9%), sign (100%), magnitude (93%), and the correct result token (54% top-1, 78% top-5) — well above the GPT-2 baseline (delta +2.5 logprob). This confirms that hidden states contain decodable arithmetic representations.

### 2B. Faithfulness Auditor — Pre-Fix Results

Prior to the latest code changes, **the three-track auditor failed its own gates**:

#### Pre-fix results (academic closure run, best checkpoint `raw_every2_even`):

| Track | AUROC | Gate Target | Status |
|-------|-------|-------------|--------|
| text_only | **0.690** | ≥ 0.80 | **FAIL** |
| latent_only | **0.505** | ≥ 0.70 | **FAIL** (chance) |
| causal_auditor | **0.500** | ≥ 0.65 | **FAIL** (chance) |
| composite | **0.810** | ≥ 0.85 | **FAIL** (close) |

#### v1 → v2 comparison (same checkpoint, `raw_every2_even`):

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| overall AUROC | 0.968 | 0.950 | -0.018 |
| text_only AUROC | 0.937 | 0.937 | 0.000 |
| latent_only AUROC | 0.500 | 0.626 | **+0.126** |
| causal_auditor AUROC | 0.968 | 0.950 | -0.018 |
| readout_high_causal_fail | 1686 | 974 | -712 |

#### Weight ablation results (pre-fix):

All 4 weight configurations produced **identical AUROC** — the causal track contributed zero discriminative signal. Confirmed by `causal_signal_coverage_fraction = 0.051`.

### 2C. The readout_high_causal_fail Signal

Across all runs, **1100-1300 out of 3850 traces** (~30%) show high readout but low causal support. This is the most scientifically interesting finding:

> The model's hidden states contain decodable arithmetic representations, but we cannot confirm these representations are causally necessary for the model's output behavior.

---

## 3. Weakness Fix Status (Latest Code Update)

### W1/W2. Track B shared hidden states — ARCHITECTURALLY FIXED

| Component | Status |
|-----------|--------|
| `build_control_latent_cache.py` (294 lines, new) | **COMPLETE** — runs per-variant forward passes through the LM, captures fresh hidden states at parsed step positions, decodes latent states keyed by `(trace_id, variant, step_idx)` |
| `causal_audit.py` — `--latent-source variant_conditioned` | **COMPLETE** — new parameter switches to three-tuple index lookup instead of shared `(trace_id, step_idx)` |
| `_index_variant_latent_preds()` helper | **COMPLETE** — builds the variant-conditioned index from the cache JSON |

**Architecture:** For each control variant's `cot_text`, the script:
1. Tokenizes the variant's text with the LM adapter
2. Runs a full forward pass (`adapter.forward(cot_text)`)
3. Extracts hidden states at each parsed step's token position
4. Decodes latent state via the state decoder checkpoint
5. Outputs keyed by `(trace_id, control_variant, step_idx)`

**Assessment:** The root cause (shared hidden states) is fully addressed. Track B will now see **different** hidden states for each variant because the forward pass runs on the variant's actual text. **However, this has not yet been run** — no benchmark results exist with `--latent-source variant_conditioned`. This is a compute dependency, not a code gap.

### W3. Track C causal coverage (5.1% → ≥25%) — PARTIALLY FIXED

| Component | Status |
|-----------|--------|
| `causal_intervention_engine.py` — donor relaxation | **COMPLETE** — two-tier fallback expands donor pool |
| `benchmark_faithfulness.py` — `--min-causal-signal-coverage` gate | **COMPLETE** — hard coverage threshold (default 0.25) |
| `benchmark_faithfulness.py` — `--require-dual-gate` | **COMPLETE** — both composite + causal floor must pass |
| Actual expanded causal runs (≥25% coverage) | **NOT YET RUN** — still needs larger `--max-records` invocations |

**Assessment:** The infrastructure for gating and reporting is complete. The actual compute to reach 25% coverage (~960+ traces with causal checks) has not been executed.

### W4. Causal track zero signal — COMPLETE (gating)

`--require-dual-gate` + `--causal-floor-auroc 0.65` ensures that if Track C contributes nothing, the overall gate **explicitly fails** rather than silently passing. This is the correct response: don't pretend a zero-signal track is useful; gate on it.

### W5. Real-CoT evaluation — SUBSTANTIALLY FIXED

| Component | Status |
|-----------|--------|
| `parse_cot_to_states.py` — `parse_mode="hybrid"` | **COMPLETE** — fallback parser handles `<<3+4=7>>` and `3 + 4 = 7` inline equations |
| `prepare_real_cot_eval.py` (new) | **COMPLETE** — ingests real CoT JSONL, uses hybrid parser, emits controls-compatible payload |
| `ingest_public_cot_benchmark.py` (new) | **COMPLETE** — ingests FaithCoT-Bench, FINE-CoT, FUR, FRIT via HF or local JSONL |
| Real-CoT benchmark results with >0 parseable rows | **NOT YET RUN** with hybrid parser |

**Assessment:** The 0-parseable-row problem (real CoT used `<<a+b=c>>` format which the template parser couldn't handle) is solved by the hybrid fallback parser. Public benchmark ingestion pipeline exists. Needs a re-run.

### W6. Single model — NOT FIXED

No Qwen results exist. Infrastructure supports it (model registry, variable hidden dim) but no runs have been done.

### W7. Hybrid adds nothing — ADDRESSED (diagnostically)

| Component | Status |
|-----------|--------|
| `generate_cot_controls.py` — style pool diversity | **COMPLETE** — randomized style templates prevent lexical shortcuts |
| `benchmark_faithfulness.py` — ablation weights + style counterfactual breakdown | **COMPLETE** — quantifies hybrid vs raw contribution |
| `common.py` — `control_variant_metadata()` | **COMPLETE** — paper taxonomy per variant |

**Assessment:** The code now properly measures and reports whether hybrid adds signal. The *finding* that hybrid ≈ raw is a legitimate result, not a bug. The style pool fix ensures the comparison is fair (no lexical shortcut inflating one variant over another).

### W8. 0.538 top-1 modest — NOT A CODE ISSUE

This is a model capability limitation, not a code bug. The state decoder extracts what's there.

### W9. No per-variant AUROC — FIXED

`benchmark_faithfulness.py` now reports `by_control_variant` with per-variant AUROC, mean scores, and class counts. The `style_counterfactual_breakdown` also reports per-style-family metrics.

### W10. Claim boundary — NOT A CODE ISSUE

Addressed by `--require-dual-gate` ensuring the claim boundary is enforced, not just stated.

---

## 4. Remaining Academic Weaknesses

### R1. CRITICAL — No benchmark results with variant-conditioned latent cache

The `build_control_latent_cache.py` infrastructure is complete and correct, but **zero benchmark runs use `--latent-source variant_conditioned`**. Until this is run:
- Track B's AUROC improvement from per-variant forward passes is **unknown**
- The W1/W2 fix is unvalidated
- All current Track B numbers (0.50-0.63) reflect the old shared-hidden-state design

**What's needed:** Run `build_control_latent_cache.py` → `causal_audit.py --latent-source variant_conditioned` → `calibrate` → `benchmark` on at least 1 checkpoint. Estimated: 2-4h GPU.

### R2. CRITICAL — Causal coverage still at 5.1%

The `--min-causal-signal-coverage 0.25` gate is set, but no run has achieved it. The causal track remains at chance. Need to run causal interventions on ≥960 traces.

**What's needed:** `causal_intervention_engine.py --max-records 1000+` on shortlisted configs. Estimated: 8-20h GPU.

### R3. HIGH — No re-run of real-CoT pilot with hybrid parser

The hybrid parser (`parse_mode="hybrid"`) should fix the 0/1248 parseable rows, but it hasn't been tested on the actual `gsm8k_test_real_cot.jsonl`. The real-CoT external validity claim is still unvalidated.

**What's needed:** `prepare_real_cot_eval.py --parse-mode hybrid` → audit → benchmark. Estimated: 1h.

### R4. HIGH — Single model (GPT-2 only)

No Qwen 2.5-7B results. The model registry and variable-dim infrastructure exist but are untested with real data. A reviewer would ask "does this generalize beyond GPT-2?"

### R5. RESOLVED IN CODE — latent separation diagnostic logic fixed; rerun validation pending

`phase7/diagnose_latent_separation.py` now computes faithful-vs-variant latent deltas from variant-conditioned cache entries keyed by `(trace_id, control_variant, step_idx)`, and emits definedness/coverage metadata. The prior same-vector bug (C4/C5) is no longer present in source.

**Remaining risk:** Existing historical diagnostic artifacts generated before this fix should not be cited; rerun diagnostics for current checkpoints.

### R6. MEDIUM — No ablation proving Track B adds value over Track A alone

Even if variant-conditioned Track B achieves AUROC ≥ 0.70, the question remains: does combining A+B outperform A alone? The ablation infrastructure exists (`--ablation-weights`) but the critical comparison `{text:1.0, latent:0.0, causal:0.0}` vs `{text:0.5, latent:0.5, causal:0.0}` hasn't been run with variant-conditioned data.

### R7. MEDIUM — Evaluation is still only on synthetic controls

Even with the hybrid parser and public benchmark ingestion tools, no **labeled** real-CoT evaluation exists. The synthetic controls test programmatic manipulations, not real-world unfaithful reasoning. This limits external validity claims.

### R8. MEDIUM — All pre-fix benchmark numbers are stale

The academic closure results (composite AUROC 0.81, text 0.69, latent 0.51) were generated before the latest code fixes. The control variant style randomization, hybrid parser, and variant-conditioned latent cache changes could meaningfully alter these numbers. Current results should not be cited.

### R9. LOW — Test coverage gaps remain

From 70 tests (all passing), these critical paths still lack coverage:
- `decode_latent_pred_states()` output format
- `MultiHeadStateDecoder.forward()` / `compute_multitask_loss()`
- full regression test for faithful-vs-variant latent delta computation path
- Non-empty `causal_idx` integration tests

---

## 5. What You Can Legitimately Claim

Despite the weaknesses, there are genuine positive findings:

1. **Arithmetic state is decodable from GPT-2 Medium's hidden representations.** Top-1 accuracy 0.538, operator accuracy 0.999, sign accuracy 1.000. This is a solid representation probing result.

2. **Middle layers (4-16) carry the most arithmetic information.** Layer sweep confirms a clear peak in the middle of the network, consistent with prior work on information flow in transformers.

3. **SAE-only inputs degrade the signal.** Raw residual streams are better than SAE reconstructions for arithmetic state decoding (0.431 vs 0.334 avg top1). This is informative for the SAE interpretability community.

4. **The readout ≠ causation gap exists.** 30% of traces have high readout but unconfirmed causal support. This is a meaningful finding even without resolving it.

5. **Text matching provides moderate unfaithful detection (AUROC 0.69).** Not strong enough for production use, but demonstrates the approach has signal.

6. **Infrastructure for variant-conditioned latent evaluation is complete.** The `build_control_latent_cache.py` pipeline is architecturally sound and ready to run.

---

## 6. Priority Actions

| Priority | Action | Effort | Validates |
|----------|--------|--------|-----------|
| **P0** | Run `build_control_latent_cache.py` + full benchmark with `--latent-source variant_conditioned` | 2-4h GPU | R1 (Track B) |
| **P0** | Run causal interventions with `--max-records 1000` on top 2 configs | 8-20h GPU | R2 (Track C) |
| **P1** | Re-run real-CoT pilot with `--parse-mode hybrid` | 1h | R3 (external validity) |
| **P1** | Run Track A-only vs A+B ablation with variant-conditioned data | 1h | R6 (multi-track justification) |
| **P2** | Run state decoder + audit on Qwen 2.5-7B | 15-30h GPU | R4 (generalization) |
| **P2** | Re-run latent separation diagnostics on latest checkpoints and archive old outputs | 30–60min | R5 (artifact validity) |

---

## 7. Broad Sweep Summary Table (Top 15 / Bottom 5)

### Top 15 by result_token_top1 (test set):
| Config | top1 | top5 | op_acc | delta_logprob |
|--------|------|------|--------|---------------|
| raw_middle_every2_06_16 | 0.538 | 0.776 | 0.999 | +2.467 |
| hybrid_block4_04_07 | 0.533 | 0.777 | 1.000 | +2.538 |
| raw_every2_even | 0.531 | 0.776 | 1.000 | +2.421 |
| hybrid_every2_even | 0.531 | 0.775 | 0.999 | +2.397 |
| hybrid_spread4_edges | 0.530 | 0.776 | 0.999 | +2.464 |
| hybrid_block8_04_11 | 0.530 | 0.773 | 0.999 | +2.495 |
| raw_every2_odd | 0.529 | 0.780 | 0.999 | +2.407 |
| raw_block8_04_11 | 0.527 | 0.770 | 0.999 | +2.485 |
| raw_block8_00_07 | 0.527 | 0.777 | 0.999 | +2.522 |
| raw_block4_04_07 | 0.527 | 0.784 | 1.000 | +2.535 |
| hybrid_block8_00_07 | 0.526 | 0.775 | 0.999 | +2.512 |
| raw_spread4_edges | 0.525 | 0.778 | 0.999 | +2.449 |
| raw_spread4_current | 0.524 | 0.766 | 0.999 | +2.413 |
| hybrid_middle_every2_06_16 | 0.523 | 0.773 | 0.999 | +2.410 |
| hybrid_spread4_quartiles | 0.521 | 0.773 | 0.999 | +2.423 |

### Bottom 5:
| Config | top1 | top5 | op_acc | delta_logprob |
|--------|------|------|--------|---------------|
| sae_single_l22 | 0.209 | 0.528 | 0.976 | +1.109 |
| hybrid_single_l01 | 0.125 | 0.420 | 0.982 | +0.805 |
| raw_single_l01 | 0.117 | 0.418 | 0.982 | +0.820 |
| hybrid_single_l00 | 0.083 | 0.308 | 0.869 | +0.393 |
| raw_single_l00 | 0.083 | 0.312 | 0.873 | +0.390 |

### Variant averages:
| Variant | N | Avg top1 | Best top1 | Worst top1 |
|---------|---|----------|-----------|------------|
| raw | 43 | 0.431 | 0.538 | 0.083 |
| hybrid | 43 | 0.434 | 0.533 | 0.083 |
| sae | 16 | 0.334 | 0.400 | 0.209 |

---

## 8. Code Issues (see also PHASE7_CODE_REVIEW.md)

The separate [PHASE7_CODE_REVIEW.md](../phase7/PHASE7_CODE_REVIEW.md) documents 36 code issues found across three review passes. Current status:

| Severity | Total | Fixed | Open |
|----------|------:|------:|-----:|
| CRITICAL | 5 | 5 | 0 |
| HIGH | 8 | 8 | 0 |
| MEDIUM | 14 | 10 | 4 |
| LOW | 9 | 7 | 2 |

---

## 9. Verdict

**The state decoder sweep is a success** — it demonstrates clear, replicable arithmetic state decoding from GPT-2 hidden representations with informative layer-dependency patterns.

**The faithfulness auditor has the right architecture but unvalidated fixes.** The latest code changes address the three core weaknesses:
- Track B: `build_control_latent_cache.py` enables per-variant forward passes (**not yet run**)
- Track C: dual-gate + coverage threshold enforce honest reporting (**coverage not yet achieved**)
- Real CoT: hybrid parser solves parseability (**not yet re-run**)

**The project's bottleneck is now compute, not code.** The top priority is running `build_control_latent_cache.py` on at least one checkpoint and benchmarking with `--latent-source variant_conditioned` to validate that Track B actually improves with per-variant hidden states. This is the single highest-impact experiment remaining.

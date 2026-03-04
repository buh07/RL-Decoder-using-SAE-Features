# Phase 7 Code Review — Issues and Recommendations

**Original date:** March 3, 2026
**Last updated:** March 3, 2026 (third pass — all claims verified against code)
**Scope:** All `phase7/` source files, `phase6/pipeline_utils.py`, sweep worker, and experimental methodology
**Sweep status at time of review:** 102/102 configs completed, 0 failed, all 32 tests passing

---

## Summary

| Severity | Total | Fixed | Open |
|----------|------:|------:|-----:|
| CRITICAL | 5 | 5 | 0 |
| HIGH | 8 | 8 | 0 |
| MEDIUM | 14 | 12 | 2 |
| LOW | 9 | 7 | 2 |

---

## CRITICAL Issues

### C1. Unverifiable steps scored as 0.0 inflate text/latent track AUROC — FIXED

**Files:** `causal_audit.py:118-131,278`
**Category:** STATISTICAL / METHODOLOGY

Unverifiable steps now receive a configurable neutral score (`thr["unverifiable_step_score"]`, default 0.05) instead of 0.0. In `_paper_metrics_for_trace()`, parseable critical steps are explicitly filtered: `parseable_critical = [s for s in crit if not s.get("unverifiable_text")]`, and only parseable steps are included in text/latent track means.

---

### C2. Three control variants trivially detectable by format, not content — FIXED

**Files:** `generate_cot_controls.py:266-300`
**Category:** METHODOLOGY / ACADEMIC_RIGOR

All three variants (`irrelevant_rationale`, `prompt_bias_rationalization`, `shortcut_rationalization`) have been redesigned to produce syntactically valid, parseable CoT text with role markers. They now generate complete step text plus variant-specific rationale lines, rather than unparseable text.

---

### C3. `perturb_number` always increases magnitude — FIXED

**File:** `common.py:403`
**Category:** METHODOLOGY / ACADEMIC_RIGOR

`perturb_number()` now randomizes perturbation direction with `delta_sign = float(rv.choice([-1.0, 1.0]))`. The systematic magnitude-direction shortcut is eliminated.

---

### C4. `diagnose_latent_separation.py` — latent delta always zero — FIXED

**File:** `diagnose_latent_separation.py:214-229`
**Category:** BUG / LOGIC_ERROR

The diagnostic now compares faithful-vs-variant latent predictions per `(trace_id, step_idx)` using the variant-conditioned latent cache index and reports definedness/coverage fields. The same-vector bug is removed.

---

### C5. Missing faithful state lookup in latent separation diagnostic — FIXED

**File:** `diagnose_latent_separation.py:207-229`
**Category:** DESIGN_FLAW

Faithful latent predictions are now explicitly looked up and compared against each unfaithful variant row. Missing faithful/variant rows are skipped and tracked via coverage counters.

---

## HIGH Issues

### H1. ROC label polarity is inverted in threshold calibration — FIXED

**File:** `calibrate_audit_thresholds.py:88-94`
**Category:** LOGIC_ERROR

The `_calibrate_track()` function now uses `_to_positive_score()` which properly inverts scores when `positive_label == "unfaithful"` (`return 1.0 - s` at line 93). Label assignment correctly maps `1 if lbl == positive_label else 0`.

---

### H2. Hybrid input variant loses feature identity — FIXED

**File:** `phase6/pipeline_utils.py:105-111`
**Category:** METHODOLOGY

The `hybrid_indexed` variant now includes both feature indices AND values. Indices are normalized at line 110 and concatenated at line 111: `torch.cat([raw_sel, topvals, topidx_norm], dim=-1)`.

---

### H3. `result.update(sweep_metadata)` silently overwrites result keys — FIXED

**Files:** `train_state_decoders.py:454`, `evaluate_state_decoders.py:221-240`
**Category:** BUG

Both files now use `result.setdefault(k, v)` / `md.setdefault()` which does NOT overwrite existing keys, preventing key collisions.

---

### H4. `_record_eq_pos` silently defaults to position 0 — FIXED

**File:** `causal_intervention_engine.py:74-86`
**Category:** BUG

Now raises `KeyError("missing eq_tok_idx")` if the field is absent (line 75), validates the value is positive (lines 77-79), and checks it doesn't exceed `token_ids` length (lines 80-85).

---

### H5. `_bool_rate` drops None values, inflating causal pass rates — FIXED

**File:** `causal_audit.py:247-263`
**Category:** STATISTICAL

Replaced with `_bool_rate_stats()` which returns three metrics: `rate_observed`, `coverage_fraction`, and `rate_with_missing_as_fail`. This prevents inflation when causal checks are incomplete.

---

### H6. Train/test leakage fallback in record filtering — FIXED

**Files:** `train_state_decoders.py`, `evaluate_state_decoders.py`
**Category:** BUG / METHODOLOGY

The `or records` fallback pattern has been fully removed. `evaluate_state_decoders.py` now raises `RuntimeError` when train records are None. No instances of the pattern remain in either file.

---

### H7. `answer_first_order_flip` confounds two failure signals — FIXED

**File:** `generate_cot_controls.py:242-251`
**Category:** METHODOLOGY

The confounded variant has been split into three separate variants: `answer_first_order_flip` (both manipulations, line 242), `answer_first_only` (line 246), and `order_flip_only` (line 249). Each variant has distinct semantics.

---

### H8. Ablation blend uses wrong threshold for confusion matrix metrics — FIXED

**File:** `benchmark_faithfulness.py:479-504`
**Category:** STATISTICAL

Ablation blend now calibrates its own threshold from the calibration split when available, falling back to `causal_thr` only when calibration data is unavailable.

---

## MEDIUM Issues

### M1. Duplicate 50K-vocab linear head wastes ~12.9M parameters — FIXED

**File:** `state_decoder_core.py:345`

`MultiHeadStateDecoder` now passes `use_output_head=False` to the backbone, preventing the duplicate linear head from being instantiated.

---

### M2. Early stopping driven by weighted multitask loss, not primary metric — FIXED

**File:** `train_state_decoders.py:105-108,362-380`

Early stopping now respects `--early-stop-metric` argument with default `"result_top1"`. Supports three modes: `result_top1`, `loss_total`, and `composite`.

---

### M3. `compare_states` fragile to type mismatches — FIXED

**File:** `common.py:407-420`

Now uses `_coerce_numeric_like()` to safely convert both string and numeric types to floats before comparison.

---

### M4. `select_matched_donor` uses `is` for identity check — FIXED

**File:** `causal_intervention_engine.py:179-184,314-342`

Now uses `_record_identity_key(r) == src_key` (value equality on tuples) instead of `r is source`.

---

### M5. `necessity_ablation` patches at `eq_pos` but measures at `pred_pos` — FIXED

Fixed indirectly by H4 — `_record_eq_pos()` now raises on missing data instead of silently defaulting to 0.

---

### M6. Population std used instead of sample std for normalization — FIXED

**File:** `state_decoder_core.py:200`

Now uses `arr.std(ddof=1)` (Bessel's correction) when `arr.size > 1`, defaulting to 0.0 when size is 1.

---

### M7. Gradient saliency computed on train records, not test — FIXED

**File:** `evaluate_state_decoders.py:79-82,555-563`

New `--saliency-split` argument accepts `"val"`, `"test"`, or `"train"`. Default is `"test"` for out-of-sample analysis.

---

### M8. `_quantile` uses nearest-rank, unstable for small N — FIXED

**File:** `benchmark_faithfulness.py:70-83`

Now uses linear interpolation between nearest ranks: `(1-w)*vals[lo] + w*vals[hi]`.

---

### M9. `soundness` applies marker penalty unconditionally — FIXED

**File:** `causal_audit.py:314`

`soundness` now checks `bool(thr.get("apply_marker_penalties_to_soundness", False))` and applies trace-only marker penalties only when enabled.

---

### M10. `state_decoder_core.py:input_dim()` hardcodes GPT-2 dimensions — FIXED

**File:** `state_decoder_core.py:50,74-83,151-167`

Now returns `int(self.model_sae_dim)` via a configurable dataclass field (default 12288). Overridable via `apply_model_metadata_to_config()` at line 162.

---

### M11. Latent track definedness uses same condition as text track — FIXED

**File:** `causal_audit.py:288-303`
**Category:** SEMANTIC

`latent_track_defined` now uses `latent_available_critical` (parseable steps with latent predictions), not `parseable_critical`. This prevents latent track from being marked defined when latent data is absent.

---

### M12. Undefined track fallback to 0.5 not documented — FIXED

**File:** `causal_audit.py:722-735`
**Category:** DOCUMENTATION / SEMANTIC

Undefined track fallback policy is now explicit in outputs via:
- `benchmark_track_definedness`
- `undefined_track_policy`
- `claim_scope_metadata`
The neutral fallback remains intentional and documented as a bounded/diagnostic choice.

---

### M13. (NEW) Incomplete claim scope mapping in `_paper_metrics_for_trace`

**File:** `causal_audit.py:381-383`
**Category:** LOGIC_CLARITY

```python
claim_scope = "partial_causal_support"
if ctrl.get("paper_failure_family") in {"prompt_bias_rationalization", "shortcut_rationalization"}:
    claim_scope = "broad_explanation_claim"
```

Only two failure families map to `"broad_explanation_claim"`. It's unclear whether other families (e.g., `silent_error_correction`, `answer_first_order_flip`) should also map there. The mapping should be documented or generalized.

---

### M14. (NEW) `_quantile` returns 0.0 for all-NaN inputs without warning

**File:** `benchmark_faithfulness.py:70-73`
**Category:** EDGE_CASE

```python
if not vals:
    return 0.0  # silent fallback
```

When all input values are NaN or non-finite, `_quantile` returns 0.0 without indication. Downstream code (e.g., `latent_high_threshold` at line 430) could silently use 0.0, affecting `readout_high` case detection. Should return NaN or log a warning.

---

## LOW Issues

### L1. `except Exception` for import fallback catches too broadly — FIXED

**Files:** All phase7 source files
Now use `except ImportError` (verified in `state_decoder_core.py:25-28`, `causal_audit.py:10-19`, `diagnose_latent_separation.py:13-40`).

---

### L2. `_mean` returns 0.0 for empty lists instead of NaN — FIXED

**File:** `causal_audit.py:238-244`
Now returns `float("nan")` when no valid values and no default provided.

---

### L3. `_is_number_char` doesn't track multiple decimal points — NOT FIXED

**File:** `common.py:94-95`
Edge case — `ch.isdigit() or ch == "."` allows unlimited decimal points. GSM8K uses integers so not triggered in practice.

---

### L4. `verdict` can be `"causally_faithful"` when all critical steps are unverifiable — FIXED

**File:** `causal_audit.py:595,608-609`
Now guarded by `has_parseable_critical` check. If all critical steps are unverifiable, `causally_supported_all_parseable=False`.

---

### L5. `from_dict` doesn't handle unknown keys gracefully — FIXED

**File:** `state_decoder_core.py:92-98`
Now filters to only allowed fields: `filtered = {k: v for k, v in d.items() if k in allowed}`.

---

### L6. `_compute_grad_sae_saliency` hardcodes 256-example cap — FIXED

**File:** `evaluate_state_decoders.py:84-89`
Now parameterized as `--grad-saliency-max-records` with default 256 and documented help text.

---

### L7. No CUDA determinism enforcement for reproducibility — FIXED

**File:** `train_state_decoders.py:111-114,469-470`
Now respects `--deterministic` flag: `if args.deterministic: torch.use_deterministic_algorithms(True)`.

---

### L8. No validation for empty controls in `diagnose_latent_separation.py` — NOT FIXED

**File:** `diagnose_latent_separation.py:112-115`
**Category:** VALIDATION

If the controls JSON is malformed or the `"controls"` key is missing, falls back to empty list without error. Should add `if not controls: raise ValueError(...)`.

---

### L9. Marker penalty weights undocumented — NOT FIXED

**File:** `causal_audit.py:34-37`
**Category:** DOCUMENTATION

```python
"marker_penalty_prompt_bias": 0.25,
"marker_penalty_shortcut": 0.30,
"marker_penalty_generic_rationale": 0.15,
```

These values have no justification comments, paper citations, or calibration references. A reviewer would ask why these specific numbers.

---

## Test Coverage Gaps

| Code Path | Risk | Status |
|-----------|------|--------|
| `decode_latent_pred_states()` — produces latent predictions for the audit | HIGH — output format unvalidated | OPEN |
| Causal check integration (`causal_idx` always empty in tests) | HIGH — necessity/sufficiency/specificity untested | OPEN |
| `silent_error_correction` and `shortcut_rationalization` variants | MEDIUM — two of eight variants untested | OPEN |
| `_parse_ablation_weights()` in benchmark | MEDIUM — ablation feature untested | OPEN |
| `readout_high_causal_fail_cases` detection | MEDIUM — key diagnostic metric untested | OPEN |
| `MultiHeadStateDecoder.forward()` | MEDIUM — model architecture untested | OPEN |
| `compute_multitask_loss()` | MEDIUM — loss computation untested | OPEN |
| `diagnose_latent_separation.py` — latent delta logic (C4/C5) | CRITICAL — broken logic, zero test coverage | OPEN |
| `split_audit_dataset.py` — edge cases with single item | LOW — single-item split produces empty eval set | OPEN |

---

## Recommendations for Academic Rigor

1. ~~**Report per-variant AUROC prominently.**~~ Addressed — variants redesigned to be parseable.

2. ~~**Randomize perturbation direction in `perturb_number`.**~~ FIXED.

3. ~~**Redesign format-unfaithful variants.**~~ FIXED — all variants produce parseable steps.

4. ~~**Separate "unverifiable" from "contradicted" in scoring.**~~ FIXED — unverifiable steps filtered.

5. ~~**Fix ROC label polarity.**~~ FIXED — proper `_to_positive_score()` inversion.

6. ~~**Document the hybrid input design decision.**~~ FIXED — `hybrid_indexed` variant now includes feature indices.

7. **Add a "lexical shortcut ablation" experiment.** (STILL OPEN) Run the auditor on faithful controls that have been reformatted to verify the auditor doesn't rely on lexical cues.

8. **(NEW) Fix `diagnose_latent_separation.py` latent delta logic.** The central diagnostic of this file is broken (C4/C5). Must compare faithful vs unfaithful latent predictions, not each variant against itself.

9. **(NEW) Document marker penalty weight rationale.** Add justification for the specific penalty values (0.25, 0.30, 0.15) or cite a calibration experiment.

10. **(NEW) Document the 0.5 neutral fallback.** Explain in the results schema why undefined tracks default to 0.5 and how this affects composite scores.

---

## Fix History

- **Session 1 (March 3, 2026):** Fixed `_masked_mae` NaN propagation, `build_structured_state` zero-falsy, double marker penalty on `causal_track`
- **Post-session fixes:** All CRITICAL (C1-C3), all HIGH (H1-H8), 10/10 original MEDIUM (M1-M10), 6/7 original LOW (L1-L2, L4-L7) issues addressed. L1 (`except Exception` → `except ImportError`) also fixed.
- **Third pass (March 3, 2026):** Verified all fix claims against actual code. Corrected L1 status (now FIXED), H6 status (now FIXED), M8 status (now FIXED). Identified 2 new CRITICAL (C4, C5 in `diagnose_latent_separation.py`), 2 new MEDIUM (M13 claim scope, M14 quantile NaN), 1 correction (L3 still open). Total open: 2 CRITICAL, 4 MEDIUM, 2 LOW.

# Phase 7 Auditor Protocol Scope

**Last Updated:** March 2, 2026  
**Status:** Active design baseline for Phase 7 implementation and benchmarking

## 1) Problem Statement and Claim Boundary

Goal: build a **causal faithfulness verifier** for chain-of-thought (CoT), not a better decoder.

The protocol separates three objects:
1. `text claim` (what the model says in CoT)
2. `latent readout` (what is decodable from internal states)
3. `causal mediation` (what interventions show is actually used)

Primary claim boundary:
- `causally_faithful` means causally supported under measured variables/subspaces and tested interventions.
- It does **not** mean complete explanation of all internal reasoning.

## 2) Existing Protocol Families in Literature

This protocol is positioned against major CoT-faithfulness families:

1. **Text-only plausibility and consistency checks**
   - Example: self-consistency-style agreement over sampled chains.
   - Strength: improves answer reliability.
   - Limitation: consistency is not faithfulness of internal mechanism.
   - Reference: Wang et al., 2022, *Self-Consistency Improves CoT Reasoning*  
     https://arxiv.org/abs/2203.11171

2. **Prompt-sensitivity / black-box faithfulness stress tests**
   - Strength: reveals CoT can be manipulated or post-hoc.
   - Limitation: usually cannot localize internal mediators.
   - Reference: Turpin et al., 2023, *Language Models Don’t Always Say What They Think*  
     https://arxiv.org/abs/2305.04388

3. **Faithfulness-oriented CoT training/inference methods**
   - Strength: encourages better rationale-answer coupling.
   - Limitation: often still behavior-level; causal internal mediation remains underdetermined.
   - Reference: Lyu et al., 2023, *Faithful Chain-of-Thought Reasoning*  
     https://arxiv.org/abs/2301.13379

4. **Intervention-inspired/step-removal paradigms**
   - Strength: closer to causal relevance of reasoning steps.
   - Limitation: typically text-level or model-output-level interventions, not latent white-box mediation checks.
   - Reference: Tutek et al., 2025, *FUR*  
     https://arxiv.org/abs/2502.14829

5. **Benchmark-centric faithfulness suites**
   - Strength: standardized stress tests and taxonomies.
   - Limitation: benchmark success alone does not prove internal causal mediation.
   - References:
     - Shen et al., 2025, *FaithCoT-Bench*  
       https://arxiv.org/abs/2510.04040
     - Swaroop et al., 2025, *FRIT*  
       https://arxiv.org/abs/2509.13334
     - Barez et al., 2025, *CoT Is Not Explainability*  
       https://aigi.ox.ac.uk/publications/chain-of-thought-is-not-explainability/

## 3) What Is New Here

Phase 7 contributes a **triangulated verifier**:

1. **Text-Latent-Causal Triangulation**
   - Track A: `text_only`
   - Track B: `latent_only`
   - Track C: `causal_auditor`
   - Explicit target: show cases where Track B is high but Track C fails.

2. **White-box necessity + sufficiency + specificity**
   - Uses subspace patching on identified latent subspaces.
   - Requires all three checks for causal support.

3. **Failure-family aware controls**
   - Includes paper-core families:
     - prompt bias rationalization
     - silent error correction
     - answer-first/order-flip
     - shortcut rationalization
   - Reports by family, not just aggregate AUROC.

4. **Bounded claims with completeness caveat**
   - Output schemas include claim-boundary and completeness-scope fields.
   - Prevents overclaiming “full explainability.”

5. **Model-portable execution layer**
   - Adapter contract + model registry allow protocol reuse across model families.
   - GPT-2 remains default for fast protocol iteration; Qwen2.5-7B is first external relevance pilot.

## 4) Failure Modes and Non-Claims

Known failure modes:
1. Parser failures (`unverifiable_text`) due to template/format mismatch.
2. Off-manifold interventions that produce destructive side effects.
3. Donor matching scarcity for strict sufficiency controls.
4. High decodability but weak causal support (expected and informative).
5. Marker cue dependence risk (prompt-bias/shortcut lexical cues).
   - Marker penalties are heuristic diagnostics only by default.
   - They are not used as causal evidence unless explicitly enabled.
   - Penalty defaults are intentionally conservative and should be calibration-backed before gate use.

Non-claims:
1. This does not prove complete internal reasoning decomposition.
2. This does not guarantee faithfulness transfer across all domains/tasks.
3. This is not yet an optimization method (reranking/reward shaping) until gates pass.

## 5) Go/No-Go Criteria for Downstream CoT Improvement

Do not use auditor scores for reranking/filtering/reward shaping unless all pass:

1. **AUROC gate:** `causal_auditor` AUROC `>= 0.85`
2. **FPR gate:** unfaithful-control FPR `<= 0.05`
3. **Separation gate:** `readout_high_causal_fail_cases_n >= 1`
4. **Reporting gate:** benchmark output includes A/B/C tracks and claim-boundary disclaimer
5. **Metric-definedness gate:** per-family outputs explicitly report metric definedness
   (for example `metric_defined.auroc=false` with `auroc: null` for single-class slices)

Actionable test sequence:
1. Generate controls (`phase7/generate_cot_controls.py`)
2. Parse/align controls (`phase7/parse_cot_to_states.py`)
3. Run shortlist causal checks (`phase7/causal_intervention_engine.py`)
4. Run audits (`phase7/causal_audit.py`)
5. Calibrate thresholds (`phase7/calibrate_audit_thresholds.py`)
6. Benchmark (`phase7/benchmark_faithfulness.py`)

## 6) Two-Track Execution Policy

Track A (GPT-2 protocol validation):
1. Validate mechanics and gate thresholds quickly with existing white-box tooling.
2. Treat failures as protocol diagnostics, not model-capability failures.

Track B (Qwen2.5-7B portability pilot):
1. Validate adapter load/forward and schema compatibility.
2. Run small text/latent benchmark, then limited causal pilot.
3. Expand only after Track A gates and Track B smoke checks are stable.

## 7) Implementation Status Mapping (March 2026 Hardening Sprint)

Implemented now:
1. Model-portable adapter contract and registry
   - `phase7/model_adapters/base.py`
   - `phase7/model_registry.py`
2. Qwen pilot adapter scaffold
   - `phase7/model_adapters/qwen25_7b_adapter.py`
3. Dual invocation hardening
   - Phase 7 CLIs support both script-style and module-style invocation.
4. Causal engine preflight support
   - `phase7/causal_intervention_engine.py --dry-run` emits structured readiness/unsupported status.
5. Phase 7 smoke test suite
   - `phase7/tests/test_model_registry.py`
   - `phase7/tests/test_invocation_modes.py`
   - `phase7/tests/test_phase7_metadata_contract.py`

Still pending:
1. Non-GPT full causal patching parity
   - Requires model-specific SAE assets/loaders and subspace tooling.
2. Qwen pilot full execution path
   - Adapter load + forward on target GPU environment
   - Small state-decoder run
   - Causal pilot once SAE path exists
3. Track A gate runs at full scale
   - AUROC/FPR/separation gates on control benchmark outputs.

## References

1. Turpin et al., 2023, *Language Models Don’t Always Say What They Think*  
   https://arxiv.org/abs/2305.04388
2. Lyu et al., 2023, *Faithful Chain-of-Thought Reasoning*  
   https://arxiv.org/abs/2301.13379
3. Tutek et al., 2025, *Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps (FUR)*  
   https://arxiv.org/abs/2502.14829
4. Shen et al., 2025, *FaithCoT-Bench*  
   https://arxiv.org/abs/2510.04040
5. Swaroop et al., 2025, *FRIT*  
   https://arxiv.org/abs/2509.13334
6. Barez et al., 2025, *Chain-of-Thought Is Not Explainability*  
   https://aigi.ox.ac.uk/publications/chain-of-thought-is-not-explainability/
7. Wang et al., 2022, *Self-Consistency Improves CoT Reasoning*  
   https://arxiv.org/abs/2203.11171

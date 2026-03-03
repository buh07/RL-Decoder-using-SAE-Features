# Phase 8 Reranking Plan

**Last Updated:** March 2, 2026  
**Status:** Planned, gated on Phase 7 protocol criteria

## 1) Purpose

Phase 8 is the first CoT-improvement track. It uses the Phase 7 auditor as a
selection signal, not as a direct training loss.

Primary objective:
- improve answer quality without sacrificing causal-faithfulness quality.

## 2) Hard Gate Before Any Run

Do not start Phase 8 until all Phase 7 gates pass on controls:
1. `causal_auditor` AUROC `>= 0.85`
2. Unfaithful-control FPR `<= 0.05`
3. `readout_high_causal_fail_cases_n >= 1`
4. Benchmark output includes A/B/C tracks and claim-boundary disclaimer
5. Per-family metric definedness is explicit (`metric_defined`) and acceptable

## 3) Candidate Generation Protocol

For each prompt:
1. Sample `K` CoT candidates (baseline: `K=8`, configurable).
2. Keep decoder/model settings fixed within a run.
3. Store candidate text, final answer token, and logprob metadata.

Output artifact:
- `phase8_results/candidates/<RUN_ID>_candidates.jsonl`

## 4) Auditor-Scored Reranking

For each candidate:
1. Parse CoT to structured states.
2. Compute Phase 7 audit outputs (text/latent/causal tracks).
3. Compute gated reranking score:
   - baseline: `score = causal_auditor_score`
   - only valid if candidate is not `unverifiable_text` hard-fail and does not trip intervention safety flags

Select highest-scoring candidate per prompt.

Output artifacts:
- `phase8_results/reranked/<RUN_ID>_reranked.jsonl`
- `phase8_results/reranked/<RUN_ID>_scoring_summary.json`

## 5) Offline Evaluation

Compare reranked decoding vs baseline decoding on:
1. Answer metrics:
   - exact-match accuracy
   - top-1 / top-5 answer token quality (if applicable)
2. Faithfulness metrics:
   - auditor verdict distribution
   - by-family unfaithfulness detection behavior
3. Tradeoff metrics:
   - answer gain vs faithfulness gain/loss

Required report:
- `phase8_results/results/<RUN_ID>_phase8_reranking_eval.json`

## 6) Baselines

Minimum baselines for every Phase 8 run:
1. Standard decoding without reranking.
2. Text-only reranking proxy (Track A score) as comparator.
3. Latent-only reranking proxy (Track B score) as comparator.

The primary method is Track C (`causal_auditor`) reranking.

## 7) Failure Analysis

For regressions, log:
1. prompts where reranking reduced answer accuracy
2. prompts where reranking selected contradiction-prone traces
3. family-level error shifts (`prompt_bias`, `silent_correction`, `answer_first`, `shortcut`)

Required artifact:
- `phase8_results/analysis/<RUN_ID>_failure_analysis.json`

## 8) Rollback Conditions

Disable reranking by default for production-style runs if any are true:
1. answer accuracy regression exceeds predefined tolerance
2. unfaithful-control false positives rise above calibration target
3. intervention safety/off-manifold flags spike materially

Rollback decision must be recorded in:
- `phase8_results/results/<RUN_ID>_go_no_go.json`

## 9) Claim Boundary

Phase 8 claims stay bounded:
- reranking can improve operational faithfulness under measured variables,
  tested subspaces, and tested interventions.
- it is not a claim of complete explanation of all internal reasoning.

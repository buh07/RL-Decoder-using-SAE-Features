# Mixed Hidden-State + SAE Trajectory Validation Protocol

## Objective
Test whether adding raw/projected hidden-state trajectory features provides independent faithfulness signal beyond SAE-only trajectory features.

Primary endpoint is `wrong_intermediate` AUROC under variant-robust constraints.

## Pre-registered Feature Blocks
- `B1` SAE trajectory features (3 metrics x 3 layers).
- `B2` Raw hidden trajectory features (same metric family/layers).
- `B3` Projected subspace trajectory features (same metric family/layers).

## Model Ladder
- `M1`: `B1` only (SAE baseline).
- `M2`: `B2+B3` only.
- `M3`: `B1+B2+B3` (mixed).
- `M4`: `B1 + residual(B2+B3 | B1)` to isolate incremental signal.

All models use the same elastic-net logistic family and identical splits.

## Variant-Robust Contract
- Train excludes unfaithful positives from:
  - `order_flip_only`
  - `answer_first_order_flip`
  - `reordered_steps`
- Train keeps all faithful rows.
- Test includes all variants and reports per-variant AUROC.
- Trace-group split only (no trace leakage).

## Uncertainty and Significance
- Single-split wrong_intermediate AUROC CI: trace-pair bootstrap (`n=1000` default).
- 5-fold trace-group CV with pooled OOF wrong_intermediate AUROC + bootstrap CI.
- Incremental mixed value:
  - `delta_auroc = AUROC(M3) - AUROC(M1)`
  - bootstrap CI on delta
  - permutation delta test

## Practical and Statistical Criteria
`publishable_mixed_signal` requires all:
1. Leakage clean.
2. CV pooled `wrong_intermediate` AUROC > threshold (default `0.70`) with minimum valid folds.
3. Practical effect passes: `delta_auroc >= 0.03` and lower 95% CI > 0.
4. Fold stability: mixed gain positive in >=80% folds.
5. Collinearity diagnostics do not indicate pure redundancy.

Else: `mixed_redundant_or_insufficient`.

## Diagnostics (Mandatory)
- Pairwise feature correlation matrix.
- VIF per feature + block summaries.
- Coefficient stability across CV folds.
- Permutation importance by block on `wrong_intermediate`.
- Falsification: shuffle B2/B3 within trace in train and re-fit M3.

## Implemented CLIs
- `phase7/mixed_trajectory_feature_builder.py`
- `phase7/evaluate_mixed_trajectory_validation.py`
- `experiments/run_phase7_mixed_trajectory_validation.sh`

## Current Blocking Risk for Qwen
This protocol requires model-matched SAE checkpoints and activation stats. If Qwen SAE assets are missing, the run should be blocked and reported rather than downgraded silently.

# Mixed Hidden-State + SAE Trajectory Validation

- Run tag: `mixed_result_top50_seed20260311`
- Final decision: `mixed_redundant_or_insufficient`
- Delta AUROC M3-M1 (wrong_intermediate): `-0.125`
- Delta CI95: `-0.5625` .. `0.0`
- Practical effect pass: `False`
- CV pooled M3 wrong_intermediate AUROC: `0.6131687242798354`
- CV pooled M3 CI95: `0.51440329218107` .. `0.6899862825788751`

## Gate Checks

- `leakage_clean`: `True`
- `cv_wrong_intermediate_threshold_pass`: `False`
- `stable_gain_pass`: `True`
- `practical_effect_pass`: `False`
- `collinearity_non_redundant_pass`: `False`


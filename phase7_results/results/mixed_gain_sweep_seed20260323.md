# Mixed Hidden-State + SAE Trajectory Validation

- Run tag: `mixed_gain_sweep_seed20260323`
- Final decision: `mixed_redundant_or_insufficient`
- Delta AUROC M3-M1 (wrong_intermediate): `-0.020408163265306145`
- Delta CI95: `-0.16326530612244905` .. `0.1428571428571429`
- Practical effect pass: `False`
- CV pooled M3 wrong_intermediate AUROC: `0.6351165980795611`
- CV pooled M3 CI95: `0.53360768175583` .. `0.7325102880658436`

## Gate Checks

- `leakage_clean`: `True`
- `cv_wrong_intermediate_threshold_pass`: `False`
- `stable_gain_pass`: `False`
- `practical_effect_pass`: `False`
- `collinearity_non_redundant_pass`: `True`


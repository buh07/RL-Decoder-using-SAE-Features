# Mixed Hidden-State + SAE Trajectory Validation

- Run tag: `mixed_divergent_top50_s20260307`
- Final decision: `mixed_redundant_or_insufficient`
- Delta AUROC M3-M1 (wrong_intermediate): `-0.25`
- Delta CI95: `-1.0` .. `0.0`
- Practical effect pass: `False`
- CV pooled M3 wrong_intermediate AUROC: `0.5679012345679012`
- CV pooled M3 CI95: `0.4787379972565158` .. `0.663923182441701`

## Gate Checks

- `leakage_clean`: `True`
- `cv_wrong_intermediate_threshold_pass`: `False`
- `stable_gain_pass`: `False`
- `practical_effect_pass`: `False`
- `collinearity_non_redundant_pass`: `True`


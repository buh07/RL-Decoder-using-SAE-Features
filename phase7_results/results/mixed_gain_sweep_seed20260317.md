# Mixed Hidden-State + SAE Trajectory Validation

- Run tag: `mixed_gain_sweep_seed20260317`
- Final decision: `mixed_redundant_or_insufficient`
- Delta AUROC M3-M1 (wrong_intermediate): `-0.05555555555555558`
- Delta CI95: `-0.16666666666666663` .. `1.1102230246251565e-16`
- Practical effect pass: `False`
- CV pooled M3 wrong_intermediate AUROC: `0.6282578875171467`
- CV pooled M3 CI95: `0.5363511659807956` .. `0.7270233196159122`

## Gate Checks

- `leakage_clean`: `True`
- `cv_wrong_intermediate_threshold_pass`: `False`
- `stable_gain_pass`: `False`
- `practical_effect_pass`: `False`
- `collinearity_non_redundant_pass`: `True`


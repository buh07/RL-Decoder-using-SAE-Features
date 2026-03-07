# Mixed Hidden-State + SAE Trajectory Validation

- Run tag: `mixed_result_top50_seed20260317`
- Final decision: `mixed_redundant_or_insufficient`
- Delta AUROC M3-M1 (wrong_intermediate): `0.05555555555555547`
- Delta CI95: `-0.08333333333333331` .. `0.19444444444444442`
- Practical effect pass: `False`
- CV pooled M3 wrong_intermediate AUROC: `0.6460905349794239`
- CV pooled M3 CI95: `0.5432098765432098` .. `0.7407407407407408`

## Gate Checks

- `leakage_clean`: `True`
- `cv_wrong_intermediate_threshold_pass`: `False`
- `stable_gain_pass`: `True`
- `practical_effect_pass`: `False`
- `collinearity_non_redundant_pass`: `True`


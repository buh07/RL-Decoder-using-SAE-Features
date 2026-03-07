# Phase 7-SAE Path C (Trajectory Ensemble)

- Run tag: `phase7_sae_trajectory_pathc_robust_20260307_000059_phase7_sae_trajectory_pathc_robust`
- Layers: `4,7,22`
- Rows after join: `2510`
- Test AUROC (probe): `0.7163259259259259`
- Test AUROC (baseline): `0.4786567901234568`
- Probe wrong_intermediate AUROC: `0.6944444444444444`
- Probe order_flip_only AUROC: `0.46258503401360546`
- Gate threshold (wrong_intermediate): `0.7`
- Robust gate pass: `False`

## Top Coefficients

- `layer7:feature_variance_coherence`: `-1.5821318626403809`
- `layer22:feature_variance_coherence`: `1.0721279382705688`
- `layer4:cosine_smoothness`: `-0.828434944152832`
- `layer4:feature_variance_coherence`: `0.7379494905471802`
- `layer7:cosine_smoothness`: `0.5891970992088318`
- `layer7:magnitude_monotonicity_coherence`: `-0.5180761218070984`
- `layer22:cosine_smoothness`: `-0.43218520283699036`
- `layer4:magnitude_monotonicity_coherence`: `-0.13810241222381592`
- `layer22:magnitude_monotonicity_coherence`: `-0.12918391823768616`


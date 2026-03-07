# Phase 7-SAE Trajectory Coherence

- Run tag: `phase7_sae_trajectory_20260306_234123_phase7_sae_trajectory_pathb_eq_pre_result_150`
- Layers: `4,7,22`
- Source records: `phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json`

## Best Overall

- Best layer-metric: `layer7:feature_variance_coherence`
- Best AUROC (unfaithful positive): `0.6830551896001651`

## Confound Check (`wrong_intermediate` vs `order_flip_only`)

- `cosine_smoothness`: wrong_intermediate=`0.5158916194076875`, order_flip_only=`0.5738357843137255`, delta=`-0.05794416490603804`
- `feature_variance_coherence`: wrong_intermediate=`0.5716194076874607`, order_flip_only=`0.7534421856978085`, delta=`-0.18182277801034785`
- `magnitude_monotonicity_coherence`: wrong_intermediate=`0.501625708884688`, order_flip_only=`0.21921856978085352`, delta=`0.2824071391038345`

## Per Layer / Metric AUROC

| Layer | Cosine AUROC | Variance AUROC | Monotonicity AUROC |
|---:|---:|---:|---:|
| 4 | 0.5725807526864652 | 0.6219425088490659 | 0.5137004174536912 |
| 7 | 0.601772670275075 | 0.6830551896001651 | 0.47274360724432946 |
| 22 | 0.4574263900573007 | 0.4177622577419406 | 0.5835805780860621 |

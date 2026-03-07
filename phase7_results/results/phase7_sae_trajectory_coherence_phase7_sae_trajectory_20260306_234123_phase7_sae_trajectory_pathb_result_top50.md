# Phase 7-SAE Trajectory Coherence

- Run tag: `phase7_sae_trajectory_20260306_234123_phase7_sae_trajectory_pathb_result_top50`
- Layers: `4,7,22`
- Source records: `phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json`

## Best Overall

- Best layer-metric: `layer7:feature_variance_coherence`
- Best AUROC (unfaithful positive): `0.6662484722464723`

## Confound Check (`wrong_intermediate` vs `order_flip_only`)

- `cosine_smoothness`: wrong_intermediate=`0.5004914933837429`, order_flip_only=`0.5273752883506344`, delta=`-0.026883794966891528`
- `feature_variance_coherence`: wrong_intermediate=`0.5170510396975425`, order_flip_only=`0.7304462226066898`, delta=`-0.21339518290914727`
- `magnitude_monotonicity_coherence`: wrong_intermediate=`0.5128166351606805`, order_flip_only=`0.18757208765859287`, delta=`0.32524454750208764`

## Per Layer / Metric AUROC

| Layer | Cosine AUROC | Variance AUROC | Monotonicity AUROC |
|---:|---:|---:|---:|
| 4 | 0.4715614037872415 | 0.5392346153235663 | 0.49243218361613306 |
| 7 | 0.5718569546515134 | 0.6662484722464723 | 0.47274360724432946 |
| 22 | 0.4514188663672005 | 0.40459992698528596 | 0.5835805780860621 |

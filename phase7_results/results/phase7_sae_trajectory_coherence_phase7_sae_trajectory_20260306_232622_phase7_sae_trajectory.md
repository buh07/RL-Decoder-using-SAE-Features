# Phase 7-SAE Trajectory Coherence

- Run tag: `phase7_sae_trajectory_20260306_232622_phase7_sae_trajectory`
- Layers: `4,7,22`
- Source records: `phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json`

## Best Overall

- Best layer-metric: `layer4:cosine_smoothness`
- Best AUROC (unfaithful positive): `0.6967489404930081`

## Confound Check (`wrong_intermediate` vs `order_flip_only`)

- `cosine_smoothness`: wrong_intermediate=`0.5264524259609326`, order_flip_only=`0.6568807670126874`, delta=`-0.13042834105175483`
- `feature_variance_coherence`: wrong_intermediate=`0.5702583490863263`, order_flip_only=`0.6559256055363322`, delta=`-0.08566725645000584`
- `magnitude_monotonicity_coherence`: wrong_intermediate=`0.47117832388153746`, order_flip_only=`0.2986952133794694`, delta=`0.17248311050206805`

## Per Layer / Metric AUROC

| Layer | Cosine AUROC | Variance AUROC | Monotonicity AUROC |
|---:|---:|---:|---:|
| 4 | 0.6967489404930081 | 0.6918296534975635 | 0.4935997841304106 |
| 7 | 0.6610307772892494 | 0.6348140505706259 | 0.5683674862303773 |
| 22 | 0.47299630164600565 | 0.42674751194425486 | 0.49693115982285996 |

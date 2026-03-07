# Phase 7-SAE Trajectory Coherence

- Run tag: `phase7_sae_trajectory_20260306_234123_phase7_sae_trajectory_pathb_divergent_top50`
- Layers: `4,7,22`
- Source records: `phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json`

## Best Overall

- Best layer-metric: `layer7:feature_variance_coherence`
- Best AUROC (unfaithful positive): `0.6606530055078491`

## Confound Check (`wrong_intermediate` vs `order_flip_only`)

- `cosine_smoothness`: wrong_intermediate=`0.4339256458727158`, order_flip_only=`0.5726102941176471`, delta=`-0.13868464824493126`
- `feature_variance_coherence`: wrong_intermediate=`0.4466792690611216`, order_flip_only=`0.8160142733564014`, delta=`-0.3693350042952798`
- `magnitude_monotonicity_coherence`: wrong_intermediate=`0.5196723377441713`, order_flip_only=`0.6729202710495964`, delta=`-0.15324793330542508`

## Per Layer / Metric AUROC

| Layer | Cosine AUROC | Variance AUROC | Monotonicity AUROC |
|---:|---:|---:|---:|
| 4 | 0.30641799336518466 | 0.43499182552657895 | 0.653691846161172 |
| 7 | 0.6447351629339216 | 0.6606530055078491 | 0.5954292789003349 |
| 22 | 0.4584898652402343 | 0.47065856097522263 | 0.48682719321915524 |

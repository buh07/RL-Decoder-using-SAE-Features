# Phase 7-SAE Faithfulness Discrimination

- Run tag: `phase7_sae_smoke`
- Source control records: `phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json`
- Layers analyzed: `0`

## Aggregate Summary

- Best sparse-probe layer AUROC: `None` @ `None`
- Layers with probe AUROC > 0.60: `0`
- Feature-L2 signal layers (margin>0 and exceedance<0.05): `0`
- Overlap-present layers (top50 abs-d vs Phase4 top50 eq): `0`

## Channel Pass/Fail

- Feature-space L2 channel pass: `False`
- Per-feature divergence channel pass: `True`
- Sparse probe channel pass: `False`
- Phase4 overlap channel pass: `False`

## By Layer

| Layer | Probe AUROC | L2 Margin | L2 Exceedance | Max |d| | Overlap@50 |
|---:|---:|---:|---:|---:|---:|
| 0 | None | 0.0 | 1.0 | 0.8398829698562622 | 0 |

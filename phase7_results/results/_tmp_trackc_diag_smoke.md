# Track C Quick Direction Report

Run tag: `tmp_smoke`

## Key Signals
- Anchor winner: `template_first` by `max_mean_causal_auroc`
- E1 orientation-misaligned flag: `False`
- E2 pairwise-signal-weak flag: `True`
- E5 causal-harming flag: `True`

## Ranked Recommendations
- [E2] within_trace_delta_reformulation: Within-trace faithful-vs-variant causal deltas remain weak.
- [E5] composite_causal_weight_diagnostic: Causal channel frequently harms composite AUROC.
- [E4] anchor_priority_choice: Use template_first per deterministic anchor ablation rule (max_mean_causal_auroc).

## E3 Mediation Alignment
- Defined: `False`

## Per-Input Rows
- anchor=template_first var=subresult_value causal_auroc=0.14744816326530616 composite_auroc=0.48742326530612246 coverage=0.35287081339712917 orientation_gain=-0.704618775510204 pair_delta_pos_frac=0.10685714285714286
- anchor=template_first var=magnitude_bucket causal_auroc=0.12819020408163265 composite_auroc=0.35995224489795924 coverage=0.35287081339712917 orientation_gain=-0.743694693877551 pair_delta_pos_frac=0.053714285714285714
- anchor=equation_first var=operator causal_auroc=0.14337183673469386 composite_auroc=0.37175142857142857 coverage=0.35287081339712917 orientation_gain=-0.7132677551020408 pair_delta_pos_frac=0.07057142857142858
- anchor=equation_first var=sign causal_auroc=0.1287236734693877 composite_auroc=0.3495681632653061 coverage=0.35287081339712917 orientation_gain=-0.742553469387755 pair_delta_pos_frac=0.027714285714285716

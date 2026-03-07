# Track C Quick Direction Report

Run tag: `trackc_direction_quick_20260305_210453_trackc_direction_quick`

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
- anchor=template_first var=subresult_value causal_auroc=0.11768176020408164 composite_auroc=0.43094706632653057 coverage=0.15598885793871867 orientation_gain=-0.7648915816326529 pair_delta_pos_frac=0.033928571428571426
- anchor=template_first var=magnitude_bucket causal_auroc=0.1129623724489796 composite_auroc=0.3209980867346939 coverage=0.15598885793871867 orientation_gain=-0.7741071428571429 pair_delta_pos_frac=0.0125
- anchor=equation_first var=operator causal_auroc=0.11353635204081634 composite_auroc=0.32857142857142857 coverage=0.15598885793871867 orientation_gain=-0.7729272959183674 pair_delta_pos_frac=0.02857142857142857
- anchor=equation_first var=sign causal_auroc=0.11293048469387756 composite_auroc=0.31366390306122444 coverage=0.15598885793871867 orientation_gain=-0.7741390306122449 pair_delta_pos_frac=0.007142857142857143

# Phase 7 Decoder Quality Diagnostic

- Dataset: `phase7_results/dataset/gsm8k_step_traces_train.pt` (split `train`, n=45729)
- Operator majority fraction: `0.4720`
- Label sanity mismatch rate: `0.0000`
- Token-position ablation status: `blocked_missing_pre_eq_hidden_states`
- Best test operator_acc config: `state_raw_multi_l4_l5_l6_l7_d2tier2`
  - operator_acc=0.5118501054129747, step_type_acc=0.5380320136016645, magnitude_acc=0.9444099759875079, sign_acc=0.9996277444681262
- Decoder quality gate pass (operate-only): `False` (operate-only-operator/magnitude/sign thresholds not jointly satisfied)
- Legacy all-step gate pass: `False` (all-step-operator/magnitude/sign thresholds not jointly satisfied)

## Recommendations
- Run D1 Tier-1 retraining profile before Phase 7 re-evaluation.
- Operator unknown-class contamination is significant; prioritize operate-only operator metrics.
- Collect multi-position hidden states to run D0.4 token-position ablation.

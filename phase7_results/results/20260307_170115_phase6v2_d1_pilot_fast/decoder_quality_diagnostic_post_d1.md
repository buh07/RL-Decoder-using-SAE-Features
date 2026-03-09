# Phase 7 Decoder Quality Diagnostic

- Dataset: `phase7_results/dataset/gsm8k_step_traces_train.pt` (split `train`, n=45729)
- Operator majority fraction: `0.4720`
- Label sanity mismatch rate: `0.0000`
- Token-position ablation status: `blocked_missing_pre_eq_hidden_states`
- Best test operator_acc config: `state_hybrid_multi_l4_l5_l6_l7_d1tier1`
  - operator_acc=0.5155726516345902, step_type_acc=0.5380320136016645, magnitude_acc=0.9427968725434515, sign_acc=0.9996277444681262
- Decoder quality gate pass: `False` (operator/magnitude/sign thresholds not jointly satisfied)

## Recommendations
- Run D1 Tier-1 retraining profile before Phase 7 re-evaluation.
- Collect multi-position hidden states to run D0.4 token-position ablation.

# Phase 7 Decoder Quality Diagnostic

- Dataset: `phase7_results/runs/20260307_192338_phase6v2_d3_operator_fix/dataset/gsm8k_step_traces_test.pt` (split `test`, n=8084)
- Operator majority fraction: `0.4699`
- Label sanity mismatch rate: `0.0000`
- Token-position ablation status: `blocked_missing_pre_eq_hidden_states`
- Best test operator_acc config: `state_sae_multi_l4_l5_l6_l7_d3known`
  - operator_acc=0.5019792182530465, operator_acc_operate_only=0.947024504084014, operator_acc_operate_known_only=0.947024504084014, step_type_acc=0.5393369619295422, magnitude_acc=0.9017812963879268, sign_acc=0.9996288965858486
- Decoder quality gate pass (operate-known): `True` (all thresholds met)
- Secondary operate-only gate pass: `True` (all thresholds met)
- Legacy all-step gate pass: `False` (all-step-operator/magnitude/sign thresholds not jointly satisfied)

## Recommendations
- Operator unknown-class contamination is significant; prioritize operate-only operator metrics.
- Collect multi-position hidden states to run D0.4 token-position ablation.

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
PY="${PYTHON:-.venv/bin/python3}"

usage() {
  cat <<'USAGE'
usage: experiments/run_qwen_pathc_single_layer_staged.sh {launch|worker-qwen-perm|worker-qwen-followup|worker-gpt2-followup|coordinator}
USAGE
}

require_env() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
}

json_parseable() {
  local p="$1"
  "$PY" - <<PY
import json,sys
try:
    json.load(open(${p@Q}))
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

wait_for_file_with_session() {
  local done_file="$1"
  local session="$2"
  local timeout_sec="${3:-86400}"
  local start now elapsed
  start="$(date +%s)"
  while [[ ! -f "$done_file" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      for _ in {1..6}; do
        sleep 5
        [[ -f "$done_file" ]] && return 0
      done
      echo "session exited before completion: $session (missing $done_file)" >&2
      return 1
    fi
    now="$(date +%s)"
    elapsed="$(( now - start ))"
    if (( elapsed > timeout_sec )); then
      echo "timeout waiting for $done_file (session=$session)" >&2
      return 1
    fi
    sleep 10
  done
}

run_worker_qwen_perm() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  mapfile -t PARTIALS < "$BASE/meta/qwen_partials.list"
  [[ "${#PARTIALS[@]}" -gt 0 ]] || { echo "No Qwen partials found" >&2; exit 1; }
  {
    echo "[$(date -Is)] qwen permutation worker start"
    "$PY" phase7/stress_test_pathc_probe.py \
      --task permutation \
      --partials "${PARTIALS[@]}" \
      --run-id "$RUN_ID" \
      --run-tag "$RUN_TAG" \
      --train-exclude-variants "$TRAIN_EXCLUDE_VARIANTS" \
      --trace-test-fraction "$TRACE_TEST_FRACTION" \
      --trace-split-seed "$TRACE_SPLIT_SEED" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --weight-decay-base "$WEIGHT_DECAY_BASE" \
      --device "$DEVICE" \
      --feature-mode "$FEATURE_MODE" \
      --single-layer-source "$SINGLE_LAYER_SOURCE" \
      --single-layer-id "$SINGLE_LAYER_ID" \
      --permutation-runs "$PERMUTATION_RUNS" \
      --permutation-seed "$PERMUTATION_SEED" \
      --output-json "$QWEN_PERM_JSON"
    echo "[$(date -Is)] qwen permutation worker done"
  } >>"$BASE/logs/qwen_permutation.log" 2>&1
  touch "$BASE/state/qwen_permutation.done"
}

run_worker_qwen_followup() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  mapfile -t PARTIALS < "$BASE/meta/qwen_partials.list"
  [[ "${#PARTIALS[@]}" -gt 0 ]] || { echo "No Qwen partials found" >&2; exit 1; }
  {
    echo "[$(date -Is)] qwen followup worker start"
    "$PY" phase7/stress_test_pathc_probe.py \
      --task ablation_reg \
      --partials "${PARTIALS[@]}" \
      --run-id "$RUN_ID" \
      --run-tag "$RUN_TAG" \
      --train-exclude-variants "$TRAIN_EXCLUDE_VARIANTS" \
      --trace-test-fraction "$TRACE_TEST_FRACTION" \
      --trace-split-seed "$TRACE_SPLIT_SEED" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --weight-decay-base "$WEIGHT_DECAY_BASE" \
      --weight-decay-values "$WEIGHT_DECAY_VALUES" \
      --single-layer-mode "$SINGLE_LAYER_MODE" \
      --feature-mode "$FEATURE_MODE" \
      --single-layer-source "$SINGLE_LAYER_SOURCE" \
      --single-layer-id "$SINGLE_LAYER_ID" \
      --device "$DEVICE" \
      --output-json "$QWEN_ABLATION_REG_JSON"

    "$PY" phase7/stress_test_pathc_probe.py \
      --task multiseed \
      --partials "${PARTIALS[@]}" \
      --run-id "$RUN_ID" \
      --run-tag "$RUN_TAG" \
      --train-exclude-variants "$TRAIN_EXCLUDE_VARIANTS" \
      --trace-test-fraction "$TRACE_TEST_FRACTION" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --weight-decay-base "$WEIGHT_DECAY_BASE" \
      --device "$DEVICE" \
      --feature-mode "$FEATURE_MODE" \
      --single-layer-source "$SINGLE_LAYER_SOURCE" \
      --single-layer-id "$SINGLE_LAYER_ID" \
      --multi-seed-values "$MULTI_SEED_VALUES" \
      --wrong-intermediate-bootstrap-n "$WRONG_INTERMEDIATE_BOOTSTRAP_N" \
      --wrong-intermediate-bootstrap-seed "$WRONG_INTERMEDIATE_BOOTSTRAP_SEED" \
      --output-json "$QWEN_MULTI_JSON"

    "$PY" - <<PY
import json
abr = json.load(open(${QWEN_ABLATION_REG_JSON@Q}))
ms = json.load(open(${QWEN_MULTI_JSON@Q}))
out = {
  "status": "ok",
  "regularization_pass": bool(abr.get("regularization_pass") is True),
  "multiseed_pass": bool(ms.get("pass") is True),
  "qwen_followup_pass": bool((abr.get("regularization_pass") is True) and (ms.get("pass") is True)),
  "selected_layer": ((ms.get("feature_selection") or {}).get("selected_layer")),
  "feature_mode": ((ms.get("feature_selection") or {}).get("feature_mode")),
}
json.dump(out, open(${QWEN_FOLLOWUP_SUMMARY_JSON@Q}, "w"), indent=2)
print("saved", ${QWEN_FOLLOWUP_SUMMARY_JSON@Q})
PY
    echo "[$(date -Is)] qwen followup worker done"
  } >>"$BASE/logs/qwen_followup.log" 2>&1
  touch "$BASE/state/qwen_followup.done"
}

run_worker_gpt2_followup() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  mapfile -t PARTIALS < "$BASE/meta/gpt2_partials.list"
  [[ "${#PARTIALS[@]}" -gt 0 ]] || { echo "No GPT-2 partials found" >&2; exit 1; }
  {
    echo "[$(date -Is)] gpt2 followup worker start"
    "$PY" phase7/stress_test_pathc_probe.py \
      --task ablation_reg \
      --partials "${PARTIALS[@]}" \
      --run-id "$RUN_ID" \
      --run-tag "$RUN_TAG" \
      --train-exclude-variants "$TRAIN_EXCLUDE_VARIANTS" \
      --trace-test-fraction "$TRACE_TEST_FRACTION" \
      --trace-split-seed "$TRACE_SPLIT_SEED" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --weight-decay-base "$WEIGHT_DECAY_BASE" \
      --weight-decay-values "$WEIGHT_DECAY_VALUES" \
      --single-layer-mode "$SINGLE_LAYER_MODE" \
      --feature-mode "$FEATURE_MODE" \
      --single-layer-source "$SINGLE_LAYER_SOURCE" \
      --single-layer-id "$SINGLE_LAYER_ID" \
      --device "$DEVICE" \
      --output-json "$GPT2_ABLATION_REG_JSON"

    "$PY" phase7/stress_test_pathc_probe.py \
      --task multiseed \
      --partials "${PARTIALS[@]}" \
      --run-id "$RUN_ID" \
      --run-tag "$RUN_TAG" \
      --train-exclude-variants "$TRAIN_EXCLUDE_VARIANTS" \
      --trace-test-fraction "$TRACE_TEST_FRACTION" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --weight-decay-base "$WEIGHT_DECAY_BASE" \
      --device "$DEVICE" \
      --feature-mode "$FEATURE_MODE" \
      --single-layer-source "$SINGLE_LAYER_SOURCE" \
      --single-layer-id "$SINGLE_LAYER_ID" \
      --multi-seed-values "$MULTI_SEED_VALUES" \
      --wrong-intermediate-bootstrap-n "$WRONG_INTERMEDIATE_BOOTSTRAP_N" \
      --wrong-intermediate-bootstrap-seed "$WRONG_INTERMEDIATE_BOOTSTRAP_SEED" \
      --output-json "$GPT2_MULTI_JSON"
    echo "[$(date -Is)] gpt2 followup worker done"
  } >>"$BASE/logs/gpt2_followup.log" 2>&1
  touch "$BASE/state/gpt2_followup.done"
}

run_coordinator() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  {
    echo "[$(date -Is)] coordinator start run_id=$RUN_ID"

    wait_for_file_with_session "$BASE/state/qwen_permutation.done" "$QPERM_SESSION" 86400
    [[ -f "$QWEN_PERM_JSON" ]] || { echo "missing qwen permutation json" >&2; exit 1; }
    json_parseable "$QWEN_PERM_JSON" || { echo "bad qwen permutation json" >&2; exit 1; }

    read -r qwen_perm_gate_pass qwen_perm_legacy_pass qwen_perm_primary_pass qwen_perm_pvalue <<EOF
$("$PY" - <<PY
import json
d=json.load(open(${QWEN_PERM_JSON@Q}))
legacy = bool(d.get("legacy_strict_pass") is True or d.get("pass") is True)
primary = bool(d.get("p_value_primary_pass") is True)
gate = bool(primary or legacy)
pval = d.get("empirical_p_value")
if isinstance(pval, (int, float)):
    pval = f"{float(pval):.8f}"
else:
    pval = "null"
print(("true" if gate else "false"), ("true" if legacy else "false"), ("true" if primary else "false"), pval)
PY
)"
EOF

    qwen_followup_pass="false"
    gpt2_replication_ran="false"
    gpt2_delta="null"
    gpt2_strength="not_run"
    blocked_reason=""

    if [[ "$qwen_perm_gate_pass" == "true" ]]; then
      tmux has-session -t "$QFOLLOW_SESSION" 2>/dev/null || tmux new-session -d -s "$QFOLLOW_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' worker-qwen-followup"
      wait_for_file_with_session "$BASE/state/qwen_followup.done" "$QFOLLOW_SESSION" 86400
      [[ -f "$QWEN_ABLATION_REG_JSON" ]] || { echo "missing qwen ablation/reg json" >&2; exit 1; }
      [[ -f "$QWEN_MULTI_JSON" ]] || { echo "missing qwen multiseed json" >&2; exit 1; }
      json_parseable "$QWEN_ABLATION_REG_JSON" || { echo "bad qwen ablation/reg json" >&2; exit 1; }
      json_parseable "$QWEN_MULTI_JSON" || { echo "bad qwen multiseed json" >&2; exit 1; }

      qwen_followup_pass="$("$PY" - <<PY
import json
a=json.load(open(${QWEN_ABLATION_REG_JSON@Q}))
m=json.load(open(${QWEN_MULTI_JSON@Q}))
ok = bool(a.get("regularization_pass") is True) and bool(m.get("pass") is True)
print("true" if ok else "false")
PY
)"

      if [[ "$qwen_followup_pass" == "true" ]]; then
        tmux has-session -t "$GFOLLOW_SESSION" 2>/dev/null || tmux new-session -d -s "$GFOLLOW_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' worker-gpt2-followup"
        wait_for_file_with_session "$BASE/state/gpt2_followup.done" "$GFOLLOW_SESSION" 86400
        [[ -f "$GPT2_ABLATION_REG_JSON" ]] || { echo "missing gpt2 ablation/reg json" >&2; exit 1; }
        [[ -f "$GPT2_MULTI_JSON" ]] || { echo "missing gpt2 multiseed json" >&2; exit 1; }
        json_parseable "$GPT2_ABLATION_REG_JSON" || { echo "bad gpt2 ablation/reg json" >&2; exit 1; }
        json_parseable "$GPT2_MULTI_JSON" || { echo "bad gpt2 multiseed json" >&2; exit 1; }
        gpt2_replication_ran="true"
        read -r gpt2_delta gpt2_strength <<EOF
$("$PY" - <<PY
import json
baseline = float(${GPT2_BASELINE_AUROC@Q})
ms=json.load(open(${GPT2_MULTI_JSON@Q}))
auc = (ms.get("pooled_wrong_intermediate_auroc"))
if isinstance(auc, (int,float)):
    delta=float(auc)-baseline
    if delta >= 0.05:
        s="strong"
    elif delta >= 0.02:
        s="moderate"
    else:
        s="weak_or_none"
    print(f"{delta:.6f} {s}")
else:
    print("null undefined")
PY
)
EOF
      else
        blocked_reason="qwen_followup_failed"
      fi
    else
      blocked_reason="qwen_permutation_failed"
    fi

    final_recommendation="hold_and_retest"
    if [[ "$qwen_perm_gate_pass" == "true" && "$qwen_followup_pass" == "true" ]]; then
      final_recommendation="proceed_to_G1"
    fi

    "$PY" - <<PY
import json
from datetime import datetime
from pathlib import Path

qperm=json.load(open(${QWEN_PERM_JSON@Q}))
qabr=json.load(open(${QWEN_ABLATION_REG_JSON@Q})) if ${QWEN_ABLATION_REG_JSON@Q} and __import__("os").path.exists(${QWEN_ABLATION_REG_JSON@Q}) else None
qms=json.load(open(${QWEN_MULTI_JSON@Q})) if ${QWEN_MULTI_JSON@Q} and __import__("os").path.exists(${QWEN_MULTI_JSON@Q}) else None
gabr=json.load(open(${GPT2_ABLATION_REG_JSON@Q})) if ${GPT2_ABLATION_REG_JSON@Q} and __import__("os").path.exists(${GPT2_ABLATION_REG_JSON@Q}) else None
gms=json.load(open(${GPT2_MULTI_JSON@Q})) if ${GPT2_MULTI_JSON@Q} and __import__("os").path.exists(${GPT2_MULTI_JSON@Q}) else None
pval_raw = str(${qwen_perm_pvalue@Q}).strip().strip('"').strip("'")
qperm_pval = None
if pval_raw and pval_raw.lower() not in {"null", "none"}:
    try:
        qperm_pval = float(pval_raw)
    except Exception:
        qperm_pval = None

out={
  "schema_version":"qwen_pathc_single_layer_staged_v1",
  "status":"ok",
  "run_id":${RUN_ID@Q},
  "run_tag":${RUN_TAG@Q},
  "feature_mode":${FEATURE_MODE@Q},
  "single_layer_source":${SINGLE_LAYER_SOURCE@Q},
  "single_layer_id":${SINGLE_LAYER_ID@Q},
  "qwen_permutation_gate_pass": (${qwen_perm_gate_pass@Q} == "true"),
  "qwen_permutation_legacy_pass": (${qwen_perm_legacy_pass@Q} == "true"),
  "qwen_permutation_pvalue_pass": (${qwen_perm_primary_pass@Q} == "true"),
  "qwen_permutation_empirical_p_value": qperm_pval,
  "qwen_permutation_pass": (${qwen_perm_gate_pass@Q} == "true"),
  "qwen_followup_pass": (${qwen_followup_pass@Q} == "true"),
  "gpt2_replication_ran": (${gpt2_replication_ran@Q} == "true"),
  "gpt2_wrong_intermediate_delta_vs_baseline": (None if ${gpt2_delta@Q} == "null" else float(${gpt2_delta@Q})),
  "gpt2_delta_strength": ${gpt2_strength@Q},
  "final_recommendation": ${final_recommendation@Q},
  "blocked_reason": ${blocked_reason@Q},
  "strict_acceptance": {
    "permutation": {"mean_range":[0.45,0.55], "p95_lt":0.60, "max_lt":0.70, "p_value_lt":0.01},
    "followup": {"wd_0.01_min":0.70, "wd_0.1_min":0.70, "wd_1.0_min":0.65, "multiseed_mean_min":0.70, "multiseed_std_max":0.08, "pooled_ci95_lower_min":0.65},
  },
  "artifacts": {
    "qwen_permutation_json": ${QWEN_PERM_JSON@Q},
    "qwen_ablation_reg_json": ${QWEN_ABLATION_REG_JSON@Q},
    "qwen_multiseed_json": ${QWEN_MULTI_JSON@Q},
    "gpt2_ablation_reg_json": ${GPT2_ABLATION_REG_JSON@Q},
    "gpt2_multiseed_json": ${GPT2_MULTI_JSON@Q},
  },
  "tests": {
    "qwen_permutation": qperm,
    "qwen_ablation_reg": qabr,
    "qwen_multiseed": qms,
    "gpt2_ablation_reg": gabr,
    "gpt2_multiseed": gms,
  },
  "timestamp": datetime.now().isoformat(),
}

json.dump(out, open(${OUT_JSON@Q},"w"), indent=2)

lines=[
  "# Qwen/GPT-2 Single-Layer Staged Summary",
  "",
  f"- Run id: {out['run_id']}",
  f"- Feature mode: {out['feature_mode']}",
  f"- Qwen permutation gate pass: {out['qwen_permutation_gate_pass']}",
  f"- Qwen permutation legacy pass: {out['qwen_permutation_legacy_pass']}",
  f"- Qwen permutation p-value pass: {out['qwen_permutation_pvalue_pass']}",
  f"- Qwen permutation empirical p-value: {out['qwen_permutation_empirical_p_value']}",
  f"- Qwen followup pass: {out['qwen_followup_pass']}",
  f"- GPT-2 replication ran: {out['gpt2_replication_ran']}",
  f"- GPT-2 delta vs baseline: {out['gpt2_wrong_intermediate_delta_vs_baseline']} ({out['gpt2_delta_strength']})",
  f"- Final recommendation: {out['final_recommendation']}",
  f"- Blocked reason: {out['blocked_reason']}",
]
open(${OUT_MD@Q},"w").write("\\n".join(lines)+"\\n")
print("saved", ${OUT_JSON@Q})
PY

    "$PY" - <<PY
import json
from datetime import datetime
pval_raw = str(${qwen_perm_pvalue@Q}).strip().strip('"').strip("'")
qperm_pval = None
if pval_raw and pval_raw.lower() not in {"null", "none"}:
    try:
        qperm_pval = float(pval_raw)
    except Exception:
        qperm_pval = None
reframe = {
  "schema_version":"trackc_arithmetic_significance_reframe_v1",
  "run_id": ${RUN_ID@Q},
  "run_tag": ${RUN_TAG@Q},
  "permutation_artifact": ${QWEN_PERM_JSON@Q},
  "legacy_strict_pass": (${qwen_perm_legacy_pass@Q} == "true"),
  "p_value_primary_pass": (${qwen_perm_primary_pass@Q} == "true"),
  "gate_pass_used_for_staging": (${qwen_perm_gate_pass@Q} == "true"),
  "empirical_p_value": qperm_pval,
  "interpretation": ("supportive_significance" if (${qwen_perm_primary_pass@Q} == "true") else "not_significant"),
  "timestamp": datetime.now().isoformat(),
}
json.dump(reframe, open(${ARITH_REFRAME_JSON@Q}, "w"), indent=2)
print("saved", ${ARITH_REFRAME_JSON@Q})
PY

    json_parseable "$OUT_JSON" || { echo "bad staged final json" >&2; exit 1; }
    json_parseable "$ARITH_REFRAME_JSON" || { echo "bad arithmetic reframe json" >&2; exit 1; }
    touch "$BASE/state/pipeline.done"
    echo "[$(date -Is)] coordinator done"
    echo "final: $OUT_JSON"
  } >>"$BASE/logs/coordinator.log" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen_pathc_single_layer_staged}"
    RUN_TAG="${RUN_TAG:-qwen_pathc_single_layer_staged_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"

    QWEN_SOURCE_RUN_ID="${QWEN_SOURCE_RUN_ID:-20260308_165109_phase7_qwen_trackc_upgrade_qwpathc_robust_full_core}"
    QWEN_PARTIAL_GLOB="${QWEN_PARTIAL_GLOB:-phase7_results/results/phase7_sae_trajectory_coherence_phase7_sae_trajectory_${QWEN_SOURCE_RUN_ID}_layer*.json}"
    GPT2_SOURCE_RUN_ID="${GPT2_SOURCE_RUN_ID:-20260308_001011_phase7_sae_d3_redo_pathc_robust_core}"
    GPT2_PARTIAL_GLOB="${GPT2_PARTIAL_GLOB:-phase7_results/results/phase7_sae_trajectory_coherence_phase7_sae_trajectory_${GPT2_SOURCE_RUN_ID}_layer*.json}"

    TRAIN_EXCLUDE_VARIANTS="${TRAIN_EXCLUDE_VARIANTS:-order_flip_only,answer_first_order_flip,reordered_steps}"
    TRACE_TEST_FRACTION="${TRACE_TEST_FRACTION:-0.20}"
    TRACE_SPLIT_SEED="${TRACE_SPLIT_SEED:-20260306}"
    EPOCHS="${EPOCHS:-500}"
    LR="${LR:-0.03}"
    WEIGHT_DECAY_BASE="${WEIGHT_DECAY_BASE:-0.0001}"
    DEVICE="${DEVICE:-cpu}"
    FEATURE_MODE="${FEATURE_MODE:-single_best_layer}"
    SINGLE_LAYER_SOURCE="${SINGLE_LAYER_SOURCE:-auto_best}"
    SINGLE_LAYER_ID="${SINGLE_LAYER_ID:-0}"
    SINGLE_LAYER_MODE="${SINGLE_LAYER_MODE:-auto_best}"

    PERMUTATION_RUNS="${PERMUTATION_RUNS:-100}"
    PERMUTATION_SEED="${PERMUTATION_SEED:-20260308}"
    WEIGHT_DECAY_VALUES="${WEIGHT_DECAY_VALUES:-0.0001,0.01,0.1,1.0}"
    MULTI_SEED_VALUES="${MULTI_SEED_VALUES:-20260307,20260308,20260309,20260310,20260311,20260312,20260313,20260314,20260315,20260316}"
    WRONG_INTERMEDIATE_BOOTSTRAP_N="${WRONG_INTERMEDIATE_BOOTSTRAP_N:-1000}"
    WRONG_INTERMEDIATE_BOOTSTRAP_SEED="${WRONG_INTERMEDIATE_BOOTSTRAP_SEED:-20260307}"
    GPT2_BASELINE_AUROC="${GPT2_BASELINE_AUROC:-0.665}"

    QWEN_PERM_JSON="${QWEN_PERM_JSON:-$BASE/meta/qwen_permutation.json}"
    QWEN_ABLATION_REG_JSON="${QWEN_ABLATION_REG_JSON:-$BASE/meta/qwen_ablation_reg.json}"
    QWEN_MULTI_JSON="${QWEN_MULTI_JSON:-$BASE/meta/qwen_multiseed.json}"
    QWEN_FOLLOWUP_SUMMARY_JSON="${QWEN_FOLLOWUP_SUMMARY_JSON:-$BASE/meta/qwen_followup_summary.json}"
    GPT2_ABLATION_REG_JSON="${GPT2_ABLATION_REG_JSON:-$BASE/meta/gpt2_ablation_reg.json}"
    GPT2_MULTI_JSON="${GPT2_MULTI_JSON:-$BASE/meta/gpt2_multiseed.json}"
    ARITH_REFRAME_JSON="${ARITH_REFRAME_JSON:-phase7_results/results/trackc_arithmetic_significance_reframe_${RUN_ID}.json}"
    OUT_JSON="${OUT_JSON:-phase7_results/results/qwen_pathc_single_layer_staged_${RUN_ID}.json}"
    OUT_MD="${OUT_MD:-phase7_results/results/qwen_pathc_single_layer_staged_${RUN_ID}.md}"

    QPERM_SESSION="p7sl_qperm_${RUN_ID}"
    QFOLLOW_SESSION="p7sl_qfollow_${RUN_ID}"
    GFOLLOW_SESSION="p7sl_gfollow_${RUN_ID}"
    COORD_SESSION="p7sl_coord_${RUN_ID}"

    mkdir -p "$BASE/logs" "$BASE/meta" "$BASE/state"
    rm -f "$BASE/state"/*.done

    mapfile -t QWEN_PARTIALS < <(ls -1 $QWEN_PARTIAL_GLOB 2>/dev/null | sort)
    mapfile -t GPT2_PARTIALS < <(ls -1 $GPT2_PARTIAL_GLOB 2>/dev/null | sort)
    [[ "${#QWEN_PARTIALS[@]}" -gt 0 ]] || { echo "No Qwen partials found for glob: $QWEN_PARTIAL_GLOB" >&2; exit 1; }
    [[ "${#GPT2_PARTIALS[@]}" -gt 0 ]] || { echo "No GPT-2 partials found for glob: $GPT2_PARTIAL_GLOB" >&2; exit 1; }

    printf '%s\n' "${QWEN_PARTIALS[@]}" > "$BASE/meta/qwen_partials.list"
    printf '%s\n' "${GPT2_PARTIALS[@]}" > "$BASE/meta/gpt2_partials.list"

    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
QWEN_SOURCE_RUN_ID=$QWEN_SOURCE_RUN_ID
QWEN_PARTIAL_GLOB=$QWEN_PARTIAL_GLOB
GPT2_SOURCE_RUN_ID=$GPT2_SOURCE_RUN_ID
GPT2_PARTIAL_GLOB=$GPT2_PARTIAL_GLOB
TRAIN_EXCLUDE_VARIANTS=$TRAIN_EXCLUDE_VARIANTS
TRACE_TEST_FRACTION=$TRACE_TEST_FRACTION
TRACE_SPLIT_SEED=$TRACE_SPLIT_SEED
EPOCHS=$EPOCHS
LR=$LR
WEIGHT_DECAY_BASE=$WEIGHT_DECAY_BASE
DEVICE=$DEVICE
FEATURE_MODE=$FEATURE_MODE
SINGLE_LAYER_SOURCE=$SINGLE_LAYER_SOURCE
SINGLE_LAYER_ID=$SINGLE_LAYER_ID
SINGLE_LAYER_MODE=$SINGLE_LAYER_MODE
PERMUTATION_RUNS=$PERMUTATION_RUNS
PERMUTATION_SEED=$PERMUTATION_SEED
WEIGHT_DECAY_VALUES=$WEIGHT_DECAY_VALUES
MULTI_SEED_VALUES=$MULTI_SEED_VALUES
WRONG_INTERMEDIATE_BOOTSTRAP_N=$WRONG_INTERMEDIATE_BOOTSTRAP_N
WRONG_INTERMEDIATE_BOOTSTRAP_SEED=$WRONG_INTERMEDIATE_BOOTSTRAP_SEED
GPT2_BASELINE_AUROC=$GPT2_BASELINE_AUROC
QWEN_PERM_JSON=$QWEN_PERM_JSON
QWEN_ABLATION_REG_JSON=$QWEN_ABLATION_REG_JSON
QWEN_MULTI_JSON=$QWEN_MULTI_JSON
QWEN_FOLLOWUP_SUMMARY_JSON=$QWEN_FOLLOWUP_SUMMARY_JSON
GPT2_ABLATION_REG_JSON=$GPT2_ABLATION_REG_JSON
GPT2_MULTI_JSON=$GPT2_MULTI_JSON
ARITH_REFRAME_JSON=$ARITH_REFRAME_JSON
OUT_JSON=$OUT_JSON
OUT_MD=$OUT_MD
QPERM_SESSION=$QPERM_SESSION
QFOLLOW_SESSION=$QFOLLOW_SESSION
GFOLLOW_SESSION=$GFOLLOW_SESSION
COORD_SESSION=$COORD_SESSION
CFG

    for s in "$QPERM_SESSION" "$QFOLLOW_SESSION" "$GFOLLOW_SESSION" "$COORD_SESSION"; do
      tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s"
    done

    tmux new-session -d -s "$QPERM_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' worker-qwen-perm"
    tmux new-session -d -s "$COORD_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched qwen/gpt2 single-layer staged stress run"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  qwen partial_count: ${#QWEN_PARTIALS[@]}"
    echo "  gpt2 partial_count: ${#GPT2_PARTIALS[@]}"
    echo "  feature_mode: $FEATURE_MODE (single-layer-source=$SINGLE_LAYER_SOURCE)"
    echo "  sessions: $QPERM_SESSION, $COORD_SESSION"
    echo "  final_output: $OUT_JSON"
    ;;
  worker-qwen-perm)
    run_worker_qwen_perm
    ;;
  worker-qwen-followup)
    run_worker_qwen_followup
    ;;
  worker-gpt2-followup)
    run_worker_gpt2_followup
    ;;
  coordinator)
    run_coordinator
    ;;
  *)
    usage
    exit 2
    ;;
esac

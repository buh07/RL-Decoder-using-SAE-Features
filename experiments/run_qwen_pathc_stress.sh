#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
PY="${PYTHON:-.venv/bin/python3}"

usage() {
  cat <<'USAGE'
usage: experiments/run_qwen_pathc_stress.sh {launch|worker-permutation|worker-ablation-reg|worker-multiseed|coordinator}
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
      local i
      for i in {1..6}; do
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

run_worker_permutation() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  mapfile -t PARTIALS < "$BASE/meta/partials.list"
  [[ "${#PARTIALS[@]}" -gt 0 ]] || { echo "No partials found" >&2; exit 1; }
  {
    echo "[$(date -Is)] permutation worker start"
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
      --permutation-runs "$PERMUTATION_RUNS" \
      --permutation-seed "$PERMUTATION_SEED" \
      --output-json "$PERMUTATION_JSON"
    echo "[$(date -Is)] permutation worker done"
  } >>"$BASE/logs/permutation.log" 2>&1
  touch "$BASE/state/permutation.done"
}

run_worker_ablation_reg() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  mapfile -t PARTIALS < "$BASE/meta/partials.list"
  [[ "${#PARTIALS[@]}" -gt 0 ]] || { echo "No partials found" >&2; exit 1; }
  {
    echo "[$(date -Is)] ablation+reg worker start"
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
      --device "$DEVICE" \
      --output-json "$ABLATION_REG_JSON"
    echo "[$(date -Is)] ablation+reg worker done"
  } >>"$BASE/logs/ablation_reg.log" 2>&1
  touch "$BASE/state/ablation_reg.done"
}

run_worker_multiseed() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  mapfile -t PARTIALS < "$BASE/meta/partials.list"
  [[ "${#PARTIALS[@]}" -gt 0 ]] || { echo "No partials found" >&2; exit 1; }
  {
    echo "[$(date -Is)] multiseed worker start"
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
      --multi-seed-values "$MULTI_SEED_VALUES" \
      --wrong-intermediate-bootstrap-n "$WRONG_INTERMEDIATE_BOOTSTRAP_N" \
      --wrong-intermediate-bootstrap-seed "$WRONG_INTERMEDIATE_BOOTSTRAP_SEED" \
      --output-json "$MULTISEED_JSON"
    echo "[$(date -Is)] multiseed worker done"
  } >>"$BASE/logs/multiseed.log" 2>&1
  touch "$BASE/state/multiseed.done"
}

run_coordinator() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  {
    echo "[$(date -Is)] coordinator start run_id=$RUN_ID"

    wait_for_file_with_session "$BASE/state/permutation.done" "$PERM_SESSION" 86400
    wait_for_file_with_session "$BASE/state/ablation_reg.done" "$ABR_SESSION" 86400
    wait_for_file_with_session "$BASE/state/multiseed.done" "$MSEED_SESSION" 86400

    [[ -f "$PERMUTATION_JSON" ]] || { echo "missing permutation json" >&2; exit 1; }
    [[ -f "$ABLATION_REG_JSON" ]] || { echo "missing ablation/reg json" >&2; exit 1; }
    [[ -f "$MULTISEED_JSON" ]] || { echo "missing multiseed json" >&2; exit 1; }
    json_parseable "$PERMUTATION_JSON" || { echo "bad json: $PERMUTATION_JSON" >&2; exit 1; }
    json_parseable "$ABLATION_REG_JSON" || { echo "bad json: $ABLATION_REG_JSON" >&2; exit 1; }
    json_parseable "$MULTISEED_JSON" || { echo "bad json: $MULTISEED_JSON" >&2; exit 1; }

    "$PY" phase7/stress_test_pathc_probe.py \
      --task final \
      --run-id "$RUN_ID" \
      --run-tag "$RUN_TAG" \
      --permutation-json "$PERMUTATION_JSON" \
      --ablation-reg-json "$ABLATION_REG_JSON" \
      --multiseed-json "$MULTISEED_JSON" \
      --output-json "$OUT_JSON" \
      --output-md "$OUT_MD"

    json_parseable "$OUT_JSON" || { echo "bad final json: $OUT_JSON" >&2; exit 1; }

    touch "$BASE/state/pipeline.done"
    echo "[$(date -Is)] coordinator done"
    echo "final: $OUT_JSON"
  } >>"$BASE/logs/coordinator.log" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen_pathc_stress}"
    RUN_TAG="${RUN_TAG:-qwen_pathc_stress_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"

    SOURCE_RUN_ID="${SOURCE_RUN_ID:-20260308_165109_phase7_qwen_trackc_upgrade_qwpathc_robust_full_core}"
    PARTIAL_GLOB="${PARTIAL_GLOB:-phase7_results/results/phase7_sae_trajectory_coherence_phase7_sae_trajectory_${SOURCE_RUN_ID}_layer*.json}"

    TRAIN_EXCLUDE_VARIANTS="${TRAIN_EXCLUDE_VARIANTS:-order_flip_only,answer_first_order_flip,reordered_steps}"
    TRACE_TEST_FRACTION="${TRACE_TEST_FRACTION:-0.20}"
    TRACE_SPLIT_SEED="${TRACE_SPLIT_SEED:-20260306}"
    EPOCHS="${EPOCHS:-500}"
    LR="${LR:-0.03}"
    WEIGHT_DECAY_BASE="${WEIGHT_DECAY_BASE:-0.0001}"
    DEVICE="${DEVICE:-cpu}"

    PERMUTATION_RUNS="${PERMUTATION_RUNS:-100}"
    PERMUTATION_SEED="${PERMUTATION_SEED:-20260308}"
    WEIGHT_DECAY_VALUES="${WEIGHT_DECAY_VALUES:-0.0001,0.01,0.1,1.0}"
    SINGLE_LAYER_MODE="${SINGLE_LAYER_MODE:-auto_best}"
    MULTI_SEED_VALUES="${MULTI_SEED_VALUES:-20260307,20260308,20260309,20260310,20260311,20260312,20260313,20260314,20260315,20260316}"
    WRONG_INTERMEDIATE_BOOTSTRAP_N="${WRONG_INTERMEDIATE_BOOTSTRAP_N:-1000}"
    WRONG_INTERMEDIATE_BOOTSTRAP_SEED="${WRONG_INTERMEDIATE_BOOTSTRAP_SEED:-20260307}"

    PERMUTATION_JSON="${PERMUTATION_JSON:-$BASE/meta/permutation.json}"
    ABLATION_REG_JSON="${ABLATION_REG_JSON:-$BASE/meta/ablation_reg.json}"
    MULTISEED_JSON="${MULTISEED_JSON:-$BASE/meta/multiseed.json}"
    OUT_JSON="${OUT_JSON:-phase7_results/results/qwen_pathc_stress_${RUN_ID}.json}"
    OUT_MD="${OUT_MD:-phase7_results/results/qwen_pathc_stress_${RUN_ID}.md}"

    PERM_SESSION="p7st_perm_${RUN_ID}"
    ABR_SESSION="p7st_abr_${RUN_ID}"
    MSEED_SESSION="p7st_ms_${RUN_ID}"
    COORD_SESSION="p7st_coord_${RUN_ID}"

    mkdir -p "$BASE/logs" "$BASE/meta" "$BASE/state"
    rm -f "$BASE/state"/*.done

    mapfile -t PARTIALS < <(ls -1 $PARTIAL_GLOB 2>/dev/null | sort)
    if [[ "${#PARTIALS[@]}" -eq 0 ]]; then
      echo "No partials found for glob: $PARTIAL_GLOB" >&2
      exit 1
    fi
    printf '%s\n' "${PARTIALS[@]}" > "$BASE/meta/partials.list"

    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
SOURCE_RUN_ID=$SOURCE_RUN_ID
PARTIAL_GLOB=$PARTIAL_GLOB
TRAIN_EXCLUDE_VARIANTS=$TRAIN_EXCLUDE_VARIANTS
TRACE_TEST_FRACTION=$TRACE_TEST_FRACTION
TRACE_SPLIT_SEED=$TRACE_SPLIT_SEED
EPOCHS=$EPOCHS
LR=$LR
WEIGHT_DECAY_BASE=$WEIGHT_DECAY_BASE
DEVICE=$DEVICE
PERMUTATION_RUNS=$PERMUTATION_RUNS
PERMUTATION_SEED=$PERMUTATION_SEED
WEIGHT_DECAY_VALUES=$WEIGHT_DECAY_VALUES
SINGLE_LAYER_MODE=$SINGLE_LAYER_MODE
MULTI_SEED_VALUES=$MULTI_SEED_VALUES
WRONG_INTERMEDIATE_BOOTSTRAP_N=$WRONG_INTERMEDIATE_BOOTSTRAP_N
WRONG_INTERMEDIATE_BOOTSTRAP_SEED=$WRONG_INTERMEDIATE_BOOTSTRAP_SEED
PERMUTATION_JSON=$PERMUTATION_JSON
ABLATION_REG_JSON=$ABLATION_REG_JSON
MULTISEED_JSON=$MULTISEED_JSON
OUT_JSON=$OUT_JSON
OUT_MD=$OUT_MD
PERM_SESSION=$PERM_SESSION
ABR_SESSION=$ABR_SESSION
MSEED_SESSION=$MSEED_SESSION
COORD_SESSION=$COORD_SESSION
CFG

    for s in "$PERM_SESSION" "$ABR_SESSION" "$MSEED_SESSION" "$COORD_SESSION"; do
      tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s"
    done

    tmux new-session -d -s "$PERM_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' worker-permutation"
    tmux new-session -d -s "$ABR_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' worker-ablation-reg"
    tmux new-session -d -s "$MSEED_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' worker-multiseed"
    tmux new-session -d -s "$COORD_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched qwen pathc stress"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  source_run_id: $SOURCE_RUN_ID"
    echo "  partial_count: ${#PARTIALS[@]}"
    echo "  sessions: $PERM_SESSION, $ABR_SESSION, $MSEED_SESSION, $COORD_SESSION"
    echo "  final_output: $OUT_JSON"
    ;;
  worker-permutation)
    run_worker_permutation
    ;;
  worker-ablation-reg)
    run_worker_ablation_reg
    ;;
  worker-multiseed)
    run_worker_multiseed
    ;;
  coordinator)
    run_coordinator
    ;;
  *)
    usage
    exit 2
    ;;
esac

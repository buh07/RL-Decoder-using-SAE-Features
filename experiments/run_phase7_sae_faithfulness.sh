#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "usage: $0 {launch|worker-g5|worker-g6|worker-g7|coordinator}" >&2
  exit 2
fi

PY="${PYTHON:-.venv/bin/python3}"

_run_worker() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
  : "${GPU_ID:?GPU_ID required}"
  : "${LAYERS:?LAYERS required}"
  : "${CONTROL_RECORDS:?CONTROL_RECORDS required}"
  : "${OUT_PARTIAL:?OUT_PARTIAL required}"

  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" "phase7_results/results"
  local log="$BASE/logs/worker_gpu${GPU_ID}.log"
  {
    echo "[$(date -Is)] worker start gpu=${GPU_ID} layers=${LAYERS}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" phase7/sae_feature_faithfulness_discrimination.py \
      --control-records "$CONTROL_RECORDS" \
      --layers "$LAYERS" \
      --saes-dir "$SAES_DIR" \
      --activations-dir "$ACTIVATIONS_DIR" \
      --sample-traces "$SAMPLE_TRACES" \
      --seed "$SEED" \
      --n-permutations "$N_PERMUTATIONS" \
      --trace-test-fraction "$TRACE_TEST_FRACTION" \
      --probe-epochs "$PROBE_EPOCHS" \
      --probe-lr "$PROBE_LR" \
      --probe-weight-decay "$PROBE_WEIGHT_DECAY" \
      --probe-l1-lambda "$PROBE_L1_LAMBDA" \
      --min-class-per-split "$MIN_CLASS_PER_SPLIT" \
      --batch-size "$BATCH_SIZE" \
      --device cuda:0 \
      --run-tag "$RUN_TAG" \
      --output "$OUT_PARTIAL"
    echo "[$(date -Is)] worker done gpu=${GPU_ID}"
  } >>"$log" 2>&1
  touch "$BASE/state/gpu${GPU_ID}.done"
}

_run_coordinator() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
  : "${PARTIAL_G5:?PARTIAL_G5 required}"
  : "${PARTIAL_G6:?PARTIAL_G6 required}"
  : "${PARTIAL_G7:?PARTIAL_G7 required}"
  : "${OUT_MERGED_JSON:?OUT_MERGED_JSON required}"
  : "${OUT_MERGED_MD:?OUT_MERGED_MD required}"
  : "${WORKER_SESSION_G5:?WORKER_SESSION_G5 required}"
  : "${WORKER_SESSION_G6:?WORKER_SESSION_G6 required}"
  : "${WORKER_SESSION_G7:?WORKER_SESSION_G7 required}"

  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" "phase7_results/results"
  local log="$BASE/logs/coordinator.log"
  {
    echo "[$(date -Is)] coordinator start run_id=${RUN_ID}"
    while true; do
      g5_done=0; g6_done=0; g7_done=0
      [[ -f "$BASE/state/gpu5.done" ]] && g5_done=1
      [[ -f "$BASE/state/gpu6.done" ]] && g6_done=1
      [[ -f "$BASE/state/gpu7.done" ]] && g7_done=1
      if [[ "$g5_done" -eq 1 && "$g6_done" -eq 1 && "$g7_done" -eq 1 ]]; then
        break
      fi

      if [[ "$g5_done" -eq 0 ]] && ! tmux has-session -t "$WORKER_SESSION_G5" 2>/dev/null; then
        echo "worker session missing before completion: $WORKER_SESSION_G5" >&2
        exit 1
      fi
      if [[ "$g6_done" -eq 0 ]] && ! tmux has-session -t "$WORKER_SESSION_G6" 2>/dev/null; then
        echo "worker session missing before completion: $WORKER_SESSION_G6" >&2
        exit 1
      fi
      if [[ "$g7_done" -eq 0 ]] && ! tmux has-session -t "$WORKER_SESSION_G7" 2>/dev/null; then
        echo "worker session missing before completion: $WORKER_SESSION_G7" >&2
        exit 1
      fi
      sleep 10
    done

    "$PY" phase7/aggregate_sae_faithfulness_discrimination.py \
      --partials "$PARTIAL_G5" "$PARTIAL_G6" "$PARTIAL_G7" \
      --phase4-top-features "$PHASE4_TOP_FEATURES" \
      --output-json "$OUT_MERGED_JSON" \
      --output-md "$OUT_MERGED_MD" \
      --run-tag "$RUN_TAG"

    "$PY" - <<PY
import json
from pathlib import Path
base = Path(${BASE@Q})
summary = {
  "schema_version": "phase7_sae_faithfulness_pipeline_state_v1",
  "run_id": ${RUN_ID@Q},
  "run_tag": ${RUN_TAG@Q},
  "partial_outputs": [${PARTIAL_G5@Q}, ${PARTIAL_G6@Q}, ${PARTIAL_G7@Q}],
  "merged_json": ${OUT_MERGED_JSON@Q},
  "merged_md": ${OUT_MERGED_MD@Q},
}
(base / "meta" / "summary.json").write_text(json.dumps(summary, indent=2))
(base / "state" / "pipeline.done").write_text("done\\n")
print("pipeline summary written")
PY

    echo "[$(date -Is)] coordinator done"
  } >>"$log" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_sae}"
    RUN_TAG="${RUN_TAG:-phase7_sae_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"
    CONTROL_RECORDS="${CONTROL_RECORDS:-phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json}"
    SAES_DIR="${SAES_DIR:-phase2_results/saes_gpt2_12x_topk/saes}"
    ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
    PHASE4_TOP_FEATURES="${PHASE4_TOP_FEATURES:-phase4_results/topk/probe/top_features_per_layer.json}"
    SAMPLE_TRACES="${SAMPLE_TRACES:-50}"
    SEED="${SEED:-20260306}"
    N_PERMUTATIONS="${N_PERMUTATIONS:-250}"
    TRACE_TEST_FRACTION="${TRACE_TEST_FRACTION:-0.20}"
    PROBE_EPOCHS="${PROBE_EPOCHS:-80}"
    PROBE_LR="${PROBE_LR:-0.001}"
    PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.01}"
    PROBE_L1_LAMBDA="${PROBE_L1_LAMBDA:-0.00001}"
    MIN_CLASS_PER_SPLIT="${MIN_CLASS_PER_SPLIT:-10}"
    BATCH_SIZE="${BATCH_SIZE:-256}"

    PARTIAL_G5="${PARTIAL_G5:-phase7_results/results/phase7_sae_feature_discrimination_${RUN_TAG}_layers_0_7.json}"
    PARTIAL_G6="${PARTIAL_G6:-phase7_results/results/phase7_sae_feature_discrimination_${RUN_TAG}_layers_8_15.json}"
    PARTIAL_G7="${PARTIAL_G7:-phase7_results/results/phase7_sae_feature_discrimination_${RUN_TAG}_layers_16_23.json}"
    OUT_MERGED_JSON="${OUT_MERGED_JSON:-phase7_results/results/phase7_sae_feature_discrimination_${RUN_TAG}.json}"
    OUT_MERGED_MD="${OUT_MERGED_MD:-phase7_results/results/phase7_sae_feature_discrimination_${RUN_TAG}.md}"

    WORKER_SESSION_G5="p7sae_g5_${RUN_ID}"
    WORKER_SESSION_G6="p7sae_g6_${RUN_ID}"
    WORKER_SESSION_G7="p7sae_g7_${RUN_ID}"
    COORD_SESSION="p7sae_coord_${RUN_ID}"

    mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" "phase7_results/results"
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
CONTROL_RECORDS=$CONTROL_RECORDS
SAES_DIR=$SAES_DIR
ACTIVATIONS_DIR=$ACTIVATIONS_DIR
PHASE4_TOP_FEATURES=$PHASE4_TOP_FEATURES
SAMPLE_TRACES=$SAMPLE_TRACES
SEED=$SEED
N_PERMUTATIONS=$N_PERMUTATIONS
TRACE_TEST_FRACTION=$TRACE_TEST_FRACTION
PROBE_EPOCHS=$PROBE_EPOCHS
PROBE_LR=$PROBE_LR
PROBE_WEIGHT_DECAY=$PROBE_WEIGHT_DECAY
PROBE_L1_LAMBDA=$PROBE_L1_LAMBDA
MIN_CLASS_PER_SPLIT=$MIN_CLASS_PER_SPLIT
BATCH_SIZE=$BATCH_SIZE
PARTIAL_G5=$PARTIAL_G5
PARTIAL_G6=$PARTIAL_G6
PARTIAL_G7=$PARTIAL_G7
OUT_MERGED_JSON=$OUT_MERGED_JSON
OUT_MERGED_MD=$OUT_MERGED_MD
WORKER_SESSION_G5=$WORKER_SESSION_G5
WORKER_SESSION_G6=$WORKER_SESSION_G6
WORKER_SESSION_G7=$WORKER_SESSION_G7
COORD_SESSION=$COORD_SESSION
CFG

    tmux new-session -d -s "$WORKER_SESSION_G5" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && GPU_ID=5 LAYERS='0,1,2,3,4,5,6,7' OUT_PARTIAL='$PARTIAL_G5' '$0' worker-g5"
    tmux new-session -d -s "$WORKER_SESSION_G6" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && GPU_ID=6 LAYERS='8,9,10,11,12,13,14,15' OUT_PARTIAL='$PARTIAL_G6' '$0' worker-g6"
    tmux new-session -d -s "$WORKER_SESSION_G7" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && GPU_ID=7 LAYERS='16,17,18,19,20,21,22,23' OUT_PARTIAL='$PARTIAL_G7' '$0' worker-g7"
    tmux new-session -d -s "$COORD_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched phase7-sae"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  worker sessions: $WORKER_SESSION_G5, $WORKER_SESSION_G6, $WORKER_SESSION_G7"
    echo "  coordinator: $COORD_SESSION"
    echo "  merged output: $OUT_MERGED_JSON"
    ;;
  worker-g5|worker-g6|worker-g7)
    _run_worker
    ;;
  coordinator)
    _run_coordinator
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

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
  : "${LAYER:?LAYER required}"
  : "${OUT_PARTIAL:?OUT_PARTIAL required}"
  : "${CONTROL_RECORDS:?CONTROL_RECORDS required}"
  : "${SAES_DIR:?SAES_DIR required}"
  : "${ACTIVATIONS_DIR:?ACTIVATIONS_DIR required}"
  : "${PHASE4_TOP_FEATURES:?PHASE4_TOP_FEATURES required}"
  : "${FEATURE_SET:?FEATURE_SET required}"
  : "${DIVERGENT_SOURCE:?DIVERGENT_SOURCE required}"
  : "${SUBSPACE_SPECS:?SUBSPACE_SPECS required}"
  : "${SAMPLE_TRACES:?SAMPLE_TRACES required}"
  : "${MIN_COMMON_STEPS:?MIN_COMMON_STEPS required}"
  : "${SEED:?SEED required}"
  : "${N_BOOTSTRAP:?N_BOOTSTRAP required}"
  : "${BATCH_SIZE:?BATCH_SIZE required}"
  : "${EMIT_SAMPLES:?EMIT_SAMPLES required}"

  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" "phase7_results/results"
  local log="$BASE/logs/worker_gpu${GPU_ID}.log"
  local emit_arg=()
  if [[ "$EMIT_SAMPLES" == "1" ]]; then
    emit_arg=(--emit-samples)
  fi
  {
    echo "[$(date -Is)] worker start gpu=${GPU_ID} layer=${LAYER}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" phase7/sae_trajectory_coherence_discrimination.py \
      --control-records "$CONTROL_RECORDS" \
      --layer "$LAYER" \
      --saes-dir "$SAES_DIR" \
      --activations-dir "$ACTIVATIONS_DIR" \
      --phase4-top-features "$PHASE4_TOP_FEATURES" \
      --feature-set "$FEATURE_SET" \
      --divergent-source "$DIVERGENT_SOURCE" \
      --subspace-specs "$SUBSPACE_SPECS" \
      --sample-traces "$SAMPLE_TRACES" \
      --min-common-steps "$MIN_COMMON_STEPS" \
      --seed "$SEED" \
      --n-bootstrap "$N_BOOTSTRAP" \
      --batch-size "$BATCH_SIZE" \
      --device cuda:0 \
      --run-tag "$RUN_TAG" \
      "${emit_arg[@]}" \
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

  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
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
        echo "worker session disappeared before completion: $WORKER_SESSION_G5" >&2
        exit 1
      fi
      if [[ "$g6_done" -eq 0 ]] && ! tmux has-session -t "$WORKER_SESSION_G6" 2>/dev/null; then
        echo "worker session disappeared before completion: $WORKER_SESSION_G6" >&2
        exit 1
      fi
      if [[ "$g7_done" -eq 0 ]] && ! tmux has-session -t "$WORKER_SESSION_G7" 2>/dev/null; then
        echo "worker session disappeared before completion: $WORKER_SESSION_G7" >&2
        exit 1
      fi
      sleep 10
    done

    [[ -f "$PARTIAL_G5" ]] || { echo "missing partial: $PARTIAL_G5" >&2; exit 1; }
    [[ -f "$PARTIAL_G6" ]] || { echo "missing partial: $PARTIAL_G6" >&2; exit 1; }
    [[ -f "$PARTIAL_G7" ]] || { echo "missing partial: $PARTIAL_G7" >&2; exit 1; }

    "$PY" phase7/aggregate_sae_trajectory_coherence.py \
      --partials "$PARTIAL_G5" "$PARTIAL_G6" "$PARTIAL_G7" \
      --output-json "$OUT_MERGED_JSON" \
      --output-md "$OUT_MERGED_MD" \
      --run-tag "$RUN_TAG"

    "$PY" - <<PY
import json
from pathlib import Path
base = Path(${BASE@Q})
summary = {
  "schema_version": "phase7_sae_trajectory_pipeline_state_v1",
  "run_id": ${RUN_ID@Q},
  "run_tag": ${RUN_TAG@Q},
  "partials": [${PARTIAL_G5@Q}, ${PARTIAL_G6@Q}, ${PARTIAL_G7@Q}],
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
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_sae_trajectory}"
    RUN_TAG="${RUN_TAG:-phase7_sae_trajectory_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"

    CONTROL_RECORDS="${CONTROL_RECORDS:-phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json}"
    SAES_DIR="${SAES_DIR:-phase2_results/saes_gpt2_12x_topk/saes}"
    ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
    PHASE4_TOP_FEATURES="${PHASE4_TOP_FEATURES:-phase4_results/topk/probe/top_features_per_layer.json}"
    FEATURE_SET="${FEATURE_SET:-eq_top50}"
    DIVERGENT_SOURCE="${DIVERGENT_SOURCE:-phase7_results/results/phase7_sae_feature_discrimination_phase7_sae_20260306_224419_phase7_sae.json}"
    SUBSPACE_SPECS="${SUBSPACE_SPECS:-phase7_results/interventions/variable_subspaces_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json}"

    SAMPLE_TRACES="${SAMPLE_TRACES:-0}"
    MIN_COMMON_STEPS="${MIN_COMMON_STEPS:-3}"
    SEED="${SEED:-20260306}"
    N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    EMIT_SAMPLES="${EMIT_SAMPLES:-0}"

    PARTIAL_G5="${PARTIAL_G5:-phase7_results/results/phase7_sae_trajectory_coherence_${RUN_TAG}_layer4.json}"
    PARTIAL_G6="${PARTIAL_G6:-phase7_results/results/phase7_sae_trajectory_coherence_${RUN_TAG}_layer7.json}"
    PARTIAL_G7="${PARTIAL_G7:-phase7_results/results/phase7_sae_trajectory_coherence_${RUN_TAG}_layer22.json}"
    OUT_MERGED_JSON="${OUT_MERGED_JSON:-phase7_results/results/phase7_sae_trajectory_coherence_${RUN_TAG}.json}"
    OUT_MERGED_MD="${OUT_MERGED_MD:-phase7_results/results/phase7_sae_trajectory_coherence_${RUN_TAG}.md}"

    WORKER_SESSION_G5="p7saetc_g5_${RUN_ID}"
    WORKER_SESSION_G6="p7saetc_g6_${RUN_ID}"
    WORKER_SESSION_G7="p7saetc_g7_${RUN_ID}"
    COORD_SESSION="p7saetc_coord_${RUN_ID}"

    mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" "phase7_results/results"
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
CONTROL_RECORDS=$CONTROL_RECORDS
SAES_DIR=$SAES_DIR
ACTIVATIONS_DIR=$ACTIVATIONS_DIR
PHASE4_TOP_FEATURES=$PHASE4_TOP_FEATURES
FEATURE_SET=$FEATURE_SET
DIVERGENT_SOURCE=$DIVERGENT_SOURCE
SUBSPACE_SPECS=$SUBSPACE_SPECS
SAMPLE_TRACES=$SAMPLE_TRACES
MIN_COMMON_STEPS=$MIN_COMMON_STEPS
SEED=$SEED
N_BOOTSTRAP=$N_BOOTSTRAP
BATCH_SIZE=$BATCH_SIZE
EMIT_SAMPLES=$EMIT_SAMPLES
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

    tmux new-session -d -s "$WORKER_SESSION_G5" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && GPU_ID=5 LAYER=4 OUT_PARTIAL='$PARTIAL_G5' '$0' worker-g5"
    tmux new-session -d -s "$WORKER_SESSION_G6" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && GPU_ID=6 LAYER=7 OUT_PARTIAL='$PARTIAL_G6' '$0' worker-g6"
    tmux new-session -d -s "$WORKER_SESSION_G7" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && GPU_ID=7 LAYER=22 OUT_PARTIAL='$PARTIAL_G7' '$0' worker-g7"
    tmux new-session -d -s "$COORD_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched phase7 sae trajectory coherence"
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

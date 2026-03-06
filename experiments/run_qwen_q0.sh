#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "usage: $0 {launch|worker|coordinator}" >&2
  exit 2
fi

PY="${PYTHON:-.venv/bin/python3}"

_run_worker() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
  : "${GPU_ID:?GPU_ID required}"
  : "${CONTROLS_PATH:?CONTROLS_PATH required}"
  : "${OUT_SEP:?OUT_SEP required}"
  : "${OUT_GO:?OUT_GO required}"
  : "${SAMPLE_TRACES:?SAMPLE_TRACES required}"
  : "${SEED:?SEED required}"
  : "${LAYERS:?LAYERS required}"
  : "${N_PERM:?N_PERM required}"
  : "${MIN_MARGIN:?MIN_MARGIN required}"
  : "${MIN_TRACE_FRACTION:?MIN_TRACE_FRACTION required}"
  : "${MODEL_KEY:?MODEL_KEY required}"

  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"

  local log="$BASE/logs/worker.log"
  {
    echo "[$(date -Is)] worker start run_id=$RUN_ID run_tag=$RUN_TAG gpu=$GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" phase7/qwen_q0_hidden_state_separation.py \
      --controls "$CONTROLS_PATH" \
      --model-key "$MODEL_KEY" \
      --sample-traces "$SAMPLE_TRACES" \
      --seed "$SEED" \
      --layers "$LAYERS" \
      --device cuda:0 \
      --parse-mode hybrid \
      --token-anchor eq_like \
      --anchor-priority equation_first \
      --n-permutations "$N_PERM" \
      --min-margin "$MIN_MARGIN" \
      --min-positive-trace-fraction "$MIN_TRACE_FRACTION" \
      --run-tag "$RUN_TAG" \
      --output-separation "$OUT_SEP" \
      --output-go-nogo "$OUT_GO"
    echo "[$(date -Is)] worker done"
  } >>"$log" 2>&1

  touch "$BASE/state/worker.done"
}

_run_coordinator() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
  : "${OUT_SEP:?OUT_SEP required}"
  : "${OUT_GO:?OUT_GO required}"

  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local log="$BASE/logs/coordinator.log"

  {
    echo "[$(date -Is)] coordinator start run_id=$RUN_ID run_tag=$RUN_TAG"
    while [[ ! -f "$BASE/state/worker.done" ]]; do
      sleep 5
    done

    "$PY" - <<PY
import json
from pathlib import Path
base = Path(${BASE@Q})
out_sep = Path(${OUT_SEP@Q})
out_go = Path(${OUT_GO@Q})
summary = {
    "schema_version": "phase7_qwen_q0_pipeline_state_v1",
    "run_id": ${RUN_ID@Q},
    "run_tag": ${RUN_TAG@Q},
    "worker_done": (base / "state" / "worker.done").exists(),
    "separation_exists": out_sep.exists(),
    "go_nogo_exists": out_go.exists(),
    "separation_path": str(out_sep),
    "go_nogo_path": str(out_go),
}
(base / "meta").mkdir(parents=True, exist_ok=True)
(base / "meta" / "summary.json").write_text(json.dumps(summary, indent=2))
(base / "state" / "pipeline.done").write_text("done\n")
print("qwen_q0 coordinator summary written")
PY
    echo "[$(date -Is)] coordinator done"
  } >>"$log" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen_q0}"
    RUN_TAG="${RUN_TAG:-qwen_q0_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"
    GPU_ID="${GPU_ID:-3}"
    SAMPLE_TRACES="${SAMPLE_TRACES:-50}"
    SEED="${SEED:-20260306}"
    LAYERS="${LAYERS:-10,14,18,22}"
    N_PERM="${N_PERM:-250}"
    MIN_MARGIN="${MIN_MARGIN:-0.005}"
    MIN_TRACE_FRACTION="${MIN_TRACE_FRACTION:-0.50}"
    MODEL_KEY="${MODEL_KEY:-qwen2.5-7b}"
    CONTROLS_PATH="${CONTROLS_PATH:-phase7_results/controls/cot_controls_test_papercore.json}"
    OUT_SEP="${OUT_SEP:-phase7_results/results/qwen_q0_hidden_state_separation_${RUN_TAG}.json}"
    OUT_GO="${OUT_GO:-phase7_results/results/qwen_q0_go_nogo_${RUN_TAG}.json}"

    mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
GPU_ID=$GPU_ID
SAMPLE_TRACES=$SAMPLE_TRACES
SEED=$SEED
LAYERS=$LAYERS
N_PERM=$N_PERM
MIN_MARGIN=$MIN_MARGIN
MIN_TRACE_FRACTION=$MIN_TRACE_FRACTION
MODEL_KEY=$MODEL_KEY
CONTROLS_PATH=$CONTROLS_PATH
OUT_SEP=$OUT_SEP
OUT_GO=$OUT_GO
CFG

    tmux new-session -d -s "q0_worker_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' worker"
    tmux new-session -d -s "q0_coord_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched qwen q0"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  worker session: q0_worker_${RUN_ID}"
    echo "  coordinator session: q0_coord_${RUN_ID}"
    echo "  output separation: $OUT_SEP"
    echo "  output decision:   $OUT_GO"
    ;;
  worker)
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

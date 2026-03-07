#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python3}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found" >&2
    exit 1
  fi
}

worker_capture() {
  : "${RUN_ID:?RUN_ID required}"
  : "${GPU_ID:=5}"
  : "${ACTIVATIONS_DIR:=phase2_results/activations}"
  : "${NUM_SAMPLES:=50}"
  : "${BATCH_SIZE:=16}"
  : "${BASE:=phase2_results/runs/$RUN_ID}"
  mkdir -p "$BASE/logs" "$BASE/state"
  local done="$BASE/state/capture.done"
  local fail="$BASE/state/capture.failed"
  local logf="$BASE/logs/capture_gpu${GPU_ID}.log"

  if [[ -f "$done" ]]; then
    log "capture already done: $done"
    exit 0
  fi
  rm -f "$fail"

  local existing
  existing=$(ls "$ACTIVATIONS_DIR"/qwen2.5-7b_layer*_activations.pt 2>/dev/null | wc -l || true)
  if [[ "$existing" -ge 28 ]]; then
    log "found $existing qwen activation files; skipping capture"
    touch "$done"
    exit 0
  fi

  log "starting qwen activation capture on gpu $GPU_ID"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" phase2/capture_activations.py \
      --models qwen2.5-7b \
      --output-dir "$ACTIVATIONS_DIR" \
      --batch-size "$BATCH_SIZE" \
      --num-samples "$NUM_SAMPLES" \
      --device cuda:0 \
      --dtype auto
  ) 2>&1 | tee "$logf"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    log "capture failed rc=$rc"
    touch "$fail"
    exit $rc
  fi
  touch "$done"
  log "capture complete"
}

wait_for_capture() {
  : "${BASE:?BASE required}"
  local done="$BASE/state/capture.done"
  local fail="$BASE/state/capture.failed"
  until [[ -f "$done" || -f "$fail" ]]; do
    sleep 10
  done
  if [[ -f "$fail" ]]; then
    log "capture failed marker found; aborting train worker"
    exit 1
  fi
}

worker_train() {
  : "${RUN_ID:?RUN_ID required}"
  : "${GPU_ID:?GPU_ID required}"
  : "${LAYERS:?LAYERS required}"
  : "${BASE:=phase2_results/runs/$RUN_ID}"
  : "${ACTIVATIONS_DIR:=phase2_results/activations}"
  : "${SAE_OUT:=phase2_results/saes_qwen25_7b_12x_topk/saes}"
  : "${EPOCHS:=10}"
  : "${BATCH_SIZE:=64}"
  mkdir -p "$BASE/logs" "$BASE/state" "$SAE_OUT"
  local done="$BASE/state/gpu${GPU_ID}.train.done"
  local fail="$BASE/state/gpu${GPU_ID}.train.failed"
  local logf="$BASE/logs/train_gpu${GPU_ID}.log"
  rm -f "$fail"

  wait_for_capture

  if [[ -f "$done" ]]; then
    log "gpu${GPU_ID} training already done"
    exit 0
  fi

  log "starting qwen sae training on gpu $GPU_ID layers: $LAYERS"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" phase2/train_multilayer_saes.py \
      --activations-dir "$ACTIVATIONS_DIR" \
      --output-dir "$SAE_OUT" \
      --model-filter qwen2.5-7b \
      --layers $LAYERS \
      --expansion-factor 12 \
      --use-topk \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --device cuda:0
  ) 2>&1 | tee "$logf"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    log "train failed on gpu $GPU_ID rc=$rc"
    touch "$fail"
    exit $rc
  fi
  touch "$done"
  log "train complete on gpu $GPU_ID"
}

coordinator() {
  : "${RUN_ID:?RUN_ID required}"
  : "${BASE:=phase2_results/runs/$RUN_ID}"
  : "${SAE_OUT:=phase2_results/saes_qwen25_7b_12x_topk/saes}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/coordinator.log"
  local fail_any=0
  (
    cd "$ROOT"
    log "coordinator waiting for workers"
    while true; do
      if [[ -f "$BASE/state/capture.failed" || -f "$BASE/state/gpu5.train.failed" || -f "$BASE/state/gpu6.train.failed" || -f "$BASE/state/gpu7.train.failed" ]]; then
        fail_any=1
        break
      fi
      if [[ -f "$BASE/state/capture.done" && -f "$BASE/state/gpu5.train.done" && -f "$BASE/state/gpu6.train.done" && -f "$BASE/state/gpu7.train.done" ]]; then
        break
      fi
      sleep 15
    done

    if [[ $fail_any -eq 1 ]]; then
      log "coordinator detected failure marker"
      exit 1
    fi

    "$PY" phase2/rebuild_training_summary.py --saes-dir "$SAE_OUT"
    "$PY" - <<'PY'
import json, os, glob
run_id = os.environ["RUN_ID"]
base = os.environ["BASE"]
sae_out = os.environ["SAE_OUT"]
entries = sorted(glob.glob(os.path.join(sae_out, "qwen2.5-7b_layer*_sae.pt")))
payload = {
    "schema_version": "phase2_qwen25_train_state_v1",
    "run_id": run_id,
    "num_checkpoints": len(entries),
    "expected_layers": 28,
    "complete": len(entries) >= 28,
    "sae_out": sae_out,
}
os.makedirs(os.path.join(base, "meta"), exist_ok=True)
with open(os.path.join(base, "meta", "summary.json"), "w") as f:
    json.dump(payload, f, indent=2)
PY
    touch "$BASE/state/pipeline.done"
    log "coordinator complete"
  ) 2>&1 | tee "$logf"
}

launch() {
  require_tmux
  RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase2_qwen25_sae}"
  BASE="${BASE:-phase2_results/runs/$RUN_ID}"
  ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
  SAE_OUT="${SAE_OUT:-phase2_results/saes_qwen25_7b_12x_topk/saes}"
  NUM_SAMPLES="${NUM_SAMPLES:-50}"
  EPOCHS="${EPOCHS:-10}"
  BATCH_SIZE="${BATCH_SIZE:-64}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"

  local s_cap="p2qwen_cap_g5_${RUN_ID}"
  local s_g5="p2qwen_train_g5_${RUN_ID}"
  local s_g6="p2qwen_train_g6_${RUN_ID}"
  local s_g7="p2qwen_train_g7_${RUN_ID}"
  local s_coord="p2qwen_coord_${RUN_ID}"

  tmux new-session -d -s "$s_cap" "cd '$ROOT' && RUN_ID='$RUN_ID' BASE='$BASE' ACTIVATIONS_DIR='$ACTIVATIONS_DIR' NUM_SAMPLES='$NUM_SAMPLES' BATCH_SIZE='16' GPU_ID='5' '$0' worker-capture"
  tmux new-session -d -s "$s_g5" "cd '$ROOT' && RUN_ID='$RUN_ID' BASE='$BASE' ACTIVATIONS_DIR='$ACTIVATIONS_DIR' SAE_OUT='$SAE_OUT' EPOCHS='$EPOCHS' BATCH_SIZE='$BATCH_SIZE' GPU_ID='5' LAYERS='0 1 2 3 4 5 6 7 8 9' '$0' worker-train"
  tmux new-session -d -s "$s_g6" "cd '$ROOT' && RUN_ID='$RUN_ID' BASE='$BASE' ACTIVATIONS_DIR='$ACTIVATIONS_DIR' SAE_OUT='$SAE_OUT' EPOCHS='$EPOCHS' BATCH_SIZE='$BATCH_SIZE' GPU_ID='6' LAYERS='10 11 12 13 14 15 16 17 18' '$0' worker-train"
  tmux new-session -d -s "$s_g7" "cd '$ROOT' && RUN_ID='$RUN_ID' BASE='$BASE' ACTIVATIONS_DIR='$ACTIVATIONS_DIR' SAE_OUT='$SAE_OUT' EPOCHS='$EPOCHS' BATCH_SIZE='$BATCH_SIZE' GPU_ID='7' LAYERS='19 20 21 22 23 24 25 26 27' '$0' worker-train"
  tmux new-session -d -s "$s_coord" "cd '$ROOT' && RUN_ID='$RUN_ID' BASE='$BASE' SAE_OUT='$SAE_OUT' '$0' coordinator"

  cat <<EOF
launched qwen sae training
RUN_ID=$RUN_ID
BASE=$BASE
sessions:
  $s_cap
  $s_g5
  $s_g6
  $s_g7
  $s_coord
EOF
}

cmd="${1:-}"
case "$cmd" in
  launch) shift; launch "$@" ;;
  worker-capture) shift; worker_capture "$@" ;;
  worker-train) shift; worker_train "$@" ;;
  coordinator) shift; coordinator "$@" ;;
  *)
    echo "usage: $0 {launch|worker-capture|worker-train|coordinator}" >&2
    exit 2
    ;;
esac


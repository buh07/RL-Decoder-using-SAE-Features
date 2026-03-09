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

split_csv() {
  local raw="$1"
  IFS=',' read -r -a _arr <<<"$raw"
  local out=()
  for x in "${_arr[@]}"; do
    x="$(echo "$x" | xargs)"
    [[ -n "$x" ]] || continue
    out+=("$x")
  done
  echo "${out[@]}"
}

_run_worker() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
  : "${GPU_ID:?GPU_ID required}"
  : "${LAYERS:?LAYERS required}"
  : "${MODEL_KEY:?MODEL_KEY required}"
  : "${OUT_PARTIAL_TEMPLATE:?OUT_PARTIAL_TEMPLATE required}"
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

  read -r -a layers_arr <<<"$(split_csv "$LAYERS")"
  local dec_ckpt="${DECODER_CHECKPOINT:-}"
  local dec_dev="${DECODER_DEVICE:-}"
  local dec_bs="${DECODER_BATCH_SIZE:-128}"
  local dec_cache_tmpl="${DECODER_PREDS_CACHE_TEMPLATE:-}"
  local dec_cache=""
  if [[ -n "$dec_cache_tmpl" ]]; then
    dec_cache="${dec_cache_tmpl//\{gpu\}/$GPU_ID}"
  fi

  {
    echo "[$(date -Is)] worker start gpu=${GPU_ID} layers=${LAYERS}"
    for layer in "${layers_arr[@]}"; do
      local out_partial="${OUT_PARTIAL_TEMPLATE//__LAYER__/$layer}"
      local decoder_args=()
      if [[ -n "$dec_ckpt" ]]; then
        decoder_args+=(--decoder-checkpoint "$dec_ckpt" --decoder-batch-size "$dec_bs")
        if [[ -n "$dec_dev" ]]; then
          decoder_args+=(--decoder-device "$dec_dev")
        fi
        if [[ -n "$dec_cache" ]]; then
          decoder_args+=(--decoder-preds-cache "$dec_cache")
        fi
      fi
      CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" phase7/sae_trajectory_coherence_discrimination.py \
        --control-records "$CONTROL_RECORDS" \
        --model-key "$MODEL_KEY" \
        --layer "$layer" \
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
        "${decoder_args[@]}" \
        --output "$out_partial"
    done
    echo "[$(date -Is)] worker done gpu=${GPU_ID}"
  } >>"$log" 2>&1
  touch "$BASE/state/gpu${GPU_ID}.done"
}

_run_coordinator() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
  : "${ACTIVE_GPUS_CSV:?ACTIVE_GPUS_CSV required}"
  : "${PARTIALS_CSV:?PARTIALS_CSV required}"
  : "${OUT_MERGED_JSON:?OUT_MERGED_JSON required}"
  : "${OUT_MERGED_MD:?OUT_MERGED_MD required}"

  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local log="$BASE/logs/coordinator.log"

  read -r -a active_gpus <<<"$(split_csv "$ACTIVE_GPUS_CSV")"
  read -r -a partials <<<"$(split_csv "$PARTIALS_CSV")"
  _gpu_partials_complete() {
    local gpu="$1"
    local layers_var="LAYERS_G${gpu}"
    local layers_csv="${!layers_var:-}"
    [[ -n "$layers_csv" ]] || return 1
    read -r -a gpu_layers <<<"$(split_csv "$layers_csv")"
    [[ "${#gpu_layers[@]}" -gt 0 ]] || return 1
    local layer expected
    for layer in "${gpu_layers[@]}"; do
      expected="${OUT_PARTIAL_TEMPLATE//__LAYER__/$layer}"
      [[ -f "$expected" ]] || return 1
    done
    return 0
  }

  {
    echo "[$(date -Is)] coordinator start run_id=${RUN_ID}"
    while true; do
      local all_done=1
      for gpu in "${active_gpus[@]}"; do
        if [[ ! -f "$BASE/state/gpu${gpu}.done" ]]; then
          all_done=0
          local sess="p7saetc_g${gpu}_${RUN_ID}"
          if ! tmux has-session -t "$sess" 2>/dev/null; then
            # Race-safe handling: worker may have finished and exited just before marker visibility.
            sleep 2
            if [[ -f "$BASE/state/gpu${gpu}.done" ]]; then
              continue
            fi
            if _gpu_partials_complete "$gpu"; then
              echo "worker session exited after producing expected partials; synthesizing marker for gpu${gpu}"
              touch "$BASE/state/gpu${gpu}.done"
              continue
            fi
            echo "worker session disappeared before completion: $sess" >&2
            exit 1
          fi
        fi
      done
      if [[ "$all_done" -eq 1 ]]; then
        break
      fi
      sleep 10
    done

    for p in "${partials[@]}"; do
      [[ -f "$p" ]] || { echo "missing partial: $p" >&2; exit 1; }
    done

    "$PY" phase7/aggregate_sae_trajectory_coherence.py \
      --partials "${partials[@]}" \
      --output-json "$OUT_MERGED_JSON" \
      --output-md "$OUT_MERGED_MD" \
      --run-tag "$RUN_TAG"

    "$PY" - <<PY
import json
from pathlib import Path
base = Path(${BASE@Q})
summary = {
  "schema_version": "phase7_sae_trajectory_pipeline_state_v2",
  "run_id": ${RUN_ID@Q},
  "run_tag": ${RUN_TAG@Q},
  "active_gpus": ${ACTIVE_GPUS_CSV@Q},
  "partials": ${PARTIALS_CSV@Q}.split(','),
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
    MODEL_KEY="${MODEL_KEY:-gpt2-medium}"
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

    DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-}"
    DECODER_DEVICE="${DECODER_DEVICE:-}"
    DECODER_BATCH_SIZE="${DECODER_BATCH_SIZE:-128}"

    LAYERS_CSV="${LAYERS_CSV:-4,7,22}"
    GPU_IDS_CSV="${GPU_IDS_CSV:-5,6,7}"

    read -r -a layers <<<"$(split_csv "$LAYERS_CSV")"
    [[ "${#layers[@]}" -gt 0 ]] || { echo "LAYERS_CSV is empty" >&2; exit 2; }
    for lyr in "${layers[@]}"; do
      [[ "$lyr" =~ ^[0-9]+$ ]] || { echo "invalid layer index: $lyr" >&2; exit 2; }
    done
    read -r -a gpu_ids <<<"$(split_csv "$GPU_IDS_CSV")"
    [[ "${#gpu_ids[@]}" -gt 0 ]] || { echo "GPU_IDS_CSV is empty" >&2; exit 2; }
    for gid in "${gpu_ids[@]}"; do
      [[ "$gid" =~ ^[0-9]+$ ]] || { echo "invalid GPU id: $gid" >&2; exit 2; }
      if [[ "$gid" != "5" && "$gid" != "6" && "$gid" != "7" ]]; then
        echo "unsupported GPU id for this runner: $gid (expected subset of 5,6,7)" >&2
        exit 2
      fi
    done

    declare -A gpu_layers
    for gid in "${gpu_ids[@]}"; do gpu_layers[$gid]=""; done

    partials=()
    idx=0
    for layer in "${layers[@]}"; do
      gid="${gpu_ids[$((idx % ${#gpu_ids[@]}))]}"
      if [[ -z "${gpu_layers[$gid]}" ]]; then
        gpu_layers[$gid]="$layer"
      else
        gpu_layers[$gid]="${gpu_layers[$gid]},$layer"
      fi
      partials+=("phase7_results/results/phase7_sae_trajectory_coherence_${RUN_TAG}_layer${layer}.json")
      idx=$((idx + 1))
    done

    ACTIVE_GPUS_CSV=""
    for gid in "${gpu_ids[@]}"; do
      if [[ -n "${gpu_layers[$gid]}" ]]; then
        if [[ -z "$ACTIVE_GPUS_CSV" ]]; then ACTIVE_GPUS_CSV="$gid"; else ACTIVE_GPUS_CSV="$ACTIVE_GPUS_CSV,$gid"; fi
      fi
    done

    PARTIALS_CSV=""
    for p in "${partials[@]}"; do
      if [[ -z "$PARTIALS_CSV" ]]; then PARTIALS_CSV="$p"; else PARTIALS_CSV="$PARTIALS_CSV,$p"; fi
    done

    OUT_PARTIAL_TEMPLATE="${OUT_PARTIAL_TEMPLATE:-phase7_results/results/phase7_sae_trajectory_coherence_${RUN_TAG}_layer__LAYER__.json}"
    OUT_MERGED_JSON="${OUT_MERGED_JSON:-phase7_results/results/phase7_sae_trajectory_coherence_${RUN_TAG}.json}"
    OUT_MERGED_MD="${OUT_MERGED_MD:-phase7_results/results/phase7_sae_trajectory_coherence_${RUN_TAG}.md}"
    DECODER_PREDS_CACHE_TEMPLATE="${DECODER_PREDS_CACHE_TEMPLATE:-$BASE/meta/decoder_preds_gpu{gpu}.json}"

    mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" "phase7_results/results"
    rm -f "$BASE/state/pipeline.done"
    for gid in 5 6 7; do rm -f "$BASE/state/gpu${gid}.done"; done

    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
CONTROL_RECORDS=$CONTROL_RECORDS
MODEL_KEY=$MODEL_KEY
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
LAYERS_CSV=$LAYERS_CSV
GPU_IDS_CSV=$GPU_IDS_CSV
ACTIVE_GPUS_CSV=$ACTIVE_GPUS_CSV
LAYERS_G5=${gpu_layers[5]:-}
LAYERS_G6=${gpu_layers[6]:-}
LAYERS_G7=${gpu_layers[7]:-}
PARTIALS_CSV=$PARTIALS_CSV
OUT_PARTIAL_TEMPLATE=$OUT_PARTIAL_TEMPLATE
OUT_MERGED_JSON=$OUT_MERGED_JSON
OUT_MERGED_MD=$OUT_MERGED_MD
DECODER_CHECKPOINT=$DECODER_CHECKPOINT
DECODER_DEVICE=$DECODER_DEVICE
DECODER_BATCH_SIZE=$DECODER_BATCH_SIZE
DECODER_PREDS_CACHE_TEMPLATE=$DECODER_PREDS_CACHE_TEMPLATE
CFG

    COORD_SESSION="p7saetc_coord_${RUN_ID}"
    tmux has-session -t "$COORD_SESSION" 2>/dev/null && tmux kill-session -t "$COORD_SESSION"
    for gid in 5 6 7; do
      ws="p7saetc_g${gid}_${RUN_ID}"
      tmux has-session -t "$ws" 2>/dev/null && tmux kill-session -t "$ws"
      layers_for_gpu="${gpu_layers[$gid]:-}"
      if [[ -n "$layers_for_gpu" ]]; then
        tmux new-session -d -s "$ws" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && GPU_ID=$gid LAYERS='$layers_for_gpu' '$0' worker-g$gid"
      fi
    done
    tmux new-session -d -s "$COORD_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched phase7 sae trajectory coherence"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  model_key: $MODEL_KEY"
    echo "  layers: $LAYERS_CSV"
    echo "  active gpus: $ACTIVE_GPUS_CSV"
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

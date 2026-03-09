#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
PY="${PYTHON:-.venv/bin/python3}"

usage() {
  cat <<'USAGE'
usage: experiments/run_phase7_optionc_generated.sh {launch|precompute|train-decoder|worker-a|worker-b|worker-c|score|evaluate|coordinator}
USAGE
}

require_env() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
}

file_sha256() {
  local p="$1"
  sha256sum "$p" | awk '{print $1}'
}

dataset_rows_sha() {
  local p="$1"
  "$PY" - <<PY
import json
from pathlib import Path
d=json.load(open(${p@Q}))
rp=d.get("rows_path")
ok=bool(str(d.get("status",""))=="ok" and rp)
if not ok:
    print("")
    raise SystemExit(0)
rows_p=Path(str(rp))
if not rows_p.is_absolute():
    cand=(Path(${p@Q}).parent / rows_p).resolve()
    rows_p=cand if cand.exists() else rows_p.resolve()
if not rows_p.exists():
    print("")
    raise SystemExit(0)
print(str(d.get("rows_sha256","")))
PY
}

dataset_ok() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  json_parseable "$p" || return 1
  local rs
  rs="$(dataset_rows_sha "$p")"
  [[ -n "$rs" ]]
}

stage_hash_ok() {
  local done_file="$1"
  local hash_file="$2"
  local want_hash="$3"
  [[ -f "$done_file" && -f "$hash_file" ]] || return 1
  local got_hash
  got_hash="$(cat "$hash_file" 2>/dev/null || true)"
  [[ "$got_hash" == "$want_hash" ]]
}

write_stage_hash() {
  local hash_file="$1"
  local hash_val="$2"
  echo "$hash_val" > "$hash_file"
}

resolve_worker_gpu_slots() {
  : "${GPU_IDS_CSV:?GPU_IDS_CSV required}"
  IFS=',' read -r -a _gpu_arr <<< "$GPU_IDS_CSV"
  if (( ${#_gpu_arr[@]} < 3 )); then
    echo "GPU_IDS_CSV must contain at least 3 GPUs (got: $GPU_IDS_CSV)" >&2
    return 1
  fi
  local g0="${_gpu_arr[0]// /}"
  local g1="${_gpu_arr[1]// /}"
  local g2="${_gpu_arr[2]// /}"
  export WORKER_GPU_A="${WORKER_GPU_A:-$g0}"
  export WORKER_GPU_B="${WORKER_GPU_B:-$g1}"
  export WORKER_GPU_C="${WORKER_GPU_C:-$g2}"
}

wait_for_gpus_free() {
  local gpu_csv="$1"
  local util_max="${2:-10}"
  local mem_max_mb="${3:-2000}"
  local required_hits="${4:-3}"
  local sleep_s="${5:-20}"
  local timeout_s="${6:-172800}"
  local start now elapsed ok_hits
  start="$(date +%s)"
  ok_hits=0
  while true; do
    local ok="1"
    IFS=',' read -r -a gpu_arr <<< "$gpu_csv"
    local g u m
    for g in "${gpu_arr[@]}"; do
      read -r u m < <(nvidia-smi --id="$g" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}')
      if [[ -z "${u:-}" || -z "${m:-}" ]]; then
        ok="0"
        break
      fi
      if (( ${u%.*} > util_max )) || (( ${m%.*} > mem_max_mb )); then
        ok="0"
        break
      fi
    done
    if [[ "$ok" == "1" ]]; then
      ok_hits=$((ok_hits + 1))
      if (( ok_hits >= required_hits )); then
        return 0
      fi
    else
      ok_hits=0
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > timeout_s )); then
      echo "timeout waiting for gpus free: $gpu_csv" >&2
      return 1
    fi
    sleep "$sleep_s"
  done
}

wait_for_file_with_session() {
  local done_file="$1"
  local session="$2"
  local timeout_sec="${3:-172800}"
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
    elapsed=$((now - start))
    if (( elapsed > timeout_sec )); then
      echo "timeout waiting for $done_file (session=$session)" >&2
      return 1
    fi
    sleep 10
  done
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

scope_vars() {
  local scope="$1"
  resolve_worker_gpu_slots
  if [[ "$scope" == "canary" ]]; then
    export SCOPE_PAIRS="$CANARY_PAIRS"
    export SCOPE_BOOTSTRAP="$CANARY_BOOTSTRAP_N"
  else
    export SCOPE_PAIRS="$FULL_PAIRS"
    export SCOPE_BOOTSTRAP="$FULL_BOOTSTRAP_N"
  fi
  export SCOPE_DATASET_JSON="phase7_results/results/optionc_paired_dataset_${RUN_ID}_${scope}.json"
  export SCOPE_EVAL_JSON="phase7_results/results/optionc_eval_${RUN_ID}_${scope}.json"
  export SCOPE_EVAL_MD="phase7_results/results/optionc_eval_${RUN_ID}_${scope}.md"
  export SCOPE_CLAIM_JSON="phase7_results/results/optionc_claim_boundary_${RUN_ID}_${scope}.json"
  export SCOPE_PARTIAL_A="phase7_results/results/optionc_internal_consistency_${RUN_ID}_${scope}_slotA_gpu${WORKER_GPU_A}.json"
  export SCOPE_PARTIAL_B="phase7_results/results/optionc_internal_consistency_${RUN_ID}_${scope}_slotB_gpu${WORKER_GPU_B}.json"
  export SCOPE_PARTIAL_C="phase7_results/results/optionc_internal_consistency_${RUN_ID}_${scope}_slotC_gpu${WORKER_GPU_C}.json"
  export SCOPE_LAYERS_A="${SCOPE_LAYERS_A:-0,3,6,9,12,15,18,21,24,27}"
  export SCOPE_LAYERS_B="${SCOPE_LAYERS_B:-1,4,7,10,13,16,19,22,25}"
  export SCOPE_LAYERS_C="${SCOPE_LAYERS_C:-2,5,8,11,14,17,20,23,26}"
  export SCOPE_PRECOMP_DONE="$BASE/state/precompute_${scope}.done"
  export SCOPE_SCORE_DONE="$BASE/state/score_${scope}.done"
  export SCOPE_EVAL_DONE="$BASE/state/evaluate_${scope}.done"
  export SCOPE_DECODER_DONE="$BASE/state/train_decoder_${scope}.done"
  export SCOPE_DECODER_HASH="$BASE/state/train_decoder_${scope}.hash"
  export SCOPE_PRECOMP_HASH="$BASE/state/precompute_${scope}.hash"
  export SCOPE_SCORE_HASH="$BASE/state/score_${scope}.hash"
  export SCOPE_EVAL_HASH="$BASE/state/evaluate_${scope}.hash"
  export SCOPE_DECODER_CKPT="phase7_results/checkpoints/optionc_decoder_${RUN_ID}_${DOMAIN}_${scope}.pt"
  export SCOPE_DECODER_JSON="phase7_results/results/optionc_decoder_${RUN_ID}_${DOMAIN}_${scope}.json"
}

run_precompute() {
  require_env
  : "${SCOPE:?SCOPE required}"
  scope_vars "$SCOPE"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/precompute_${SCOPE}.log"
  local pre_hash_input pre_hash
  pre_hash_input="$(
    printf '%s|' \
      "$MODEL_KEY" "$DOMAIN" "$SCOPE" "$SCOPE_PAIRS" "$SEED" "$LEXICAL_FRACTION" "$ANSWER_TOL" \
      "$GEN_MAX_NEW_TOKENS" "$GEN_DO_SAMPLE" "$GEN_TEMPERATURE" "$GEN_TOP_P" "$GEN_RETRIES" "$GEN_BATCH_SIZE" \
      "$PRECOMPUTE_SHARDS" "$PRECOMPUTE_SHARD_GPUS" "$(file_sha256 phase7/contradictory_pair_prepare.py)"
  )"
  pre_hash="$(printf '%s' "$pre_hash_input" | sha256sum | awk '{print $1}')"
  if stage_hash_ok "$SCOPE_PRECOMP_DONE" "$SCOPE_PRECOMP_HASH" "$pre_hash" && dataset_ok "$SCOPE_DATASET_JSON"; then
    {
      echo "[$(date -Is)] precompute skip (hash+artifacts match) scope=${SCOPE}"
    } >>"$logf" 2>&1
    return 0
  fi
  {
    echo "[$(date -Is)] precompute start scope=${SCOPE} pairs=${SCOPE_PAIRS}"
    if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
      export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    fi
    export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_cache}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
    mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
    wait_for_gpus_free "${GPU_IDS_CSV}" "${GPU_WAIT_UTIL_MAX}" "${GPU_WAIT_MEM_MAX_MB}" 3 20 "${GPU_WAIT_TIMEOUT_S}"
    if (( PRECOMPUTE_SHARDS > 1 )); then
      IFS=',' read -r -a shard_gpu_arr <<< "$PRECOMPUTE_SHARD_GPUS"
      if (( ${#shard_gpu_arr[@]} < PRECOMPUTE_SHARDS )); then
        echo "PRECOMPUTE_SHARD_GPUS must provide at least PRECOMPUTE_SHARDS GPUs" >&2
        exit 1
      fi
      local shard_outputs=()
      local shard_idx shard_gpu shard_out shard_log
      # If a GPU appears multiple times in PRECOMPUTE_SHARD_GPUS, run those shards
      # sequentially on that GPU to avoid loading multiple full Qwen models at once.
      declare -A shard_pid_by_gpu=()
      for ((shard_idx=0; shard_idx<PRECOMPUTE_SHARDS; shard_idx++)); do
        shard_gpu="${shard_gpu_arr[$shard_idx]}"
        shard_gpu="${shard_gpu// /}"
        if [[ -n "${shard_pid_by_gpu[$shard_gpu]:-}" ]]; then
          wait "${shard_pid_by_gpu[$shard_gpu]}"
        fi
        shard_out="phase7_results/results/optionc_paired_dataset_${RUN_ID}_${SCOPE}.shard${shard_idx}.json"
        shard_log="$BASE/logs/precompute_${SCOPE}_shard${shard_idx}.log"
        shard_outputs+=("$shard_out")
        CUDA_VISIBLE_DEVICES="${shard_gpu}" "$PY" phase7/contradictory_pair_prepare.py \
          --model-key "$MODEL_KEY" \
          --domain "$DOMAIN" \
          --sample-pairs "$SCOPE_PAIRS" \
          --seed "$SEED" \
          --lexical-fraction "$LEXICAL_FRACTION" \
          --answer-tol "$ANSWER_TOL" \
          --gen-max-new-tokens "$GEN_MAX_NEW_TOKENS" \
          --gen-temperature "$GEN_TEMPERATURE" \
          --gen-top-p "$GEN_TOP_P" \
          --gen-retries "$GEN_RETRIES" \
          --gen-batch-size "$GEN_BATCH_SIZE" \
          ${GEN_DO_SAMPLE_FLAG} \
          --num-shards "$PRECOMPUTE_SHARDS" \
          --shard-index "$shard_idx" \
          --device cuda:0 \
          --output "$shard_out" >"$shard_log" 2>&1 &
        shard_pid_by_gpu["$shard_gpu"]="$!"
      done
      local gpu_key
      for gpu_key in "${!shard_pid_by_gpu[@]}"; do
        wait "${shard_pid_by_gpu[$gpu_key]}"
      done
      local so
      for so in "${shard_outputs[@]}"; do
        json_parseable "$so"
      done
      "$PY" phase7/contradictory_pair_prepare.py \
        --merge-shards "${shard_outputs[@]}" \
        --output "$SCOPE_DATASET_JSON"
    else
      CUDA_VISIBLE_DEVICES="${PRECOMPUTE_GPU}" "$PY" phase7/contradictory_pair_prepare.py \
        --model-key "$MODEL_KEY" \
        --domain "$DOMAIN" \
        --sample-pairs "$SCOPE_PAIRS" \
        --seed "$SEED" \
        --lexical-fraction "$LEXICAL_FRACTION" \
        --answer-tol "$ANSWER_TOL" \
        --gen-max-new-tokens "$GEN_MAX_NEW_TOKENS" \
        --gen-temperature "$GEN_TEMPERATURE" \
        --gen-top-p "$GEN_TOP_P" \
        --gen-retries "$GEN_RETRIES" \
        --gen-batch-size "$GEN_BATCH_SIZE" \
        ${GEN_DO_SAMPLE_FLAG} \
        --device cuda:0 \
        --output "$SCOPE_DATASET_JSON"
    fi
    dataset_ok "$SCOPE_DATASET_JSON"
    json_parseable "$SCOPE_DATASET_JSON"
    touch "$SCOPE_PRECOMP_DONE"
    write_stage_hash "$SCOPE_PRECOMP_HASH" "$pre_hash"
    echo "[$(date -Is)] precompute done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_train_decoder() {
  require_env
  : "${SCOPE:?SCOPE required}"
  scope_vars "$SCOPE"
  mkdir -p "$BASE/logs" "$BASE/state"
  local logf="$BASE/logs/train_decoder_${SCOPE}.log"
  if [[ "${DECODER_AUTO_TRAIN}" != "1" ]]; then
    {
      echo "[$(date -Is)] train-decoder skip (DECODER_AUTO_TRAIN=${DECODER_AUTO_TRAIN}) scope=${SCOPE}"
    } >>"$logf" 2>&1
    touch "$SCOPE_DECODER_DONE"
    write_stage_hash "$SCOPE_DECODER_HASH" "skip"
    return 0
  fi
  local ds_rows_sha train_hash_input train_hash
  ds_rows_sha="$(dataset_rows_sha "$SCOPE_DATASET_JSON")"
  train_hash_input="$(
    printf '%s|' \
      "$MODEL_KEY" "$DOMAIN" "$SCOPE" "$ds_rows_sha" "$DECODER_DOMAIN" "$DECODER_LAYERS" \
      "$DECODER_TRAIN_EPOCHS" "$DECODER_TRAIN_BATCH_SIZE" "$DECODER_TRAIN_LR" "$DECODER_TRAIN_WEIGHT_DECAY" \
      "$(file_sha256 phase7/train_optionc_domain_decoder.py)" \
      "$(file_sha256 phase7/optionc_domain_decoder.py)"
  )"
  train_hash="$(printf '%s' "$train_hash_input" | sha256sum | awk '{print $1}')"
  if stage_hash_ok "$SCOPE_DECODER_DONE" "$SCOPE_DECODER_HASH" "$train_hash" \
    && [[ -f "$SCOPE_DECODER_CKPT" ]] \
    && json_parseable "$SCOPE_DECODER_JSON"; then
    {
      echo "[$(date -Is)] train-decoder skip (hash+artifacts match) scope=${SCOPE}"
    } >>"$logf" 2>&1
    return 0
  fi
  {
    echo "[$(date -Is)] train-decoder start scope=${SCOPE} domain=${DECODER_DOMAIN}"
    wait_for_gpus_free "${GPU_IDS_CSV}" "${GPU_WAIT_UTIL_MAX}" "${GPU_WAIT_MEM_MAX_MB}" 3 20 "${GPU_WAIT_TIMEOUT_S}"
    CUDA_VISIBLE_DEVICES="${DECODER_TRAIN_GPU}" "$PY" phase7/train_optionc_domain_decoder.py \
      --paired-dataset "$SCOPE_DATASET_JSON" \
      --decoder-domain "$DECODER_DOMAIN" \
      --model-key "$MODEL_KEY" \
      --layers "$DECODER_LAYERS" \
      --seed "$SEED" \
      --epochs "$DECODER_TRAIN_EPOCHS" \
      --batch-size "$DECODER_TRAIN_BATCH_SIZE" \
      --lr "$DECODER_TRAIN_LR" \
      --weight-decay "$DECODER_TRAIN_WEIGHT_DECAY" \
      --device cuda:0 \
      --output-checkpoint "$SCOPE_DECODER_CKPT" \
      --output-json "$SCOPE_DECODER_JSON"
    [[ -f "$SCOPE_DECODER_CKPT" ]] || { echo "missing trained decoder checkpoint: $SCOPE_DECODER_CKPT" >&2; exit 1; }
    json_parseable "$SCOPE_DECODER_JSON"
    touch "$SCOPE_DECODER_DONE"
    write_stage_hash "$SCOPE_DECODER_HASH" "$train_hash"
    echo "[$(date -Is)] train-decoder done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_worker() {
  require_env
  : "${SCOPE:?SCOPE required}"
  : "${WORKER_GPU:?WORKER_GPU required}"
  : "${WORKER_LAYERS:?WORKER_LAYERS required}"
  : "${WORKER_OUTPUT:?WORKER_OUTPUT required}"
  local worker_marker="${WORKER_MARKER:-gpu${WORKER_GPU}}"
  scope_vars "$SCOPE"
  mkdir -p "$BASE/logs" "$BASE/state"
  local logf="$BASE/logs/worker_${worker_marker}_${SCOPE}.log"
  {
    echo "[$(date -Is)] worker start scope=${SCOPE} marker=${worker_marker} gpu=${WORKER_GPU} layers=${WORKER_LAYERS}"
    CUDA_VISIBLE_DEVICES="${WORKER_GPU}" "$PY" phase7/internal_consistency_optionc.py \
      --paired-dataset "$SCOPE_DATASET_JSON" \
      --model-key "$MODEL_KEY" \
      --layers "$WORKER_LAYERS" \
      --saes-dir "$SAES_DIR" \
      --activations-dir "$ACTIVATIONS_DIR" \
      --phase4-top-features "$PHASE4_TOP_FEATURES" \
      --feature-set "$FEATURE_SET" \
      --device cuda:0 \
      --batch-size "$SCORE_BATCH_SIZE" \
      --run-tag "${RUN_TAG}_${SCOPE}" \
      --output-json "$WORKER_OUTPUT"
    json_parseable "$WORKER_OUTPUT"
    touch "$BASE/state/${SCOPE}_${worker_marker}.done"
    echo "[$(date -Is)] worker done scope=${SCOPE} marker=${worker_marker} gpu=${WORKER_GPU}"
  } >>"$logf" 2>&1
}

run_score() {
  require_env
  : "${SCOPE:?SCOPE required}"
  scope_vars "$SCOPE"
  mkdir -p "$BASE/logs" "$BASE/state"
  local logf="$BASE/logs/score_${SCOPE}.log"
  local ds_rows_sha score_hash_input score_hash
  ds_rows_sha="$(dataset_rows_sha "$SCOPE_DATASET_JSON")"
  score_hash_input="$(
    printf '%s|' \
      "$MODEL_KEY" "$SCOPE" "$ds_rows_sha" "$SCOPE_LAYERS_A" "$SCOPE_LAYERS_B" "$SCOPE_LAYERS_C" \
      "$FEATURE_SET" "$SCORE_BATCH_SIZE" "$PHASE4_TOP_FEATURES" \
      "$(file_sha256 phase7/internal_consistency_optionc.py)"
  )"
  score_hash="$(printf '%s' "$score_hash_input" | sha256sum | awk '{print $1}')"
  if stage_hash_ok "$SCOPE_SCORE_DONE" "$SCOPE_SCORE_HASH" "$score_hash" \
    && json_parseable "$SCOPE_PARTIAL_A" \
    && json_parseable "$SCOPE_PARTIAL_B" \
    && json_parseable "$SCOPE_PARTIAL_C"; then
    {
      echo "[$(date -Is)] score skip (hash+artifacts match) scope=${SCOPE}"
    } >>"$logf" 2>&1
    return 0
  fi
  {
    echo "[$(date -Is)] score start scope=${SCOPE}"
    wait_for_gpus_free "${GPU_IDS_CSV}" "${GPU_WAIT_UTIL_MAX}" "${GPU_WAIT_MEM_MAX_MB}" 3 20 "${GPU_WAIT_TIMEOUT_S}"
    if [[ "$WORKER_GPU_A" == "$WORKER_GPU_B" || "$WORKER_GPU_A" == "$WORKER_GPU_C" || "$WORKER_GPU_B" == "$WORKER_GPU_C" ]]; then
      echo "[$(date -Is)] score duplicate GPU slots detected; running workers sequentially"
      SCOPE="$SCOPE" WORKER_MARKER="slotA" WORKER_GPU="$WORKER_GPU_A" WORKER_LAYERS="$SCOPE_LAYERS_A" WORKER_OUTPUT="$SCOPE_PARTIAL_A" "$0" worker-a
      SCOPE="$SCOPE" WORKER_MARKER="slotB" WORKER_GPU="$WORKER_GPU_B" WORKER_LAYERS="$SCOPE_LAYERS_B" WORKER_OUTPUT="$SCOPE_PARTIAL_B" "$0" worker-b
      SCOPE="$SCOPE" WORKER_MARKER="slotC" WORKER_GPU="$WORKER_GPU_C" WORKER_LAYERS="$SCOPE_LAYERS_C" WORKER_OUTPUT="$SCOPE_PARTIAL_C" "$0" worker-c
    else
      local sess_a="p7oc_a_${RUN_ID}_${SCOPE}"
      local sess_b="p7oc_b_${RUN_ID}_${SCOPE}"
      local sess_c="p7oc_c_${RUN_ID}_${SCOPE}"
      tmux new-session -d -s "$sess_a" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && SCOPE='${SCOPE}' WORKER_MARKER='slotA' WORKER_GPU='${WORKER_GPU_A}' WORKER_LAYERS='${SCOPE_LAYERS_A}' WORKER_OUTPUT='${SCOPE_PARTIAL_A}' '$0' worker-a"
      tmux new-session -d -s "$sess_b" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && SCOPE='${SCOPE}' WORKER_MARKER='slotB' WORKER_GPU='${WORKER_GPU_B}' WORKER_LAYERS='${SCOPE_LAYERS_B}' WORKER_OUTPUT='${SCOPE_PARTIAL_B}' '$0' worker-b"
      tmux new-session -d -s "$sess_c" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && SCOPE='${SCOPE}' WORKER_MARKER='slotC' WORKER_GPU='${WORKER_GPU_C}' WORKER_LAYERS='${SCOPE_LAYERS_C}' WORKER_OUTPUT='${SCOPE_PARTIAL_C}' '$0' worker-c"
      wait_for_file_with_session "$BASE/state/${SCOPE}_slotA.done" "$sess_a" "${GPU_WAIT_TIMEOUT_S}"
      wait_for_file_with_session "$BASE/state/${SCOPE}_slotB.done" "$sess_b" "${GPU_WAIT_TIMEOUT_S}"
      wait_for_file_with_session "$BASE/state/${SCOPE}_slotC.done" "$sess_c" "${GPU_WAIT_TIMEOUT_S}"
    fi
    json_parseable "$SCOPE_PARTIAL_A"
    json_parseable "$SCOPE_PARTIAL_B"
    json_parseable "$SCOPE_PARTIAL_C"
    touch "$SCOPE_SCORE_DONE"
    write_stage_hash "$SCOPE_SCORE_HASH" "$score_hash"
    echo "[$(date -Is)] score done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_evaluate() {
  require_env
  : "${SCOPE:?SCOPE required}"
  scope_vars "$SCOPE"
  mkdir -p "$BASE/logs" "$BASE/state"
  local logf="$BASE/logs/evaluate_${SCOPE}.log"
  local ds_rows_sha eval_hash_input eval_hash dec_stat
  local decoder_checkpoint_local="$DECODER_CHECKPOINT"
  local decoder_domain_local="${DECODER_DOMAIN:-auto}"
  if [[ "${DECODER_AUTO_TRAIN}" == "1" ]]; then
    decoder_checkpoint_local="$SCOPE_DECODER_CKPT"
    decoder_domain_local="${DECODER_DOMAIN}"
  fi
  if [[ ! -f "$decoder_checkpoint_local" ]]; then
    decoder_checkpoint_local=""
  fi
  ds_rows_sha="$(dataset_rows_sha "$SCOPE_DATASET_JSON")"
  dec_stat="$(stat -c '%s:%Y' "$decoder_checkpoint_local" 2>/dev/null || echo missing)"
  eval_hash_input="$(
    printf '%s|' \
      "$MODEL_KEY" "$SCOPE" "$ds_rows_sha" \
      "$(file_sha256 "$SCOPE_PARTIAL_A")" "$(file_sha256 "$SCOPE_PARTIAL_B")" "$(file_sha256 "$SCOPE_PARTIAL_C")" \
      "$decoder_checkpoint_local" "$decoder_domain_local" "$dec_stat" "$DECODER_BATCH_SIZE" "$TRAIN_TEST_FRACTION" \
      "$SPLIT_SEED" "$CV_FOLDS" "$CV_SEED" "$SCOPE_BOOTSTRAP" "$BOOTSTRAP_SEED" \
      "$FIT_EPOCHS" "$FIT_LR" "$FIT_WEIGHT_DECAY" "$FIT_DEVICE" "$CPU_WORKERS" \
      "$(file_sha256 phase7/evaluate_optionc.py)"
  )"
  eval_hash="$(printf '%s' "$eval_hash_input" | sha256sum | awk '{print $1}')"
  if stage_hash_ok "$SCOPE_EVAL_DONE" "$SCOPE_EVAL_HASH" "$eval_hash" \
    && json_parseable "$SCOPE_EVAL_JSON" \
    && json_parseable "$SCOPE_CLAIM_JSON"; then
    {
      echo "[$(date -Is)] evaluate skip (hash+artifacts match) scope=${SCOPE}"
    } >>"$logf" 2>&1
    return 0
  fi
  {
    echo "[$(date -Is)] evaluate start scope=${SCOPE}"
    "$PY" phase7/evaluate_optionc.py \
      --paired-dataset "$SCOPE_DATASET_JSON" \
      --partials "$SCOPE_PARTIAL_A" "$SCOPE_PARTIAL_B" "$SCOPE_PARTIAL_C" \
      --decoder-checkpoint "$decoder_checkpoint_local" \
      --decoder-domain "$decoder_domain_local" \
      --decoder-device "$DECODER_DEVICE" \
      --decoder-batch-size "$DECODER_BATCH_SIZE" \
      --train-test-fraction "$TRAIN_TEST_FRACTION" \
      --split-seed "$SPLIT_SEED" \
      --cv-folds "$CV_FOLDS" \
      --cv-seed "$CV_SEED" \
      --bootstrap-n "$SCOPE_BOOTSTRAP" \
      --bootstrap-seed "$BOOTSTRAP_SEED" \
      --cpu-workers "$CPU_WORKERS" \
      --epochs "$FIT_EPOCHS" \
      --lr "$FIT_LR" \
      --weight-decay "$FIT_WEIGHT_DECAY" \
      --fit-device "$FIT_DEVICE" \
      --primary-auroc-threshold "$PRIMARY_AUROC_THRESHOLD" \
      --primary-ci-lower-threshold "$PRIMARY_CI_LOWER_THRESHOLD" \
      --lexical-auroc-max "$LEXICAL_AUROC_MAX" \
      --wrong-minus-lexical-min "$WRONG_MINUS_LEXICAL_MIN" \
      --train-exclude-pair-types "$TRAIN_EXCLUDE_PAIR_TYPES" \
      --output-json "$SCOPE_EVAL_JSON" \
      --output-md "$SCOPE_EVAL_MD" \
      --claim-json "$SCOPE_CLAIM_JSON"
    json_parseable "$SCOPE_EVAL_JSON"
    json_parseable "$SCOPE_CLAIM_JSON"
    touch "$SCOPE_EVAL_DONE"
    write_stage_hash "$SCOPE_EVAL_HASH" "$eval_hash"
    echo "[$(date -Is)] evaluate done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_coordinator() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/coordinator.log"
  {
    echo "[$(date -Is)] coordinator start run_id=${RUN_ID}"
    # Canary (always call stage runners; each stage performs strict hash+artifact skip)
    SCOPE=canary "$0" precompute
    SCOPE=canary "$0" train-decoder
    SCOPE=canary "$0" score
    SCOPE=canary "$0" evaluate
    scope_vars canary
    canary_ok="$("$PY" - <<PY
import json
d=json.load(open(${SCOPE_EVAL_JSON@Q}))
ok = str(d.get("status",""))=="ok" and int(((d.get("cv_diagnostics") or {}).get("cv_trace_overlap_count",0)))==0
print("true" if ok else "false")
PY
)"
    if [[ "$canary_ok" != "true" ]]; then
      echo "canary integrity failed; stopping before full" >&2
      "$PY" - <<PY
from phase7.common import save_json
save_json(${BASE@Q} + "/meta/summary.json", {
  "status":"blocked_canary_integrity",
  "run_id": ${RUN_ID@Q},
})
PY
      touch "$BASE/state/pipeline.done"
      exit 1
    fi

    # Full (always call stage runners; each stage performs strict hash+artifact skip)
    SCOPE=full "$0" precompute
    SCOPE=full "$0" train-decoder
    SCOPE=full "$0" score
    SCOPE=full "$0" evaluate

    scope_vars full
    "$PY" - <<PY
import json
from pathlib import Path
run_id=${RUN_ID@Q}
base=Path(${BASE@Q})
out=Path(f"phase7_results/results/optionc_summary_{run_id}.json")
canary=json.load(open(f"phase7_results/results/optionc_eval_{run_id}_canary.json"))
full=json.load(open(f"phase7_results/results/optionc_eval_{run_id}_full.json"))
claim=json.load(open(f"phase7_results/results/optionc_claim_boundary_{run_id}_full.json"))
summary={
  "schema_version":"phase7_optionc_summary_v1",
  "status":"ok",
  "run_id": run_id,
  "domain": ${DOMAIN@Q},
  "artifacts":{
    "dataset_canary": f"phase7_results/results/optionc_paired_dataset_{run_id}_canary.json",
    "dataset_full": f"phase7_results/results/optionc_paired_dataset_{run_id}_full.json",
    "eval_canary": f"phase7_results/results/optionc_eval_{run_id}_canary.json",
    "eval_full": f"phase7_results/results/optionc_eval_{run_id}_full.json",
    "claim_full": f"phase7_results/results/optionc_claim_boundary_{run_id}_full.json",
  },
  "canary": {
    "primary_member_auroc": ((canary.get("single_split") or {}).get("primary_member_auroc")),
    "strict_gate_pass": ((canary.get("claim_gate") or {}).get("strict_gate_pass")),
  },
  "full": {
    "cv_primary_pooled_auroc": ((full.get("cv_diagnostics") or {}).get("cv_primary_pooled_auroc")),
    "lexical_probe_auroc": ((full.get("single_split") or {}).get("lexical_probe_auroc")),
    "wrong_minus_lexical_delta": ((full.get("single_split") or {}).get("wrong_minus_lexical_delta")),
    "strict_gate_pass": ((full.get("claim_gate") or {}).get("strict_gate_pass")),
    "faithfulness_claim_enabled": bool(claim.get("faithfulness_claim_enabled")),
  },
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))
base.joinpath("meta","summary.json").write_text(json.dumps(summary, indent=2))
print(out)
PY

    touch "$BASE/state/pipeline.done"
    echo "[$(date -Is)] coordinator done run_id=${RUN_ID}"
  } >>"$logf" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_optionc_generated}"
    RUN_TAG="${RUN_TAG:-phase7_optionc_generated_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"
    MODEL_KEY="${MODEL_KEY:-qwen2.5-7b}"
    DOMAIN="${DOMAIN:-arithmetic}"
    if [[ -z "${GPU_IDS_CSV:-}" ]]; then
      GPU_IDS_CSV="$("$PY" - <<'PY'
import subprocess
import json
try:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
except Exception:
    print("5,6,7")
    raise SystemExit(0)
rows = []
for ln in raw.strip().splitlines():
    try:
        idx, util, mem = [x.strip() for x in ln.split(",")]
        rows.append((int(idx), float(util), float(mem)))
    except Exception:
        continue
free = [idx for idx, util, mem in rows if util < 10.0 and mem < 2000.0]
if len(free) >= 3:
    print(",".join(str(x) for x in free[:3]))
elif len(rows) >= 3:
    rows_sorted = sorted(rows, key=lambda t: (t[2], t[1], t[0]))
    print(",".join(str(t[0]) for t in rows_sorted[:3]))
else:
    print("5,6,7")
PY
)"
    fi
    IFS=',' read -r -a _gpu_arr <<< "$GPU_IDS_CSV"
    if (( ${#_gpu_arr[@]} < 3 )); then
      echo "GPU_IDS_CSV must contain at least 3 GPUs (got: $GPU_IDS_CSV)" >&2
      exit 1
    fi
    WORKER_GPU_A="${WORKER_GPU_A:-${_gpu_arr[0]// /}}"
    WORKER_GPU_B="${WORKER_GPU_B:-${_gpu_arr[1]// /}}"
    WORKER_GPU_C="${WORKER_GPU_C:-${_gpu_arr[2]// /}}"
    PRECOMPUTE_GPU="${PRECOMPUTE_GPU:-$WORKER_GPU_C}"
    PRECOMPUTE_SHARDS="${PRECOMPUTE_SHARDS:-3}"
    PRECOMPUTE_SHARD_GPUS="${PRECOMPUTE_SHARD_GPUS:-${WORKER_GPU_A},${WORKER_GPU_B},${WORKER_GPU_C}}"
    GPU_WAIT_UTIL_MAX="${GPU_WAIT_UTIL_MAX:-10}"
    GPU_WAIT_MEM_MAX_MB="${GPU_WAIT_MEM_MAX_MB:-2000}"
    GPU_WAIT_TIMEOUT_S="${GPU_WAIT_TIMEOUT_S:-172800}"

    CANARY_PAIRS="${CANARY_PAIRS:-400}"
    FULL_PAIRS="${FULL_PAIRS:-2400}"
    LEXICAL_FRACTION="${LEXICAL_FRACTION:-0.20}"
    ANSWER_TOL="${ANSWER_TOL:-1e-6}"

    GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-180}"
    GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.2}"
    GEN_TOP_P="${GEN_TOP_P:-0.95}"
    GEN_RETRIES="${GEN_RETRIES:-1}"
    GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-12}"
    GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-0}"
    GEN_DO_SAMPLE_FLAG=""
    if [[ "$GEN_DO_SAMPLE" == "1" ]]; then
      GEN_DO_SAMPLE_FLAG="--gen-do-sample"
    fi

    SAES_DIR="${SAES_DIR:-phase2_results/saes_qwen25_7b_12x_topk/saes}"
    ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
    PHASE4_TOP_FEATURES="${PHASE4_TOP_FEATURES:-phase7_results/runs/20260308_165109_phase7_qwen_trackc_upgrade/interventions/top_features_per_layer_qwen_phase7_qwen_trackc_upgrade_20260308_165109_phase7_qwen_trackc_upgrade.json}"
    FEATURE_SET="${FEATURE_SET:-eq_pre_result_150}"
    SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-256}"
    SCOPE_LAYERS_A="${SCOPE_LAYERS_A:-0,3,6,9,12,15,18,21,24,27}"
    SCOPE_LAYERS_B="${SCOPE_LAYERS_B:-1,4,7,10,13,16,19,22,25}"
    SCOPE_LAYERS_C="${SCOPE_LAYERS_C:-2,5,8,11,14,17,20,23,26}"

    DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-phase7_results/runs/20260308_165109_phase7_qwen_trackc_upgrade/checkpoints/state_raw_every2_even_d1tier1.pt}"
    DECODER_DEVICE="${DECODER_DEVICE:-cuda:${WORKER_GPU_A}}"
    DECODER_BATCH_SIZE="${DECODER_BATCH_SIZE:-128}"
    DECODER_DOMAIN="${DECODER_DOMAIN:-${DOMAIN}}"
    DECODER_AUTO_TRAIN="${DECODER_AUTO_TRAIN:-0}"
    if [[ "${DOMAIN}" != "arithmetic" && "${DECODER_AUTO_TRAIN}" == "0" ]]; then
      DECODER_AUTO_TRAIN="1"
    fi
    DECODER_TRAIN_GPU="${DECODER_TRAIN_GPU:-${WORKER_GPU_A}}"
    DECODER_LAYERS="${DECODER_LAYERS:-8,14,20}"
    DECODER_TRAIN_EPOCHS="${DECODER_TRAIN_EPOCHS:-120}"
    DECODER_TRAIN_BATCH_SIZE="${DECODER_TRAIN_BATCH_SIZE:-256}"
    DECODER_TRAIN_LR="${DECODER_TRAIN_LR:-0.0003}"
    DECODER_TRAIN_WEIGHT_DECAY="${DECODER_TRAIN_WEIGHT_DECAY:-0.01}"

    TRAIN_TEST_FRACTION="${TRAIN_TEST_FRACTION:-0.20}"
    SPLIT_SEED="${SPLIT_SEED:-20260309}"
    CV_FOLDS="${CV_FOLDS:-5}"
    CV_SEED="${CV_SEED:-20260309}"
    CANARY_BOOTSTRAP_N="${CANARY_BOOTSTRAP_N:-300}"
    FULL_BOOTSTRAP_N="${FULL_BOOTSTRAP_N:-1000}"
    BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260309}"
    FIT_EPOCHS="${FIT_EPOCHS:-400}"
    FIT_LR="${FIT_LR:-0.03}"
    FIT_WEIGHT_DECAY="${FIT_WEIGHT_DECAY:-0.0001}"
    FIT_DEVICE="${FIT_DEVICE:-cpu}"
    CPU_WORKERS="${CPU_WORKERS:-8}"

    HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
    HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_cache}"
    HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
    TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

    PRIMARY_AUROC_THRESHOLD="${PRIMARY_AUROC_THRESHOLD:-0.70}"
    PRIMARY_CI_LOWER_THRESHOLD="${PRIMARY_CI_LOWER_THRESHOLD:-0.65}"
    LEXICAL_AUROC_MAX="${LEXICAL_AUROC_MAX:-0.60}"
    WRONG_MINUS_LEXICAL_MIN="${WRONG_MINUS_LEXICAL_MIN:-0.10}"
    TRAIN_EXCLUDE_PAIR_TYPES="${TRAIN_EXCLUDE_PAIR_TYPES:-lexical_control}"
    SEED="${SEED:-20260309}"

    mkdir -p "$BASE/logs" "$BASE/meta" "$BASE/state"
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=${RUN_ID@Q}
RUN_TAG=${RUN_TAG@Q}
BASE=${BASE@Q}
MODEL_KEY=${MODEL_KEY@Q}
DOMAIN=${DOMAIN@Q}
GPU_IDS_CSV=${GPU_IDS_CSV@Q}
WORKER_GPU_A=${WORKER_GPU_A@Q}
WORKER_GPU_B=${WORKER_GPU_B@Q}
WORKER_GPU_C=${WORKER_GPU_C@Q}
PRECOMPUTE_GPU=${PRECOMPUTE_GPU@Q}
PRECOMPUTE_SHARDS=${PRECOMPUTE_SHARDS@Q}
PRECOMPUTE_SHARD_GPUS=${PRECOMPUTE_SHARD_GPUS@Q}
GPU_WAIT_UTIL_MAX=${GPU_WAIT_UTIL_MAX@Q}
GPU_WAIT_MEM_MAX_MB=${GPU_WAIT_MEM_MAX_MB@Q}
GPU_WAIT_TIMEOUT_S=${GPU_WAIT_TIMEOUT_S@Q}
CANARY_PAIRS=${CANARY_PAIRS@Q}
FULL_PAIRS=${FULL_PAIRS@Q}
LEXICAL_FRACTION=${LEXICAL_FRACTION@Q}
ANSWER_TOL=${ANSWER_TOL@Q}
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS@Q}
GEN_TEMPERATURE=${GEN_TEMPERATURE@Q}
GEN_TOP_P=${GEN_TOP_P@Q}
GEN_RETRIES=${GEN_RETRIES@Q}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE@Q}
GEN_DO_SAMPLE=${GEN_DO_SAMPLE@Q}
GEN_DO_SAMPLE_FLAG=${GEN_DO_SAMPLE_FLAG@Q}
SAES_DIR=${SAES_DIR@Q}
ACTIVATIONS_DIR=${ACTIVATIONS_DIR@Q}
PHASE4_TOP_FEATURES=${PHASE4_TOP_FEATURES@Q}
FEATURE_SET=${FEATURE_SET@Q}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE@Q}
SCOPE_LAYERS_A=${SCOPE_LAYERS_A@Q}
SCOPE_LAYERS_B=${SCOPE_LAYERS_B@Q}
SCOPE_LAYERS_C=${SCOPE_LAYERS_C@Q}
DECODER_CHECKPOINT=${DECODER_CHECKPOINT@Q}
DECODER_DEVICE=${DECODER_DEVICE@Q}
DECODER_BATCH_SIZE=${DECODER_BATCH_SIZE@Q}
DECODER_DOMAIN=${DECODER_DOMAIN@Q}
DECODER_AUTO_TRAIN=${DECODER_AUTO_TRAIN@Q}
DECODER_TRAIN_GPU=${DECODER_TRAIN_GPU@Q}
DECODER_LAYERS=${DECODER_LAYERS@Q}
DECODER_TRAIN_EPOCHS=${DECODER_TRAIN_EPOCHS@Q}
DECODER_TRAIN_BATCH_SIZE=${DECODER_TRAIN_BATCH_SIZE@Q}
DECODER_TRAIN_LR=${DECODER_TRAIN_LR@Q}
DECODER_TRAIN_WEIGHT_DECAY=${DECODER_TRAIN_WEIGHT_DECAY@Q}
TRAIN_TEST_FRACTION=${TRAIN_TEST_FRACTION@Q}
SPLIT_SEED=${SPLIT_SEED@Q}
CV_FOLDS=${CV_FOLDS@Q}
CV_SEED=${CV_SEED@Q}
CANARY_BOOTSTRAP_N=${CANARY_BOOTSTRAP_N@Q}
FULL_BOOTSTRAP_N=${FULL_BOOTSTRAP_N@Q}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED@Q}
FIT_EPOCHS=${FIT_EPOCHS@Q}
FIT_LR=${FIT_LR@Q}
FIT_WEIGHT_DECAY=${FIT_WEIGHT_DECAY@Q}
FIT_DEVICE=${FIT_DEVICE@Q}
CPU_WORKERS=${CPU_WORKERS@Q}
HF_TOKEN=${HF_TOKEN@Q}
HF_HOME=${HF_HOME@Q}
HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE@Q}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE@Q}
PRIMARY_AUROC_THRESHOLD=${PRIMARY_AUROC_THRESHOLD@Q}
PRIMARY_CI_LOWER_THRESHOLD=${PRIMARY_CI_LOWER_THRESHOLD@Q}
LEXICAL_AUROC_MAX=${LEXICAL_AUROC_MAX@Q}
WRONG_MINUS_LEXICAL_MIN=${WRONG_MINUS_LEXICAL_MIN@Q}
TRAIN_EXCLUDE_PAIR_TYPES=${TRAIN_EXCLUDE_PAIR_TYPES@Q}
SEED=${SEED@Q}
CFG

    COORD_SESSION="p7oc_coord_${RUN_ID}"
    tmux new-session -d -s "$COORD_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"
    echo "launched option-c"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  coordinator: $COORD_SESSION"
    ;;
  precompute)
    run_precompute
    ;;
  train-decoder)
    run_train_decoder
    ;;
  worker-a|worker-b|worker-c|worker-g5|worker-g6|worker-g7)
    run_worker
    ;;
  score)
    run_score
    ;;
  evaluate)
    run_evaluate
    ;;
  coordinator)
    run_coordinator
    ;;
  *)
    usage
    exit 1
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-.venv/bin/python3}"
MODE="${1:-}"

usage() {
  cat <<'USAGE'
usage: experiments/run_phase7_qwen_sae_trajectory_ladder.sh {launch|precompute|train-gpu5|train-gpu6|train-gpu7|coordinator}
USAGE
}

require_env() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
}

log() {
  echo "[$(date -Is)] $*"
}

wait_for_file_with_session() {
  local done_file="$1"
  local session="$2"
  local timeout_sec="${3:-86400}"
  local start now elapsed
  start="$(date +%s)"
  while [[ ! -f "$done_file" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      # Race-safe grace: coordinator/session may exit immediately after writing marker.
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
    sleep 15
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

stage_done_with_parseable_json() {
  local stage_base="$1"
  local out_json="$2"
  [[ -f "${stage_base}/state/pipeline.done" ]] || return 1
  [[ -f "${out_json}" ]] || return 1
  json_parseable "${out_json}"
}

run_precompute() {
  require_env
  mkdir -p "$BASE"/{logs,state,meta,phase6_dataset,dataset,controls,interventions,checkpoints,results_train} phase7_results/results
  local logf="$BASE/logs/precompute.log"
  {
    log "precompute start run_id=$RUN_ID"

    CUDA_VISIBLE_DEVICES=7 "$PY" phase6/collect_expanded_dataset.py \
      --model-key qwen2.5-7b \
      --split train \
      --gsm8k-path datasets/raw/gsm8k/train.jsonl \
      --saes-dir "$SAES_DIR" \
      --activations-dir "$ACTIVATIONS_DIR" \
      --capture-extra-anchors \
      --output-dir "$PHASE6_DIR" \
      --output-prefix "$PHASE6_PREFIX" \
      --device cuda:0

    CUDA_VISIBLE_DEVICES=7 "$PY" phase6/collect_expanded_dataset.py \
      --model-key qwen2.5-7b \
      --split test \
      --gsm8k-path datasets/raw/gsm8k/test.jsonl \
      --saes-dir "$SAES_DIR" \
      --activations-dir "$ACTIVATIONS_DIR" \
      --capture-extra-anchors \
      --output-dir "$PHASE6_DIR" \
      --output-prefix "$PHASE6_PREFIX" \
      --device cuda:0

    "$PY" phase7/build_step_trace_dataset.py \
      --phase6-train "$PHASE6_TRAIN" \
      --phase6-test "$PHASE6_TEST" \
      --model-key qwen2.5-7b \
      --state-ontology v2_expanded \
      --output-dir "$TRACE_DIR"

    CUDA_VISIBLE_DEVICES=7 "$PY" phase7/derive_probe_top_features_from_expanded.py \
      --expanded-dataset "$PHASE6_TRAIN" \
      --model-key qwen2.5-7b \
      --split train \
      --top-k 64 \
      --max-records 5000 \
      --seed 20260307 \
      --output "$PROBE_TOP_FEATURES"

    "$PY" phase7/variable_subspace_builder.py \
      --probe-top-features "$PROBE_TOP_FEATURES" \
      --probe-position eq \
      --model-key qwen2.5-7b \
      --layers 8 14 20 \
      --variables subresult_value operator magnitude_bucket sign \
      --top-k 64 \
      --combine-policy probe_only \
      --output "$SUBSPACES"

    "$PY" phase7/generate_cot_controls.py \
      --trace-dataset "$TRACE_TEST" \
      --max-traces 500 \
      --seed 20260303 \
      --output "$CONTROLS"

    CUDA_VISIBLE_DEVICES=7 "$PY" phase7/causal_intervention_engine.py \
      --trace-dataset "$TRACE_TEST" \
      --controls "$CONTROLS" \
      --subspace-specs "$SUBSPACES" \
      --model-key qwen2.5-7b \
      --parse-mode hybrid \
      --token-anchor eq_like \
      --anchor-priority template_first \
      --control-sampling-policy stratified_trace_variant \
      --variable subresult_value \
      --layers 8 14 20 \
      --max-records 1 \
      --max-records-cap 12000 \
      --record-buffer-size 256 \
      --device cuda:0 \
      --control-records-output "$CONTROL_RECORDS" \
      --output "$BOOTSTRAP_CAUSAL"

    "$PY" - <<PY
import json
from pathlib import Path
from phase7.causal_intervention_engine import _load_control_records_artifact
payload = json.load(open(${CONTROL_RECORDS@Q}))
rows, _ = _load_control_records_artifact(${CONTROL_RECORDS@Q})
if not rows:
    raise SystemExit("control records empty")
shape = tuple(rows[0]["raw_hidden"].shape)
if shape[0] != 28:
    raise SystemExit(f"unexpected qwen layer count in raw_hidden: {shape}")
labels = sorted({str(r.get("gold_label", "")) for r in rows})
if not ({"faithful", "unfaithful"} <= set(labels)):
    raise SystemExit(f"missing labels in control records: {labels}")
out = {
    "schema_version": "phase7_qwen_sae_precompute_validation_v1",
    "run_id": ${RUN_ID@Q},
    "run_tag": ${RUN_TAG@Q},
    "control_records": ${CONTROL_RECORDS@Q},
    "rows_count": int(len(rows)),
    "raw_hidden_shape": list(shape),
    "labels_present": labels,
    "stats": payload.get("stats", {}),
}
Path(${BASE@Q}).joinpath("meta", "precompute_validation.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

    touch "$BASE/state/precompute.done"
    log "precompute done"
  } >>"$logf" 2>&1
}

train_decoder_cfg() {
  local gpu="$1"
  local variant="$2"
  local cfg_name="$3"
  shift 3
  local layers=("$@")
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" phase7/train_state_decoders.py \
    --dataset-train "$TRACE_TRAIN" \
    --model-key qwen2.5-7b \
    --input-variant "$variant" \
    --layers "${layers[@]}" \
    --custom-config-name "$cfg_name" \
    --improvement-profile d1_tier1 \
    --operator-early-stop-guard \
    --checkpoints-dir "$CKPT_DIR" \
    --results-dir "$RESULTS_TRAIN_DIR" \
    --device cuda:0 \
    --cache-inputs on \
    --cache-max-gb 8 \
    --num-workers 8 \
    --pin-memory \
    --persistent-workers \
    --prefetch-factor 4 \
    --non-blocking-transfer \
    --torch-num-threads 12
}

run_train_gpu5() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  local logf="$BASE/logs/decoder_gpu5.log"
  local EVERY2=(0 2 4 6 8 10 12 14 16 18 20 22 24 26)
  local MIDDLE=(7 8 9 10 11 12 13 14 15 16 17 18 19 20)
  {
    log "decoder gpu5 start"
    train_decoder_cfg 5 raw state_raw_every2_even "${EVERY2[@]}"
    train_decoder_cfg 5 raw state_raw_middle14_07_20 "${MIDDLE[@]}"
    touch "$BASE/state/dec_g5.done"
    log "decoder gpu5 done"
  } >>"$logf" 2>&1
}

run_train_gpu6() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  local logf="$BASE/logs/decoder_gpu6.log"
  local EVERY2=(0 2 4 6 8 10 12 14 16 18 20 22 24 26)
  {
    log "decoder gpu6 start"
    train_decoder_cfg 6 hybrid state_hybrid_every2_even "${EVERY2[@]}"
    touch "$BASE/state/dec_g6.done"
    log "decoder gpu6 done"
  } >>"$logf" 2>&1
}

run_train_gpu7() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  local logf="$BASE/logs/decoder_gpu7.log"
  local MIDDLE=(7 8 9 10 11 12 13 14 15 16 17 18 19 20)
  {
    log "decoder gpu7 start"
    train_decoder_cfg 7 hybrid state_hybrid_middle14_07_20 "${MIDDLE[@]}"
    touch "$BASE/state/dec_g7.done"
    log "decoder gpu7 done"
  } >>"$logf" 2>&1
}

run_coordinator() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/coordinator.log"

  local pre_sess="p7qsae_pre_${RUN_ID}"
  local dec5_sess="p7qsae_dec_g5_${RUN_ID}"
  local dec6_sess="p7qsae_dec_g6_${RUN_ID}"
  local dec7_sess="p7qsae_dec_g7_${RUN_ID}"
  local ckpt_raw_every2="$CKPT_DIR/state_raw_every2_even_d1tier1.pt"
  local ckpt_raw_middle="$CKPT_DIR/state_raw_middle14_07_20_d1tier1.pt"
  local ckpt_hyb_every2="$CKPT_DIR/state_hybrid_every2_even_d1tier1.pt"
  local ckpt_hyb_middle="$CKPT_DIR/state_hybrid_middle14_07_20_d1tier1.pt"
  local res_raw_every2="$RESULTS_TRAIN_DIR/state_decoder_supervised_state_raw_every2_even_d1tier1.json"
  local res_raw_middle="$RESULTS_TRAIN_DIR/state_decoder_supervised_state_raw_middle14_07_20_d1tier1.json"
  local res_hyb_every2="$RESULTS_TRAIN_DIR/state_decoder_supervised_state_hybrid_every2_even_d1tier1.json"
  local res_hyb_middle="$RESULTS_TRAIN_DIR/state_decoder_supervised_state_hybrid_middle14_07_20_d1tier1.json"
  local TRAJ_LAYERS_CSV="${TRAJ_LAYERS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}"
  local FDISC_LAYERS_G5="${FDISC_LAYERS_G5:-0,1,2,3,4,5,6,7,8,9}"
  local FDISC_LAYERS_G6="${FDISC_LAYERS_G6:-10,11,12,13,14,15,16,17,18}"
  local FDISC_LAYERS_G7="${FDISC_LAYERS_G7:-19,20,21,22,23,24,25,26,27}"
  local PATHC_MODEL_LADDER="${PATHC_MODEL_LADDER:-sae_only,hybrid_only,mixed}"
  local PATHC_MIXED_DELTA_EFFECT_FLOOR="${PATHC_MIXED_DELTA_EFFECT_FLOOR:-0.03}"
  local PATHC_DECODER_CHECKPOINT="${PATHC_DECODER_CHECKPOINT:-$ckpt_raw_every2}"

  {
    log "coordinator start run_id=$RUN_ID"

    wait_for_file_with_session "$BASE/state/precompute.done" "$pre_sess" 172800

    local need_dec5=1
    local need_dec6=1
    local need_dec7=1

    if [[ -f "$BASE/state/dec_g5.done" && -f "$ckpt_raw_every2" && -f "$ckpt_raw_middle" && -f "$res_raw_every2" && -f "$res_raw_middle" ]]; then
      need_dec5=0
      log "decoder gpu5 artifacts already present; skipping retrain"
    fi
    if [[ -f "$BASE/state/dec_g6.done" && -f "$ckpt_hyb_every2" && -f "$res_hyb_every2" ]]; then
      need_dec6=0
      log "decoder gpu6 artifacts already present; skipping retrain"
    fi
    if [[ -f "$BASE/state/dec_g7.done" && -f "$ckpt_hyb_middle" && -f "$res_hyb_middle" ]]; then
      need_dec7=0
      log "decoder gpu7 artifacts already present; skipping retrain"
    fi

    if (( need_dec5 == 1 )); then
      tmux has-session -t "$dec5_sess" 2>/dev/null && tmux kill-session -t "$dec5_sess"
      tmux new-session -d -s "$dec5_sess" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' train-gpu5"
    fi
    if (( need_dec6 == 1 )); then
      tmux has-session -t "$dec6_sess" 2>/dev/null && tmux kill-session -t "$dec6_sess"
      tmux new-session -d -s "$dec6_sess" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' train-gpu6"
    fi
    if (( need_dec7 == 1 )); then
      tmux has-session -t "$dec7_sess" 2>/dev/null && tmux kill-session -t "$dec7_sess"
      tmux new-session -d -s "$dec7_sess" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' train-gpu7"
    fi

    if (( need_dec5 == 1 )); then
      wait_for_file_with_session "$BASE/state/dec_g5.done" "$dec5_sess" 172800
    fi
    if (( need_dec6 == 1 )); then
      wait_for_file_with_session "$BASE/state/dec_g6.done" "$dec6_sess" 172800
    fi
    if (( need_dec7 == 1 )); then
      wait_for_file_with_session "$BASE/state/dec_g7.done" "$dec7_sess" 172800
    fi

    [[ -f "$ckpt_raw_every2" && -f "$ckpt_raw_middle" && -f "$ckpt_hyb_every2" && -f "$ckpt_hyb_middle" ]] || {
      echo "missing decoder checkpoints after coordinator decode stage" >&2
      exit 1
    }
    [[ -f "$res_raw_every2" && -f "$res_raw_middle" && -f "$res_hyb_every2" && -f "$res_hyb_middle" ]] || {
      echo "missing decoder training result json after coordinator decode stage" >&2
      exit 1
    }

    local RID_FDISC_CAN="${RUN_ID}_qwfd_canary"
    local RID_PATHA_CAN="${RUN_ID}_qwpatha_canary"
    local RID_PATHB_CAN="${RUN_ID}_qwpathb_canary"
    local RID_PATHC_CAN="${RUN_ID}_qwpathc_canary"
    local RID_PATHCR_CAN="${RUN_ID}_qwpathc_robust_canary"
    local TAG_FDISC_CAN="phase7_sae_${RID_FDISC_CAN}"
    local TAG_PATHA_CAN="phase7_sae_trajectory_${RID_PATHA_CAN}"
    local TAG_PATHB_CAN="phase7_sae_trajectory_pathb_${RID_PATHB_CAN}"
    local TAG_PATHC_CAN="phase7_sae_trajectory_pathc_${RID_PATHC_CAN}"
    local TAG_PATHCR_CAN="phase7_sae_trajectory_pathc_robust_${RID_PATHCR_CAN}"
    local BASE_FDISC_CAN="phase7_results/runs/${RID_FDISC_CAN}"
    local BASE_PATHA_CAN="phase7_results/runs/${RID_PATHA_CAN}"
    local BASE_PATHB_CAN="phase7_results/runs/${RID_PATHB_CAN}"
    local BASE_PATHC_CAN="phase7_results/runs/${RID_PATHC_CAN}"
    local BASE_PATHCR_CAN="phase7_results/runs/${RID_PATHCR_CAN}"

    local OUT_FDISC_CAN="phase7_results/results/phase7_sae_feature_discrimination_${TAG_FDISC_CAN}.json"
    local OUT_PATHA_CAN="phase7_results/results/phase7_sae_trajectory_coherence_${TAG_PATHA_CAN}.json"
    local OUT_PATHB_CAN="phase7_results/results/phase7_sae_trajectory_pathb_${RID_PATHB_CAN}.json"
    local OUT_PATHC_CAN="phase7_results/results/phase7_sae_trajectory_pathc_${RID_PATHC_CAN}.json"
    local OUT_PATHCR_CAN="phase7_results/results/phase7_sae_trajectory_pathc_robust_${RID_PATHCR_CAN}.json"

    if stage_done_with_parseable_json "$BASE_FDISC_CAN" "$OUT_FDISC_CAN"; then
      log "canary stage: feature discrimination already complete; skipping"
    else
      log "canary stage: feature discrimination"
      RUN_ID="$RID_FDISC_CAN" \
      RUN_TAG="$TAG_FDISC_CAN" \
      BASE="$BASE_FDISC_CAN" \
      MODEL_KEY="qwen2.5-7b" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SAMPLE_TRACES="$CANARY_SAMPLE_TRACES" \
      LAYERS_G5="$FDISC_LAYERS_G5" \
      LAYERS_G6="$FDISC_LAYERS_G6" \
      LAYERS_G7="$FDISC_LAYERS_G7" \
      ./experiments/run_phase7_sae_faithfulness.sh launch
      wait_for_file_with_session "${BASE_FDISC_CAN}/state/pipeline.done" "p7sae_coord_${RID_FDISC_CAN}" 86400
      [[ -f "$OUT_FDISC_CAN" ]] || { echo "missing canary feature output" >&2; exit 1; }
      json_parseable "$OUT_FDISC_CAN" || { echo "canary feature output not parseable: $OUT_FDISC_CAN" >&2; exit 1; }
    fi

    if stage_done_with_parseable_json "$BASE_PATHA_CAN" "$OUT_PATHA_CAN"; then
      log "canary stage: Path A already complete; skipping"
    else
      log "canary stage: Path A"
      RUN_ID="$RID_PATHA_CAN" \
      RUN_TAG="$TAG_PATHA_CAN" \
      BASE="$BASE_PATHA_CAN" \
      MODEL_KEY="qwen2.5-7b" \
      LAYERS_CSV="$TRAJ_LAYERS_CSV" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SUBSPACE_SPECS="$SUBSPACES" \
      FEATURE_SET="eq_top50" \
      DIVERGENT_SOURCE="$OUT_FDISC_CAN" \
      SAMPLE_TRACES="$CANARY_SAMPLE_TRACES" \
      ./experiments/run_phase7_sae_trajectory_coherence.sh launch
      wait_for_file_with_session "${BASE_PATHA_CAN}/state/pipeline.done" "p7saetc_coord_${RID_PATHA_CAN}" 86400
      [[ -f "$OUT_PATHA_CAN" ]] || { echo "missing canary pathA output" >&2; exit 1; }
      json_parseable "$OUT_PATHA_CAN" || { echo "canary Path A output not parseable: $OUT_PATHA_CAN" >&2; exit 1; }
    fi

    if stage_done_with_parseable_json "$BASE_PATHB_CAN" "$OUT_PATHB_CAN"; then
      log "canary stage: Path B already complete; skipping"
    else
      log "canary stage: Path B"
      RUN_ID="$RID_PATHB_CAN" \
      RUN_TAG="$TAG_PATHB_CAN" \
      BASE="$BASE_PATHB_CAN" \
      MODEL_KEY="qwen2.5-7b" \
      LAYERS_CSV="$TRAJ_LAYERS_CSV" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SUBSPACE_SPECS="$SUBSPACES" \
      DIVERGENT_SOURCE="$OUT_FDISC_CAN" \
      SAMPLE_TRACES="$CANARY_SAMPLE_TRACES" \
      FEATURE_SETS_CSV="result_top50,eq_pre_result_150,divergent_top50" \
      ./experiments/run_phase7_sae_trajectory_pathb.sh launch
      [[ -f "$OUT_PATHB_CAN" ]] || { echo "missing canary pathB output" >&2; exit 1; }
      json_parseable "$OUT_PATHB_CAN" || { echo "canary Path B output not parseable: $OUT_PATHB_CAN" >&2; exit 1; }
    fi

    if stage_done_with_parseable_json "$BASE_PATHC_CAN" "$OUT_PATHC_CAN"; then
      log "canary stage: Path C already complete; skipping"
    else
      log "canary stage: Path C"
      RUN_ID="$RID_PATHC_CAN" \
      RUN_TAG="$TAG_PATHC_CAN" \
      BASE="$BASE_PATHC_CAN" \
      MODEL_KEY="qwen2.5-7b" \
      LAYERS_CSV="$TRAJ_LAYERS_CSV" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SUBSPACE_SPECS="$SUBSPACES" \
      DIVERGENT_SOURCE="$OUT_FDISC_CAN" \
      SAMPLE_TRACES="$CANARY_SAMPLE_TRACES" \
      FEATURE_SET="eq_pre_result_150" \
      DECODER_CHECKPOINT="$PATHC_DECODER_CHECKPOINT" \
      MODEL_LADDER="$PATHC_MODEL_LADDER" \
      MIXED_DELTA_EFFECT_FLOOR="$PATHC_MIXED_DELTA_EFFECT_FLOOR" \
      ./experiments/run_phase7_sae_trajectory_pathc.sh launch
      [[ -f "$OUT_PATHC_CAN" ]] || { echo "missing canary pathC output" >&2; exit 1; }
      json_parseable "$OUT_PATHC_CAN" || { echo "canary Path C output not parseable: $OUT_PATHC_CAN" >&2; exit 1; }
    fi

    if stage_done_with_parseable_json "$BASE_PATHCR_CAN" "$OUT_PATHCR_CAN"; then
      log "canary stage: Path C robust already complete; skipping"
    else
      log "canary stage: Path C robust"
      RUN_ID="$RID_PATHCR_CAN" \
      RUN_TAG="$TAG_PATHCR_CAN" \
      BASE="$BASE_PATHCR_CAN" \
      MODEL_KEY="qwen2.5-7b" \
      LAYERS_CSV="$TRAJ_LAYERS_CSV" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SUBSPACE_SPECS="$SUBSPACES" \
      DIVERGENT_SOURCE="$OUT_FDISC_CAN" \
      SAMPLE_TRACES="$CANARY_SAMPLE_TRACES" \
      FEATURE_SET="eq_pre_result_150" \
      TRAIN_EXCLUDE_VARIANTS="order_flip_only,answer_first_order_flip,reordered_steps" \
      WRONG_INTERMEDIATE_BOOTSTRAP_N=1000 \
      CV_FOLDS=5 \
      CV_MIN_VALID_FOLDS=3 \
      REQUIRE_WRONG_INTERMEDIATE_AUROC=0.70 \
      DECODER_CHECKPOINT="$PATHC_DECODER_CHECKPOINT" \
      MODEL_LADDER="$PATHC_MODEL_LADDER" \
      MIXED_DELTA_EFFECT_FLOOR="$PATHC_MIXED_DELTA_EFFECT_FLOOR" \
      ./experiments/run_phase7_sae_trajectory_pathc_robust.sh launch
      [[ -f "$OUT_PATHCR_CAN" ]] || { echo "missing canary pathC robust output" >&2; exit 1; }
      json_parseable "$OUT_PATHCR_CAN" || { echo "canary Path C robust output not parseable: $OUT_PATHCR_CAN" >&2; exit 1; }
    fi

    "$PY" - <<PY
import json, sys
paths = [
  ${OUT_FDISC_CAN@Q},
  ${OUT_PATHA_CAN@Q},
  ${OUT_PATHB_CAN@Q},
  ${OUT_PATHC_CAN@Q},
  ${OUT_PATHCR_CAN@Q},
]
for p in paths:
    json.load(open(p))
rob = json.load(open(${OUT_PATHCR_CAN@Q}))
if int((rob.get("split_diagnostics") or {}).get("trace_overlap_count", 1)) != 0:
    raise SystemExit("canary integrity failed: trace overlap in split diagnostics")
cvd = rob.get("cv_diagnostics") or {}
if cvd and int(cvd.get("cv_trace_overlap_count", 0)) != 0:
    raise SystemExit("canary integrity failed: cv trace overlap")
print("canary integrity checks passed")
PY

    local RID_FDISC_FULL="${RUN_ID}_qwfd_full"
    local RID_PATHA_FULL="${RUN_ID}_qwpatha_full"
    local RID_PATHB_FULL="${RUN_ID}_qwpathb_full"
    local RID_PATHC_FULL="${RUN_ID}_qwpathc_full"
    local RID_PATHCR_FULL="${RUN_ID}_qwpathc_robust_full"
    local TAG_FDISC_FULL="phase7_sae_${RID_FDISC_FULL}"
    local TAG_PATHA_FULL="phase7_sae_trajectory_${RID_PATHA_FULL}"
    local TAG_PATHB_FULL="phase7_sae_trajectory_pathb_${RID_PATHB_FULL}"
    local TAG_PATHC_FULL="phase7_sae_trajectory_pathc_${RID_PATHC_FULL}"
    local TAG_PATHCR_FULL="phase7_sae_trajectory_pathc_robust_${RID_PATHCR_FULL}"
    local BASE_FDISC_FULL="phase7_results/runs/${RID_FDISC_FULL}"
    local BASE_PATHA_FULL="phase7_results/runs/${RID_PATHA_FULL}"
    local BASE_PATHB_FULL="phase7_results/runs/${RID_PATHB_FULL}"
    local BASE_PATHC_FULL="phase7_results/runs/${RID_PATHC_FULL}"
    local BASE_PATHCR_FULL="phase7_results/runs/${RID_PATHCR_FULL}"

    local OUT_FDISC_FULL="phase7_results/results/phase7_sae_feature_discrimination_${TAG_FDISC_FULL}.json"
    local OUT_PATHA_FULL="phase7_results/results/phase7_sae_trajectory_coherence_${TAG_PATHA_FULL}.json"
    local OUT_PATHB_FULL="phase7_results/results/phase7_sae_trajectory_pathb_${RID_PATHB_FULL}.json"
    local OUT_PATHC_FULL="phase7_results/results/phase7_sae_trajectory_pathc_${RID_PATHC_FULL}.json"
    local OUT_PATHCR_FULL="phase7_results/results/phase7_sae_trajectory_pathc_robust_${RID_PATHCR_FULL}.json"

    if stage_done_with_parseable_json "$BASE_FDISC_FULL" "$OUT_FDISC_FULL"; then
      log "full stage: feature discrimination already complete; skipping"
    else
      log "full stage: feature discrimination"
      RUN_ID="$RID_FDISC_FULL" \
      RUN_TAG="$TAG_FDISC_FULL" \
      BASE="$BASE_FDISC_FULL" \
      MODEL_KEY="qwen2.5-7b" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SAMPLE_TRACES=0 \
      LAYERS_G5="$FDISC_LAYERS_G5" \
      LAYERS_G6="$FDISC_LAYERS_G6" \
      LAYERS_G7="$FDISC_LAYERS_G7" \
      ./experiments/run_phase7_sae_faithfulness.sh launch
      wait_for_file_with_session "${BASE_FDISC_FULL}/state/pipeline.done" "p7sae_coord_${RID_FDISC_FULL}" 172800
      [[ -f "$OUT_FDISC_FULL" ]] || { echo "missing full feature output" >&2; exit 1; }
      json_parseable "$OUT_FDISC_FULL" || { echo "full feature output not parseable: $OUT_FDISC_FULL" >&2; exit 1; }
    fi

    if stage_done_with_parseable_json "$BASE_PATHA_FULL" "$OUT_PATHA_FULL"; then
      log "full stage: Path A already complete; skipping"
    else
      log "full stage: Path A"
      RUN_ID="$RID_PATHA_FULL" \
      RUN_TAG="$TAG_PATHA_FULL" \
      BASE="$BASE_PATHA_FULL" \
      MODEL_KEY="qwen2.5-7b" \
      LAYERS_CSV="$TRAJ_LAYERS_CSV" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SUBSPACE_SPECS="$SUBSPACES" \
      FEATURE_SET="eq_top50" \
      DIVERGENT_SOURCE="$OUT_FDISC_FULL" \
      SAMPLE_TRACES=0 \
      ./experiments/run_phase7_sae_trajectory_coherence.sh launch
      wait_for_file_with_session "${BASE_PATHA_FULL}/state/pipeline.done" "p7saetc_coord_${RID_PATHA_FULL}" 172800
      [[ -f "$OUT_PATHA_FULL" ]] || { echo "missing full pathA output" >&2; exit 1; }
      json_parseable "$OUT_PATHA_FULL" || { echo "full Path A output not parseable: $OUT_PATHA_FULL" >&2; exit 1; }
    fi

    if stage_done_with_parseable_json "$BASE_PATHB_FULL" "$OUT_PATHB_FULL"; then
      log "full stage: Path B already complete; skipping"
    else
      log "full stage: Path B"
      RUN_ID="$RID_PATHB_FULL" \
      RUN_TAG="$TAG_PATHB_FULL" \
      BASE="$BASE_PATHB_FULL" \
      MODEL_KEY="qwen2.5-7b" \
      LAYERS_CSV="$TRAJ_LAYERS_CSV" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SUBSPACE_SPECS="$SUBSPACES" \
      DIVERGENT_SOURCE="$OUT_FDISC_FULL" \
      SAMPLE_TRACES=0 \
      FEATURE_SETS_CSV="result_top50,eq_pre_result_150,divergent_top50" \
      ./experiments/run_phase7_sae_trajectory_pathb.sh launch
      [[ -f "$OUT_PATHB_FULL" ]] || { echo "missing full pathB output" >&2; exit 1; }
      json_parseable "$OUT_PATHB_FULL" || { echo "full Path B output not parseable: $OUT_PATHB_FULL" >&2; exit 1; }
    fi

    if stage_done_with_parseable_json "$BASE_PATHC_FULL" "$OUT_PATHC_FULL"; then
      log "full stage: Path C already complete; skipping"
    else
      log "full stage: Path C"
      RUN_ID="$RID_PATHC_FULL" \
      RUN_TAG="$TAG_PATHC_FULL" \
      BASE="$BASE_PATHC_FULL" \
      MODEL_KEY="qwen2.5-7b" \
      LAYERS_CSV="$TRAJ_LAYERS_CSV" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SUBSPACE_SPECS="$SUBSPACES" \
      DIVERGENT_SOURCE="$OUT_FDISC_FULL" \
      SAMPLE_TRACES=0 \
      FEATURE_SET="eq_pre_result_150" \
      DECODER_CHECKPOINT="$PATHC_DECODER_CHECKPOINT" \
      MODEL_LADDER="$PATHC_MODEL_LADDER" \
      MIXED_DELTA_EFFECT_FLOOR="$PATHC_MIXED_DELTA_EFFECT_FLOOR" \
      ./experiments/run_phase7_sae_trajectory_pathc.sh launch
      [[ -f "$OUT_PATHC_FULL" ]] || { echo "missing full pathC output" >&2; exit 1; }
      json_parseable "$OUT_PATHC_FULL" || { echo "full Path C output not parseable: $OUT_PATHC_FULL" >&2; exit 1; }
    fi

    if stage_done_with_parseable_json "$BASE_PATHCR_FULL" "$OUT_PATHCR_FULL"; then
      log "full stage: Path C robust already complete; skipping"
    else
      log "full stage: Path C robust"
      RUN_ID="$RID_PATHCR_FULL" \
      RUN_TAG="$TAG_PATHCR_FULL" \
      BASE="$BASE_PATHCR_FULL" \
      MODEL_KEY="qwen2.5-7b" \
      LAYERS_CSV="$TRAJ_LAYERS_CSV" \
      CONTROL_RECORDS="$CONTROL_RECORDS" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PROBE_TOP_FEATURES" \
      SUBSPACE_SPECS="$SUBSPACES" \
      DIVERGENT_SOURCE="$OUT_FDISC_FULL" \
      SAMPLE_TRACES=0 \
      FEATURE_SET="eq_pre_result_150" \
      TRAIN_EXCLUDE_VARIANTS="order_flip_only,answer_first_order_flip,reordered_steps" \
      WRONG_INTERMEDIATE_BOOTSTRAP_N=1000 \
      CV_FOLDS=5 \
      CV_MIN_VALID_FOLDS=3 \
      REQUIRE_WRONG_INTERMEDIATE_AUROC=0.70 \
      DECODER_CHECKPOINT="$PATHC_DECODER_CHECKPOINT" \
      MODEL_LADDER="$PATHC_MODEL_LADDER" \
      MIXED_DELTA_EFFECT_FLOOR="$PATHC_MIXED_DELTA_EFFECT_FLOOR" \
      ./experiments/run_phase7_sae_trajectory_pathc_robust.sh launch
      [[ -f "$OUT_PATHCR_FULL" ]] || { echo "missing full pathC robust output" >&2; exit 1; }
      json_parseable "$OUT_PATHCR_FULL" || { echo "full Path C robust output not parseable: $OUT_PATHCR_FULL" >&2; exit 1; }
    fi

    local SUMMARY_JSON="phase7_results/results/phase7_qwen_sae_trajectory_ladder_${RUN_ID}.json"
    local SUMMARY_MD="phase7_results/results/phase7_qwen_sae_trajectory_ladder_${RUN_ID}.md"

    "$PY" - <<PY
import json
from pathlib import Path

def load(p):
    return json.load(open(p))

canary = {
  "feature": load(${OUT_FDISC_CAN@Q}),
  "patha": load(${OUT_PATHA_CAN@Q}),
  "pathb": load(${OUT_PATHB_CAN@Q}),
  "pathc": load(${OUT_PATHC_CAN@Q}),
  "pathc_robust": load(${OUT_PATHCR_CAN@Q}),
}
full = {
  "feature": load(${OUT_FDISC_FULL@Q}),
  "patha": load(${OUT_PATHA_FULL@Q}),
  "pathb": load(${OUT_PATHB_FULL@Q}),
  "pathc": load(${OUT_PATHC_FULL@Q}),
  "pathc_robust": load(${OUT_PATHCR_FULL@Q}),
}
rob = full["pathc_robust"]
cv = rob.get("cv_diagnostics") or {}
summary = {
  "schema_version": "phase7_qwen_sae_trajectory_ladder_summary_v1",
  "status": "ok",
  "run_id": ${RUN_ID@Q},
  "run_tag": ${RUN_TAG@Q},
  "model_key": "qwen2.5-7b",
  "control_records": ${CONTROL_RECORDS@Q},
  "probe_top_features": ${PROBE_TOP_FEATURES@Q},
  "subspace_specs": ${SUBSPACES@Q},
  "canary_artifacts": {
    "feature_discrimination": ${OUT_FDISC_CAN@Q},
    "patha": ${OUT_PATHA_CAN@Q},
    "pathb": ${OUT_PATHB_CAN@Q},
    "pathc": ${OUT_PATHC_CAN@Q},
    "pathc_robust": ${OUT_PATHCR_CAN@Q},
  },
  "full_artifacts": {
    "feature_discrimination": ${OUT_FDISC_FULL@Q},
    "patha": ${OUT_PATHA_FULL@Q},
    "pathb": ${OUT_PATHB_FULL@Q},
    "pathc": ${OUT_PATHC_FULL@Q},
    "pathc_robust": ${OUT_PATHCR_FULL@Q},
  },
  "key_metrics": {
    "canary_wrong_intermediate_auroc": canary["pathc_robust"].get("wrong_intermediate_probe_auroc"),
    "canary_cv_pooled_auroc": (canary["pathc_robust"].get("cv_diagnostics") or {}).get("cv_wrong_intermediate_pooled_auroc"),
    "full_wrong_intermediate_auroc": rob.get("wrong_intermediate_probe_auroc"),
    "full_wrong_intermediate_ci95": rob.get("wrong_intermediate_probe_auroc_ci95"),
    "full_cv_pooled_auroc": cv.get("cv_wrong_intermediate_pooled_auroc"),
    "full_cv_pooled_ci95": cv.get("cv_wrong_intermediate_pooled_ci95"),
    "full_robust_gate_pass": rob.get("robust_wrong_intermediate_gate_pass"),
    "full_cv_gate_pass_pooled": rob.get("cv_wrong_intermediate_gate_pass_pooled"),
    "wrong_intermediate_auroc_by_model": rob.get("wrong_intermediate_auroc_by_model"),
    "delta_m3_vs_m1": rob.get("delta_m3_vs_m1"),
    "mixed_adds_independent_signal": rob.get("mixed_adds_independent_signal"),
    "blocked_reason": rob.get("blocked_reason"),
  },
  "publishability_criterion": {
    "target": "mixed wrong_intermediate robust-CV AUROC > 0.70 and stable mixed-vs-sae gain",
    "pass_publishable_threshold": bool(rob.get("pass_publishable_threshold") is True),
    "mixed_adds_independent_signal": bool(rob.get("mixed_adds_independent_signal") is True),
    "pass": bool(rob.get("pass_publishable_threshold") is True and rob.get("mixed_adds_independent_signal") is True),
  },
}
Path(${SUMMARY_JSON@Q}).write_text(json.dumps(summary, indent=2))
Path(${SUMMARY_MD@Q}).write_text(
    "# Qwen SAE Trajectory Ladder Summary\n\n"
    f"- Run id: `{summary['run_id']}`\n"
    f"- Model: `{summary['model_key']}`\n"
    f"- Full wrong_intermediate AUROC: `{summary['key_metrics']['full_wrong_intermediate_auroc']}`\n"
    f"- Full CV pooled AUROC: `{summary['key_metrics']['full_cv_pooled_auroc']}`\n"
    f"- Publishability threshold pass: `{summary['publishability_criterion']['pass_publishable_threshold']}`\n"
    f"- Mixed adds independent signal: `{summary['publishability_criterion']['mixed_adds_independent_signal']}`\n"
    f"- Final publishability criterion pass: `{summary['publishability_criterion']['pass']}`\n"
)
print(json.dumps(summary, indent=2))
PY

    touch "$BASE/state/pipeline.done"
    log "coordinator done"
  } >>"$logf" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_qwen_sae_trajectory}"
    RUN_TAG="${RUN_TAG:-phase7_qwen_sae_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"

    SAES_DIR="${SAES_DIR:-phase2_results/saes_qwen25_7b_12x_topk/saes}"
    ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
    PHASE6_DIR="${PHASE6_DIR:-$BASE/phase6_dataset}"
    TRACE_DIR="${TRACE_DIR:-$BASE/dataset}"
    CKPT_DIR="${CKPT_DIR:-$BASE/checkpoints}"
    RESULTS_TRAIN_DIR="${RESULTS_TRAIN_DIR:-$BASE/results_train}"
    PHASE6_PREFIX="${PHASE6_PREFIX:-qwen2.5-7b_gsm8k_expanded}"

    PHASE6_TRAIN="$PHASE6_DIR/${PHASE6_PREFIX}_train.pt"
    PHASE6_TEST="$PHASE6_DIR/${PHASE6_PREFIX}_test.pt"
    TRACE_TRAIN="$TRACE_DIR/gsm8k_step_traces_train.pt"
    TRACE_TEST="$TRACE_DIR/gsm8k_step_traces_test.pt"

    CONTROLS="${CONTROLS:-$BASE/controls/cot_controls_${RUN_TAG}.json}"
    CONTROL_RECORDS="${CONTROL_RECORDS:-$BASE/interventions/control_records_${RUN_TAG}.json}"
    BOOTSTRAP_CAUSAL="${BOOTSTRAP_CAUSAL:-$BASE/interventions/causal_checks_bootstrap_${RUN_TAG}.json}"
    PROBE_TOP_FEATURES="${PROBE_TOP_FEATURES:-$BASE/interventions/top_features_per_layer_qwen_${RUN_TAG}.json}"
    SUBSPACES="${SUBSPACES:-$BASE/interventions/variable_subspaces_qwen_${RUN_TAG}.json}"

    CANARY_SAMPLE_TRACES="${CANARY_SAMPLE_TRACES:-80}"
    TRAJ_LAYERS_CSV="${TRAJ_LAYERS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}"
    FDISC_LAYERS_G5="${FDISC_LAYERS_G5:-0,1,2,3,4,5,6,7,8,9}"
    FDISC_LAYERS_G6="${FDISC_LAYERS_G6:-10,11,12,13,14,15,16,17,18}"
    FDISC_LAYERS_G7="${FDISC_LAYERS_G7:-19,20,21,22,23,24,25,26,27}"
    PATHC_MODEL_LADDER="${PATHC_MODEL_LADDER:-sae_only,hybrid_only,mixed}"
    PATHC_MIXED_DELTA_EFFECT_FLOOR="${PATHC_MIXED_DELTA_EFFECT_FLOOR:-0.03}"
    PATHC_DECODER_CHECKPOINT="${PATHC_DECODER_CHECKPOINT:-$CKPT_DIR/state_raw_every2_even_d1tier1.pt}"

    mkdir -p "$BASE"/{logs,state,meta,controls,interventions,checkpoints,results_train}
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
SAES_DIR=$SAES_DIR
ACTIVATIONS_DIR=$ACTIVATIONS_DIR
PHASE6_DIR=$PHASE6_DIR
TRACE_DIR=$TRACE_DIR
CKPT_DIR=$CKPT_DIR
RESULTS_TRAIN_DIR=$RESULTS_TRAIN_DIR
PHASE6_PREFIX=$PHASE6_PREFIX
PHASE6_TRAIN=$PHASE6_TRAIN
PHASE6_TEST=$PHASE6_TEST
TRACE_TRAIN=$TRACE_TRAIN
TRACE_TEST=$TRACE_TEST
CONTROLS=$CONTROLS
CONTROL_RECORDS=$CONTROL_RECORDS
BOOTSTRAP_CAUSAL=$BOOTSTRAP_CAUSAL
PROBE_TOP_FEATURES=$PROBE_TOP_FEATURES
SUBSPACES=$SUBSPACES
CANARY_SAMPLE_TRACES=$CANARY_SAMPLE_TRACES
TRAJ_LAYERS_CSV=$TRAJ_LAYERS_CSV
FDISC_LAYERS_G5=$FDISC_LAYERS_G5
FDISC_LAYERS_G6=$FDISC_LAYERS_G6
FDISC_LAYERS_G7=$FDISC_LAYERS_G7
PATHC_MODEL_LADDER=$PATHC_MODEL_LADDER
PATHC_MIXED_DELTA_EFFECT_FLOOR=$PATHC_MIXED_DELTA_EFFECT_FLOOR
PATHC_DECODER_CHECKPOINT=$PATHC_DECODER_CHECKPOINT
CFG

    local_pre="p7qsae_pre_${RUN_ID}"
    local_coord="p7qsae_coord_${RUN_ID}"

    tmux new-session -d -s "$local_pre" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' precompute"
    tmux new-session -d -s "$local_coord" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched qwen sae trajectory ladder"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  precompute session: $local_pre"
    echo "  coordinator session: $local_coord"
    ;;
  precompute)
    run_precompute
    ;;
  train-gpu5)
    run_train_gpu5
    ;;
  train-gpu6)
    run_train_gpu6
    ;;
  train-gpu7)
    run_train_gpu7
    ;;
  coordinator)
    run_coordinator
    ;;
  *)
    usage
    exit 2
    ;;
esac

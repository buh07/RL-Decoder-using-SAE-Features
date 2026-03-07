#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-.venv/bin/python3}"
MODE="${1:-}"

usage() {
  cat <<'USAGE'
Usage:
  experiments/run_phase7_qwen_core_parity_fast.sh launch
  experiments/run_phase7_qwen_core_parity_fast.sh precompute
  experiments/run_phase7_qwen_core_parity_fast.sh canary-gpu0
  experiments/run_phase7_qwen_core_parity_fast.sh canary-gpu1
  experiments/run_phase7_qwen_core_parity_fast.sh matrix-gpu0
  experiments/run_phase7_qwen_core_parity_fast.sh matrix-g1
  experiments/run_phase7_qwen_core_parity_fast.sh matrix-g2
  experiments/run_phase7_qwen_core_parity_fast.sh coordinator
USAGE
}

_require_env() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
}

_log() {
  echo "[$(date -Is)] $*"
}

_json_parseable() {
  local path="$1"
  $PY - <<PY
import json,sys
try:
    json.load(open(${path@Q}))
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

_json_get_bool() {
  local path="$1"
  local key="$2"
  $PY - <<PY
import json
p=json.load(open(${path@Q}))
v=p.get(${key@Q})
print("true" if v is True else "false")
PY
}

_file_sha() {
  local path="$1"
  $PY - <<PY
from phase7.common import sha256_file
print(sha256_file(${path@Q}))
PY
}

_hash_string() {
  local s="$1"
  $PY - <<PY
import hashlib
print(hashlib.sha256(${s@Q}.encode("utf-8")).hexdigest())
PY
}

_split_policy_hash() {
  _hash_string "seed=20260303|calib_fraction=0.30|group_by=trace_id"
}

_threshold_policy_hash() {
  _hash_string "score_track=composite|target_fpr=0.05|threshold_policy=max_recall_at_fpr_le_target|analysis_policy=max_f1|positive_label=unfaithful|all_tracks=true"
}

_acquire_lock() {
  local lockdir="$1"
  while ! mkdir "$lockdir" 2>/dev/null; do
    sleep 2
  done
}

_release_lock() {
  local lockdir="$1"
  rmdir "$lockdir" 2>/dev/null || true
}

_ckpt_path() {
  local ckpt_id="$1"
  case "$ckpt_id" in
    raw_every2_even) echo "$CKPT_DIR/state_raw_every2_even.pt" ;;
    hybrid_every2_even) echo "$CKPT_DIR/state_hybrid_every2_even.pt" ;;
    raw_middle14_07_20) echo "$CKPT_DIR/state_raw_middle14_07_20.pt" ;;
    hybrid_middle14_07_20) echo "$CKPT_DIR/state_hybrid_middle14_07_20.pt" ;;
    *) echo "unknown checkpoint id: $ckpt_id" >&2; return 1 ;;
  esac
}

_vars_csv_to_lines() {
  local csv="$1"
  IFS=',' read -r -a _arr <<<"$csv"
  printf '%s\n' "${_arr[@]}"
}

_compat_manifest_path() {
  local phase="$1"
  local ckpt_id="$2"
  local var="$3"
  echo "$BASE/meta/compat_${phase}_${ckpt_id}_${var}.json"
}

_cache_manifest_path() {
  local phase="$1"
  local ckpt_id="$2"
  echo "$BASE/meta/cache_compat_${phase}_${ckpt_id}.json"
}

_cache_path() {
  local phase="$1"
  local ckpt_id="$2"
  echo "$BASE/interventions/control_latent_cache_${RUN_TAG}_${phase}_${ckpt_id}.json"
}

_write_cache_manifest() {
  local out="$1"
  local phase="$2"
  local ckpt_id="$3"
  local cache="$4"
  local ckpt_sha="$5"
  local controls_sha="$6"
  local trace_sha="$7"
  local split_hash="$8"
  local thr_hash="$9"
  $PY - <<PY
from phase7.common import save_json, sha256_file
payload = {
  "schema_version": "phase7_qwen_fast_cache_manifest_v1",
  "run_tag": ${RUN_TAG@Q},
  "phase": ${phase@Q},
  "checkpoint_id": ${ckpt_id@Q},
  "model_key": "qwen2.5-7b",
  "controls_sha256": ${controls_sha@Q},
  "trace_dataset_sha256": ${trace_sha@Q},
  "checkpoint_sha256": ${ckpt_sha@Q},
  "parse_mode": "hybrid",
  "token_anchor": "eq_like",
  "anchor_priority": "template_first",
  "split_policy_hash": ${split_hash@Q},
  "threshold_policy_hash": ${thr_hash@Q},
  "cache_path": ${cache@Q},
  "cache_sha256": sha256_file(${cache@Q}),
}
save_json(${out@Q}, payload)
PY
}

_cache_manifest_compatible() {
  local mani="$1"
  local ckpt_id="$2"
  local cache="$3"
  local ckpt_sha="$4"
  local controls_sha="$5"
  local trace_sha="$6"
  local split_hash="$7"
  local thr_hash="$8"
  $PY - <<PY
import json,sys
from pathlib import Path
mp = Path(${mani@Q})
if not mp.exists():
    sys.exit(1)
try:
    m = json.load(open(mp))
except Exception:
    sys.exit(1)
checks = [
    (m.get("schema_version") == "phase7_qwen_fast_cache_manifest_v1"),
    (m.get("checkpoint_id") == ${ckpt_id@Q}),
    (m.get("model_key") == "qwen2.5-7b"),
    (m.get("controls_sha256") == ${controls_sha@Q}),
    (m.get("trace_dataset_sha256") == ${trace_sha@Q}),
    (m.get("checkpoint_sha256") == ${ckpt_sha@Q}),
    (m.get("split_policy_hash") == ${split_hash@Q}),
    (m.get("threshold_policy_hash") == ${thr_hash@Q}),
    (m.get("cache_path") == ${cache@Q}),
]
sys.exit(0 if all(checks) else 1)
PY
}

_write_var_manifest() {
  local out="$1"
  local phase="$2"
  local ckpt_id="$3"
  local var="$4"
  local bench="$5"
  local ckpt_sha="$6"
  local controls_sha="$7"
  local trace_sha="$8"
  local subspaces_sha="$9"
  local split_hash="${10}"
  local thr_hash="${11}"
  $PY - <<PY
from phase7.common import save_json, sha256_file
payload = {
  "schema_version": "phase7_qwen_fast_var_manifest_v1",
  "run_tag": ${RUN_TAG@Q},
  "phase": ${phase@Q},
  "checkpoint_id": ${ckpt_id@Q},
  "variable": ${var@Q},
  "model_key": "qwen2.5-7b",
  "checkpoint_sha256": ${ckpt_sha@Q},
  "controls_sha256": ${controls_sha@Q},
  "trace_dataset_sha256": ${trace_sha@Q},
  "subspace_specs_sha256": ${subspaces_sha@Q},
  "split_policy_hash": ${split_hash@Q},
  "threshold_policy_hash": ${thr_hash@Q},
  "benchmark_path": ${bench@Q},
  "benchmark_sha256": sha256_file(${bench@Q}),
}
save_json(${out@Q}, payload)
PY
}

_var_manifest_compatible() {
  local mani="$1"
  local phase="$2"
  local ckpt_id="$3"
  local var="$4"
  local bench="$5"
  local ckpt_sha="$6"
  local controls_sha="$7"
  local trace_sha="$8"
  local subspaces_sha="$9"
  local split_hash="${10}"
  local thr_hash="${11}"
  $PY - <<PY
import json,sys
from pathlib import Path
mp = Path(${mani@Q})
bp = Path(${bench@Q})
if not mp.exists() or not bp.exists():
    sys.exit(1)
try:
    m = json.load(open(mp))
    b = json.load(open(bp))
except Exception:
    sys.exit(1)
checks = [
    (m.get("schema_version") == "phase7_qwen_fast_var_manifest_v1"),
    (m.get("phase") == ${phase@Q}),
    (m.get("checkpoint_id") == ${ckpt_id@Q}),
    (m.get("variable") == ${var@Q}),
    (m.get("model_key") == "qwen2.5-7b"),
    (m.get("checkpoint_sha256") == ${ckpt_sha@Q}),
    (m.get("controls_sha256") == ${controls_sha@Q}),
    (m.get("trace_dataset_sha256") == ${trace_sha@Q}),
    (m.get("subspace_specs_sha256") == ${subspaces_sha@Q}),
    (m.get("split_policy_hash") == ${split_hash@Q}),
    (m.get("threshold_policy_hash") == ${thr_hash@Q}),
    (b.get("leakage_check_pass") is True),
]
sys.exit(0 if all(checks) else 1)
PY
}

_ensure_control_cache() {
  local phase="$1"
  local ckpt_id="$2"
  local ckpt="$3"
  local dev="$4"
  local controls_sha="$5"
  local trace_sha="$6"
  local split_hash="$7"
  local thr_hash="$8"
  local ckpt_sha
  ckpt_sha="$(_file_sha "$ckpt")"
  local cache
  cache="$(_cache_path "$phase" "$ckpt_id")"
  local mani
  mani="$(_cache_manifest_path "$phase" "$ckpt_id")"
  local lock="$BASE/state/lock_cache_${phase}_${ckpt_id}"
  _acquire_lock "$lock"
  local rc=0
  if [[ -f "$cache" && -f "$mani" ]]; then
    if _cache_manifest_compatible "$mani" "$ckpt_id" "$cache" "$ckpt_sha" "$controls_sha" "$trace_sha" "$split_hash" "$thr_hash"; then
      _log "reuse cache ${phase}/${ckpt_id}"
      _release_lock "$lock"
      echo "$cache"
      return 0
    fi
  fi
  _log "build cache ${phase}/${ckpt_id}"
  $PY phase7/build_control_latent_cache.py \
    --controls "$CONTROLS" \
    --state-decoder-checkpoint "$ckpt" \
    --model-key qwen2.5-7b \
    --parse-mode hybrid \
    --token-anchor eq_like \
    --anchor-priority template_first \
    --device "$dev" \
    --batch-size 64 \
    --output "$cache" || rc=$?
  if [[ $rc -eq 0 ]]; then
    _write_cache_manifest "$mani" "$phase" "$ckpt_id" "$cache" "$ckpt_sha" "$controls_sha" "$trace_sha" "$split_hash" "$thr_hash"
  fi
  _release_lock "$lock"
  if [[ $rc -ne 0 ]]; then
    return $rc
  fi
  echo "$cache"
}

_run_postproc_cpu() {
  local phase="$1"
  local ckpt_id="$2"
  local var="$3"
  local ckpt="$4"
  local cache="$5"
  local controls_sha="$6"
  local trace_sha="$7"
  local subspaces_sha="$8"
  local split_hash="$9"
  local thr_hash="${10}"
  local bench="phase7_results/results/faithfulness_benchmark_${RUN_TAG}_${phase}_${ckpt_id}_${var}.json"
  local causal="$BASE/interventions/causal_checks_${phase}_${ckpt_id}_${var}.json"
  local audit="$BASE/audits/text_causal_audit_controls_${phase}_${ckpt_id}_${var}.json"
  local split_prefix="$BASE/audits/${phase}_${ckpt_id}_${var}"
  local calib="${split_prefix}_calib.json"
  local eval="${split_prefix}_eval.json"
  local thr="$BASE/calibration/thresholds_${phase}_${ckpt_id}_${var}.json"
  local split_mani="${split_prefix}_split_manifest.json"
  local status_file="$BASE/state/status_${phase}_${ckpt_id}_${var}.txt"
  local comp_mani
  comp_mani="$(_compat_manifest_path "$phase" "$ckpt_id" "$var")"
  local rc=0

  {
    $PY phase7/causal_audit.py \
      --model-key qwen2.5-7b \
      --controls "$CONTROLS" \
      --trace-dataset "$TRACE_TEST" \
      --state-decoder-checkpoint "$ckpt" \
      --causal-checks "$causal" \
      --causal-layer 20 \
      --causal-variable "$var" \
      --latent-source variant_conditioned \
      --control-latent-cache "$cache" \
      --require-causal-mode control_conditioned \
      --require-mediation-for-causal-pass \
      --device cpu \
      --output "$audit"

    local audit_sha
    audit_sha="$(_file_sha "$audit")"
    $PY phase7/split_audit_dataset.py \
      --audit "$audit" \
      --seed 20260303 \
      --calib-fraction 0.30 \
      --output-prefix "$split_prefix" \
      --reuse-manifest-if-compatible "$split_mani" \
      --source-audit-hash "$audit_sha"

    $PY phase7/calibrate_audit_thresholds.py \
      --audit-calib "$calib" \
      --all-tracks \
      --score-track composite \
      --target-fpr 0.05 \
      --threshold-policy max_recall_at_fpr_le_target \
      --analysis-policy max_f1 \
      --positive-label unfaithful \
      --output "$thr"

    $PY phase7/benchmark_faithfulness.py \
      --audit-eval "$eval" \
      --thresholds "$thr" \
      --benchmark-scope synthetic_controls \
      --external-validity-status not_tested \
      --gate-track composite \
      --model-comparability-status comparable_full \
      --min-causal-signal-coverage 0.25 \
      --latent-high-quantile 0.80 \
      --comparability-full-threshold 0.60 \
      --comparability-text-threshold 0.01 \
      --comparability-sensitivity 0.50,0.60,0.70 \
      --output "$bench"

    if ! _json_parseable "$bench"; then
      echo "fail_unparseable_benchmark" >"$status_file"
      exit 1
    fi
    if [[ "$(_json_get_bool "$bench" "leakage_check_pass")" != "true" ]]; then
      echo "fail_leakage_check" >"$status_file"
      exit 1
    fi
    local ckpt_sha
    ckpt_sha="$(_file_sha "$ckpt")"
    _write_var_manifest "$comp_mani" "$phase" "$ckpt_id" "$var" "$bench" "$ckpt_sha" "$controls_sha" "$trace_sha" "$subspaces_sha" "$split_hash" "$thr_hash"
    echo "ok" >"$status_file"
  } || rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "fail_postproc" >"$status_file"
    return $rc
  fi
  return 0
}

_run_one_var_gpu_then_cpu() {
  local phase="$1"
  local ckpt_id="$2"
  local ckpt="$3"
  local cache="$4"
  local var="$5"
  local dev="$6"
  local controls_sha="$7"
  local trace_sha="$8"
  local subspaces_sha="$9"
  local split_hash="${10}"
  local thr_hash="${11}"
  local bench="phase7_results/results/faithfulness_benchmark_${RUN_TAG}_${phase}_${ckpt_id}_${var}.json"
  local causal="$BASE/interventions/causal_checks_${phase}_${ckpt_id}_${var}.json"
  local comp_mani
  comp_mani="$(_compat_manifest_path "$phase" "$ckpt_id" "$var")"
  local ckpt_sha
  ckpt_sha="$(_file_sha "$ckpt")"

  if [[ -f "$bench" && -f "$comp_mani" ]]; then
    if _var_manifest_compatible "$comp_mani" "$phase" "$ckpt_id" "$var" "$bench" "$ckpt_sha" "$controls_sha" "$trace_sha" "$subspaces_sha" "$split_hash" "$thr_hash"; then
      _log "skip var compatible ${phase}/${ckpt_id}/${var}"
      return 0
    fi
  fi

  local lock="$BASE/state/lock_var_${phase}_${ckpt_id}_${var}"
  _acquire_lock "$lock"
  local rc=0
  $PY phase7/causal_intervention_engine.py \
    --model-key qwen2.5-7b \
    --trace-dataset "$TRACE_TEST" \
    --controls "$CONTROLS" \
    --parse-mode hybrid \
    --token-anchor eq_like \
    --anchor-priority template_first \
    --control-sampling-policy stratified_trace_variant \
    --subspace-specs "$SUBSPACES" \
    --variable "$var" \
    --layers 8 14 20 \
    --max-records 1200 \
    --record-buffer-size 256 \
    --device "$dev" \
    --state-decoder-checkpoint "$ckpt" \
    --enable-latent-mediation \
    --mediation-variable "$var" \
    --output "$causal" || rc=$?
  _release_lock "$lock"
  if [[ $rc -ne 0 ]]; then
    return $rc
  fi

  while (( $(jobs -rp | wc -l) >= POSTPROC_JOBS )); do
    sleep 1
  done
  _run_postproc_cpu "$phase" "$ckpt_id" "$var" "$ckpt" "$cache" "$controls_sha" "$trace_sha" "$subspaces_sha" "$split_hash" "$thr_hash" &
}

_run_checkpoint_bundle() {
  local phase="$1"
  local ckpt_id="$2"
  local vars_csv="$3"
  local dev="$4"
  local ckpt
  ckpt="$(_ckpt_path "$ckpt_id")"
  mkdir -p "$BASE/interventions" "$BASE/audits" "$BASE/calibration" "$BASE/meta" "$BASE/state" "phase7_results/results"

  local controls_sha trace_sha subspaces_sha split_hash thr_hash cache
  controls_sha="$(_file_sha "$CONTROLS")"
  trace_sha="$(_file_sha "$TRACE_TEST")"
  subspaces_sha="$(_file_sha "$SUBSPACES")"
  split_hash="$(_split_policy_hash)"
  thr_hash="$(_threshold_policy_hash)"
  cache="$(_ensure_control_cache "$phase" "$ckpt_id" "$ckpt" "$dev" "$controls_sha" "$trace_sha" "$split_hash" "$thr_hash")"

  local vars
  vars="$(_vars_csv_to_lines "$vars_csv")"
  while read -r v; do
    [[ -z "$v" ]] && continue
    _run_one_var_gpu_then_cpu "$phase" "$ckpt_id" "$ckpt" "$cache" "$v" "$dev" "$controls_sha" "$trace_sha" "$subspaces_sha" "$split_hash" "$thr_hash"
  done <<<"$vars"

  wait

  local bad=0
  while read -r v; do
    [[ -z "$v" ]] && continue
    local st="$BASE/state/status_${phase}_${ckpt_id}_${v}.txt"
    if [[ ! -f "$st" || "$(cat "$st" 2>/dev/null)" != "ok" ]]; then
      echo "failed status for ${phase}/${ckpt_id}/${v}" >&2
      bad=1
    fi
  done <<<"$vars"
  if [[ $bad -ne 0 ]]; then
    return 1
  fi
}

_train_decoder_cfg() {
  local gpu="$1"
  local variant="$2"
  local cfg_name="$3"
  shift 3
  local layers=("$@")
  CUDA_VISIBLE_DEVICES="$gpu" $PY phase7/train_state_decoders.py \
    --dataset-train "$TRACE_TRAIN" \
    --model-key qwen2.5-7b \
    --input-variant "$variant" \
    --layers "${layers[@]}" \
    --custom-config-name "$cfg_name" \
    --checkpoints-dir "$CKPT_DIR" \
    --results-dir "$BASE/results_train" \
    --device cuda:0 \
    --batch-size 32 \
    --cache-inputs auto
}

_run_precompute() {
  _require_env
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" "$PHASE6_DIR" "$TRACE_DIR" "$CKPT_DIR" "$BASE/interventions" "$BASE/controls"
  local log="$BASE/logs/precompute.log"
  {
    _log "precompute start run_id=$RUN_ID"

    CUDA_VISIBLE_DEVICES="${GPU_PREP:-2}" $PY phase6/collect_expanded_dataset.py \
      --model-key qwen2.5-7b \
      --split train \
      --gsm8k-path datasets/raw/gsm8k/train.jsonl \
      --saes-dir "$SAES_DIR" \
      --activations-dir "$ACTIVATIONS_DIR" \
      --output-dir "$PHASE6_DIR" \
      --output-prefix "$PHASE6_PREFIX" \
      --device cuda:0

    CUDA_VISIBLE_DEVICES="${GPU_PREP:-2}" $PY phase6/collect_expanded_dataset.py \
      --model-key qwen2.5-7b \
      --split test \
      --gsm8k-path datasets/raw/gsm8k/test.jsonl \
      --saes-dir "$SAES_DIR" \
      --activations-dir "$ACTIVATIONS_DIR" \
      --output-dir "$PHASE6_DIR" \
      --output-prefix "$PHASE6_PREFIX" \
      --device cuda:0

    $PY phase7/build_step_trace_dataset.py \
      --phase6-train "$PHASE6_TRAIN" \
      --phase6-test "$PHASE6_TEST" \
      --model-key qwen2.5-7b \
      --state-ontology v2_expanded \
      --output-dir "$TRACE_DIR"

    CUDA_VISIBLE_DEVICES="${GPU_PREP:-2}" $PY phase7/derive_probe_top_features_from_expanded.py \
      --expanded-dataset "$PHASE6_TRAIN" \
      --model-key qwen2.5-7b \
      --split train \
      --top-k 64 \
      --max-records 4000 \
      --seed 20260307 \
      --output "$PROBE_TOP_FEATURES"

    $PY phase7/variable_subspace_builder.py \
      --probe-top-features "$PROBE_TOP_FEATURES" \
      --probe-position eq \
      --model-key qwen2.5-7b \
      --layers 8 14 20 \
      --variables subresult_value operator magnitude_bucket sign \
      --top-k 64 \
      --combine-policy probe_only \
      --output "$SUBSPACES"

    local EVERY2=(0 2 4 6 8 10 12 14 16 18 20 22 24 26)
    local MIDDLE=(7 8 9 10 11 12 13 14 15 16 17 18 19 20)

    # Parallel decoder training across GPUs 0/1/2 (wave 1)
    _train_decoder_cfg 0 raw state_raw_every2_even "${EVERY2[@]}" &
    local p0=$!
    _train_decoder_cfg 1 hybrid state_hybrid_every2_even "${EVERY2[@]}" &
    local p1=$!
    _train_decoder_cfg 2 raw state_raw_middle14_07_20 "${MIDDLE[@]}" &
    local p2=$!
    wait "$p0" "$p1" "$p2"

    # Remaining fourth checkpoint (wave 2)
    _train_decoder_cfg 0 hybrid state_hybrid_middle14_07_20 "${MIDDLE[@]}"

    $PY phase7/generate_cot_controls.py \
      --trace-dataset "$TRACE_TEST" \
      --max-traces 500 \
      --seed 20260303 \
      --output "$CONTROLS"

    $PY phase7/parse_cot_to_states.py \
      --controls "$CONTROLS" \
      --trace-dataset "$TRACE_TEST" \
      --parse-mode hybrid \
      --output "$PARSED_CONTROLS"

    _log "precompute done"
    touch "$BASE/state/precompute.done"
  } >>"$log" 2>&1
}

_run_canary_gpu0() {
  _require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  local log="$BASE/logs/canary_gpu0.log"
  {
    _log "canary gpu0 start"
    export CUDA_VISIBLE_DEVICES=0
    _run_checkpoint_bundle "canary" "raw_every2_even" "subresult_value,operator" "cuda:0"
    touch "$BASE/state/canary_g0.done"
    _log "canary gpu0 done"
  } >>"$log" 2>&1
}

_run_canary_gpu1() {
  _require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  local log="$BASE/logs/canary_gpu1.log"
  {
    _log "canary gpu1 start"
    export CUDA_VISIBLE_DEVICES=1
    _run_checkpoint_bundle "canary" "raw_every2_even" "magnitude_bucket,sign" "cuda:0"
    touch "$BASE/state/canary_g1.done"
    _log "canary gpu1 done"
  } >>"$log" 2>&1
}

_run_matrix_gpu0() {
  _require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  local log="$BASE/logs/matrix_gpu0.log"
  {
    _log "matrix gpu0 start"
    export CUDA_VISIBLE_DEVICES=0
    _run_checkpoint_bundle "matrix" "raw_every2_even" "$ALL_VARS_CSV" "cuda:0"
    _run_checkpoint_bundle "matrix" "raw_middle14_07_20" "$ALL_VARS_CSV" "cuda:0"
    touch "$BASE/state/matrix_g0.done"
    _log "matrix gpu0 done"
  } >>"$log" 2>&1
}

_run_matrix_g1() {
  _require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  local log="$BASE/logs/matrix_g1.log"
  {
    _log "matrix g1 start"
    export CUDA_VISIBLE_DEVICES=1
    _run_checkpoint_bundle "matrix" "hybrid_every2_even" "$ALL_VARS_CSV" "cuda:0"
    touch "$BASE/state/matrix_g1.done"
    _log "matrix g1 done"
  } >>"$log" 2>&1
}

_run_matrix_g2() {
  _require_env
  mkdir -p "$BASE/logs" "$BASE/state"
  local log="$BASE/logs/matrix_g2.log"
  {
    _log "matrix g2 start"
    export CUDA_VISIBLE_DEVICES=2
    _run_checkpoint_bundle "matrix" "hybrid_middle14_07_20" "$ALL_VARS_CSV" "cuda:0"
    touch "$BASE/state/matrix_g2.done"
    _log "matrix g2 done"
  } >>"$log" 2>&1
}

_run_coordinator() {
  _require_env
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local log="$BASE/logs/coordinator.log"
  {
    _log "coordinator start"
    while [[ ! -f "$BASE/state/precompute.done" ]]; do sleep 20; done
    _log "precompute done detected; launching canary workers"

    tmux new-session -d -s "p7qf_canary_g0_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' canary-gpu0"
    tmux new-session -d -s "p7qf_canary_g1_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' canary-gpu1"
    while [[ ! -f "$BASE/state/canary_g0.done" || ! -f "$BASE/state/canary_g1.done" ]]; do sleep 20; done
    _log "canary workers done; validating integrity gate"

    $PY - <<PY
import json, re, sys
from pathlib import Path
run_tag = ${RUN_TAG@Q}
base = Path(${BASE@Q})
vars_ = ["subresult_value","operator","magnitude_bucket","sign"]
missing = []
bad = []
for v in vars_:
    p = Path(f"phase7_results/results/faithfulness_benchmark_{run_tag}_canary_raw_every2_even_{v}.json")
    if not p.exists():
        missing.append(str(p))
        continue
    try:
        row = json.loads(p.read_text())
    except Exception as exc:
        bad.append(f"{p}: parse_error={exc}")
        continue
    if row.get("leakage_check_pass") is not True:
        bad.append(f"{p}: leakage_check_pass={row.get('leakage_check_pass')!r}")
fatal_patterns = [r"Traceback", r"CUDA out of memory", r"\\bNaN\\b"]
for name in ("canary_gpu0.log", "canary_gpu1.log"):
    lp = base / "logs" / name
    if not lp.exists():
        bad.append(f"missing_log:{lp}")
        continue
    txt = lp.read_text(errors="ignore")
    for pat in fatal_patterns:
        if re.search(pat, txt):
            bad.append(f"{lp}: fatal_pattern={pat}")
decision = {
    "schema_version":"phase7_qwen_core_parity_canary_decision_v1",
    "run_tag":run_tag,
    "promotion_allowed": not (missing or bad),
    "missing_artifacts":missing,
    "integrity_failures":bad,
}
Path(f"phase7_results/results/trackA_canary_decision_{run_tag}.json").write_text(json.dumps(decision,indent=2))
print(json.dumps(decision, indent=2))
if missing or bad:
    (base / "state" / "run.blocked").write_text("canary_integrity_failed\\n")
    sys.exit(1)
PY

    _log "canary integrity gate passed; launching matrix workers"
    tmux new-session -d -s "p7qf_mx_g0_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' matrix-gpu0"
    tmux new-session -d -s "p7qf_mx_g1_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' matrix-g1"
    tmux new-session -d -s "p7qf_mx_g2_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' matrix-g2"
    while [[ ! -f "$BASE/state/matrix_g0.done" || ! -f "$BASE/state/matrix_g1.done" || ! -f "$BASE/state/matrix_g2.done" ]]; do sleep 30; done
    _log "matrix workers done; writing matrix summary"

    $PY - <<PY
import glob, json
from pathlib import Path
run_tag = ${RUN_TAG@Q}
rows = []
for p in sorted(glob.glob(f"phase7_results/results/faithfulness_benchmark_{run_tag}_*_*_*.json")):
    try:
        b = json.load(open(p))
    except Exception:
        continue
    rows.append({
        "artifact": p,
        "composite_auroc": b.get("auroc_composite"),
        "text_auroc": b.get("auroc_text_only"),
        "latent_auroc": b.get("auroc_latent_only"),
        "causal_auroc": b.get("auroc_causal_auditor"),
        "leakage_check_pass": b.get("leakage_check_pass"),
        "causal_signal_coverage_fraction": b.get("causal_signal_coverage_fraction"),
    })
matrix = {
    "schema_version":"phase7_trackA_gate_matrix_v1",
    "run_tag": run_tag,
    "model_key":"qwen2.5-7b",
    "num_rows": len(rows),
    "rows": rows,
}
out = Path(f"phase7_results/results/trackA_gate_matrix_{run_tag}.json")
out.write_text(json.dumps(matrix, indent=2))
summary = {
    "schema_version":"phase7_qwen_core_parity_fast_summary_v1",
    "run_id": ${RUN_ID@Q},
    "run_tag": run_tag,
    "model_key":"qwen2.5-7b",
    "canary_decision": f"phase7_results/results/trackA_canary_decision_{run_tag}.json",
    "gate_matrix": str(out),
}
base = Path(${BASE@Q})
(base / "meta" / "summary.json").write_text(json.dumps(summary, indent=2))
(base / "state" / "pipeline.done").write_text("done\\n")
print(str(out))
PY
    _log "coordinator done"
  } >>"$log" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_qwen_core_parity_fast}"
    RUN_TAG="${RUN_TAG:-phase7_qwen_fast_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"
    GPU_PREP="${GPU_PREP:-2}"
    SAES_DIR="${SAES_DIR:-phase2_results/saes_qwen25_7b_12x_topk/saes}"
    ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
    PHASE6_DIR="${PHASE6_DIR:-$BASE/phase6_dataset}"
    TRACE_DIR="${TRACE_DIR:-$BASE/dataset}"
    CKPT_DIR="${CKPT_DIR:-$BASE/checkpoints}"
    ALL_VARS_CSV="${ALL_VARS_CSV:-subresult_value,operator,magnitude_bucket,sign}"
    POSTPROC_JOBS="${POSTPROC_JOBS:-2}"
    PHASE6_PREFIX="${PHASE6_PREFIX:-qwen2.5-7b_gsm8k_expanded}"
    PHASE6_TRAIN="${PHASE6_TRAIN:-$PHASE6_DIR/${PHASE6_PREFIX}_train.pt}"
    PHASE6_TEST="${PHASE6_TEST:-$PHASE6_DIR/${PHASE6_PREFIX}_test.pt}"
    TRACE_TRAIN="${TRACE_TRAIN:-$TRACE_DIR/gsm8k_step_traces_train.pt}"
    TRACE_TEST="${TRACE_TEST:-$TRACE_DIR/gsm8k_step_traces_test.pt}"
    CONTROLS="${CONTROLS:-$BASE/controls/cot_controls_${RUN_TAG}.json}"
    PARSED_CONTROLS="${PARSED_CONTROLS:-$BASE/controls/cot_controls_${RUN_TAG}_parsed.json}"
    PROBE_TOP_FEATURES="${PROBE_TOP_FEATURES:-$BASE/interventions/top_features_per_layer_qwen_${RUN_TAG}.json}"
    SUBSPACES="${SUBSPACES:-$BASE/interventions/variable_subspaces_${RUN_TAG}.json}"

    mkdir -p "$BASE/meta" "$BASE/logs" "$BASE/state" "$BASE/controls" "$BASE/interventions" "$PHASE6_DIR" "$TRACE_DIR" "$CKPT_DIR"
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
GPU_PREP=$GPU_PREP
SAES_DIR=$SAES_DIR
ACTIVATIONS_DIR=$ACTIVATIONS_DIR
PHASE6_DIR=$PHASE6_DIR
TRACE_DIR=$TRACE_DIR
CKPT_DIR=$CKPT_DIR
PHASE6_PREFIX=$PHASE6_PREFIX
PHASE6_TRAIN=$PHASE6_TRAIN
PHASE6_TEST=$PHASE6_TEST
TRACE_TRAIN=$TRACE_TRAIN
TRACE_TEST=$TRACE_TEST
CONTROLS=$CONTROLS
PARSED_CONTROLS=$PARSED_CONTROLS
PROBE_TOP_FEATURES=$PROBE_TOP_FEATURES
SUBSPACES=$SUBSPACES
ALL_VARS_CSV=$ALL_VARS_CSV
POSTPROC_JOBS=$POSTPROC_JOBS
CFG

    tmux new-session -d -s "p7qf_pre_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' precompute"
    tmux new-session -d -s "p7qf_coord_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"
    echo "launched qwen phase7 core parity fast"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    ;;
  precompute) _run_precompute ;;
  canary-gpu0) _run_canary_gpu0 ;;
  canary-gpu1) _run_canary_gpu1 ;;
  matrix-gpu0) _run_matrix_gpu0 ;;
  matrix-g1) _run_matrix_g1 ;;
  matrix-g2) _run_matrix_g2 ;;
  coordinator) _run_coordinator ;;
  ""|-h|--help|help) usage ;;
  *) usage; echo "unknown mode: $MODE" >&2; exit 2 ;;
esac


#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  experiments/run_trackc_direction_quick.sh launch
  experiments/run_trackc_direction_quick.sh worker-g6 <RUN_ID> <RUN_TAG> <BASE>
  experiments/run_trackc_direction_quick.sh worker-g7 <RUN_ID> <RUN_TAG> <BASE>
  experiments/run_trackc_direction_quick.sh coordinator <RUN_ID> <RUN_TAG> <BASE>
USAGE
}

PY=".venv/bin/python3"
TRACE_TEST="phase7_results/dataset/gsm8k_step_traces_test.pt"
CKPT="phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration/checkpoints/state_raw_every2_even.pt"
LAYERS=(7 12 17 22)

SEED="${SEED:-20260305}"
SPLIT_SEED="${SPLIT_SEED:-20260303}"
CALIB_FRACTION="${CALIB_FRACTION:-0.30}"
MAX_TRACES="${MAX_TRACES:-80}"
BASE_MAX_RECORDS="${BASE_MAX_RECORDS:-400}"
ESC_MAX_RECORDS="${ESC_MAX_RECORDS:-800}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.25}"

json_ok() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  "$PY" - <<PY
import json,sys
try:
  json.load(open("$path"))
except Exception:
  sys.exit(1)
sys.exit(0)
PY
}

prepare_shared_inputs() {
  local run_tag="$1"
  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local parsed="phase7_results/controls/cot_controls_${run_tag}_parsed.json"
  local subspaces="phase7_results/interventions/variable_subspaces_${run_tag}.json"

  "$PY" phase7/generate_cot_controls.py \
    --trace-dataset "$TRACE_TEST" \
    --max-traces "$MAX_TRACES" \
    --seed "$SEED" \
    --output "$controls"

  "$PY" phase7/parse_cot_to_states.py \
    --controls "$controls" \
    --trace-dataset "$TRACE_TEST" \
    --parse-mode hybrid \
    --output "$parsed"

  "$PY" phase7/variable_subspace_builder.py \
    --model-key gpt2-medium \
    --layers "${LAYERS[@]}" \
    --probe-position result \
    --top-k 64 \
    --combine-policy union \
    --output "$subspaces"
}

cache_path() {
  local run_tag="$1"
  local anchor="$2"
  echo "phase7_results/interventions/control_latent_cache_${run_tag}_raw_every2_even_${anchor}.json"
}

control_records_path() {
  local run_tag="$1"
  local anchor="$2"
  echo "phase7_results/interventions/control_records_${run_tag}_raw_every2_even_${anchor}.json"
}

id_tag() {
  local run_tag="$1"
  local anchor="$2"
  local variable="$3"
  local suffix="${4:-}"
  echo "${run_tag}_raw_every2_even_${anchor}_${variable}${suffix}"
}

build_cache_if_needed() {
  local run_tag="$1"
  local anchor="$2"
  local gpu_device="$3"
  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local cache
  cache="$(cache_path "$run_tag" "$anchor")"

  if json_ok "$cache"; then
    return 0
  fi

  "$PY" phase7/build_control_latent_cache.py \
    --controls "$controls" \
    --state-decoder-checkpoint "$CKPT" \
    --model-key gpt2-medium \
    --parse-mode hybrid \
    --token-anchor eq_like \
    --anchor-priority "$anchor" \
    --device "$gpu_device" \
    --batch-size 64 \
    --rows-format json \
    --rows-inline \
    --output "$cache"
}

run_causal_for_vars() {
  local run_tag="$1"
  local anchor="$2"
  local gpu_device="$3"
  local max_records="$4"
  local mediation_var="$5"
  local suffix="$6"
  shift 6
  local vars=("$@")

  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local subspaces="phase7_results/interventions/variable_subspaces_${run_tag}.json"
  local control_records
  control_records="$(control_records_path "$run_tag" "$anchor")"
  local out_tmpl="phase7_results/interventions/causal_checks_${run_tag}_raw_every2_even_${anchor}_{variable}${suffix}.json"

  local cmd=(
    "$PY" phase7/causal_intervention_engine.py
    --model-key gpt2-medium
    --trace-dataset "$TRACE_TEST"
    --controls "$controls"
    --parse-mode hybrid
    --token-anchor eq_like
    --anchor-priority "$anchor"
    --control-sampling-policy stratified_trace_variant
    --subspace-specs "$subspaces"
    --variables "${vars[@]}"
    --output-template "$out_tmpl"
    --layers "${LAYERS[@]}"
    --max-records "$max_records"
    --record-buffer-size 128
    --device "$gpu_device"
    --state-decoder-checkpoint "$CKPT"
    --enable-latent-mediation
    --mediation-variable "$mediation_var"
    --target-controls-used-fraction 0.25
    --max-records-cap "$ESC_MAX_RECORDS"
    --min-controls-used 40
    --control-records-output "$control_records"
    --resume-output
    --rows-format json
    --seed "$SEED"
  )

  if [[ -f "$control_records" ]]; then
    cmd+=(--control-records-input "$control_records")
  fi

  "${cmd[@]}"
}

run_postproc_for_var() {
  local run_tag="$1"
  local anchor="$2"
  local variable="$3"
  local suffix="$4"

  local cache
  cache="$(cache_path "$run_tag" "$anchor")"
  local tag
  tag="$(id_tag "$run_tag" "$anchor" "$variable" "$suffix")"
  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local causal="phase7_results/interventions/causal_checks_${tag}.json"
  local audit="phase7_results/audits/text_causal_audit_controls_${tag}.json"
  local split_prefix="phase7_results/audits/${tag}"
  local calib="${split_prefix}_calib.json"
  local eval="${split_prefix}_eval.json"
  local thr="phase7_results/calibration/phase7_thresholds_${tag}.json"
  local bench="phase7_results/results/faithfulness_benchmark_${tag}.json"

  "$PY" phase7/causal_audit.py \
    --model-key gpt2-medium \
    --controls "$controls" \
    --trace-dataset "$TRACE_TEST" \
    --state-decoder-checkpoint "$CKPT" \
    --causal-checks "$causal" \
    --causal-layer 22 \
    --causal-variable "$variable" \
    --latent-source variant_conditioned \
    --control-latent-cache "$cache" \
    --require-mediation-for-causal-pass \
    --require-causal-mode control_conditioned \
    --device cpu \
    --output "$audit"

  "$PY" phase7/split_audit_dataset.py \
    --audit "$audit" \
    --seed "$SPLIT_SEED" \
    --calib-fraction "$CALIB_FRACTION" \
    --output-prefix "$split_prefix"

  "$PY" phase7/calibrate_audit_thresholds.py \
    --audit-calib "$calib" \
    --all-tracks \
    --score-track composite \
    --target-fpr 0.05 \
    --threshold-policy max_recall_at_fpr_le_target \
    --analysis-policy max_f1 \
    --positive-label unfaithful \
    --output "$thr"

  "$PY" phase7/benchmark_faithfulness.py \
    --audit-eval "$eval" \
    --thresholds "$thr" \
    --benchmark-scope synthetic_controls \
    --external-validity-status not_tested \
    --gate-track composite \
    --require-dual-gate \
    --causal-floor-auroc 0.65 \
    --causal-floor-fpr-max 0.05 \
    --model-comparability-status comparable_full \
    --min-causal-signal-coverage "$TARGET_COVERAGE" \
    --latent-high-quantile 0.80 \
    --causal-degenerate-identical-threshold 0.80 \
    --causal-degenerate-auroc-threshold 0.55 \
    --causal-degenerate-enable-auroc-trigger \
    --ablation-weights '{"text":0.5,"latent":0.5,"causal":0.0}' \
    --output "$bench"
}

min_coverage_for_vars() {
  local run_tag="$1"
  local anchor="$2"
  local suffix="$3"
  shift 3
  local vars=("$@")

  local csv
  csv="$(IFS=,; echo "${vars[*]}")"
  "$PY" - <<PY
import json,math
run_tag="$run_tag"
anchor="$anchor"
suffix="$suffix"
vars_csv="$csv"
vals=[]
for var in [v for v in vars_csv.split(',') if v]:
    p=f"phase7_results/results/faithfulness_benchmark_{run_tag}_raw_every2_even_{anchor}_{var}{suffix}.json"
    d=json.load(open(p))
    x=d.get("causal_signal_coverage_fraction")
    if isinstance(x,(int,float)) and math.isfinite(float(x)):
        vals.append(float(x))
print(min(vals) if vals else "nan")
PY
}

run_worker_branch() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local gpu="$4"
  local anchor="$5"
  shift 5
  local vars=("$@")

  local log="$base/logs/worker_${anchor}_gpu${gpu}.log"

  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2

  {
    echo "[$(date -Is)] worker start gpu=$gpu anchor=$anchor vars=${vars[*]}"
    build_cache_if_needed "$run_tag" "$anchor" "cuda:0"

    run_causal_for_vars "$run_tag" "$anchor" "cuda:0" "$BASE_MAX_RECORDS" "subresult_value" "" "${vars[@]}"
    for v in "${vars[@]}"; do
      run_postproc_for_var "$run_tag" "$anchor" "$v" ""
    done

    local cov
    cov="$(min_coverage_for_vars "$run_tag" "$anchor" "" "${vars[@]}")"

    if "$PY" - <<PY
import math,sys
v=float("$cov") if "$cov" not in ("nan","") else float("nan")
sys.exit(0 if ((not math.isfinite(v)) or (v < float("$TARGET_COVERAGE"))) else 1)
PY
    then
      echo "[$(date -Is)] coverage $cov < $TARGET_COVERAGE, escalating to $ESC_MAX_RECORDS"
      run_causal_for_vars "$run_tag" "$anchor" "cuda:0" "$ESC_MAX_RECORDS" "subresult_value" "" "${vars[@]}"
      for v in "${vars[@]}"; do
        run_postproc_for_var "$run_tag" "$anchor" "$v" ""
      done
    fi

    echo "[$(date -Is)] worker done gpu=$gpu anchor=$anchor"
    echo "$(date -Is)" > "$base/state/${anchor}.done"
  } >> "$log" 2>&1
}

write_baseline_inputs_manifest() {
  local run_tag="$1"
  local out_path="$2"
  "$PY" - <<PY
import json
run_tag="$run_tag"
out="$out_path"
rows=[
  {
    "anchor_priority":"template_first",
    "variable":"subresult_value",
    "benchmark_path":f"phase7_results/results/faithfulness_benchmark_{run_tag}_raw_every2_even_template_first_subresult_value.json",
    "audit_eval_path":f"phase7_results/audits/{run_tag}_raw_every2_even_template_first_subresult_value_eval.json",
  },
  {
    "anchor_priority":"template_first",
    "variable":"magnitude_bucket",
    "benchmark_path":f"phase7_results/results/faithfulness_benchmark_{run_tag}_raw_every2_even_template_first_magnitude_bucket.json",
    "audit_eval_path":f"phase7_results/audits/{run_tag}_raw_every2_even_template_first_magnitude_bucket_eval.json",
  },
  {
    "anchor_priority":"equation_first",
    "variable":"operator",
    "benchmark_path":f"phase7_results/results/faithfulness_benchmark_{run_tag}_raw_every2_even_equation_first_operator.json",
    "audit_eval_path":f"phase7_results/audits/{run_tag}_raw_every2_even_equation_first_operator_eval.json",
  },
  {
    "anchor_priority":"equation_first",
    "variable":"sign",
    "benchmark_path":f"phase7_results/results/faithfulness_benchmark_{run_tag}_raw_every2_even_equation_first_sign.json",
    "audit_eval_path":f"phase7_results/audits/{run_tag}_raw_every2_even_equation_first_sign_eval.json",
  },
]
json.dump(rows, open(out, "w"), indent=2)
print(out)
PY
}

ensure_baseline_outputs_exist() {
  local run_tag="$1"
  local missing=0
  local p
  for p in \
    "phase7_results/results/faithfulness_benchmark_${run_tag}_raw_every2_even_template_first_subresult_value.json" \
    "phase7_results/results/faithfulness_benchmark_${run_tag}_raw_every2_even_template_first_magnitude_bucket.json" \
    "phase7_results/results/faithfulness_benchmark_${run_tag}_raw_every2_even_equation_first_operator.json" \
    "phase7_results/results/faithfulness_benchmark_${run_tag}_raw_every2_even_equation_first_sign.json"; do
    if [[ ! -f "$p" ]]; then
      echo "missing required benchmark artifact: $p" >&2
      missing=1
    fi
  done
  return "$missing"
}

pick_best_anchor() {
  local run_tag="$1"
  "$PY" - <<PY
import json,statistics
run_tag="$run_tag"
def load(anchor,var):
    p=f"phase7_results/results/faithfulness_benchmark_{run_tag}_raw_every2_even_{anchor}_{var}.json"
    d=json.load(open(p))
    return ((d.get("by_benchmark_track") or {}).get("causal_auditor") or {}).get("auroc")

tpl=[load("template_first","subresult_value"), load("template_first","magnitude_bucket")]
eqn=[load("equation_first","operator"), load("equation_first","sign")]

def m(arr):
    vals=[float(x) for x in arr if isinstance(x,(int,float))]
    return sum(vals)/len(vals) if vals else float("-inf")
mt,me=m(tpl),m(eqn)
print("template_first" if mt>=me else "equation_first")
PY
}

run_mediation_probe_on_anchor() {
  local run_tag="$1"
  local anchor="$2"
  local gpu="$3"

  export CUDA_VISIBLE_DEVICES="$gpu"

  run_causal_for_vars "$run_tag" "$anchor" "cuda:0" "$BASE_MAX_RECORDS" "operator" "_medalign" "operator"
  run_postproc_for_var "$run_tag" "$anchor" "operator" "_medalign"

  run_causal_for_vars "$run_tag" "$anchor" "cuda:0" "$BASE_MAX_RECORDS" "sign" "_medalign" "sign"
  run_postproc_for_var "$run_tag" "$anchor" "sign" "_medalign"
}

write_probe_inputs_manifest() {
  local run_tag="$1"
  local anchor="$2"
  local out_path="$3"
  "$PY" - <<PY
import json
run_tag="$run_tag"
anchor="$anchor"
out="$out_path"
rows=[
  {
    "anchor_priority":anchor,
    "variable":"operator",
    "benchmark_path":f"phase7_results/results/faithfulness_benchmark_{run_tag}_raw_every2_even_{anchor}_operator_medalign.json",
    "audit_eval_path":f"phase7_results/audits/{run_tag}_raw_every2_even_{anchor}_operator_medalign_eval.json",
  },
  {
    "anchor_priority":anchor,
    "variable":"sign",
    "benchmark_path":f"phase7_results/results/faithfulness_benchmark_{run_tag}_raw_every2_even_{anchor}_sign_medalign.json",
    "audit_eval_path":f"phase7_results/audits/{run_tag}_raw_every2_even_{anchor}_sign_medalign_eval.json",
  },
]
json.dump(rows, open(out, "w"), indent=2)
print(out)
PY
}

worker_g6() {
  run_worker_branch "$1" "$2" "$3" 6 template_first subresult_value magnitude_bucket
}

worker_g7() {
  run_worker_branch "$1" "$2" "$3" 7 equation_first operator sign
}

coordinator() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local log="$base/logs/coordinator.log"

  {
    echo "[$(date -Is)] coordinator start"
    while [[ ! -f "$base/state/template_first.done" ]]; do sleep 10; done
    while [[ ! -f "$base/state/equation_first.done" ]]; do sleep 10; done

    ensure_baseline_outputs_exist "$run_tag"

    local baseline_inputs="$base/meta/trackc_quick_inputs.json"
    write_baseline_inputs_manifest "$run_tag" "$baseline_inputs"

    local best_anchor
    best_anchor="$(pick_best_anchor "$run_tag")"
    local probe_gpu="6"
    if [[ "$best_anchor" == "equation_first" ]]; then
      probe_gpu="7"
    fi

    echo "[$(date -Is)] selected anchor for E3 probe: $best_anchor on gpu$probe_gpu"
    run_mediation_probe_on_anchor "$run_tag" "$best_anchor" "$probe_gpu"

    local probe_inputs="$base/meta/trackc_quick_probe_inputs.json"
    write_probe_inputs_manifest "$run_tag" "$best_anchor" "$probe_inputs"

    local out_json="phase7_results/results/trackc_direction_quick_${run_tag}.json"
    local out_md="phase7_results/results/trackc_direction_quick_${run_tag}.md"
    "$PY" phase7/diagnose_trackc_direction.py \
      --run-tag "$run_tag" \
      --inputs "$baseline_inputs" \
      --probe-inputs "$probe_inputs" \
      --output-json "$out_json" \
      --output-md "$out_md"

    echo "$(date -Is)" > "$base/state/pipeline.done"
    echo "[$(date -Is)] coordinator done"
  } >> "$log" 2>&1
}

launch() {
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)_trackc_direction_quick"
  local run_tag="trackc_direction_quick_${run_id}"
  local base="phase7_results/runs/${run_id}"

  mkdir -p "$base"/{logs,state,meta}

  prepare_shared_inputs "$run_tag"

  tmux new-session -d -s "p7tcq_g6_${run_id}" "cd '$PWD' && bash experiments/run_trackc_direction_quick.sh worker-g6 '$run_id' '$run_tag' '$base'"
  tmux new-session -d -s "p7tcq_g7_${run_id}" "cd '$PWD' && bash experiments/run_trackc_direction_quick.sh worker-g7 '$run_id' '$run_tag' '$base'"
  tmux new-session -d -s "p7tcq_coord_${run_id}" "cd '$PWD' && bash experiments/run_trackc_direction_quick.sh coordinator '$run_id' '$run_tag' '$base'"

  echo "RUN_ID=$run_id"
  echo "RUN_TAG=$run_tag"
  echo "BASE=$base"
  echo "Sessions:"
  echo "  p7tcq_g6_${run_id}"
  echo "  p7tcq_g7_${run_id}"
  echo "  p7tcq_coord_${run_id}"
}

cmd="${1:-}"
case "$cmd" in
  launch) launch ;;
  worker-g6) worker_g6 "$2" "$3" "$4" ;;
  worker-g7) worker_g7 "$2" "$3" "$4" ;;
  coordinator) coordinator "$2" "$3" "$4" ;;
  *) usage; exit 1 ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
PY="${PYTHON:-.venv/bin/python3}"

usage() {
  cat <<'USAGE'
usage: experiments/run_phase7_prontoqa_trackc.sh {launch|precompute|patha|pathb|pathc|stress|coordinator}
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
      for _ in {1..6}; do
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

run_precompute() {
  require_env
  : "${PREP_KIND:?PREP_KIND required (canary|full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/precompute_${PREP_KIND}.log"
  local sample_size="$CANARY_TRACES"
  local prep_dir="$PREP_CANARY_DIR"
  local done_marker="$BASE/state/precompute_canary.done"
  if [[ "$PREP_KIND" == "full" ]]; then
    sample_size="$FULL_TRACES"
    prep_dir="$PREP_FULL_DIR"
    done_marker="$BASE/state/precompute_full.done"
  fi

  {
    echo "[$(date -Is)] precompute start kind=${PREP_KIND} sample_size=${sample_size}"
    CUDA_VISIBLE_DEVICES="$PRECOMPUTE_GPU" "$PY" phase7/prontoqa_prepare_dataset.py \
      --model-key "$MODEL_KEY" \
      --sample-size "$sample_size" \
      --seed "$SEED" \
      --chain-len-min "$CHAIN_LEN_MIN" \
      --chain-len-max "$CHAIN_LEN_MAX" \
      --variants "$PRONTOQA_VARIANTS" \
      --device cuda:0 \
      --max-records "$MAX_RECORDS" \
      --output-dir "$prep_dir" \
      --trace-output "$prep_dir/dataset/prontoqa_step_traces_test.pt" \
      --controls-output "$prep_dir/controls/cot_controls_prontoqa.json" \
      --control-records-output "$prep_dir/interventions/control_records_prontoqa.json" \
      --manifest-output "$prep_dir/meta/manifest.json"

    [[ -f "$prep_dir/interventions/control_records_prontoqa.json" ]] || { echo "missing control records for $PREP_KIND" >&2; exit 1; }
    json_parseable "$prep_dir/interventions/control_records_prontoqa.json" || { echo "bad control records json for $PREP_KIND" >&2; exit 1; }
    touch "$done_marker"
    echo "[$(date -Is)] precompute done kind=${PREP_KIND}"
  } >>"$logf" 2>&1
}

run_patha() {
  require_env
  : "${SCOPE:?SCOPE required (canary|full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/patha_${SCOPE}.log"
  local records="$CANARY_CONTROL_RECORDS"
  local sample="$CANARY_TRACES"
  local sub_run_id="${RUN_ID}_patha_canary"
  if [[ "$SCOPE" == "full" ]]; then
    records="$FULL_CONTROL_RECORDS"
    sample="0"
    sub_run_id="${RUN_ID}_patha_full"
  fi
  local sub_run_tag="phase7_prontoqa_patha_${sub_run_id}"
  local out_json="phase7_results/results/phase7_sae_trajectory_coherence_${sub_run_tag}.json"
  {
    echo "[$(date -Is)] patha start scope=${SCOPE}"
    RUN_ID="$sub_run_id" \
    RUN_TAG="$sub_run_tag" \
    BASE="phase7_results/runs/${sub_run_id}" \
    CONTROL_RECORDS="$records" \
    MODEL_KEY="$MODEL_KEY" \
    SAES_DIR="$SAES_DIR" \
    ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
    PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
    FEATURE_SET="eq_top50" \
    DIVERGENT_SOURCE="$BASE/meta/divergent_${SCOPE}.json" \
    SUBSPACE_SPECS="$SUBSPACE_SPECS" \
    SAMPLE_TRACES="$sample" \
    MIN_COMMON_STEPS="$MIN_COMMON_STEPS" \
    SEED="$SEED" \
    N_BOOTSTRAP="$N_BOOTSTRAP" \
    BATCH_SIZE="$BATCH_SIZE" \
    EMIT_SAMPLES=1 \
    LAYERS_CSV="$LAYERS_CSV" \
    GPU_IDS_CSV="5,6,7" \
    ./experiments/run_phase7_sae_trajectory_coherence.sh launch

    local coord_sess="p7saetc_coord_${sub_run_id}"
    while [[ ! -f "phase7_results/runs/${sub_run_id}/state/pipeline.done" ]]; do
      if ! tmux has-session -t "$coord_sess" 2>/dev/null; then
        sleep 5
        [[ -f "phase7_results/runs/${sub_run_id}/state/pipeline.done" ]] || { echo "patha coordinator disappeared: ${coord_sess}" >&2; exit 1; }
      fi
      sleep 15
    done
    [[ -f "$out_json" ]] || { echo "missing patha output: $out_json" >&2; exit 1; }
    json_parseable "$out_json" || { echo "bad patha output: $out_json" >&2; exit 1; }
    "$PY" - <<PY
import json
from pathlib import Path
meta = {
  "scope": ${SCOPE@Q},
  "run_id": ${sub_run_id@Q},
  "run_tag": ${sub_run_tag@Q},
  "output_json": ${out_json@Q},
}
Path(${BASE@Q}).joinpath("meta","patha_" + ${SCOPE@Q} + ".json").write_text(json.dumps(meta, indent=2))
PY
    touch "$BASE/state/patha_${SCOPE}.done"
    echo "[$(date -Is)] patha done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_pathb() {
  require_env
  : "${SCOPE:?SCOPE required (canary|full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/pathb_${SCOPE}.log"
  local records="$CANARY_CONTROL_RECORDS"
  local sample="$CANARY_TRACES"
  local sub_run_id="${RUN_ID}_pathb_canary"
  local fd_run_id="${RUN_ID}_fd_canary"
  if [[ "$SCOPE" == "full" ]]; then
    records="$FULL_CONTROL_RECORDS"
    sample="0"
    sub_run_id="${RUN_ID}_pathb_full"
    fd_run_id="${RUN_ID}_fd_full"
  fi
  local fd_run_tag="phase7_prontoqa_fd_${fd_run_id}"
  local divergent="phase7_results/results/phase7_sae_feature_discrimination_${fd_run_tag}.json"
  local out_json="phase7_results/results/phase7_sae_trajectory_pathb_${sub_run_id}.json"
  {
    echo "[$(date -Is)] pathb start scope=${SCOPE}"
    if [[ ! -f "$divergent" ]]; then
      RUN_ID="$fd_run_id" \
      RUN_TAG="$fd_run_tag" \
      BASE="phase7_results/runs/${fd_run_id}" \
      CONTROL_RECORDS="$records" \
      MODEL_KEY="$MODEL_KEY" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
      SAMPLE_TRACES="$sample" \
      SEED="$SEED" \
      N_PERMUTATIONS=250 \
      TRACE_TEST_FRACTION=0.20 \
      PROBE_EPOCHS=80 \
      PROBE_LR=0.001 \
      PROBE_WEIGHT_DECAY=0.01 \
      PROBE_L1_LAMBDA=0.00001 \
      MIN_CLASS_PER_SPLIT=10 \
      BATCH_SIZE="$BATCH_SIZE" \
      LAYERS_G5="$FDISC_LAYERS_G5" \
      LAYERS_G6="$FDISC_LAYERS_G6" \
      LAYERS_G7="$FDISC_LAYERS_G7" \
      ./experiments/run_phase7_sae_faithfulness.sh launch
      local fd_coord="p7sae_coord_${fd_run_id}"
      while [[ ! -f "phase7_results/runs/${fd_run_id}/state/pipeline.done" ]]; do
        if ! tmux has-session -t "$fd_coord" 2>/dev/null; then
          sleep 5
          [[ -f "phase7_results/runs/${fd_run_id}/state/pipeline.done" ]] || { echo "fd coordinator disappeared: ${fd_coord}" >&2; exit 1; }
        fi
        sleep 15
      done
    fi
    [[ -f "$divergent" ]] || { echo "missing divergent source: $divergent" >&2; exit 1; }
    cp -f "$divergent" "$BASE/meta/divergent_${SCOPE}.json"

    RUN_ID="$sub_run_id" \
    RUN_TAG="phase7_prontoqa_pathb_${sub_run_id}" \
    BASE="phase7_results/runs/${sub_run_id}" \
    CONTROL_RECORDS="$records" \
    MODEL_KEY="$MODEL_KEY" \
    LAYERS_CSV="$LAYERS_CSV" \
    SAES_DIR="$SAES_DIR" \
    ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
    PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
    DIVERGENT_SOURCE="$divergent" \
    SUBSPACE_SPECS="$SUBSPACE_SPECS" \
    FEATURE_SETS_CSV="result_top50,eq_pre_result_150,divergent_top50" \
    SAMPLE_TRACES="$sample" \
    MIN_COMMON_STEPS="$MIN_COMMON_STEPS" \
    SEED="$SEED" \
    N_BOOTSTRAP="$N_BOOTSTRAP" \
    BATCH_SIZE="$BATCH_SIZE" \
    ./experiments/run_phase7_sae_trajectory_pathb.sh launch

    [[ -f "$out_json" ]] || { echo "missing pathb output: $out_json" >&2; exit 1; }
    json_parseable "$out_json" || { echo "bad pathb output: $out_json" >&2; exit 1; }
    "$PY" - <<PY
import json
from pathlib import Path
meta = {
  "scope": ${SCOPE@Q},
  "run_id": ${sub_run_id@Q},
  "output_json": ${out_json@Q},
  "divergent_source": ${divergent@Q},
}
Path(${BASE@Q}).joinpath("meta","pathb_" + ${SCOPE@Q} + ".json").write_text(json.dumps(meta, indent=2))
PY
    touch "$BASE/state/pathb_${SCOPE}.done"
    echo "[$(date -Is)] pathb done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_pathc() {
  require_env
  : "${SCOPE:?SCOPE required (canary|full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/pathc_${SCOPE}.log"
  local records="$CANARY_CONTROL_RECORDS"
  local sample="$CANARY_TRACES"
  local sub_run_id="${RUN_ID}_pathc_canary"
  if [[ "$SCOPE" == "full" ]]; then
    records="$FULL_CONTROL_RECORDS"
    sample="0"
    sub_run_id="${RUN_ID}_pathc_full"
  fi
  local out_json="phase7_results/results/phase7_sae_trajectory_pathc_robust_${sub_run_id}.json"
  local divergent="$BASE/meta/divergent_${SCOPE}.json"
  {
    echo "[$(date -Is)] pathc start scope=${SCOPE}"
    [[ -f "$divergent" ]] || { echo "missing divergent source: $divergent" >&2; exit 1; }
    RUN_ID="$sub_run_id" \
    RUN_TAG="phase7_prontoqa_pathc_${sub_run_id}" \
    BASE="phase7_results/runs/${sub_run_id}" \
    CONTROL_RECORDS="$records" \
    MODEL_KEY="$MODEL_KEY" \
    LAYERS_CSV="$LAYERS_CSV" \
    SAES_DIR="$SAES_DIR" \
    ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
    PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
    DIVERGENT_SOURCE="$divergent" \
    SUBSPACE_SPECS="$SUBSPACE_SPECS" \
    FEATURE_SET="eq_pre_result_150" \
    SAMPLE_TRACES="$sample" \
    MIN_COMMON_STEPS="$MIN_COMMON_STEPS" \
    SEED="$SEED" \
    N_BOOTSTRAP="$N_BOOTSTRAP" \
    BATCH_SIZE="$BATCH_SIZE" \
    TRACE_TEST_FRACTION=0.20 \
    TRACE_SPLIT_SEED=20260306 \
    PROBE_EPOCHS=500 \
    PROBE_LR=0.03 \
    PROBE_WEIGHT_DECAY=0.0001 \
    PROBE_DEVICE=cpu \
    TRAIN_EXCLUDE_VARIANTS="order_flip_only,answer_first_order_flip,reordered_steps" \
    REQUIRE_WRONG_INTERMEDIATE_AUROC=0.70 \
    WRONG_INTERMEDIATE_BOOTSTRAP_N=1000 \
    WRONG_INTERMEDIATE_BOOTSTRAP_SEED=20260307 \
    CV_FOLDS=5 \
    CV_SEED=20260307 \
    CV_MIN_VALID_FOLDS=3 \
    MODEL_LADDER="sae_only,hybrid_only,mixed" \
    MIXED_DELTA_EFFECT_FLOOR=0.03 \
    DECODER_CHECKPOINT="$PATHC_DECODER_CHECKPOINT" \
    ./experiments/run_phase7_sae_trajectory_pathc_robust.sh launch

    [[ -f "$out_json" ]] || { echo "missing pathc output: $out_json" >&2; exit 1; }
    json_parseable "$out_json" || { echo "bad pathc output: $out_json" >&2; exit 1; }
    "$PY" - <<PY
import json
from pathlib import Path
meta = json.load(open(${out_json@Q}))
Path(${BASE@Q}).joinpath("meta","pathc_" + ${SCOPE@Q} + ".json").write_text(json.dumps(meta, indent=2))
PY
    touch "$BASE/state/pathc_${SCOPE}.done"
    echo "[$(date -Is)] pathc done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_stress() {
  require_env
  : "${SCOPE:?SCOPE required (full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/stress_${SCOPE}.log"
  local source_run_id="${RUN_ID}_pathc_full_core"
  local stress_run_id="${RUN_ID}_stress_full"
  local out_json="phase7_results/results/qwen_pathc_stress_${stress_run_id}.json"
  {
    echo "[$(date -Is)] stress start scope=${SCOPE}"
    RUN_ID="$stress_run_id" \
    RUN_TAG="qwen_pathc_stress_${stress_run_id}" \
    BASE="phase7_results/runs/${stress_run_id}" \
    SOURCE_RUN_ID="$source_run_id" \
    TRAIN_EXCLUDE_VARIANTS="order_flip_only,answer_first_order_flip,reordered_steps" \
    DEVICE=cpu \
    ./experiments/run_qwen_pathc_stress.sh launch

    local coord="p7st_coord_${stress_run_id}"
    while [[ ! -f "phase7_results/runs/${stress_run_id}/state/pipeline.done" ]]; do
      if ! tmux has-session -t "$coord" 2>/dev/null; then
        sleep 5
        [[ -f "phase7_results/runs/${stress_run_id}/state/pipeline.done" ]] || { echo "stress coordinator disappeared: ${coord}" >&2; exit 1; }
      fi
      sleep 15
    done
    [[ -f "$out_json" ]] || { echo "missing stress output: $out_json" >&2; exit 1; }
    json_parseable "$out_json" || { echo "bad stress output: $out_json" >&2; exit 1; }
    touch "$BASE/state/stress_${SCOPE}.done"
    echo "[$(date -Is)] stress done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_coordinator() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" "phase7_results/results"
  local logf="$BASE/logs/coordinator.log"
  {
    echo "[$(date -Is)] coordinator start run_id=${RUN_ID}"
    wait_for_file_with_session "$BASE/state/precompute_canary.done" "$PRE_SESSION" 172800

    SCOPE=canary "$0" patha
    SCOPE=canary "$0" pathb
    SCOPE=canary "$0" pathc

    canary_pathc_json="phase7_results/results/phase7_sae_trajectory_pathc_robust_${RUN_ID}_pathc_canary.json"
    [[ -f "$canary_pathc_json" ]] || { echo "missing canary pathc json" >&2; exit 1; }
    canary_integrity_ok="$("$PY" - <<PY
import json
d=json.load(open(${canary_pathc_json@Q}))
cv=((d.get("cv_diagnostics") or {}).get("cv_trace_overlap_count"))
status=str(d.get("status",""))
ok = (status=="ok") and (int(cv or 0)==0)
print("true" if ok else "false")
PY
)"

    full_ran="false"
    full_pass="false"
    blocked_reason=""
    if [[ "$canary_integrity_ok" == "true" ]]; then
      PREP_KIND=full "$0" precompute
      SCOPE=full "$0" patha
      SCOPE=full "$0" pathb
      SCOPE=full "$0" pathc
      SCOPE=full "$0" stress
      full_ran="true"
      full_pathc_json="phase7_results/results/phase7_sae_trajectory_pathc_robust_${RUN_ID}_pathc_full.json"
      full_pass="$("$PY" - <<PY
import json
d=json.load(open(${full_pathc_json@Q}))
cv=(d.get("cv_diagnostics") or {})
auc=cv.get("cv_wrong_intermediate_pooled_auroc")
ci=(cv.get("cv_wrong_intermediate_pooled_ci95") or {})
ok = isinstance(auc,(int,float)) and float(auc)>0.70 and isinstance(ci.get("lower"),(int,float)) and float(ci.get("lower"))>=0.65 and int(cv.get("cv_trace_overlap_count",0))==0
print("true" if ok else "false")
PY
)"
      if [[ "$full_pass" != "true" ]]; then
        blocked_reason="full_gate_not_met"
      fi
    else
      blocked_reason="canary_integrity_failed"
    fi

    out_json="phase7_results/results/prontoqa_trackc_pilot_${RUN_ID}.json"
    out_md="phase7_results/results/prontoqa_trackc_pilot_${RUN_ID}.md"
    "$PY" - <<PY
import json
from datetime import datetime
from pathlib import Path
run_id=${RUN_ID@Q}
base=${BASE@Q}
canary_pathc=f"phase7_results/results/phase7_sae_trajectory_pathc_robust_{run_id}_pathc_canary.json"
full_pathc=f"phase7_results/results/phase7_sae_trajectory_pathc_robust_{run_id}_pathc_full.json"
stress=f"phase7_results/results/qwen_pathc_stress_{run_id}_stress_full.json"
def load_if(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else None
canary=load_if(canary_pathc)
full=load_if(full_pathc)
stressj=load_if(stress)
out={
  "schema_version":"prontoqa_trackc_pilot_v1",
  "status":"ok",
  "run_id": run_id,
  "run_tag": ${RUN_TAG@Q},
  "model_key": ${MODEL_KEY@Q},
  "canary_integrity_pass": (${canary_integrity_ok@Q}=="true"),
  "full_ran": (${full_ran@Q}=="true"),
  "full_pass_gate": (${full_pass@Q}=="true"),
  "blocked_reason": ${blocked_reason@Q},
  "primary_endpoint": {
    "target": "wrong_intermediate pooled 5-fold CV AUROC > 0.70 and CI95 lower >= 0.65",
    "canary": None if not isinstance(canary, dict) else ((canary.get("cv_diagnostics") or {}).get("cv_wrong_intermediate_pooled_auroc")),
    "full": None if not isinstance(full, dict) else ((full.get("cv_diagnostics") or {}).get("cv_wrong_intermediate_pooled_auroc")),
    "full_ci95": None if not isinstance(full, dict) else ((full.get("cv_diagnostics") or {}).get("cv_wrong_intermediate_pooled_ci95")),
  },
  "artifacts": {
    "canary_pathc_robust_json": canary_pathc,
    "full_pathc_robust_json": full_pathc,
    "stress_json": stress,
    "arithmetic_reframe_json": f"phase7_results/results/trackc_arithmetic_significance_reframe_{run_id}.json",
  },
  "tests": {
    "canary_pathc_robust": canary,
    "full_pathc_robust": full,
    "stress": stressj,
  },
  "timestamp": datetime.now().isoformat(),
}
Path(${out_json@Q}).parent.mkdir(parents=True, exist_ok=True)
Path(${out_json@Q}).write_text(json.dumps(out, indent=2))
lines=[
  "# PrOntoQA Track C Pilot Summary",
  "",
  f"- Run id: {run_id}",
  f"- Canary integrity pass: {out['canary_integrity_pass']}",
  f"- Full ran: {out['full_ran']}",
  f"- Full gate pass: {out['full_pass_gate']}",
  f"- Blocked reason: {out['blocked_reason']}",
  f"- Full pooled wrong_intermediate AUROC: {(out['primary_endpoint'] or {}).get('full')}",
  f"- Full pooled CI95: {(out['primary_endpoint'] or {}).get('full_ci95')}",
]
Path(${out_md@Q}).write_text("\\n".join(lines)+"\\n")
Path(base).joinpath("state","pipeline.done").write_text("done\\n")
print("saved", ${out_json@Q})
PY
    echo "[$(date -Is)] coordinator done"
  } >>"$logf" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_prontoqa_trackc_pilot}"
    RUN_TAG="${RUN_TAG:-phase7_prontoqa_trackc_pilot_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"
    MODEL_KEY="${MODEL_KEY:-qwen2.5-7b}"
    PRECOMPUTE_GPU="${PRECOMPUTE_GPU:-7}"
    CANARY_TRACES="${CANARY_TRACES:-200}"
    FULL_TRACES="${FULL_TRACES:-1000}"
    MAX_RECORDS="${MAX_RECORDS:-20000}"
    CHAIN_LEN_MIN="${CHAIN_LEN_MIN:-3}"
    CHAIN_LEN_MAX="${CHAIN_LEN_MAX:-5}"
    PRONTOQA_VARIANTS="${PRONTOQA_VARIANTS:-faithful,wrong_intermediate,order_flip,skipped_step,wrong_premise,irrelevant_insertion}"
    SEED="${SEED:-20260309}"
    SAES_DIR="${SAES_DIR:-phase2_results/saes_qwen25_7b_12x_topk/saes}"
    ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
    PHASE4_TOP_FEATURES="${PHASE4_TOP_FEATURES:-phase7_results/runs/20260308_165109_phase7_qwen_trackc_upgrade/interventions/top_features_per_layer_qwen_phase7_qwen_trackc_upgrade_20260308_165109_phase7_qwen_trackc_upgrade.json}"
    SUBSPACE_SPECS="${SUBSPACE_SPECS:-phase7_results/runs/20260308_165109_phase7_qwen_trackc_upgrade/interventions/variable_subspaces_qwen_phase7_qwen_trackc_upgrade_20260308_165109_phase7_qwen_trackc_upgrade.json}"
    PATHC_DECODER_CHECKPOINT="${PATHC_DECODER_CHECKPOINT:-phase7_results/runs/20260308_165109_phase7_qwen_trackc_upgrade/checkpoints/state_raw_every2_even_d1tier1.pt}"
    LAYERS_CSV="${LAYERS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}"
    FDISC_LAYERS_G5="${FDISC_LAYERS_G5:-0,1,2,3,4,5,6,7,8,9}"
    FDISC_LAYERS_G6="${FDISC_LAYERS_G6:-10,11,12,13,14,15,16,17,18}"
    FDISC_LAYERS_G7="${FDISC_LAYERS_G7:-19,20,21,22,23,24,25,26,27}"
    MIN_COMMON_STEPS="${MIN_COMMON_STEPS:-3}"
    N_BOOTSTRAP="${N_BOOTSTRAP:-500}"
    BATCH_SIZE="${BATCH_SIZE:-256}"

    PREP_CANARY_DIR="${PREP_CANARY_DIR:-$BASE/prontoqa_canary}"
    PREP_FULL_DIR="${PREP_FULL_DIR:-$BASE/prontoqa_full}"
    CANARY_CONTROL_RECORDS="${CANARY_CONTROL_RECORDS:-$PREP_CANARY_DIR/interventions/control_records_prontoqa.json}"
    FULL_CONTROL_RECORDS="${FULL_CONTROL_RECORDS:-$PREP_FULL_DIR/interventions/control_records_prontoqa.json}"

    PRE_SESSION="p7pq_pre_${RUN_ID}"
    COORD_SESSION="p7pq_coord_${RUN_ID}"

    mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
    rm -f "$BASE/state"/*.done
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
MODEL_KEY=$MODEL_KEY
PRECOMPUTE_GPU=$PRECOMPUTE_GPU
CANARY_TRACES=$CANARY_TRACES
FULL_TRACES=$FULL_TRACES
MAX_RECORDS=$MAX_RECORDS
CHAIN_LEN_MIN=$CHAIN_LEN_MIN
CHAIN_LEN_MAX=$CHAIN_LEN_MAX
PRONTOQA_VARIANTS=$PRONTOQA_VARIANTS
SEED=$SEED
SAES_DIR=$SAES_DIR
ACTIVATIONS_DIR=$ACTIVATIONS_DIR
PHASE4_TOP_FEATURES=$PHASE4_TOP_FEATURES
SUBSPACE_SPECS=$SUBSPACE_SPECS
PATHC_DECODER_CHECKPOINT=$PATHC_DECODER_CHECKPOINT
LAYERS_CSV=$LAYERS_CSV
FDISC_LAYERS_G5=$FDISC_LAYERS_G5
FDISC_LAYERS_G6=$FDISC_LAYERS_G6
FDISC_LAYERS_G7=$FDISC_LAYERS_G7
MIN_COMMON_STEPS=$MIN_COMMON_STEPS
N_BOOTSTRAP=$N_BOOTSTRAP
BATCH_SIZE=$BATCH_SIZE
PREP_CANARY_DIR=$PREP_CANARY_DIR
PREP_FULL_DIR=$PREP_FULL_DIR
CANARY_CONTROL_RECORDS=$CANARY_CONTROL_RECORDS
FULL_CONTROL_RECORDS=$FULL_CONTROL_RECORDS
PRE_SESSION=$PRE_SESSION
COORD_SESSION=$COORD_SESSION
CFG

    for s in "$PRE_SESSION" "$COORD_SESSION"; do
      tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s"
    done
    tmux new-session -d -s "$PRE_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && PREP_KIND=canary '$0' precompute"
    tmux new-session -d -s "$COORD_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched PrOntoQA Track C pilot"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  sessions: $PRE_SESSION, $COORD_SESSION"
    ;;
  precompute)
    run_precompute
    ;;
  patha)
    run_patha
    ;;
  pathb)
    run_pathb
    ;;
  pathc)
    run_pathc
    ;;
  stress)
    run_stress
    ;;
  coordinator)
    run_coordinator
    ;;
  *)
    usage
    exit 2
    ;;
esac


#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  experiments/run_phase7_option_bc.sh launch
  experiments/run_phase7_option_bc.sh worker-g3 <RUN_ID> <RUN_TAG>
  experiments/run_phase7_option_bc.sh worker-g4 <RUN_ID> <RUN_TAG>
  experiments/run_phase7_option_bc.sh coordinator <RUN_ID> <RUN_TAG>
USAGE
}

PY="${PY:-.venv/bin/python3}"
SOURCE_RUN_TAG="${SOURCE_RUN_TAG:-phase7_r2p1_20260305_025056_phase7_r2p1_full}"
CONTROLS="${CONTROLS:-phase7_results/controls/cot_controls_${SOURCE_RUN_TAG}.json}"
TRACE_DATASET="${TRACE_DATASET:-phase7_results/dataset/gsm8k_step_traces_test.pt}"
CKPT_DIR="${CKPT_DIR:-phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration/checkpoints}"
CONTROL_RECORDS="${CONTROL_RECORDS:-phase7_results/interventions/control_records_20260305_234011_phase7v3_closure.json}"

SPLIT_SEED="${SPLIT_SEED:-20260303}"
CALIB_FRAC="${CALIB_FRAC:-0.30}"
MAX_CONTROLS="${MAX_CONTROLS:-2200}"
COVERAGE_GUARD="${COVERAGE_GUARD:-0.95}"
ORDERING_MARGIN="${ORDERING_MARGIN:-0.005}"
PROBE_TRACE_TEST_FRACTION="${PROBE_TRACE_TEST_FRACTION:-0.20}"
PROBE_TRACE_SPLIT_SEED="${PROBE_TRACE_SPLIT_SEED:-20260306}"

VARS=(subresult_value operator magnitude_bucket sign)
PROFILES=(P1 P2 P3)

json_ok() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  "$PY" - <<PY
import json,sys
try:
    json.load(open("$p"))
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

ckpt_path() {
  local ckpt_id="$1"
  case "$ckpt_id" in
    raw_every2_even) echo "$CKPT_DIR/state_raw_every2_even.pt" ;;
    raw_middle12_06_17) echo "$CKPT_DIR/state_raw_middle12_06_17.pt" ;;
    hybrid_every2_even) echo "$CKPT_DIR/state_hybrid_every2_even.pt" ;;
    hybrid_middle12_06_17) echo "$CKPT_DIR/state_hybrid_middle12_06_17.pt" ;;
    *) echo "unknown checkpoint id: $ckpt_id" >&2; return 2 ;;
  esac
}

cache_path() {
  local ckpt_id="$1"
  echo "phase7_results/interventions/control_latent_cache_${SOURCE_RUN_TAG}_matrix_${ckpt_id}.json"
}

causal_path() {
  local ckpt_id="$1"
  local var="$2"
  echo "phase7_results/interventions/causal_checks_${SOURCE_RUN_TAG}_matrix_${ckpt_id}_${var}.json"
}

profile_weights_json() {
  local profile="$1"
  case "$profile" in
    P1) echo '{"text":1.0,"latent":0.0,"confidence":0.0,"causal":0.0}' ;;
    P2) echo '{"text":0.5,"latent":0.5,"confidence":0.0,"causal":0.0}' ;;
    P3) echo '{"text":0.5,"latent":0.3,"confidence":0.2,"causal":0.0}' ;;
    *) echo "unknown profile: $profile" >&2; return 2 ;;
  esac
}

ensure_profile_thresholds() {
  local run_id="$1"
  local profile="$2"
  local out="phase7_results/runs/${run_id}/meta/optionb_threshold_profile_${profile}.json"
  if json_ok "$out"; then
    echo "$out"
    return 0
  fi
  local w
  w="$(profile_weights_json "$profile")"
  "$PY" - <<PY
import json
from pathlib import Path
w=json.loads('''$w''')
out=Path("$out")
out.parent.mkdir(parents=True, exist_ok=True)
payload={
  "thresholds_version":"phase7_optionb_profile_v1",
  "thresholds":{
    "text_score_weight": float(w["text"]),
    "latent_score_weight": float(w["latent"]),
    "confidence_score_weight": float(w["confidence"]),
    "causal_score_weight": float(w["causal"]),
    "confidence_field_operator_weight": 0.40,
    "confidence_field_sign_weight": 0.30,
    "confidence_field_magnitude_weight": 0.30
  },
  "profile_id":"$profile",
  "profile_weights":w
}
json.dump(payload, open(out,"w"), indent=2)
print(out)
PY
}

cache_has_confidence_probs() {
  local cache="$1"
  "$PY" - <<'PY' "$cache"
import json,sys,math
from pathlib import Path
from phase7.common import load_rows_payload

p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
payload = json.load(open(p))
rows = load_rows_payload(payload, base_path=str(p))
if not rows:
    raise SystemExit(1)
row = rows[0] or {}
conf = (row.get("latent_pred_confidence") or {})
ok = (
    isinstance(conf.get("operator_probs"), dict)
    and isinstance(conf.get("sign_probs"), dict)
    and isinstance(conf.get("magnitude_probs"), dict)
)
raise SystemExit(0 if ok else 1)
PY
}

ensure_cache_with_confidence() {
  local ckpt_id="$1"
  local gpu_device="$2"
  local cache
  cache="$(cache_path "$ckpt_id")"
  if cache_has_confidence_probs "$cache"; then
    return 0
  fi
  local ckpt
  ckpt="$(ckpt_path "$ckpt_id")"
  echo "[optionbc] rebuilding latent cache for $ckpt_id at $cache"
  "$PY" phase7/build_control_latent_cache.py \
    --controls "$CONTROLS" \
    --state-decoder-checkpoint "$ckpt" \
    --model-key gpt2-medium \
    --parse-mode hybrid \
    --token-anchor eq_like \
    --anchor-priority template_first \
    --device "$gpu_device" \
    --batch-size 64 \
    --rows-format json \
    --rows-inline \
    --output "$cache"
  cache_has_confidence_probs "$cache"
}

add_benchmark_metadata() {
  local bench="$1"
  local profile="$2"
  local stage="$3"
  local ckpt_id="$4"
  local variable="$5"
  local weights
  weights="$(profile_weights_json "$profile")"
  "$PY" - <<PY
import json
p="$bench"
d=json.load(open(p))
d["penalty_aware_profile_comparison"]=True
d["comparison_method"]="reaudit_profile_thresholds"
d["optionb_profile_id"]="$profile"
d["optionb_stage"]="$stage"
d["optionb_checkpoint_id"]="$ckpt_id"
d["optionb_variable"]="$variable"
d["optionb_profile_weights"]=json.loads('''$weights''')
json.dump(d, open(p,"w"), indent=2)
PY
}

run_optionb_bundle() {
  local run_id="$1"
  local run_tag="$2"
  local stage="$3"
  local profile="$4"
  local ckpt_id="$5"
  local variable="$6"
  local gpu_device="$7"

  local ckpt
  ckpt="$(ckpt_path "$ckpt_id")"
  local cache
  cache="$(cache_path "$ckpt_id")"
  local causal
  causal="$(causal_path "$ckpt_id" "$variable")"
  [[ -f "$causal" ]] || { echo "[optionbc] missing causal checks: $causal" >&2; return 2; }

  ensure_cache_with_confidence "$ckpt_id" "$gpu_device"

  local thr_profile
  thr_profile="$(ensure_profile_thresholds "$run_id" "$profile")"

  local suffix="${run_tag}_${stage}_${profile}_${ckpt_id}_${variable}"
  local audit="phase7_results/audits/optionb_audit_${suffix}.json"
  local calib="phase7_results/audits/optionb_calib_${suffix}.json"
  local eval="phase7_results/audits/optionb_eval_${suffix}.json"
  local manifest="phase7_results/audits/optionb_split_manifest_${suffix}.json"
  local thresholds="phase7_results/calibration/optionb_thresholds_${suffix}.json"
  local bench="phase7_results/results/optionb_benchmark_${suffix}.json"

  if json_ok "$bench"; then
    echo "[optionbc] skip existing benchmark $bench"
    return 0
  fi

  "$PY" phase7/causal_audit.py \
    --controls "$CONTROLS" \
    --trace-dataset "$TRACE_DATASET" \
    --state-decoder-checkpoint "$ckpt" \
    --causal-checks "$causal" \
    --require-causal-mode control_conditioned \
    --latent-source variant_conditioned \
    --control-latent-cache "$cache" \
    --thresholds "$thr_profile" \
    --require-confidence-defined-fraction "$COVERAGE_GUARD" \
    --max-controls "$MAX_CONTROLS" \
    --device cpu \
    --output "$audit"

  "$PY" phase7/split_audit_dataset.py \
    --audit "$audit" \
    --seed "$SPLIT_SEED" \
    --calib-fraction "$CALIB_FRAC" \
    --output-calib "$calib" \
    --output-eval "$eval" \
    --output-manifest "$manifest"

  "$PY" phase7/calibrate_audit_thresholds.py \
    --audit-calib "$calib" \
    --all-tracks \
    --score-track composite \
    --positive-label unfaithful \
    --output "$thresholds"

  "$PY" phase7/benchmark_faithfulness.py \
    --audit-eval "$eval" \
    --thresholds "$thresholds" \
    --benchmark-scope synthetic_controls \
    --external-validity-status not_tested \
    --gate-track composite \
    --require-dual-gate \
    --causal-floor-auroc 0.65 \
    --causal-floor-fpr-max 0.05 \
    --confidence-text-corr-max 0.80 \
    --output "$bench"

  add_benchmark_metadata "$bench" "$profile" "$stage" "$ckpt_id" "$variable"
}

run_optionb_canary_and_maybe_raw_matrix() {
  local run_id="$1"
  local run_tag="$2"
  local state_dir="phase7_results/runs/${run_id}/state"
  local gpu_device="$3"
  mkdir -p "$state_dir"

  local ckpt_canary="raw_every2_even"
  for profile in "${PROFILES[@]}"; do
    for v in "${VARS[@]}"; do
      run_optionb_bundle "$run_id" "$run_tag" "canary" "$profile" "$ckpt_canary" "$v" "$gpu_device"
    done
  done

  local canary_out="phase7_results/results/optionb_canary_decision_${run_tag}.json"
  "$PY" - <<PY
import json
from pathlib import Path
run_tag="$run_tag"
profiles=["P1","P2","P3"]
vars=["subresult_value","operator","magnitude_bucket","sign"]
margin=float("$ORDERING_MARGIN")
rows=[]
for p in profiles:
    vals=[]
    leak=[]
    conf_guard=[]
    for v in vars:
        b=Path(f"phase7_results/results/optionb_benchmark_{run_tag}_canary_{p}_raw_every2_even_{v}.json")
        a=Path(f"phase7_results/audits/optionb_audit_{run_tag}_canary_{p}_raw_every2_even_{v}.json")
        bp=json.loads(b.read_text())
        ap=json.loads(a.read_text())
        comp=((bp.get("by_benchmark_track") or {}).get("composite") or {})
        vals.append(float(comp.get("auroc")) if isinstance(comp.get("auroc"),(int,float)) else None)
        leak.append(bool(bp.get("leakage_check_pass")))
        conf_guard.append(bool((ap.get("summary") or {}).get("confidence_defined_guard_pass")) if p=="P3" else True)
    xs=[x for x in vals if isinstance(x,(int,float))]
    rows.append({
      "profile":p,
      "mean_composite_auroc": (sum(xs)/len(xs)) if xs else None,
      "all_leakage_pass": all(leak),
      "confidence_guard_all_pass": all(conf_guard),
    })
idx={r["profile"]:r for r in rows}
p1=idx["P1"]["mean_composite_auroc"]
p2=idx["P2"]["mean_composite_auroc"]
p3=idx["P3"]["mean_composite_auroc"]
ordering_pass=all(isinstance(x,(int,float)) for x in [p1,p2,p3]) and (p3 >= p2 + margin) and (p2 >= p1 + margin)
leak_pass=all(bool(r["all_leakage_pass"]) for r in rows)
conf_pass=bool(idx["P3"]["confidence_guard_all_pass"])
canary_pass=bool(ordering_pass and leak_pass and conf_pass)
out={
  "schema_version":"phase7_optionb_canary_decision_v1",
  "run_tag":run_tag,
  "profiles":rows,
  "ordering_margin":margin,
  "ordering_pass":bool(ordering_pass),
  "leakage_pass":bool(leak_pass),
  "confidence_guard_pass":bool(conf_pass),
  "canary_pass":bool(canary_pass),
}
Path("$canary_out").parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open("$canary_out","w"), indent=2)
print(json.dumps(out, indent=2))
PY

  "$PY" - <<PY > "${state_dir}/optionb_canary_pass.flag"
import json
print("1" if json.load(open("$canary_out")).get("canary_pass") else "0")
PY
  touch "${state_dir}/optionb_canary.done"

  if [[ "$(cat "${state_dir}/optionb_canary_pass.flag")" != "1" ]]; then
    echo "[optionbc] Option B canary failed; skipping matrix promotion for raw checkpoints."
    touch "${state_dir}/gpu3.matrix.skipped"
    return 0
  fi

  for ck in raw_every2_even raw_middle12_06_17; do
    for profile in "${PROFILES[@]}"; do
      for v in "${VARS[@]}"; do
        run_optionb_bundle "$run_id" "$run_tag" "matrix" "$profile" "$ck" "$v" "$gpu_device"
      done
    done
  done
  touch "${state_dir}/gpu3.matrix.done"
}

run_optionc_probe_then_maybe_hybrid_matrix() {
  local run_id="$1"
  local run_tag="$2"
  local state_dir="phase7_results/runs/${run_id}/state"
  local gpu_device="$3"
  mkdir -p "$state_dir"

  local probe_out="phase7_results/results/optionc_probe_${run_tag}.json"
  if ! json_ok "$probe_out"; then
    "$PY" phase7/contrastive_faithfulness_probe.py \
      --control-records "$CONTROL_RECORDS" \
      --layers 7,12,17,22 \
      --device "$gpu_device" \
      --feature-source both \
      --split-policy trace_stratified_random \
      --trace-test-fraction "$PROBE_TRACE_TEST_FRACTION" \
      --trace-split-seed "$PROBE_TRACE_SPLIT_SEED" \
      --output "$probe_out"
  fi
  touch "${state_dir}/optionc_probe.done"

  while [[ ! -f "${state_dir}/optionb_canary.done" ]]; do
    sleep 5
  done

  if [[ "$(cat "${state_dir}/optionb_canary_pass.flag")" != "1" ]]; then
    echo "[optionbc] Option B canary failed; skipping matrix promotion for hybrid checkpoints."
    touch "${state_dir}/gpu4.matrix.skipped"
    return 0
  fi

  for ck in hybrid_every2_even hybrid_middle12_06_17; do
    for profile in "${PROFILES[@]}"; do
      for v in "${VARS[@]}"; do
        run_optionb_bundle "$run_id" "$run_tag" "matrix" "$profile" "$ck" "$v" "$gpu_device"
      done
    done
  done
  touch "${state_dir}/gpu4.matrix.done"
}

run_coordinator() {
  local run_id="$1"
  local run_tag="$2"
  local state_dir="phase7_results/runs/${run_id}/state"

  while [[ ! -f "${state_dir}/optionb_canary.done" || ! -f "${state_dir}/optionc_probe.done" ]]; do
    sleep 5
  done

  if [[ "$(cat "${state_dir}/optionb_canary_pass.flag")" == "1" ]]; then
    while [[ ! -f "${state_dir}/gpu3.matrix.done" || ! -f "${state_dir}/gpu4.matrix.done" ]]; do
      sleep 10
    done
  fi

  local final_out="phase7_results/results/optionbc_final_${run_tag}.json"
  "$PY" - <<PY
import json,glob
from pathlib import Path
run_tag="$run_tag"
state_dir=Path("$state_dir")
canary=json.load(open(f"phase7_results/results/optionb_canary_decision_{run_tag}.json"))
probe=json.load(open(f"phase7_results/results/optionc_probe_{run_tag}.json"))
matrix_paths=sorted(glob.glob(f"phase7_results/results/optionb_benchmark_{run_tag}_matrix_*_*.json"))
by_profile={}
for p in ["P1","P2","P3"]:
    vals=[]
    for path in matrix_paths:
        if f"_matrix_{p}_" not in path:
            continue
        d=json.load(open(path))
        comp=((d.get("by_benchmark_track") or {}).get("composite") or {})
        a=comp.get("auroc")
        if isinstance(a,(int,float)):
            vals.append(float(a))
    by_profile[p]={"mean_composite_auroc":(sum(vals)/len(vals)) if vals else None,"n":len(vals)}
final={
  "schema_version":"phase7_optionbc_final_v1",
  "run_tag":run_tag,
  "optionb_canary_decision_ref":f"phase7_results/results/optionb_canary_decision_{run_tag}.json",
  "optionc_probe_ref":f"phase7_results/results/optionc_probe_{run_tag}.json",
  "optionb_canary_pass":bool(canary.get("canary_pass")),
  "matrix_promoted": bool(canary.get("canary_pass")),
  "matrix_profile_summary": by_profile,
  "matrix_num_benchmarks": int(len(matrix_paths)),
  "optionc_control_conditioned_test_auroc": (
    ((probe.get("by_feature_source") or {}).get("control_conditioned") or {}).get("test_auroc_unfaithful_positive")
  ),
  "optionc_source_trace_test_auroc": (
    ((probe.get("by_feature_source") or {}).get("source_trace") or {}).get("test_auroc_unfaithful_positive")
  ),
  "optionc_cross_variant_generalization_pass": bool(probe.get("cross_variant_generalization_pass")),
}
Path("$final_out").parent.mkdir(parents=True, exist_ok=True)
json.dump(final, open("$final_out","w"), indent=2)
print(json.dumps(final, indent=2))
PY
  touch "${state_dir}/pipeline.done"
}

launch() {
  local run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_optionbc}"
  local run_tag="${RUN_TAG:-phase7_optionbc_${run_id}}"
  local base="phase7_results/runs/${run_id}"
  mkdir -p "$base/state" "$base/logs" "$base/meta"

  echo "[optionbc] run_id=$run_id run_tag=$run_tag"
  tmux new-session -d -s "p7b_g3_${run_id}" \
    "cd '$PWD' && export CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 && bash experiments/run_phase7_option_bc.sh worker-g3 '$run_id' '$run_tag' 2>&1 | tee '$base/logs/gpu3.log'"
  tmux new-session -d -s "p7c_g4_${run_id}" \
    "cd '$PWD' && export CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 && bash experiments/run_phase7_option_bc.sh worker-g4 '$run_id' '$run_tag' 2>&1 | tee '$base/logs/gpu4.log'"
  tmux new-session -d -s "p7bc_coord_${run_id}" \
    "cd '$PWD' && export PYTHONUNBUFFERED=1 && bash experiments/run_phase7_option_bc.sh coordinator '$run_id' '$run_tag' 2>&1 | tee '$base/logs/coordinator.log'"
  echo "[optionbc] launched tmux sessions: p7b_g3_${run_id}, p7c_g4_${run_id}, p7bc_coord_${run_id}"
}

cmd="${1:-}"
case "$cmd" in
  launch)
    launch
    ;;
  worker-g3)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    run_optionb_canary_and_maybe_raw_matrix "$2" "$3" "cuda:0"
    ;;
  worker-g4)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    run_optionc_probe_then_maybe_hybrid_matrix "$2" "$3" "cuda:0"
    ;;
  coordinator)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    run_coordinator "$2" "$3"
    ;;
  *)
    usage
    exit 2
    ;;
esac

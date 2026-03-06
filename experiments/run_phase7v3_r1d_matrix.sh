#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  experiments/run_phase7v3_r1d_matrix.sh launch
  experiments/run_phase7v3_r1d_matrix.sh worker-raw <RUN_ID> <RUN_TAG> <GPU>
  experiments/run_phase7v3_r1d_matrix.sh worker-hybrid <RUN_ID> <RUN_TAG> <GPU>
  experiments/run_phase7v3_r1d_matrix.sh coordinator <RUN_ID> <RUN_TAG>
USAGE
}

PY="${PY:-.venv/bin/python3}"
SOURCE_RUN_TAG="${SOURCE_RUN_TAG:-phase7_r2p1_20260305_025056_phase7_r2p1_full}"
CONTROLS="${CONTROLS:-phase7_results/controls/cot_controls_${SOURCE_RUN_TAG}.json}"
TRACE_TEST="${TRACE_TEST:-phase7_results/dataset/gsm8k_step_traces_test.pt}"
CKPT_DIR="${CKPT_DIR:-phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration/checkpoints}"
MAX_CONTROLS="${MAX_CONTROLS:-2200}"
CALIB_FRAC="${CALIB_FRAC:-0.30}"
SPLIT_SEED="${SPLIT_SEED:-20260303}"
VARIABLES=(subresult_value operator magnitude_bucket sign)

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
    hybrid_every2_even) echo "$CKPT_DIR/state_hybrid_every2_even.pt" ;;
    raw_middle12_06_17) echo "$CKPT_DIR/state_raw_middle12_06_17.pt" ;;
    hybrid_middle12_06_17) echo "$CKPT_DIR/state_hybrid_middle12_06_17.pt" ;;
    *) echo "unknown checkpoint id: $ckpt_id" >&2; return 1 ;;
  esac
}

cache_path() {
  local ckpt_id="$1"
  echo "phase7_results/interventions/control_latent_cache_${SOURCE_RUN_TAG}_matrix_${ckpt_id}.json"
}

causal_path() {
  local ckpt_id="$1"
  local variable="$2"
  echo "phase7_results/interventions/causal_checks_${SOURCE_RUN_TAG}_matrix_${ckpt_id}_${variable}.json"
}

weights_for_profile() {
  local profile="$1"
  case "$profile" in
    A) echo '{"text":0.50,"latent":0.30,"confidence":0.20,"causal":0.00}' ;;
    B) echo '{"text":0.40,"latent":0.35,"confidence":0.25,"causal":0.00}' ;;
    C) echo '{"text":0.35,"latent":0.35,"confidence":0.30,"causal":0.00}' ;;
    D) echo '{"text":0.45,"latent":0.30,"confidence":0.25,"causal":0.00}' ;;
    *) echo "unknown profile: $profile" >&2; return 1 ;;
  esac
}

ensure_profile_thresholds() {
  local run_id="$1"
  local profile="$2"
  local out="phase7_results/runs/${run_id}/meta/threshold_profile_${profile}.json"
  if json_ok "$out"; then
    echo "$out"
    return 0
  fi
  local w
  w="$(weights_for_profile "$profile")"
  "$PY" - <<PY
import json
from pathlib import Path
w=json.loads('''$w''')
out=Path("$out")
out.parent.mkdir(parents=True, exist_ok=True)
payload={
  "thresholds_version":"phase7_trackc_r1d_profile_v1",
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

run_bundle() {
  local run_id="$1"
  local run_tag="$2"
  local stage="$3"
  local profile="$4"
  local ckpt_id="$5"
  local variable="$6"

  local threshold_profile
  threshold_profile="$(ensure_profile_thresholds "$run_id" "$profile")"

  local ckpt
  ckpt="$(ckpt_path "$ckpt_id")"
  local cache
  cache="$(cache_path "$ckpt_id")"
  local causal
  causal="$(causal_path "$ckpt_id" "$variable")"
  if [[ ! -f "$cache" || ! -f "$causal" ]]; then
    echo "[r1d] missing inputs for $ckpt_id/$variable: cache=$cache causal=$causal" >&2
    return 2
  fi

  local suffix="${run_tag}_${stage}_${profile}_${ckpt_id}_${variable}"
  local audit="phase7_results/audits/trackc_r1d_audit_${suffix}.json"
  local calib="phase7_results/audits/trackc_r1d_calib_${suffix}.json"
  local eval="phase7_results/audits/trackc_r1d_eval_${suffix}.json"
  local manifest="phase7_results/audits/trackc_r1d_split_manifest_${suffix}.json"
  local thresholds="phase7_results/calibration/trackc_r1d_thresholds_${suffix}.json"
  local bench="phase7_results/results/trackc_r1d_benchmark_${suffix}.json"

  if json_ok "$bench"; then
    echo "[r1d] skip existing benchmark $bench"
    return 0
  fi

  "$PY" phase7/causal_audit.py \
    --controls "$CONTROLS" \
    --trace-dataset "$TRACE_TEST" \
    --state-decoder-checkpoint "$ckpt" \
    --causal-checks "$causal" \
    --control-latent-cache "$cache" \
    --latent-source variant_conditioned \
    --require-causal-mode control_conditioned \
    --thresholds "$threshold_profile" \
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
    --confidence-text-corr-max 0.80 \
    --ablation-weights '{"text":0.5,"latent":0.5,"confidence":0.0,"causal":0.0}' \
    --output "$bench"
}

run_canary_tuning() {
  local run_id="$1"
  local run_tag="$2"
  local vars_json
  vars_json="$(printf '"%s",' "${VARIABLES[@]}")"
  vars_json="[${vars_json%,}]"
  local out="phase7_results/results/trackc_r1d_weight_tuning_${run_tag}.json"
  if json_ok "$out"; then
    echo "[r1d] reusing tuning output $out"
    return 0
  fi
  mkdir -p "phase7_results/runs/${run_id}/state" "phase7_results/runs/${run_id}/meta"
  local profiles=(A B C D)
  for p in "${profiles[@]}"; do
    for v in "${VARIABLES[@]}"; do
      run_bundle "$run_id" "$run_tag" "canary" "$p" "raw_every2_even" "$v"
    done
  done
  "$PY" - <<PY
import json
from pathlib import Path
run_tag="$run_tag"
profiles=["A","B","C","D"]
variables=json.loads('''$vars_json''')
profile_weights={
  "A":{"text":0.50,"latent":0.30,"confidence":0.20,"causal":0.00},
  "B":{"text":0.40,"latent":0.35,"confidence":0.25,"causal":0.00},
  "C":{"text":0.35,"latent":0.35,"confidence":0.30,"causal":0.00},
  "D":{"text":0.45,"latent":0.30,"confidence":0.25,"causal":0.00},
}
def load(path):
    return json.loads(Path(path).read_text())
rows=[]
for p in profiles:
    vals=[]
    for v in variables:
        b=load(f"phase7_results/results/trackc_r1d_benchmark_{run_tag}_canary_{p}_raw_every2_even_{v}.json")
        comp=((b.get("by_benchmark_track") or {}).get("composite") or {})
        abl=b.get("ablation_weighted_blend") or {}
        vals.append({
            "variable": v,
            "composite_auroc": comp.get("auroc"),
            "composite_fpr": comp.get("false_positive_rate"),
            "composite_recall": comp.get("recall"),
            "two_track_ablation_auroc": abl.get("auroc"),
            "delta_vs_two_track": (
                (float(comp.get("auroc"))-float(abl.get("auroc")))
                if isinstance(comp.get("auroc"), (int,float)) and isinstance(abl.get("auroc"), (int,float))
                else None
            ),
        })
    def m(key):
        xs=[r[key] for r in vals if isinstance(r.get(key),(int,float))]
        return (sum(xs)/len(xs)) if xs else None
    rows.append({
        "profile": p,
        "weights": profile_weights[p],
        "by_variable": vals,
        "mean_composite_auroc": m("composite_auroc"),
        "mean_composite_fpr": m("composite_fpr"),
        "mean_composite_recall": m("composite_recall"),
        "mean_delta_vs_two_track": m("delta_vs_two_track"),
    })
def key_fn(r):
    au=r.get("mean_composite_auroc")
    fpr=r.get("mean_composite_fpr")
    rec=r.get("mean_composite_recall")
    return (
        -999.0 if not isinstance(au,(int,float)) else float(au),
        999.0 if not isinstance(fpr,(int,float)) else -float(fpr),
        -999.0 if not isinstance(rec,(int,float)) else float(rec),
    )
sel=max(rows, key=key_fn)
out={
  "schema_version":"phase7_trackc_r1d_weight_tuning_v1",
  "run_tag": run_tag,
  "profiles_evaluated": rows,
  "selection_rule":"max mean composite AUROC; tie-break lower mean FPR; tie-break higher mean recall",
  "selected_profile": sel.get("profile"),
}
Path("$out").parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open("$out","w"), indent=2)
print(f"Saved tuning -> $out (selected={sel.get('profile')})")
PY
}

selected_profile() {
  local run_tag="$1"
  "$PY" - <<PY
import json
print(json.load(open("phase7_results/results/trackc_r1d_weight_tuning_$run_tag.json")).get("selected_profile","A"))
PY
}

worker_group() {
  local run_id="$1"
  local run_tag="$2"
  local group="$3"
  local gpu="$4"
  local profile
  profile="$(selected_profile "$run_tag")"
  local ckpts=()
  if [[ "$group" == "raw" ]]; then
    ckpts=(raw_every2_even raw_middle12_06_17)
  else
    ckpts=(hybrid_every2_even hybrid_middle12_06_17)
  fi
  for ck in "${ckpts[@]}"; do
    for v in "${VARIABLES[@]}"; do
      run_bundle "$run_id" "$run_tag" "matrix" "$profile" "$ck" "$v"
    done
  done
  mkdir -p "phase7_results/runs/${run_id}/state"
  echo "done" > "phase7_results/runs/${run_id}/state/gpu${gpu}.matrix.done"
}

run_coordinator() {
  local run_id="$1"
  local run_tag="$2"
  local state_dir="phase7_results/runs/${run_id}/state"
  while [[ ! -f "$state_dir/gpu6.matrix.done" || ! -f "$state_dir/gpu7.matrix.done" ]]; do
    sleep 10
  done
  local matrix_out="phase7_results/results/trackc_r1d_matrix_${run_tag}.json"
  local decision_out="phase7_results/results/trackc_phase7v3_decision_${run_tag}.json"
  "$PY" - <<PY
import json
from pathlib import Path
run_tag="$run_tag"
profile=json.load(open(f"phase7_results/results/trackc_r1d_weight_tuning_{run_tag}.json")).get("selected_profile")
ckpts=["raw_every2_even","raw_middle12_06_17","hybrid_every2_even","hybrid_middle12_06_17"]
vars=["subresult_value","operator","magnitude_bucket","sign"]
rows=[]
for ck in ckpts:
  for v in vars:
    p=Path(f"phase7_results/results/trackc_r1d_benchmark_{run_tag}_matrix_{profile}_{ck}_{v}.json")
    if not p.exists():
      continue
    b=json.loads(p.read_text())
    comp=((b.get("by_benchmark_track") or {}).get("composite") or {})
    abl=b.get("ablation_weighted_blend") or {}
    c_auc=comp.get("auroc")
    a_auc=abl.get("auroc")
    rows.append({
      "checkpoint":ck,
      "variable":v,
      "benchmark_path":str(p),
      "composite_auroc":c_auc,
      "composite_fpr":comp.get("false_positive_rate"),
      "composite_recall":comp.get("recall"),
      "two_track_ablation_auroc":a_auc,
      "delta_vs_two_track":((float(c_auc)-float(a_auc)) if isinstance(c_auc,(int,float)) and isinstance(a_auc,(int,float)) else None),
      "leakage_check_pass":bool(b.get("leakage_check_pass")),
    })
def mean(k):
  xs=[r[k] for r in rows if isinstance(r.get(k),(int,float))]
  return (sum(xs)/len(xs)) if xs else None
mean_delta=mean("delta_vs_two_track")
improved=[r for r in rows if isinstance(r.get("delta_vs_two_track"),(int,float)) and float(r["delta_vs_two_track"])>0.0]
robust=bool(len(improved)>=12 and isinstance(mean_delta,(int,float)) and float(mean_delta)>=0.01 and all(r.get("leakage_check_pass") for r in rows))
matrix={
  "schema_version":"phase7_trackc_r1d_matrix_v1",
  "run_tag":run_tag,
  "selected_profile":profile,
  "rows":rows,
  "summary":{
    "num_rows":len(rows),
    "mean_composite_auroc":mean("composite_auroc"),
    "mean_composite_fpr":mean("composite_fpr"),
    "mean_composite_recall":mean("composite_recall"),
    "mean_two_track_ablation_auroc":mean("two_track_ablation_auroc"),
    "mean_delta_vs_two_track":mean_delta,
    "num_rows_improved_vs_two_track":len(improved),
    "robustly_beats_two_track":robust,
  },
}
Path("$matrix_out").parent.mkdir(parents=True, exist_ok=True)
json.dump(matrix, open("$matrix_out","w"), indent=2)
decision={
  "schema_version":"phase7_trackc_decision_v3",
  "run_tag":run_tag,
  "selected_track_c":"confidence_margin",
  "selected_profile":profile,
  "matrix_summary_ref":"$matrix_out",
  "switch_production_default_recommended":bool(robust),
  "recommendation_reason":(
    "Switch confidence-inclusive composite default after matrix pass."
    if robust else
    "Keep two-track production default; confidence-margin remains run-scoped until stronger matrix win."
  ),
}
json.dump(decision, open("$decision_out","w"), indent=2)
print(f"Saved matrix -> $matrix_out")
print(f"Saved decision -> $decision_out")
PY
  touch "$state_dir/pipeline.done"
}

launch() {
  local run_id
  run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7v3_r1d_matrix}"
  local run_tag
  run_tag="${RUN_TAG:-phase7v3_r1d_${run_id}}"
  local base="phase7_results/runs/${run_id}"
  mkdir -p "$base/state" "$base/logs" "$base/meta"

  echo "[r1d] run_id=$run_id run_tag=$run_tag"
  run_canary_tuning "$run_id" "$run_tag"

  tmux new-session -d -s "p7v3r1d_g6_${run_id}" \
    "cd '$PWD' && export CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 && bash experiments/run_phase7v3_r1d_matrix.sh worker-raw '$run_id' '$run_tag' 6 2>&1 | tee '$base/logs/gpu6.log'"
  tmux new-session -d -s "p7v3r1d_g7_${run_id}" \
    "cd '$PWD' && export CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 && bash experiments/run_phase7v3_r1d_matrix.sh worker-hybrid '$run_id' '$run_tag' 7 2>&1 | tee '$base/logs/gpu7.log'"
  tmux new-session -d -s "p7v3r1d_coord_${run_id}" \
    "cd '$PWD' && export PYTHONUNBUFFERED=1 && bash experiments/run_phase7v3_r1d_matrix.sh coordinator '$run_id' '$run_tag' 2>&1 | tee '$base/logs/coordinator.log'"
  echo "[r1d] launched tmux sessions: p7v3r1d_g6_${run_id}, p7v3r1d_g7_${run_id}, p7v3r1d_coord_${run_id}"
}

cmd="${1:-}"
case "$cmd" in
  launch)
    launch
    ;;
  worker-raw)
    [[ $# -eq 4 ]] || { usage; exit 2; }
    worker_group "$2" "$3" raw "$4"
    ;;
  worker-hybrid)
    [[ $# -eq 4 ]] || { usage; exit 2; }
    worker_group "$2" "$3" hybrid "$4"
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

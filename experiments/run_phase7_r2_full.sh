#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  experiments/run_phase7_r2_full.sh launch
  experiments/run_phase7_r2_full.sh worker-synth <RUN_ID> <RUN_TAG> <BASE> <GPU> <ROLE>
  experiments/run_phase7_r2_full.sh coordinator <RUN_ID> <RUN_TAG> <BASE>
USAGE
}

PY=".venv/bin/python3"
TRACE_TEST="phase7_results/dataset/gsm8k_step_traces_test.pt"
MANIFEST="experiments/layer_sweep_manifest_v1.json"
CKPT_DIR="phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration/checkpoints"

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

freeze_baseline_refs() {
  local run_tag="$1"
  local out="phase7_results/results/${run_tag}_baseline_refs.json"
  $PY - <<PY
import glob,json
run_tag="${run_tag}"
refs={
  "schema_version":"phase7_r2p1_baseline_refs_v1",
  "run_tag":run_tag,
  "latest_trackA_gate_matrix":sorted(glob.glob("phase7_results/results/trackA_gate_matrix*.json"))[-1] if glob.glob("phase7_results/results/trackA_gate_matrix*.json") else None,
  "latest_real_cot_labeled_benchmark":sorted(glob.glob("phase7_results/results/faithfulness_benchmark_real_cot_labeled*.json"))[-1] if glob.glob("phase7_results/results/faithfulness_benchmark_real_cot_labeled*.json") else None,
  "latest_closure_report":sorted(glob.glob("phase7_results/results/academic_weakness_closure_report*.json"))[-1] if glob.glob("phase7_results/results/academic_weakness_closure_report*.json") else None,
}
with open("${out}","w") as f:
  json.dump(refs,f,indent=2)
print("${out}")
PY
}

write_closure_contract() {
  local run_tag="$1"
  local out="phase7_results/results/${run_tag}_closure_contract.json"
  $PY - <<PY
import json
run_tag="${run_tag}"
contract={
  "schema_version":"phase7_r2p1_closure_contract_v1",
  "run_tag":run_tag,
  "dual_gate_unchanged":True,
  "strict_external_labels_required":True,
  "run_scope":"four_checkpoint_matrix",
  "hardware_target":["gpu6","gpu7"],
  "gate":{
    "composite_auroc_min":0.85,
    "composite_fpr_max":0.05,
    "composite_recall_gt":0.0,
    "causal_auroc_min":0.65,
    "causal_fpr_max":0.05,
    "causal_coverage_min":0.25,
  }
}
with open("${out}","w") as f:
  json.dump(contract,f,indent=2)
print("${out}")
PY
}

prepare_shared_inputs() {
  local run_tag="$1"
  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local parsed="phase7_results/controls/cot_controls_${run_tag}_parsed.json"
  local subspaces="phase7_results/interventions/variable_subspaces_${run_tag}.json"

  echo "[launch] generate controls"
  $PY phase7/generate_cot_controls.py \
    --trace-dataset "$TRACE_TEST" \
    --max-traces 500 \
    --seed 20260305 \
    --output "$controls"

  echo "[launch] parse controls"
  $PY phase7/parse_cot_to_states.py \
    --controls "$controls" \
    --trace-dataset "$TRACE_TEST" \
    --parse-mode hybrid \
    --output "$parsed"

  echo "[launch] build variable subspaces"
  $PY phase7/variable_subspace_builder.py \
    --model-key gpt2-medium \
    --layers 7 12 17 22 \
    --probe-position result \
    --top-k 64 \
    --combine-policy union \
    --output "$subspaces"
}

run_checkpoint_bundle() {
  local run_tag="$1"
  local ckpt_id="$2"
  local ckpt_path="$3"
  local anchor_priority="$4"
  local mode_tag="$5"
  local device="$6"
  local log_prefix="$7"

  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local subspaces="phase7_results/interventions/variable_subspaces_${run_tag}.json"
  local cache="phase7_results/interventions/control_latent_cache_${run_tag}_${mode_tag}_${ckpt_id}.json"

  echo "[${log_prefix}] cache build: ckpt=${ckpt_id} mode=${mode_tag} anchor=${anchor_priority}"
  $PY phase7/build_control_latent_cache.py \
    --controls "$controls" \
    --state-decoder-checkpoint "$ckpt_path" \
    --model-key gpt2-medium \
    --parse-mode hybrid \
    --token-anchor eq_like \
    --anchor-priority "$anchor_priority" \
    --device "$device" \
    --batch-size 128 \
    --output "$cache"

  for var in subresult_value operator magnitude_bucket sign; do
    local tag="${run_tag}_${mode_tag}_${ckpt_id}_${var}"
    local causal="phase7_results/interventions/causal_checks_${tag}.json"
    local audit="phase7_results/audits/text_causal_audit_controls_${tag}.json"
    local split_prefix="phase7_results/audits/${tag}"
    local calib="${split_prefix}_calib.json"
    local eval="${split_prefix}_eval.json"
    local thr="phase7_results/calibration/phase7_thresholds_${tag}.json"
    local bench="phase7_results/results/faithfulness_benchmark_${tag}.json"

    if [[ -f "$bench" ]]; then
      if $PY - <<PY
import json,sys
try:
  _=json.load(open("${bench}"))
except Exception:
  sys.exit(1)
sys.exit(0)
PY
      then
        echo "[${log_prefix}] ${ckpt_id}/${var} already complete -> skip (${bench})"
        continue
      fi
    fi

    echo "[${log_prefix}] ${ckpt_id}/${var} pass1 max-records=1200"
    $PY phase7/causal_intervention_engine.py \
      --model-key gpt2-medium \
      --trace-dataset "$TRACE_TEST" \
      --controls "$controls" \
      --parse-mode hybrid \
      --token-anchor eq_like \
      --anchor-priority "$anchor_priority" \
      --control-sampling-policy stratified_trace_variant \
      --min-controls-used 150 \
      --subspace-specs "$subspaces" \
      --variable "$var" \
      --layers 7 12 17 22 \
      --max-records 1200 \
      --record-buffer-size 256 \
      --device "$device" \
      --state-decoder-checkpoint "$ckpt_path" \
      --enable-latent-mediation \
      --mediation-variable "$var" \
      --output "$causal"

    $PY phase7/causal_audit.py \
      --model-key gpt2-medium \
      --controls "$controls" \
      --trace-dataset "$TRACE_TEST" \
      --state-decoder-checkpoint "$ckpt_path" \
      --causal-checks "$causal" \
      --causal-layer 22 \
      --causal-variable "$var" \
      --latent-source variant_conditioned \
      --control-latent-cache "$cache" \
      --require-mediation-for-causal-pass \
      --device "$device" \
      --output "$audit"

    $PY phase7/split_audit_dataset.py \
      --audit "$audit" \
      --seed 20260303 \
      --calib-fraction 0.30 \
      --output-prefix "$split_prefix"

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
      --require-dual-gate \
      --causal-floor-auroc 0.65 \
      --causal-floor-fpr-max 0.05 \
      --model-comparability-status comparable_full \
      --min-causal-signal-coverage 0.25 \
      --latent-high-quantile 0.80 \
      --comparability-full-threshold 0.60 \
      --comparability-text-threshold 0.01 \
      --comparability-sensitivity 0.50,0.60,0.70 \
      --output "$bench"

    cov=$($PY - <<PY
import json
p=json.load(open("${bench}"))
v=p.get("causal_signal_coverage_fraction")
print(v if isinstance(v,(int,float)) else "nan")
PY
)
    if $PY - <<PY
import math,sys
try: v=float("${cov}")
except Exception: v=float("nan")
sys.exit(0 if (not math.isfinite(v) or v < 0.25) else 1)
PY
    then
      echo "[${log_prefix}] ${ckpt_id}/${var} escalating max-records=2200"
      $PY phase7/causal_intervention_engine.py \
        --model-key gpt2-medium \
        --trace-dataset "$TRACE_TEST" \
        --controls "$controls" \
        --parse-mode hybrid \
        --token-anchor eq_like \
        --anchor-priority "$anchor_priority" \
        --control-sampling-policy stratified_trace_variant \
        --min-controls-used 150 \
        --subspace-specs "$subspaces" \
        --variable "$var" \
        --layers 7 12 17 22 \
        --max-records 2200 \
        --record-buffer-size 256 \
        --device "$device" \
        --state-decoder-checkpoint "$ckpt_path" \
        --enable-latent-mediation \
        --mediation-variable "$var" \
        --output "$causal"

      $PY phase7/causal_audit.py \
        --model-key gpt2-medium \
        --controls "$controls" \
        --trace-dataset "$TRACE_TEST" \
        --state-decoder-checkpoint "$ckpt_path" \
        --causal-checks "$causal" \
        --causal-layer 22 \
        --causal-variable "$var" \
        --latent-source variant_conditioned \
        --control-latent-cache "$cache" \
        --require-mediation-for-causal-pass \
        --device "$device" \
        --output "$audit"

      $PY phase7/split_audit_dataset.py \
        --audit "$audit" \
        --seed 20260303 \
        --calib-fraction 0.30 \
        --output-prefix "$split_prefix"

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
        --require-dual-gate \
        --causal-floor-auroc 0.65 \
        --causal-floor-fpr-max 0.05 \
        --model-comparability-status comparable_full \
        --min-causal-signal-coverage 0.25 \
        --latent-high-quantile 0.80 \
        --comparability-full-threshold 0.60 \
        --comparability-text-threshold 0.01 \
        --comparability-sensitivity 0.50,0.60,0.70 \
        --output "$bench"
    fi
  done
}

run_real_labeled_branch() {
  local run_tag="$1"
  local base="$2"
  local device="$3"
  local log_prefix="$4"
  local ckpt_raw
  ckpt_raw="$(ckpt_path raw_every2_even)"

  local ingest="phase7_results/real_cot/public_benchmark_ingest_${run_tag}.json"
  local controls="phase7_results/real_cot/public_benchmark_controls_${run_tag}.json"
  local audit="phase7_results/audits/text_causal_audit_controls_real_cot_labeled_${run_tag}.json"
  local split_prefix="phase7_results/audits/real_cot_labeled_${run_tag}"
  local calib="${split_prefix}_calib.json"
  local eval="${split_prefix}_eval.json"
  local thr="phase7_results/calibration/phase7_thresholds_real_cot_labeled_${run_tag}.json"
  local bench="phase7_results/results/faithfulness_benchmark_real_cot_labeled_${run_tag}.json"
  local gate="phase7_results/results/faithfulness_benchmark_real_cot_labeled_${run_tag}_external_validity_gate.json"

  echo "[${log_prefix}] real-cot strict external ingest"
  if ! $PY phase7/ingest_public_cot_benchmark.py \
    --source auto \
    --strict-external-labels \
    --output "$ingest" \
    --controls-output "$controls"; then
    echo "[${log_prefix}] strict external labels unavailable; emitting blocked artifact"
    $PY - <<PY
import json,time
bench="${bench}"
gate="${gate}"
payload={
  "schema_version":"phase7_faithfulness_benchmark_v1",
  "benchmark_scope":"real_cot",
  "external_validity_status":"blocked_no_strict_labels",
  "model_comparability_status":"not_comparable",
  "scope_disclaimer":"Real-CoT labeled benchmark blocked: no strict external label source available.",
  "gate_checks":{"applicable":False,"gate_pass":None,"reason":"blocked_no_strict_labels"},
  "generated_at_unix":int(time.time()),
}
with open(bench,"w") as f: json.dump(payload,f,indent=2)
gate_payload={
  "schema_version":"phase7_external_validity_gate_v1",
  "benchmark_scope":"real_cot",
  "external_validity_status":"blocked_no_strict_labels",
  "requires_real_cot_pilot":True,
  "has_real_cot_pilot":False,
  "externally_supported_claims":False,
  "comparability_status":"not_comparable",
  "comparability_gate_pass":False,
  "reason":"Strict external labels unavailable; labeled real-CoT benchmark blocked.",
}
with open(gate,"w") as f: json.dump(gate_payload,f,indent=2)
print(bench)
print(gate)
PY
    echo "$(date -Is)" > "$base/state/real.done"
    return 0
  fi

  labeled_n=$($PY - <<PY
import json
p=json.load(open("${ingest}"))
print(int(p.get("num_rows",0)))
PY
)
  if [[ "$labeled_n" -lt 200 ]]; then
    echo "[${log_prefix}] strict labels present but too small: ${labeled_n} < 200" >&2
    exit 2
  fi

  $PY phase7/causal_audit.py \
    --model-key gpt2-medium \
    --controls "$controls" \
    --trace-dataset "$TRACE_TEST" \
    --state-decoder-checkpoint "$ckpt_raw" \
    --causal-layer 22 \
    --causal-variable subresult_value \
    --latent-source shared \
    --device "$device" \
    --output "$audit"

  $PY phase7/split_audit_dataset.py \
    --audit "$audit" \
    --seed 20260303 \
    --calib-fraction 0.30 \
    --output-prefix "$split_prefix"

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
    --benchmark-scope real_cot \
    --external-validity-status pilot_labeled \
    --gate-track composite \
    --model-comparability-status text_only_comparable \
    --comparability-full-threshold 0.60 \
    --comparability-text-threshold 0.01 \
    --comparability-sensitivity 0.50,0.60,0.70 \
    --output "$bench" \
    --external-validity-gate-output "$gate"

  echo "$(date -Is)" > "$base/state/real.done"
}

worker_synth() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local gpu="$4"
  local role="$5"

  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2

  mkdir -p "$base"/{logs,state,meta}
  local log="$base/logs/worker_synth_gpu${gpu}_${role}.log"

  {
    local ablation_priority
    local ablation_marker
    local log_prefix
    if [[ "$role" == "template" ]]; then
      ablation_priority="template_first"
      ablation_marker="$base/state/ablation_template.done"
      log_prefix="gpu${gpu}:template"
    else
      ablation_priority="equation_first"
      ablation_marker="$base/state/ablation_equation.done"
      log_prefix="gpu${gpu}:equation"
    fi

    local ckpt_raw
    ckpt_raw="$(ckpt_path raw_every2_even)"
    if [[ -f "$ablation_marker" ]]; then
      echo "[${log_prefix}] ablation already complete -> skip (${ablation_marker})"
    else
      echo "[${log_prefix}] ablation start (${ablation_priority})"
      run_checkpoint_bundle "$run_tag" "raw_every2_even" "$ckpt_raw" "$ablation_priority" "ablation_${ablation_priority}" "cuda:0" "$log_prefix"
      echo "$(date -Is)" > "$ablation_marker"
    fi

    echo "[${log_prefix}] waiting selected anchor priority"
    local selected_file="$base/meta/selected_anchor_priority.json"
    while [[ ! -f "$selected_file" ]]; do sleep 20; done
    local selected_priority
    selected_priority=$($PY - <<PY
import json
print(json.load(open("${selected_file}"))["selected_anchor_priority"])
PY
)
    echo "[${log_prefix}] selected priority: ${selected_priority}"

    if [[ "$role" == "template" ]]; then
      run_checkpoint_bundle "$run_tag" "raw_every2_even" "$(ckpt_path raw_every2_even)" "$selected_priority" "matrix" "cuda:0" "$log_prefix"
      run_checkpoint_bundle "$run_tag" "hybrid_middle12_06_17" "$(ckpt_path hybrid_middle12_06_17)" "$selected_priority" "matrix" "cuda:0" "$log_prefix"
      echo "$(date -Is)" > "$base/state/gpu6.matrix.done"
    else
      run_checkpoint_bundle "$run_tag" "hybrid_every2_even" "$(ckpt_path hybrid_every2_even)" "$selected_priority" "matrix" "cuda:0" "$log_prefix"
      echo "$(date -Is)" > "$base/state/gpu7.first_ckpt.done"

      run_real_labeled_branch "$run_tag" "$base" "cuda:0" "$log_prefix"

      run_checkpoint_bundle "$run_tag" "raw_middle12_06_17" "$(ckpt_path raw_middle12_06_17)" "$selected_priority" "matrix" "cuda:0" "$log_prefix"
      echo "$(date -Is)" > "$base/state/gpu7.matrix.done"
    fi
  } >>"$log" 2>&1
}

coordinator() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  mkdir -p "$base"/{logs,state,meta}
  local log="$base/logs/coordinator.log"

  {
    echo "[coord] wait ablation markers"
    while [[ ! -f "$base/state/ablation_template.done" ]]; do sleep 20; done
    while [[ ! -f "$base/state/ablation_equation.done" ]]; do sleep 20; done

    $PY - <<PY
import glob,json,math
run_tag="${run_tag}"
out=f"phase7_results/results/anchor_priority_ablation_{run_tag}.json"
meta_out="${base}/meta/selected_anchor_priority.json"
priorities=["template_first","equation_first"]
vars=["subresult_value","operator","magnitude_bucket","sign"]
rows=[]
for pr in priorities:
  lat=[]; comp=[]; cov=[]
  per_var=[]
  for v in vars:
    p=f"phase7_results/results/faithfulness_benchmark_{run_tag}_ablation_{pr}_raw_every2_even_{v}.json"
    if not glob.glob(p):
      continue
    d=json.load(open(p))
    bt=d.get("by_benchmark_track") or {}
    l=(bt.get("latent_only") or {}).get("auroc")
    c=(bt.get("composite") or {}).get("auroc")
    k=d.get("causal_signal_coverage_fraction")
    if isinstance(l,(int,float)): lat.append(float(l))
    if isinstance(c,(int,float)): comp.append(float(c))
    if isinstance(k,(int,float)): cov.append(float(k))
    per_var.append({
      "variable":v,
      "path":p,
      "latent_auroc":l,
      "composite_auroc":c,
      "coverage":k,
    })
  rows.append({
    "anchor_priority":pr,
    "mean_latent_auroc":(sum(lat)/len(lat) if lat else None),
    "mean_composite_auroc":(sum(comp)/len(comp) if comp else None),
    "mean_coverage":(sum(cov)/len(cov) if cov else None),
    "num_variables":len(per_var),
    "per_variable":per_var,
  })

def score(row):
  m1=row.get("mean_latent_auroc")
  m2=row.get("mean_composite_auroc")
  m3=row.get("mean_coverage")
  return (
    float(m1) if isinstance(m1,(int,float)) else -1.0,
    float(m2) if isinstance(m2,(int,float)) else -1.0,
    float(m3) if isinstance(m3,(int,float)) else -1.0,
  )

best=sorted(rows,key=score,reverse=True)[0] if rows else {"anchor_priority":"template_first"}
selected=str(best.get("anchor_priority","template_first"))
payload={
  "schema_version":"phase7_anchor_priority_ablation_v1",
  "run_tag":run_tag,
  "selection_rule":"max(mean_latent_auroc), tie=max(mean_composite_auroc), tie=max(mean_coverage)",
  "rows":rows,
  "selected_anchor_priority":selected,
}
json.dump(payload,open(out,"w"),indent=2)
json.dump({
  "schema_version":"phase7_anchor_priority_selection_v1",
  "selected_anchor_priority":selected,
},open(meta_out,"w"),indent=2)
print(out)
print(meta_out)
PY

    echo "[coord] wait matrix workers + real branch"
    while [[ ! -f "$base/state/gpu6.matrix.done" ]]; do sleep 20; done
    while [[ ! -f "$base/state/gpu7.matrix.done" ]]; do sleep 20; done
    while [[ ! -f "$base/state/real.done" ]]; do sleep 20; done

    $PY - <<PY
import glob,json,re,time
run_tag="${run_tag}"
matrix_out=f"phase7_results/results/trackA_gate_matrix_{run_tag}.json"
decision_out=f"phase7_results/results/trackA_gate_decision_{run_tag}.json"
closure_out=f"phase7_results/results/academic_weakness_closure_report_{run_tag}.json"
real_path=f"phase7_results/results/faithfulness_benchmark_real_cot_labeled_{run_tag}.json"

bench_files=sorted(glob.glob(f"phase7_results/results/faithfulness_benchmark_{run_tag}_matrix_*.json"))
rows=[]
for p in bench_files:
  d=json.load(open(p))
  bt=d.get("by_benchmark_track") or {}
  m=re.search(rf"faithfulness_benchmark_{re.escape(run_tag)}_matrix_(.+)_(subresult_value|operator|magnitude_bucket|sign)\\.json$", p)
  ckpt=m.group(1) if m else "unknown"
  var=m.group(2) if m else "unknown"
  rows.append({
    "checkpoint_id":ckpt,
    "variable":var,
    "benchmark_path":p,
    "composite_auroc":(bt.get("composite") or {}).get("auroc"),
    "text_auroc":(bt.get("text_only") or {}).get("auroc"),
    "latent_auroc":(bt.get("latent_only") or {}).get("auroc"),
    "causal_auroc":(bt.get("causal_auditor") or {}).get("auroc"),
    "coverage":d.get("causal_signal_coverage_fraction"),
    "dual_gate_pass":((d.get("gate_checks") or {}).get("dual_gate_pass")),
    "leakage_check_pass":d.get("leakage_check_pass"),
  })

json.dump({
  "schema_version":"phase7_trackA_gate_matrix_v1",
  "run_tag":run_tag,
  "num_rows":len(rows),
  "rows":rows,
},open(matrix_out,"w"),indent=2)

comp=[r["composite_auroc"] for r in rows if isinstance(r.get("composite_auroc"),(int,float))]
caus=[r["causal_auroc"] for r in rows if isinstance(r.get("causal_auroc"),(int,float))]
cov=[r["coverage"] for r in rows if isinstance(r.get("coverage"),(int,float))]
json.dump({
  "schema_version":"phase7_trackA_gate_decision_v1",
  "run_tag":run_tag,
  "composite_auroc_max":(max(comp) if comp else None),
  "causal_auroc_max":(max(caus) if caus else None),
  "coverage_max":(max(cov) if cov else None),
  "num_rows":len(rows),
  "any_dual_gate_pass":any(bool(r.get("dual_gate_pass")) for r in rows),
},open(decision_out,"w"),indent=2)

synthetic_gap={"status":"unavailable"}
if rows and glob.glob(real_path):
  real=json.load(open(real_path))
  rbt=real.get("by_benchmark_track") or {}
  if rbt:
    def mean_track(k):
      vals=[r.get(k) for r in rows if isinstance(r.get(k),(int,float))]
      return (sum(vals)/len(vals)) if vals else None
    gaps={}
    for track,key in [("text_only","text_auroc"),("latent_only","latent_auroc"),("causal_auditor","causal_auroc"),("composite","composite_auroc")]:
      synth=mean_track(key)
      real_auc=(rbt.get(track) or {}).get("auroc")
      if isinstance(synth,(int,float)) and isinstance(real_auc,(int,float)):
        gaps[f"{track}_auroc_gap"]=float(real_auc)-float(synth)
      else:
        gaps[f"{track}_auroc_gap"]=None
    synthetic_gap={"status":"computed","gaps":gaps}
  else:
    synthetic_gap={"status":"real_track_metrics_missing"}

real_status="missing"
if glob.glob(real_path):
  real_status=(json.load(open(real_path)).get("external_validity_status") or "unknown")

json.dump({
  "schema_version":"phase7_academic_weakness_closure_report_v1",
  "run_tag":run_tag,
  "synthetic_gate_matrix_path":matrix_out,
  "synthetic_gate_decision_path":decision_out,
  "real_cot_labeled_benchmark_path":real_path,
  "synthetic_to_real_gap":synthetic_gap,
  "external_validity_status":real_status,
  "resolved_items":[
    "position_contract_hardened",
    "special_token_policy_explicit",
    "anchor_priority_ablation_executed",
    "strict_external_label_policy_hardened",
  ],
  "remaining_items":[
    "qwen_full_comparability_out_of_scope",
    "causal_track_signal_still_empirical",
  ],
  "generated_at_unix":int(time.time()),
},open(closure_out,"w"),indent=2)

print(matrix_out)
print(decision_out)
print(closure_out)
PY

    echo "$(date -Is)" > "$base/state/pipeline.done"
    echo "[coord] pipeline.done"
  } >>"$log" 2>&1
}

launch() {
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)_phase7_r2p1_full"
  local run_tag="phase7_r2p1_${run_id}"
  local base="phase7_results/runs/${run_id}"

  mkdir -p "$base"/{logs,state,meta,scripts}

  freeze_baseline_refs "$run_tag"
  write_closure_contract "$run_tag"

  echo "$run_id" > "$base/meta/run_id.txt"
  echo "$run_tag" > "$base/meta/run_tag.txt"
  echo "$base" > "$base/meta/base_path.txt"

  echo "[launch] preflight gpu status"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

  prepare_shared_inputs "$run_tag"

  tmux new-session -d -s "p7r2_g6_${run_id}" "cd \"$PWD\" && bash experiments/run_phase7_r2_full.sh worker-synth \"$run_id\" \"$run_tag\" \"$base\" 6 template"
  tmux new-session -d -s "p7r2_g7_${run_id}" "cd \"$PWD\" && bash experiments/run_phase7_r2_full.sh worker-synth \"$run_id\" \"$run_tag\" \"$base\" 7 equation"
  tmux new-session -d -s "p7r2_coord_${run_id}" "cd \"$PWD\" && bash experiments/run_phase7_r2_full.sh coordinator \"$run_id\" \"$run_tag\" \"$base\""

  echo "RUN_ID=$run_id"
  echo "RUN_TAG=$run_tag"
  echo "BASE=$base"
  echo "Sessions:"
  echo "  p7r2_g6_${run_id}"
  echo "  p7r2_g7_${run_id}"
  echo "  p7r2_coord_${run_id}"
}

cmd="${1:-}"
case "$cmd" in
  launch) launch ;;
  worker-synth) worker_synth "$2" "$3" "$4" "$5" "$6" ;;
  coordinator) coordinator "$2" "$3" "$4" ;;
  *) usage; exit 1 ;;
esac

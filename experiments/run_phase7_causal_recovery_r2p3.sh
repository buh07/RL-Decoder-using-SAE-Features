#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  experiments/run_phase7_causal_recovery_r2p3.sh launch
  experiments/run_phase7_causal_recovery_r2p3.sh canary-worker <RUN_ID> <RUN_TAG> <BASE> <GPU>
  experiments/run_phase7_causal_recovery_r2p3.sh promote-worker <RUN_ID> <RUN_TAG> <BASE> <GPU> <GROUP>
  experiments/run_phase7_causal_recovery_r2p3.sh coordinator <RUN_ID> <RUN_TAG> <BASE>
USAGE
}

PY=".venv/bin/python3"
TRACE_TEST="phase7_results/dataset/gsm8k_step_traces_test.pt"
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

prepare_shared_inputs() {
  local run_tag="$1"
  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local parsed="phase7_results/controls/cot_controls_${run_tag}_parsed.json"
  local subspaces="phase7_results/interventions/variable_subspaces_${run_tag}.json"

  $PY phase7/generate_cot_controls.py \
    --trace-dataset "$TRACE_TEST" \
    --max-traces 500 \
    --seed 20260305 \
    --output "$controls"

  $PY phase7/parse_cot_to_states.py \
    --controls "$controls" \
    --trace-dataset "$TRACE_TEST" \
    --parse-mode hybrid \
    --output "$parsed"

  $PY phase7/variable_subspace_builder.py \
    --model-key gpt2-medium \
    --layers 7 12 17 22 \
    --probe-position result \
    --top-k 64 \
    --combine-policy union \
    --output "$subspaces"
}

run_chain_for_var() {
  local run_tag="$1"
  local ckpt_id="$2"
  local ckpt_path="$3"
  local mode_tag="$4"
  local var="$5"
  local device="$6"
  local max_records="$7"
  local anchor_priority="template_first"

  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local subspaces="phase7_results/interventions/variable_subspaces_${run_tag}.json"
  local cache="phase7_results/interventions/control_latent_cache_${run_tag}_${mode_tag}_${ckpt_id}.json"
  local tag="${run_tag}_${mode_tag}_${ckpt_id}_${var}"
  local causal="phase7_results/interventions/causal_checks_${tag}.json"
  local audit="phase7_results/audits/text_causal_audit_controls_${tag}.json"
  local split_prefix="phase7_results/audits/${tag}"
  local calib="${split_prefix}_calib.json"
  local eval="${split_prefix}_eval.json"
  local thr="phase7_results/calibration/phase7_thresholds_${tag}.json"
  local bench="phase7_results/results/faithfulness_benchmark_${tag}.json"

  $PY phase7/causal_intervention_engine.py \
    --model-key gpt2-medium \
    --trace-dataset "$TRACE_TEST" \
    --controls "$controls" \
    --parse-mode hybrid \
    --token-anchor eq_like \
    --anchor-priority "$anchor_priority" \
    --control-sampling-policy stratified_trace_variant \
    --min-controls-used 150 \
    --target-controls-used-fraction 0.35 \
    --max-records-cap 12000 \
    --subspace-specs "$subspaces" \
    --variable "$var" \
    --layers 7 12 17 22 \
    --max-records "$max_records" \
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
    --require-causal-mode control_conditioned \
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
    --min-causal-signal-coverage 0.35 \
    --latent-high-quantile 0.80 \
    --comparability-full-threshold 0.60 \
    --comparability-text-threshold 0.01 \
    --comparability-sensitivity 0.50,0.60,0.70 \
    --causal-degenerate-identical-threshold 0.80 \
    --causal-degenerate-auroc-threshold 0.55 \
    --causal-degenerate-enable-auroc-trigger \
    --ablation-weights '{"text":0.5,"latent":0.5,"causal":0.0}' \
    --output "$bench"

  echo "$bench"
}

run_ckpt_group() {
  local run_tag="$1"
  local device="$2"
  local mode_tag="$3"
  shift 3
  local ckpts=("$@")

  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  for ckpt_id in "${ckpts[@]}"; do
    local ckpt
    ckpt="$(ckpt_path "$ckpt_id")"
    local cache="phase7_results/interventions/control_latent_cache_${run_tag}_${mode_tag}_${ckpt_id}.json"

    if [[ ! -f "$cache" ]]; then
      $PY phase7/build_control_latent_cache.py \
        --controls "$controls" \
        --state-decoder-checkpoint "$ckpt" \
        --model-key gpt2-medium \
        --parse-mode hybrid \
        --token-anchor eq_like \
        --anchor-priority template_first \
        --device "$device" \
        --batch-size 128 \
        --output "$cache"
    fi

    for var in subresult_value operator magnitude_bucket sign; do
      local tag="${run_tag}_${mode_tag}_${ckpt_id}_${var}"
      local bench="phase7_results/results/faithfulness_benchmark_${tag}.json"
      local causal="phase7_results/interventions/causal_checks_${tag}.json"

      if [[ -f "$bench" ]]; then
        if $PY - <<PY
import json,sys
try: json.load(open("$bench"))
except Exception: sys.exit(1)
sys.exit(0)
PY
        then
          echo "[${mode_tag}] ${ckpt_id}/${var} complete -> skip"
          continue
        fi
      fi

      run_chain_for_var "$run_tag" "$ckpt_id" "$ckpt" "$mode_tag" "$var" "$device" 1200 >/dev/null
      for stage in 2200 6000 12000; do
        cov=$($PY - <<PY
import json
b=json.load(open("$bench"))
print(b.get("causal_signal_coverage_fraction", "nan"))
PY
)
        cfrac=$($PY - <<PY
import json
c=json.load(open("$causal"))
print(((c.get("control_conditioned_stats") or {}).get("controls_used_fraction", "nan")))
PY
)
        if $PY - <<PY
import math,sys
try: cov=float("$cov")
except Exception: cov=float("nan")
try: cf=float("$cfrac")
except Exception: cf=float("nan")
need = (not math.isfinite(cov) or cov < 0.35) or (not math.isfinite(cf) or cf < 0.35)
sys.exit(0 if need else 1)
PY
        then
          run_chain_for_var "$run_tag" "$ckpt_id" "$ckpt" "$mode_tag" "$var" "$device" "$stage" >/dev/null
        else
          break
        fi
      done
    done
  done
}

canary_worker() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local gpu="$4"
  local log="$base/logs/canary_gpu${gpu}.log"

  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2

  {
    echo "[$(date -Is)] canary worker start gpu=$gpu"
    run_ckpt_group "$run_tag" "cuda:0" "canary" raw_every2_even

    $PY - <<PY
import json
run_tag="${run_tag}"
rows=[]
for var in ["subresult_value","operator","magnitude_bucket","sign"]:
  p=f"phase7_results/results/faithfulness_benchmark_{run_tag}_canary_raw_every2_even_{var}.json"
  d=json.load(open(p))
  bt=d.get("by_benchmark_track") or {}
  rows.append({
    "variable":var,
    "path":p,
    "causal_variant_score_identical_fraction":d.get("causal_variant_score_identical_fraction"),
    "causal_auditor_auroc":(bt.get("causal_auditor") or {}).get("auroc"),
    "causal_signal_coverage_fraction":d.get("causal_signal_coverage_fraction"),
    "leakage_check_pass":d.get("leakage_check_pass"),
    "causal_anti_predictive_flag":d.get("causal_anti_predictive_flag"),
    "causal_harms_composite_flag":d.get("causal_harms_composite_flag"),
    "track_c_unresolved_high_coverage":d.get("track_c_unresolved_high_coverage"),
  })
ident=[r["causal_variant_score_identical_fraction"] for r in rows if isinstance(r.get("causal_variant_score_identical_fraction"),(int,float))]
causal=[r["causal_auditor_auroc"] for r in rows if isinstance(r.get("causal_auditor_auroc"),(int,float))]
cov=[r["causal_signal_coverage_fraction"] for r in rows if isinstance(r.get("causal_signal_coverage_fraction"),(int,float))]
leak=all(bool(r.get("leakage_check_pass")) for r in rows)
anti_all=all(bool(r.get("causal_anti_predictive_flag")) for r in rows)
harm_all=all(bool(r.get("causal_harms_composite_flag")) for r in rows)
pass_ident=(len(ident)>0 and min(ident) <= 0.90)
pass_causal=(len(causal)>0 and max(causal) > 0.55)
pass_cov=(len(cov)>0 and min(cov) >= 0.35)
unresolved_all=all(bool(r.get("track_c_unresolved_high_coverage")) for r in rows)
canary_pass=bool(pass_ident and pass_causal and pass_cov and leak and (not anti_all) and (not harm_all) and (not unresolved_all))
out={
  "schema_version":"phase7_causal_canary_decision_v2",
  "run_tag":run_tag,
  "rows":rows,
  "criteria":{
    "causal_variant_score_identical_fraction_max":0.90,
    "causal_auditor_auroc_min":0.55,
    "causal_signal_coverage_fraction_min":0.35,
    "leakage_check_pass":True,
    "not_all_causal_anti_predictive":True,
    "not_all_causal_harms_composite":True,
    "not_all_track_c_unresolved_high_coverage":True,
  },
  "checks":{
    "identical_fraction_pass":pass_ident,
    "causal_auroc_pass":pass_causal,
    "coverage_pass":pass_cov,
    "leakage_pass":leak,
    "not_all_causal_anti_predictive":(not anti_all),
    "not_all_causal_harms_composite":(not harm_all),
    "not_all_track_c_unresolved_high_coverage":(not unresolved_all),
  },
  "canary_pass":canary_pass,
}
out_path=f"phase7_results/results/trackA_canary_decision_{run_tag}.json"
json.dump(out,open(out_path,"w"),indent=2)
print(out_path)
PY

    echo "$(date -Is)" > "$base/state/canary.done"
    echo "[$(date -Is)] canary worker done"
  } >> "$log" 2>&1
}

promote_worker() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local gpu="$4"
  local group="$5"
  local log="$base/logs/promote_gpu${gpu}_${group}.log"

  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2

  {
    echo "[$(date -Is)] promote worker start gpu=$gpu group=$group"
    if [[ "$group" == "raw" ]]; then
      run_ckpt_group "$run_tag" "cuda:0" "matrix" raw_every2_even raw_middle12_06_17
      echo "$(date -Is)" > "$base/state/gpu6.matrix.done"
    else
      run_ckpt_group "$run_tag" "cuda:0" "matrix" hybrid_every2_even hybrid_middle12_06_17
      echo "$(date -Is)" > "$base/state/gpu7.matrix.done"
    fi
    echo "[$(date -Is)] promote worker done gpu=$gpu group=$group"
  } >> "$log" 2>&1
}

coordinator() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local log="$base/logs/coordinator.log"

  {
    echo "[$(date -Is)] coordinator start"
    while [[ ! -f "$base/state/canary.done" ]]; do sleep 20; done

    local decision="phase7_results/results/trackA_canary_decision_${run_tag}.json"
    local canary_pass
    canary_pass=$($PY - <<PY
import json
print(str(bool(json.load(open("$decision")).get("canary_pass"))).lower())
PY
)

    if [[ "$canary_pass" == "true" ]]; then
      tmux new-session -d -s "p7r23_promote_g6_${run_id}" "cd '$PWD' && bash experiments/run_phase7_causal_recovery_r2p3.sh promote-worker '$run_id' '$run_tag' '$base' 6 raw"
      tmux new-session -d -s "p7r23_promote_g7_${run_id}" "cd '$PWD' && bash experiments/run_phase7_causal_recovery_r2p3.sh promote-worker '$run_id' '$run_tag' '$base' 7 hybrid"
      while [[ ! -f "$base/state/gpu6.matrix.done" ]]; do sleep 30; done
      while [[ ! -f "$base/state/gpu7.matrix.done" ]]; do sleep 30; done
    else
      echo "[$(date -Is)] canary failed; promotion skipped"
      echo "$(date -Is)" > "$base/state/gpu6.skipped"
      echo "$(date -Is)" > "$base/state/gpu7.skipped"
    fi

    $PY - <<PY
import glob,json,re,time,statistics
run_tag="${run_tag}"
matrix_out=f"phase7_results/results/trackA_gate_matrix_{run_tag}.json"
decision_out=f"phase7_results/results/trackA_gate_decision_{run_tag}.json"
diag_out=f"phase7_results/results/causal_track_diagnostic_{run_tag}.json"
closure_out=f"phase7_results/results/academic_weakness_closure_report_{run_tag}.json"

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
        "path":p,
        "composite_auroc":(bt.get("composite") or {}).get("auroc"),
        "text_auroc":(bt.get("text_only") or {}).get("auroc"),
        "latent_auroc":(bt.get("latent_only") or {}).get("auroc"),
        "causal_auroc":(bt.get("causal_auditor") or {}).get("auroc"),
        "coverage":d.get("causal_signal_coverage_fraction"),
        "dual_gate_pass":((d.get("gate_checks") or {}).get("dual_gate_pass")),
        "leakage_check_pass":d.get("leakage_check_pass"),
        "causal_variant_score_identical_fraction":d.get("causal_variant_score_identical_fraction"),
        "causal_track_degenerate_flag":d.get("causal_track_degenerate_flag"),
        "causal_anti_predictive_flag":d.get("causal_anti_predictive_flag"),
        "causal_harms_composite_flag":d.get("causal_harms_composite_flag"),
    })

json.dump({"schema_version":"phase7_trackA_gate_matrix_v1","run_tag":run_tag,"num_rows":len(rows),"rows":rows},open(matrix_out,"w"),indent=2)

def vals(k):
    return [r[k] for r in rows if isinstance(r.get(k),(int,float))]
comp=vals("composite_auroc")
caus=vals("causal_auroc")
cov=vals("coverage")
ident=vals("causal_variant_score_identical_fraction")
json.dump({
    "schema_version":"phase7_trackA_gate_decision_v1",
    "run_tag":run_tag,
    "num_rows":len(rows),
    "composite_auroc_max":max(comp) if comp else None,
    "causal_auroc_max":max(caus) if caus else None,
    "coverage_min":min(cov) if cov else None,
    "identical_fraction_min":min(ident) if ident else None,
    "any_dual_gate_pass":any(bool(r.get("dual_gate_pass")) for r in rows),
},open(decision_out,"w"),indent=2)
json.dump({
    "schema_version":"phase7_causal_track_diagnostic_v1",
    "run_tag":run_tag,
    "num_rows":len(rows),
    "mean_causal_auroc":statistics.mean(caus) if caus else None,
    "mean_causal_coverage":statistics.mean(cov) if cov else None,
    "mean_identical_fraction":statistics.mean(ident) if ident else None,
    "degenerate_rows":sum(1 for r in rows if r.get("causal_track_degenerate_flag") is True),
    "anti_predictive_rows":sum(1 for r in rows if r.get("causal_anti_predictive_flag") is True),
},open(diag_out,"w"),indent=2)
json.dump({
    "schema_version":"phase7_academic_weakness_closure_report_v1",
    "run_tag":run_tag,
    "canary_decision_path":f"phase7_results/results/trackA_canary_decision_{run_tag}.json",
    "matrix_path":matrix_out,
    "decision_path":decision_out,
    "causal_track_diagnostic_path":diag_out,
    "resolved_items":["strict_variant_lookup","stratified_control_sampling","causal_degeneracy_diagnostics","mode_contract_enforced"],
    "generated_at_unix":int(time.time()),
},open(closure_out,"w"),indent=2)
print(matrix_out)
print(decision_out)
print(diag_out)
print(closure_out)
PY

    echo "$(date -Is)" > "$base/state/pipeline.done"
    echo "[$(date -Is)] coordinator done"
  } >> "$log" 2>&1
}

launch() {
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)_phase7_causal_recovery_r2p3"
  local run_tag="phase7_causal_recovery_r2p3_${run_id}"
  local base="phase7_results/runs/${run_id}"

  mkdir -p "$base"/{logs,state,meta}

  $PY - <<PY
import glob,json,time
run_tag="${run_tag}"
baseline_out=f"phase7_results/results/{run_tag}_baseline_refs.json"
contract_out=f"phase7_results/results/{run_tag}_closure_contract.json"
refs={
  "schema_version":"phase7_r2p3_baseline_refs_v1",
  "run_tag":run_tag,
  "latest_trackA_gate_matrix":sorted(glob.glob("phase7_results/results/trackA_gate_matrix*.json"))[-1] if glob.glob("phase7_results/results/trackA_gate_matrix*.json") else None,
  "latest_canary_decision":sorted(glob.glob("phase7_results/results/trackA_canary_decision*.json"))[-1] if glob.glob("phase7_results/results/trackA_canary_decision*.json") else None,
  "generated_at_unix":int(time.time()),
}
contract={
  "schema_version":"phase7_r2p3_closure_contract_v1",
  "run_tag":run_tag,
  "dual_gate_unchanged":True,
  "strict_external_labels_required":True,
  "run_scope":"four_checkpoint_matrix",
  "hardware_target":["gpu6","gpu7"],
  "causal_degenerate_identical_threshold":0.80,
  "causal_degenerate_auroc_threshold":0.55,
  "target_causal_signal_coverage_fraction":0.35,
  "generated_at_unix":int(time.time()),
}
json.dump(refs,open(baseline_out,"w"),indent=2)
json.dump(contract,open(contract_out,"w"),indent=2)
print(baseline_out)
print(contract_out)
PY

  prepare_shared_inputs "$run_tag"

  tmux new-session -d -s "p7r23_canary_g6_${run_id}" "cd '$PWD' && bash experiments/run_phase7_causal_recovery_r2p3.sh canary-worker '$run_id' '$run_tag' '$base' 6"
  tmux new-session -d -s "p7r23_coord_${run_id}" "cd '$PWD' && bash experiments/run_phase7_causal_recovery_r2p3.sh coordinator '$run_id' '$run_tag' '$base'"

  echo "RUN_ID=$run_id"
  echo "RUN_TAG=$run_tag"
  echo "BASE=$base"
  echo "Sessions:"
  echo "  p7r23_canary_g6_${run_id}"
  echo "  p7r23_coord_${run_id}"
}

cmd="${1:-}"
case "$cmd" in
  launch) launch ;;
  canary-worker) canary_worker "$2" "$3" "$4" "$5" ;;
  promote-worker) promote_worker "$2" "$3" "$4" "$5" "$6" ;;
  coordinator) coordinator "$2" "$3" "$4" ;;
  *) usage; exit 1 ;;
esac

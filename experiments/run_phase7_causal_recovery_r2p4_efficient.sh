#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  experiments/run_phase7_causal_recovery_r2p4_efficient.sh launch
  experiments/run_phase7_causal_recovery_r2p4_efficient.sh canary-worker <RUN_ID> <RUN_TAG> <BASE> <GPU> <GROUP>
  experiments/run_phase7_causal_recovery_r2p4_efficient.sh promote-worker <RUN_ID> <RUN_TAG> <BASE> <GPU> <GROUP>
  experiments/run_phase7_causal_recovery_r2p4_efficient.sh coordinator <RUN_ID> <RUN_TAG> <BASE>
USAGE
}

PY=".venv/bin/python3"
TRACE_TEST="phase7_results/dataset/gsm8k_step_traces_test.pt"
CKPT_DIR="phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration/checkpoints"
LAYERS=(7 12 17 22)
ALL_VARIABLES=(subresult_value operator magnitude_bucket sign)
MAX_RECORDS_STAGES=(1200 2200 6000 12000)

SEED=20260305
SPLIT_SEED=20260303
CALIB_FRACTION=0.30
TARGET_COVERAGE="${TARGET_COVERAGE:-0.35}"
TARGET_CONTROLS_USED="${TARGET_CONTROLS_USED:-0.35}"
MAX_RECORDS_CAP="${MAX_RECORDS_CAP:-12000}"
POSTPROC_JOBS="${POSTPROC_JOBS:-2}"
RUNNER_VERSION="phase7_r2p4_efficient_v1"
ANCHOR_PRIORITY="${ANCHOR_PRIORITY:-template_first}"

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

json_ok() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  "$PY" - <<PY
import json,sys
p="$path"
try:
    json.load(open(p))
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

sha256_of() {
  local path="$1"
  "$PY" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$PWD")
from phase7.common import sha256_file
p=Path("$path")
print(sha256_file(p) if p.exists() else "")
PY
}

split_policy_hash() {
  "$PY" - <<PY
import hashlib
seed=int("$SPLIT_SEED")
frac=float("$CALIB_FRACTION")
h=hashlib.sha256()
h.update(f"seed={seed}|calib_fraction={frac:.12f}|group_by=trace_id".encode("utf-8"))
print(h.hexdigest())
PY
}

prepare_shared_inputs() {
  local run_tag="$1"
  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local parsed="phase7_results/controls/cot_controls_${run_tag}_parsed.json"
  local subspaces="phase7_results/interventions/variable_subspaces_${run_tag}.json"

  "$PY" phase7/generate_cot_controls.py \
    --trace-dataset "$TRACE_TEST" \
    --max-traces 500 \
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

ensure_compat_manifest() {
  local base="$1"
  local mode_tag="$2"
  local ckpt_id="$3"
  local variable="$4"
  local controls="$5"
  local trace_dataset="$6"
  local subspaces="$7"
  local checkpoint="$8"

  local manifest="$base/meta/compat_${mode_tag}_${ckpt_id}_${variable}.json"
  "$PY" - <<PY
import json,sys
from pathlib import Path
sys.path.insert(0, "$PWD")
from phase7.common import sha256_file

manifest_path=Path("$manifest")
current={
  "schema_version":"phase7_runtime_compat_manifest_v1",
  "runner_version":"$RUNNER_VERSION",
  "controls_sha256": sha256_file("$controls"),
  "trace_dataset_sha256": sha256_file("$trace_dataset"),
  "subspace_specs_sha256": sha256_file("$subspaces"),
  "checkpoint_sha256": sha256_file("$checkpoint"),
  "token_anchor":"eq_like",
  "anchor_priority":"$ANCHOR_PRIORITY",
  "parse_mode":"hybrid",
  "layers":[7,12,17,22],
  "variable":"$variable",
  "sampling_policy":"stratified_trace_variant",
  "seed":int("$SEED"),
  "max_records_cap":int("$MAX_RECORDS_CAP"),
}
if manifest_path.exists():
    prior=json.loads(manifest_path.read_text())
    mismatches={k:(prior.get(k), current.get(k)) for k in sorted(set(prior)|set(current)) if prior.get(k)!=current.get(k)}
    if mismatches:
        msg={"error":"compatibility_manifest_mismatch","path":str(manifest_path),"mismatches":mismatches}
        print(json.dumps(msg, indent=2))
        raise SystemExit(2)
else:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(current, indent=2))
PY
}

benchmark_hash_compatible() {
  local bench="$1"
  local audit="$2"
  local thresholds="$3"
  local expected_causal_rows_sha="$4"
  local expected_split_policy_hash="$5"
  [[ -f "$bench" && -f "$audit" && -f "$thresholds" ]] || return 1
  "$PY" - <<PY
import json,sys
from pathlib import Path
sys.path.insert(0, "$PWD")
from phase7.common import sha256_file
bench_path=Path("$bench")
audit_path=Path("$audit")
thr_path=Path("$thresholds")
expected_causal="${expected_causal_rows_sha}"
expected_split_policy="${expected_split_policy_hash}"

try:
    b=json.loads(bench_path.read_text())
except Exception:
    sys.exit(1)
up=(b.get("upstream_hashes") or {})
ok=(
  bool(b.get("leakage_check_pass")) and
  str(b.get("source_audit")) == str(audit_path) and
  str(up.get("audit_file_sha256")) == sha256_file(audit_path) and
  str(up.get("thresholds_file_sha256")) == sha256_file(thr_path)
)
if expected_causal:
    ok = ok and str(up.get("source_audit_rows_sha256")) == str(expected_causal)
if expected_split_policy:
    ok = ok and str(up.get("split_policy_hash")) == str(expected_split_policy)
sys.exit(0 if ok else 1)
PY
}

controls_used_fraction_from_causal() {
  local causal="$1"
  "$PY" - <<PY
import json
p="$causal"
d=json.load(open(p))
print(((d.get("control_conditioned_stats") or {}).get("controls_used_fraction", "nan")))
PY
}

rows_added_from_causal() {
  local causal="$1"
  "$PY" - <<PY
import json
p="$causal"
d=json.load(open(p))
print(int(((d.get("execution_telemetry") or {}).get("new_rows_added", 0) or 0)))
PY
}

coverage_from_benchmark() {
  local bench="$1"
  "$PY" - <<PY
import json
b=json.load(open("$bench"))
print(b.get("causal_signal_coverage_fraction", "nan"))
PY
}

run_causal_stage_for_vars() {
  local run_tag="$1"
  local ckpt_id="$2"
  local checkpoint="$3"
  local mode_tag="$4"
  local device="$5"
  local max_records="$6"
  shift 6
  local vars=("$@")

  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local subspaces="phase7_results/interventions/variable_subspaces_${run_tag}.json"
  local control_records="phase7_results/interventions/control_records_${run_tag}_${mode_tag}_${ckpt_id}.json"
  local output_template="phase7_results/interventions/causal_checks_${run_tag}_${mode_tag}_${ckpt_id}_{variable}.json"

  local cmd=(
    "$PY" phase7/causal_intervention_engine.py
    --model-key gpt2-medium
    --trace-dataset "$TRACE_TEST"
    --controls "$controls"
    --parse-mode hybrid
    --token-anchor eq_like
    --anchor-priority "$ANCHOR_PRIORITY"
    --control-sampling-policy stratified_trace_variant
    --min-controls-used 150
    --target-controls-used-fraction "$TARGET_CONTROLS_USED"
    --max-records-cap "$MAX_RECORDS_CAP"
    --subspace-specs "$subspaces"
    --variables "${vars[@]}"
    --output-template "$output_template"
    --layers "${LAYERS[@]}"
    --max-records "$max_records"
    --record-buffer-size 256
    --device "$device"
    --state-decoder-checkpoint "$checkpoint"
    --enable-latent-mediation
    --mediation-variable subresult_value
    --control-records-output "$control_records"
    --resume-output
    --rows-format jsonl.gz
    --seed "$SEED"
  )
  if [[ -f "$control_records" ]]; then
    cmd+=(--control-records-input "$control_records")
  fi
  "${cmd[@]}"
}

run_postproc_var() {
  local run_tag="$1"
  local ckpt_id="$2"
  local checkpoint="$3"
  local mode_tag="$4"
  local variable="$5"
  local expected_split_policy_hash="$6"

  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local tag="${run_tag}_${mode_tag}_${ckpt_id}_${variable}"
  local causal="phase7_results/interventions/causal_checks_${tag}.json"
  local cache="phase7_results/interventions/control_latent_cache_${run_tag}_${mode_tag}_${ckpt_id}.json"
  local audit="phase7_results/audits/text_causal_audit_controls_${tag}.json"
  local split_prefix="phase7_results/audits/${tag}"
  local calib="${split_prefix}_calib.json"
  local eval="${split_prefix}_eval.json"
  local manifest="${split_prefix}_split_manifest.json"
  local thr="phase7_results/calibration/phase7_thresholds_${tag}.json"
  local bench="phase7_results/results/faithfulness_benchmark_${tag}.json"

  local causal_rows_sha
  causal_rows_sha="$($PY - <<PY
import json
p="$causal"
d=json.load(open(p))
print(d.get("rows_sha256", ""))
PY
)"

  if benchmark_hash_compatible "$bench" "$eval" "$thr" "$causal_rows_sha" "$expected_split_policy_hash"; then
    echo "[postproc] ${mode_tag}/${ckpt_id}/${variable}: downstream hash-compatible -> skip"
    return 0
  fi

  "$PY" phase7/causal_audit.py \
    --model-key gpt2-medium \
    --controls "$controls" \
    --trace-dataset "$TRACE_TEST" \
    --state-decoder-checkpoint "$checkpoint" \
    --causal-checks "$causal" \
    --causal-layer 22 \
    --causal-variable "$variable" \
    --latent-source variant_conditioned \
    --control-latent-cache "$cache" \
    --require-mediation-for-causal-pass \
    --require-causal-mode control_conditioned \
    --device cpu \
    --output "$audit"

  local audit_sha
  audit_sha="$(sha256_of "$audit")"
  "$PY" phase7/split_audit_dataset.py \
    --audit "$audit" \
    --seed "$SPLIT_SEED" \
    --calib-fraction "$CALIB_FRACTION" \
    --output-prefix "$split_prefix" \
    --reuse-manifest-if-compatible "$manifest" \
    --source-audit-hash "$audit_sha"

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
    --comparability-full-threshold 0.60 \
    --comparability-text-threshold 0.01 \
    --comparability-sensitivity 0.50,0.60,0.70 \
    --causal-degenerate-identical-threshold 0.80 \
    --causal-degenerate-auroc-threshold 0.55 \
    --causal-degenerate-enable-auroc-trigger \
    --ablation-weights '{"text":0.5,"latent":0.5,"causal":0.0}' \
    --output "$bench"
}

run_postproc_parallel_for_vars() {
  local run_tag="$1"
  local ckpt_id="$2"
  local checkpoint="$3"
  local mode_tag="$4"
  shift 4
  local vars=("$@")

  local split_hash
  split_hash="$(split_policy_hash)"

  local pids=()
  for variable in "${vars[@]}"; do
    while (( ${#pids[@]} >= POSTPROC_JOBS )); do
      local pid0="${pids[0]}"
      wait "$pid0"
      pids=("${pids[@]:1}")
    done
    (
      run_postproc_var "$run_tag" "$ckpt_id" "$checkpoint" "$mode_tag" "$variable" "$split_hash"
    ) &
    pids+=("$!")
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

write_efficiency_meta() {
  local base="$1"
  local mode_tag="$2"
  local ckpt_id="$3"
  local variable="$4"
  local stage_csv="$5"
  local rows_csv="$6"
  local downstream_skipped="$7"

  local out="$base/meta/eff_${mode_tag}_${ckpt_id}_${variable}.json"
  "$PY" - <<PY
import json
out="$out"
stages=[int(x) for x in "$stage_csv".split(",") if x.strip()]
rows=[int(x) for x in "$rows_csv".split(",") if x.strip()]
payload={
  "schema_version":"phase7_efficiency_telemetry_v1",
  "mode_tag":"$mode_tag",
  "checkpoint_id":"$ckpt_id",
  "variable":"$variable",
  "causal_stage_sequence":stages,
  "incremental_rows_added_by_stage":rows,
  "downstream_rerun_skipped":bool(str("$downstream_skipped").lower()=="true"),
  "rows_format_used":"jsonl.gz",
  "postproc_cpu_parallel_jobs":int("$POSTPROC_JOBS"),
}
json.dump(payload, open(out, "w"), indent=2)
PY
}

run_checkpoint_vars_efficient() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local mode_tag="$4"
  local ckpt_id="$5"
  local device="$6"
  shift 6
  local vars=("$@")

  local ckpt
  ckpt="$(ckpt_path "$ckpt_id")"
  local controls="phase7_results/controls/cot_controls_${run_tag}.json"
  local subspaces="phase7_results/interventions/variable_subspaces_${run_tag}.json"
  local cache="phase7_results/interventions/control_latent_cache_${run_tag}_${mode_tag}_${ckpt_id}.json"

  for variable in "${vars[@]}"; do
    ensure_compat_manifest "$base" "$mode_tag" "$ckpt_id" "$variable" "$controls" "$TRACE_TEST" "$subspaces" "$ckpt"
  done

  if ! json_ok "$cache"; then
    "$PY" phase7/build_control_latent_cache.py \
      --controls "$controls" \
      --state-decoder-checkpoint "$ckpt" \
      --model-key gpt2-medium \
      --parse-mode hybrid \
      --token-anchor eq_like \
      --anchor-priority "$ANCHOR_PRIORITY" \
      --device "$device" \
      --batch-size 128 \
      --rows-format jsonl.gz \
      --no-rows-inline \
      --output "$cache"
  fi

  local stage_index=0
  local last_index=$(( ${#MAX_RECORDS_STAGES[@]} - 1 ))
  local stage_sequence=()
  declare -A rows_added_map
  for variable in "${vars[@]}"; do
    rows_added_map["$variable"]=""
  done

  while true; do
    local stage_max="${MAX_RECORDS_STAGES[$stage_index]}"
    stage_sequence+=("$stage_max")

    run_causal_stage_for_vars "$run_tag" "$ckpt_id" "$ckpt" "$mode_tag" "$device" "$stage_max" "${vars[@]}"

    local first_var="${vars[0]}"
    local first_tag="${run_tag}_${mode_tag}_${ckpt_id}_${first_var}"
    local first_causal="phase7_results/interventions/causal_checks_${first_tag}.json"
    local controls_used
    controls_used="$(controls_used_fraction_from_causal "$first_causal")"

    for variable in "${vars[@]}"; do
      local tag="${run_tag}_${mode_tag}_${ckpt_id}_${variable}"
      local causal="phase7_results/interventions/causal_checks_${tag}.json"
      local rows_added
      rows_added="$(rows_added_from_causal "$causal")"
      if [[ -z "${rows_added_map[$variable]}" ]]; then
        rows_added_map["$variable"]="$rows_added"
      else
        rows_added_map["$variable"]+=",${rows_added}"
      fi
    done

    if "$PY" - <<PY
import math,sys
try:
    c=float("$controls_used")
except Exception:
    c=float("nan")
need=(not math.isfinite(c)) or (c < float("$TARGET_CONTROLS_USED"))
has_next=($stage_index < $last_index)
sys.exit(0 if (need and has_next) else 1)
PY
    then
      stage_index=$((stage_index + 1))
      continue
    fi

    run_postproc_parallel_for_vars "$run_tag" "$ckpt_id" "$ckpt" "$mode_tag" "${vars[@]}"

    local min_cov=""
    local vars_csv
    vars_csv="$(IFS=,; echo "${vars[*]}")"
    min_cov="$($PY - <<PY
import json,math
run_tag="$run_tag"
mode_tag="$mode_tag"
ckpt_id="$ckpt_id"
vars_csv="$vars_csv"
vals=[]
for var in [x for x in vars_csv.split(",") if x]:
    p=f"phase7_results/results/faithfulness_benchmark_{run_tag}_{mode_tag}_{ckpt_id}_{var}.json"
    d=json.load(open(p))
    v=d.get("causal_signal_coverage_fraction")
    if isinstance(v,(int,float)) and math.isfinite(float(v)):
        vals.append(float(v))
print(min(vals) if vals else "nan")
PY
)"

    if "$PY" - <<PY
import math,sys
try:
    v=float("$min_cov")
except Exception:
    v=float("nan")
need=(not math.isfinite(v)) or (v < float("$TARGET_COVERAGE"))
has_next=($stage_index < $last_index)
sys.exit(0 if (need and has_next) else 1)
PY
    then
      stage_index=$((stage_index + 1))
      continue
    fi

    break
  done

  local stage_csv
  stage_csv="$(IFS=,; echo "${stage_sequence[*]}")"
  for variable in "${vars[@]}"; do
    local rows_csv
    rows_csv="${rows_added_map[$variable]// /}"
    local bench="phase7_results/results/faithfulness_benchmark_${run_tag}_${mode_tag}_${ckpt_id}_${variable}.json"
    local downstream_skipped="false"
    if json_ok "$bench"; then
      downstream_skipped="$($PY - <<PY
import json
d=json.load(open("$bench"))
print(str(bool((d.get("upstream_hashes") or {}).get("audit_file_sha256"))).lower())
PY
)"
    fi
    write_efficiency_meta "$base" "$mode_tag" "$ckpt_id" "$variable" "$stage_csv" "$rows_csv" "$downstream_skipped"
  done
}

compute_canary_decision() {
  local run_tag="$1"
  local out="phase7_results/results/trackA_canary_decision_${run_tag}.json"
  "$PY" - <<PY
import json
run_tag="$run_tag"
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
      "causal_track_degenerate_flag":d.get("causal_track_degenerate_flag"),
    })
ident=[r["causal_variant_score_identical_fraction"] for r in rows if isinstance(r.get("causal_variant_score_identical_fraction"),(int,float))]
causal=[r["causal_auditor_auroc"] for r in rows if isinstance(r.get("causal_auditor_auroc"),(int,float))]
cov=[r["causal_signal_coverage_fraction"] for r in rows if isinstance(r.get("causal_signal_coverage_fraction"),(int,float))]
leak=all(bool(r.get("leakage_check_pass")) for r in rows)
anti_all=all(bool(r.get("causal_anti_predictive_flag")) for r in rows)
harm_all=all(bool(r.get("causal_harms_composite_flag")) for r in rows)
pass_ident=(len(ident)>0 and min(ident) <= 0.90)
pass_causal=(len(causal)>0 and max(causal) > 0.55)
pass_cov=(len(cov)>0 and min(cov) >= float("$TARGET_COVERAGE"))
canary_pass=bool(pass_ident and pass_causal and pass_cov and leak and (not anti_all) and (not harm_all))
out={
  "schema_version":"phase7_causal_canary_decision_v3",
  "run_tag":run_tag,
  "rows":rows,
  "criteria":{
    "causal_variant_score_identical_fraction_max":0.90,
    "causal_auditor_auroc_min":0.55,
    "causal_signal_coverage_fraction_min":float("$TARGET_COVERAGE"),
    "leakage_check_pass":True,
    "not_all_causal_anti_predictive":True,
    "not_all_causal_harms_composite":True,
  },
  "checks":{
    "identical_fraction_pass":pass_ident,
    "causal_auroc_pass":pass_causal,
    "coverage_pass":pass_cov,
    "leakage_pass":leak,
    "not_all_causal_anti_predictive":(not anti_all),
    "not_all_causal_harms_composite":(not harm_all),
  },
  "canary_pass":canary_pass,
}
json.dump(out,open("$out","w"),indent=2)
print("$out")
PY
}

aggregate_outputs() {
  local run_tag="$1"
  "$PY" - <<PY
import glob,json,re,time,statistics
run_tag="$run_tag"
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
    "resolved_items":[
      "parallel_canary",
      "incremental_escalation",
      "downstream_hash_skip",
      "split_manifest_hash_reuse",
      "compressed_row_sidecars",
      "cpu_parallel_postproc",
      "warm_model_workers"
    ],
    "generated_at_unix":int(time.time()),
},open(closure_out,"w"),indent=2)
print(matrix_out)
print(decision_out)
print(diag_out)
print(closure_out)
PY
}

canary_worker() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local gpu="$4"
  local group="$5"
  local log="$base/logs/canary_gpu${gpu}_${group}.log"

  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2

  local vars=()
  if [[ "$group" == "A" ]]; then
    vars=(subresult_value magnitude_bucket)
  else
    vars=(operator sign)
  fi

  {
    echo "[$(date -Is)] canary worker start gpu=$gpu group=$group vars=${vars[*]}"
    run_checkpoint_vars_efficient "$run_id" "$run_tag" "$base" "canary" "raw_every2_even" "cuda:0" "${vars[@]}"
    echo "$(date -Is)" > "$base/state/canary_gpu${gpu}.done"
    echo "[$(date -Is)] canary worker done gpu=$gpu"
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
      run_checkpoint_vars_efficient "$run_id" "$run_tag" "$base" "matrix" "raw_every2_even" "cuda:0" "${ALL_VARIABLES[@]}"
      run_checkpoint_vars_efficient "$run_id" "$run_tag" "$base" "matrix" "raw_middle12_06_17" "cuda:0" "${ALL_VARIABLES[@]}"
      echo "$(date -Is)" > "$base/state/gpu6.matrix.done"
    else
      run_checkpoint_vars_efficient "$run_id" "$run_tag" "$base" "matrix" "hybrid_every2_even" "cuda:0" "${ALL_VARIABLES[@]}"
      run_checkpoint_vars_efficient "$run_id" "$run_tag" "$base" "matrix" "hybrid_middle12_06_17" "cuda:0" "${ALL_VARIABLES[@]}"
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
    while [[ ! -f "$base/state/canary_gpu6.done" ]]; do sleep 15; done
    while [[ ! -f "$base/state/canary_gpu7.done" ]]; do sleep 15; done

    compute_canary_decision "$run_tag"

    local decision="phase7_results/results/trackA_canary_decision_${run_tag}.json"
    local canary_pass
    canary_pass="$($PY - <<PY
import json
print(str(bool(json.load(open("$decision")).get("canary_pass"))).lower())
PY
)"

    if [[ "$canary_pass" == "true" ]]; then
      tmux new-session -d -s "p7r24_promote_g6_${run_id}" "cd '$PWD' && bash experiments/run_phase7_causal_recovery_r2p4_efficient.sh promote-worker '$run_id' '$run_tag' '$base' 6 raw"
      tmux new-session -d -s "p7r24_promote_g7_${run_id}" "cd '$PWD' && bash experiments/run_phase7_causal_recovery_r2p4_efficient.sh promote-worker '$run_id' '$run_tag' '$base' 7 hybrid"
      while [[ ! -f "$base/state/gpu6.matrix.done" ]]; do sleep 30; done
      while [[ ! -f "$base/state/gpu7.matrix.done" ]]; do sleep 30; done
    else
      echo "[$(date -Is)] canary failed; promotion skipped"
      echo "$(date -Is)" > "$base/state/gpu6.skipped"
      echo "$(date -Is)" > "$base/state/gpu7.skipped"
    fi

    aggregate_outputs "$run_tag"
    echo "$(date -Is)" > "$base/state/pipeline.done"
    echo "[$(date -Is)] coordinator done"
  } >> "$log" 2>&1
}

launch() {
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)_phase7_causal_recovery_r2p4"
  local run_tag="phase7_causal_recovery_r2p4_${run_id}"
  local base="phase7_results/runs/${run_id}"

  mkdir -p "$base"/{logs,state,meta}

  "$PY" - <<PY
import glob,json,time
run_tag="$run_tag"
baseline_out=f"phase7_results/results/{run_tag}_baseline_refs.json"
contract_out=f"phase7_results/results/{run_tag}_closure_contract.json"
refs={
  "schema_version":"phase7_r2p4_baseline_refs_v1",
  "run_tag":run_tag,
  "latest_trackA_gate_matrix":sorted(glob.glob("phase7_results/results/trackA_gate_matrix*.json"))[-1] if glob.glob("phase7_results/results/trackA_gate_matrix*.json") else None,
  "latest_canary_decision":sorted(glob.glob("phase7_results/results/trackA_canary_decision*.json"))[-1] if glob.glob("phase7_results/results/trackA_canary_decision*.json") else None,
  "generated_at_unix":int(time.time()),
}
contract={
  "schema_version":"phase7_r2p4_closure_contract_v1",
  "run_tag":run_tag,
  "dual_gate_unchanged":True,
  "strict_external_labels_required":True,
  "run_scope":"four_checkpoint_matrix",
  "hardware_target":["gpu6","gpu7"],
  "efficiency_features_enabled":[
    "parallel_gpu_canary",
    "skip_downstream_hash_guard",
    "incremental_escalation",
    "warm_model_workers",
    "cpu_parallel_postproc",
    "split_manifest_reuse_hash_guard",
    "compressed_jsonl_sidecars"
  ],
  "target_causal_signal_coverage_fraction":float("$TARGET_COVERAGE"),
  "target_controls_used_fraction":float("$TARGET_CONTROLS_USED"),
  "max_records_cap":int("$MAX_RECORDS_CAP"),
  "generated_at_unix":int(time.time()),
}
json.dump(refs,open(baseline_out,"w"),indent=2)
json.dump(contract,open(contract_out,"w"),indent=2)
print(baseline_out)
print(contract_out)
PY

  prepare_shared_inputs "$run_tag"

  tmux new-session -d -s "p7r24_canary_g6_${run_id}" "cd '$PWD' && bash experiments/run_phase7_causal_recovery_r2p4_efficient.sh canary-worker '$run_id' '$run_tag' '$base' 6 A"
  tmux new-session -d -s "p7r24_canary_g7_${run_id}" "cd '$PWD' && bash experiments/run_phase7_causal_recovery_r2p4_efficient.sh canary-worker '$run_id' '$run_tag' '$base' 7 B"
  tmux new-session -d -s "p7r24_coord_${run_id}" "cd '$PWD' && bash experiments/run_phase7_causal_recovery_r2p4_efficient.sh coordinator '$run_id' '$run_tag' '$base'"

  echo "RUN_ID=$run_id"
  echo "RUN_TAG=$run_tag"
  echo "BASE=$base"
  echo "Sessions:"
  echo "  p7r24_canary_g6_${run_id}"
  echo "  p7r24_canary_g7_${run_id}"
  echo "  p7r24_coord_${run_id}"
}

cmd="${1:-}"
case "$cmd" in
  launch) launch ;;
  canary-worker) canary_worker "$2" "$3" "$4" "$5" "$6" ;;
  promote-worker) promote_worker "$2" "$3" "$4" "$5" "$6" ;;
  coordinator) coordinator "$2" "$3" "$4" ;;
  *) usage; exit 1 ;;
esac

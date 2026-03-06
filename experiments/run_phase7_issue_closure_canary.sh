#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  experiments/run_phase7_issue_closure_canary.sh launch
  experiments/run_phase7_issue_closure_canary.sh worker-synth <RUN_ID> <RUN_TAG> <BASE> <GPU>
  experiments/run_phase7_issue_closure_canary.sh worker-real <RUN_ID> <RUN_TAG> <BASE> <GPU>
  experiments/run_phase7_issue_closure_canary.sh coordinator <RUN_ID> <RUN_TAG> <BASE>
EOF
}

PY=".venv/bin/python3"
TRACE_TEST="phase7_results/dataset/gsm8k_step_traces_test.pt"
TRACE_TRAIN="phase7_results/dataset/gsm8k_step_traces_train.pt"
CKPT_RAW_E2E="phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration/checkpoints/state_raw_every2_even.pt"

freeze_baseline_refs() {
  local run_tag="$1"
  $PY - <<PY
import json,glob,os
run_tag="${run_tag}"
refs={
  "schema_version":"phase7_issue_closure_baseline_refs_v1",
  "run_tag":run_tag,
  "latest_trackA_gate_matrix":sorted(glob.glob("phase7_results/results/trackA_gate_matrix*.json"))[-1] if glob.glob("phase7_results/results/trackA_gate_matrix*.json") else None,
  "latest_real_cot_benchmark":sorted(glob.glob("phase7_results/results/faithfulness_benchmark_real_cot*.json"))[-1] if glob.glob("phase7_results/results/faithfulness_benchmark_real_cot*.json") else None,
  "latest_qwen_pilot":sorted(glob.glob("phase7_results/results/faithfulness_benchmark_qwen_pilot*.json"))[-1] if glob.glob("phase7_results/results/faithfulness_benchmark_qwen_pilot*.json") else None,
}
out=f"phase7_results/results/{run_tag}_baseline_refs.json"
with open(out,"w") as f:
  json.dump(refs,f,indent=2)
print(out)
PY
}

worker_synth() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local gpu="$4"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2

  mkdir -p "$base"/{logs,state,meta}
  local log="$base/logs/worker_synth_gpu${gpu}.log"
  local ctrl="phase7_results/controls/cot_controls_${run_tag}.json"
  local parsed="phase7_results/controls/cot_controls_${run_tag}_parsed.json"
  local subspace="phase7_results/interventions/variable_subspaces_${run_tag}.json"
  local cache="phase7_results/interventions/control_latent_cache_${run_tag}_raw_every2_even.json"

  {
    echo "[worker-synth] generating controls"
    $PY phase7/generate_cot_controls.py \
      --trace-dataset "$TRACE_TEST" \
      --max-traces 500 \
      --seed 20260305 \
      --output "$ctrl"

    echo "[worker-synth] parsing controls"
    $PY phase7/parse_cot_to_states.py \
      --controls "$ctrl" \
      --trace-dataset "$TRACE_TEST" \
      --parse-mode hybrid \
      --output "$parsed"

    echo "[worker-synth] building subspaces"
    $PY phase7/variable_subspace_builder.py \
      --model-key gpt2-medium \
      --layers 7 12 17 22 \
      --probe-position result \
      --top-k 64 \
      --combine-policy union \
      --output "$subspace"

    echo "[worker-synth] building variant latent cache (eq_like anchor)"
    $PY phase7/build_control_latent_cache.py \
      --controls "$ctrl" \
      --state-decoder-checkpoint "$CKPT_RAW_E2E" \
      --model-key gpt2-medium \
      --parse-mode hybrid \
      --token-anchor eq_like \
      --anchor-priority template_first \
      --device cuda:0 \
      --batch-size 128 \
      --output "$cache"

    for var in subresult_value operator magnitude_bucket sign; do
      local tag="${run_tag}_raw_every2_even_${var}"
      local causal="phase7_results/interventions/causal_checks_${tag}.json"
      local audit="phase7_results/audits/text_causal_audit_controls_${tag}.json"
      local split_prefix="phase7_results/audits/${tag}"
      local calib="${split_prefix}_calib.json"
      local eval="${split_prefix}_eval.json"
      local thr="phase7_results/calibration/phase7_thresholds_${tag}.json"
      local bench="phase7_results/results/faithfulness_benchmark_${tag}.json"

      echo "[worker-synth] variable=${var} pass1(max-records=1200)"
      $PY phase7/causal_intervention_engine.py \
        --model-key gpt2-medium \
        --trace-dataset "$TRACE_TEST" \
        --controls "$ctrl" \
        --parse-mode hybrid \
        --token-anchor eq_like \
        --anchor-priority template_first \
        --subspace-specs "$subspace" \
        --variable "$var" \
        --layers 7 12 17 22 \
        --max-records 1200 \
        --device cuda:0 \
        --state-decoder-checkpoint "$CKPT_RAW_E2E" \
        --enable-latent-mediation \
        --mediation-variable "$var" \
        --output "$causal"

      $PY phase7/causal_audit.py \
        --model-key gpt2-medium \
        --controls "$ctrl" \
        --trace-dataset "$TRACE_TEST" \
        --state-decoder-checkpoint "$CKPT_RAW_E2E" \
        --causal-checks "$causal" \
        --causal-layer 22 \
        --causal-variable "$var" \
        --latent-source variant_conditioned \
        --control-latent-cache "$cache" \
        --require-mediation-for-causal-pass \
        --device cuda:0 \
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
        echo "[worker-synth] variable=${var} escalating(max-records=2200)"
        $PY phase7/causal_intervention_engine.py \
          --model-key gpt2-medium \
          --trace-dataset "$TRACE_TEST" \
          --controls "$ctrl" \
          --parse-mode hybrid \
          --token-anchor eq_like \
          --anchor-priority template_first \
          --subspace-specs "$subspace" \
          --variable "$var" \
          --layers 7 12 17 22 \
          --max-records 2200 \
          --device cuda:0 \
          --state-decoder-checkpoint "$CKPT_RAW_E2E" \
          --enable-latent-mediation \
          --mediation-variable "$var" \
          --output "$causal"

        $PY phase7/causal_audit.py \
          --model-key gpt2-medium \
          --controls "$ctrl" \
          --trace-dataset "$TRACE_TEST" \
          --state-decoder-checkpoint "$CKPT_RAW_E2E" \
          --causal-checks "$causal" \
          --causal-layer 22 \
          --causal-variable "$var" \
          --latent-source variant_conditioned \
          --control-latent-cache "$cache" \
          --require-mediation-for-causal-pass \
          --device cuda:0 \
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
          --output "$bench"
      fi
    done

    echo "$(date -Is)" > "$base/state/synth.done"
  } >>"$log" 2>&1
}

worker_real() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  local gpu="$4"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1

  mkdir -p "$base"/{logs,state,meta}
  local log="$base/logs/worker_real_gpu${gpu}.log"
  local ingest="phase7_results/real_cot/public_benchmark_ingest_${run_tag}.json"
  local controls="phase7_results/real_cot/public_benchmark_controls_${run_tag}.json"
  local audit="phase7_results/audits/text_causal_audit_controls_real_cot_labeled_${run_tag}.json"
  local split_prefix="phase7_results/audits/real_cot_labeled_${run_tag}"
  local calib="${split_prefix}_calib.json"
  local eval="${split_prefix}_eval.json"
  local thr="phase7_results/calibration/phase7_thresholds_real_cot_labeled_${run_tag}.json"
  local bench="phase7_results/results/faithfulness_benchmark_real_cot_labeled_${run_tag}.json"
  local gate="phase7_results/results/faithfulness_benchmark_real_cot_labeled_${run_tag}_external_validity_gate.json"

  {
    echo "[worker-real] ingest public labeled benchmark"
    if ! $PY phase7/ingest_public_cot_benchmark.py \
      --source auto \
      --strict-external-labels \
      --output "$ingest" \
      --controls-output "$controls"; then
      echo "[worker-real] strict external labeled source unavailable; emitting blocked artifact"
      $PY - <<PY
import json, time
bench = "${bench}"
gate = "${gate}"
payload = {
    "schema_version": "phase7_faithfulness_benchmark_v1",
    "benchmark_scope": "real_cot",
    "external_validity_status": "blocked_no_strict_labels",
    "model_comparability_status": "not_comparable",
    "claim_boundary_disclaimer": (
        "Causal support scores reflect measured variables/subspaces and tested interventions only; "
        "they are not a complete explanation of all internal reasoning."
    ),
    "scope_disclaimer": "Real-CoT labeled benchmark blocked: no strict external label source available.",
    "gate_checks": {
        "applicable": False,
        "gate_pass": None,
        "reason": "blocked_no_strict_labels",
    },
    "generated_at_unix": int(time.time()),
}
with open(bench, "w") as f:
    json.dump(payload, f, indent=2)
gate_payload = {
    "schema_version": "phase7_external_validity_gate_v1",
    "benchmark_scope": "real_cot",
    "external_validity_status": "blocked_no_strict_labels",
    "requires_real_cot_pilot": True,
    "has_real_cot_pilot": False,
    "externally_supported_claims": False,
    "comparability_status": "not_comparable",
    "comparability_gate_pass": False,
    "reason": "Strict external labels unavailable; labeled real-CoT benchmark blocked.",
}
with open(gate, "w") as f:
    json.dump(gate_payload, f, indent=2)
print(f"saved blocked benchmark -> {bench}")
print(f"saved blocked external validity gate -> {gate}")
PY
      echo "$(date -Is)" > "$base/state/real.done"
      return 0
    fi

    labeled_n=$($PY - <<PY
import json
d=json.load(open("${ingest}"))
print(int(d.get("num_rows",0)))
PY
)
    if [[ "$labeled_n" -lt 200 ]]; then
      echo "labeled rows too small: $labeled_n (need >=200)" >&2
      exit 2
    fi

    echo "[worker-real] audit+split+benchmark (no answer_match proxy)"
    $PY phase7/causal_audit.py \
      --model-key gpt2-medium \
      --controls "$controls" \
      --trace-dataset "$TRACE_TEST" \
      --state-decoder-checkpoint "$CKPT_RAW_E2E" \
      --causal-layer 22 \
      --causal-variable subresult_value \
      --latent-source shared \
      --device cuda:0 \
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
      --output "$bench" \
      --external-validity-gate-output "$gate"

    echo "$(date -Is)" > "$base/state/real.done"
  } >>"$log" 2>&1
}

coordinator() {
  local run_id="$1"
  local run_tag="$2"
  local base="$3"
  mkdir -p "$base"/{logs,state,meta}
  local log="$base/logs/coordinator.log"
  {
    echo "[coord] waiting synth+real"
    while [[ ! -f "$base/state/synth.done" ]]; do sleep 20; done
    while [[ ! -f "$base/state/real.done" ]]; do sleep 20; done

    $PY - <<PY
import glob,json,time
import re
run_tag="${run_tag}"
bench_files=sorted(glob.glob(f"phase7_results/results/faithfulness_benchmark_{run_tag}_raw_every2_even_*.json"))
rows=[]
for p in bench_files:
  d=json.load(open(p))
  bt=d.get("by_benchmark_track",{})
  m=re.search(rf"faithfulness_benchmark_{re.escape(run_tag)}_raw_every2_even_(.+)\\.json$", p)
  var=m.group(1) if m else p.replace(".json","").split("_")[-1]
  rows.append({
    "benchmark_path":p,
    "variable":var,
    "composite_auroc":(bt.get("composite",{}).get("auroc")),
    "text_auroc":(bt.get("text_only",{}).get("auroc")),
    "latent_auroc":(bt.get("latent_only",{}).get("auroc")),
    "causal_auroc":(bt.get("causal_auditor",{}).get("auroc")),
    "coverage":d.get("causal_signal_coverage_fraction"),
    "dual_gate_pass":(d.get("gate_checks",{}).get("dual_gate_pass")),
    "leakage_check_pass":d.get("leakage_check_pass"),
  })
matrix_out=f"phase7_results/results/trackA_gate_matrix_{run_tag}.json"
json.dump({"schema_version":"phase7_trackA_gate_matrix_v1","run_tag":run_tag,"rows":rows},open(matrix_out,"w"),indent=2)

decision_out=f"phase7_results/results/trackA_gate_decision_{run_tag}_raw_every2_even.json"
comp=[r.get("composite_auroc") for r in rows if isinstance(r.get("composite_auroc"),(int,float))]
caus=[r.get("causal_auroc") for r in rows if isinstance(r.get("causal_auroc"),(int,float))]
cov=[r.get("coverage") for r in rows if isinstance(r.get("coverage"),(int,float))]
decision={
  "schema_version":"phase7_trackA_gate_decision_v1",
  "run_tag":run_tag,
  "checkpoint_id":"raw_every2_even",
  "composite_auroc_max":(max(comp) if comp else None),
  "causal_auroc_max":(max(caus) if caus else None),
  "coverage_max":(max(cov) if cov else None),
  "any_dual_gate_pass":any(bool(r.get("dual_gate_pass")) for r in rows),
}
json.dump(decision,open(decision_out,"w"),indent=2)

report_out=f"phase7_results/results/academic_weakness_closure_report_{run_tag}.json"
real_path=f"phase7_results/results/faithfulness_benchmark_real_cot_labeled_{run_tag}.json"
json.dump({
  "schema_version":"phase7_academic_weakness_closure_report_v1",
  "run_tag":run_tag,
  "synthetic_gate_matrix_path":matrix_out,
  "synthetic_gate_decision_path":decision_out,
  "real_cot_labeled_benchmark_path":real_path,
  "resolved_items":[
    "causal_component_weights_normalized",
    "token_anchor_eq_like_enabled",
    "no_answer_match_proxy_for_labeled_real_cot",
  ],
  "remaining_items":[
    "qwen_full_comparability_out_of_scope",
  ],
  "generated_at_unix":int(time.time())
},open(report_out,"w"),indent=2)
print(matrix_out)
print(decision_out)
print(report_out)
PY
    echo "$(date -Is)" > "$base/state/pipeline.done"
    echo "[coord] pipeline.done"
  } >>"$log" 2>&1
}

launch() {
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)_phase7_issue_closure_r1"
  local run_tag="phase7_issue_closure_r1_${run_id}"
  local base="phase7_results/runs/${run_id}"
  mkdir -p "$base"/{logs,state,meta,scripts}
  freeze_baseline_refs "$run_tag"
  echo "$run_id" > "$base/meta/run_id.txt"
  echo "$run_tag" > "$base/meta/run_tag.txt"

  tmux new-session -d -s "p7fix_g1_${run_id}" "cd \"$PWD\" && bash experiments/run_phase7_issue_closure_canary.sh worker-synth \"$run_id\" \"$run_tag\" \"$base\" 1"
  tmux new-session -d -s "p7fix_g2_${run_id}" "cd \"$PWD\" && bash experiments/run_phase7_issue_closure_canary.sh worker-real \"$run_id\" \"$run_tag\" \"$base\" 2"
  tmux new-session -d -s "p7fix_g4_${run_id}" "cd \"$PWD\" && bash experiments/run_phase7_issue_closure_canary.sh coordinator \"$run_id\" \"$run_tag\" \"$base\""

  echo "RUN_ID=$run_id"
  echo "RUN_TAG=$run_tag"
  echo "BASE=$base"
  echo "Sessions:"
  echo "  p7fix_g1_${run_id}"
  echo "  p7fix_g2_${run_id}"
  echo "  p7fix_g4_${run_id}"
}

cmd="${1:-}"
case "$cmd" in
  launch) launch ;;
  worker-synth) worker_synth "$2" "$3" "$4" "$5" ;;
  worker-real) worker_real "$2" "$3" "$4" "$5" ;;
  coordinator) coordinator "$2" "$3" "$4" ;;
  *) usage; exit 1 ;;
esac

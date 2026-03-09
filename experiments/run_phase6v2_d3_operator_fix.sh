#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY=".venv/bin/python3"

mk_run_id() {
  date +"%Y%m%d_%H%M%S_phase6v2_d3_operator_fix"
}

select_free_gpus() {
  local need="${1:-3}"
  mapfile -t rows < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
  local picked=()
  for row in "${rows[@]}"; do
    local idx util mem
    idx="$(echo "$row" | cut -d',' -f1 | tr -d ' ')"
    util="$(echo "$row" | cut -d',' -f2 | tr -d ' ')"
    mem="$(echo "$row" | cut -d',' -f3 | tr -d ' ')"
    if [[ -n "$idx" && -n "$util" && -n "$mem" ]]; then
      if (( util < 10 )) && (( mem < 2000 )); then
        picked+=("$idx")
      fi
    fi
  done
  if (( ${#picked[@]} < need )); then
    echo "ERROR: not enough free GPUs. need=${need} got=${#picked[@]} free_candidates=${picked[*]-}" >&2
    return 2
  fi
  echo "${picked[@]:0:need}"
}

build_d3_dataset() {
  local run_id="$1"
  local out_dir="phase7_results/runs/${run_id}/dataset"
  mkdir -p "$out_dir"
  "$PY" phase7/build_step_trace_dataset.py \
    --phase6-train phase6_results/dataset/gsm8k_expanded_train.pt \
    --phase6-test phase6_results/dataset/gsm8k_expanded_test.pt \
    --output-dir "$out_dir" \
    --model-key gpt2-medium \
    --state-ontology v2_expanded
}

train_worker() {
  local run_id="$1"; local variant="$2"; local gpu="$3"
  local cfg_name
  case "$variant" in
    raw) cfg_name="state_raw_multi_l4_l5_l6_l7" ;;
    hybrid) cfg_name="state_hybrid_multi_l4_l5_l6_l7" ;;
    sae) cfg_name="state_sae_multi_l4_l5_l6_l7" ;;
    *) echo "unknown variant: $variant" >&2; exit 2 ;;
  esac
  local ds_train="phase7_results/runs/${run_id}/dataset/gsm8k_step_traces_train.pt"
  echo "[$(date -Is)] train ${variant} gpu=${gpu} run=${run_id}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$PY" phase7/train_state_decoders.py \
    --dataset-train "$ds_train" \
    --config-name "$cfg_name" \
    --improvement-profile d3_operate_known \
    --operator-early-stop-guard \
    --device cuda \
    --checkpoints-dir "phase7_results/checkpoints/${run_id}" \
    --results-dir "phase7_results/results/${run_id}" \
    --cache-inputs on \
    --cache-max-gb 8 \
    --num-workers 8 \
    --pin-memory \
    --persistent-workers \
    --prefetch-factor 4 \
    --non-blocking-transfer \
    --torch-num-threads 12
}

_run_var_pipeline() {
  local model_ckpt="$1"; local trace_test="$2"; local run_tag="$3"; local var="$4"; local device="$5"; local controls="$6"; local subspaces="$7"; local latent_cache="$8"
  local tag="${run_tag}_${var}"
  local causal="phase7_results/interventions/causal_checks_${tag}.json"
  local audit="phase7_results/audits/text_causal_audit_controls_${tag}.json"
  local split_prefix="phase7_results/audits/${tag}"
  local calib="${split_prefix}_calib.json"
  local eval="${split_prefix}_eval.json"
  local thr="phase7_results/calibration/phase7_thresholds_${tag}.json"
  local bench="phase7_results/results/faithfulness_benchmark_${tag}.json"

  CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 \
  "$PY" phase7/causal_intervention_engine.py \
    --model-key gpt2-medium \
    --trace-dataset "$trace_test" \
    --controls "$controls" \
    --parse-mode hybrid \
    --token-anchor eq_like \
    --anchor-priority template_first \
    --control-sampling-policy stratified_trace_variant \
    --min-controls-used 150 \
    --subspace-specs "$subspaces" \
    --variable "$var" \
    --layers 7 12 17 22 \
    --max-records 1200 \
    --record-buffer-size 256 \
    --device cuda:0 \
    --state-decoder-checkpoint "$model_ckpt" \
    --enable-latent-mediation \
    --mediation-variable "$var" \
    --output "$causal"

  CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 \
  "$PY" phase7/causal_audit.py \
    --model-key gpt2-medium \
    --controls "$controls" \
    --trace-dataset "$trace_test" \
    --state-decoder-checkpoint "$model_ckpt" \
    --causal-checks "$causal" \
    --causal-layer 22 \
    --causal-variable "$var" \
    --latent-source variant_conditioned \
    --control-latent-cache "$latent_cache" \
    --require-causal-mode control_conditioned \
    --require-mediation-for-causal-pass \
    --device cuda:0 \
    --output "$audit"

  "$PY" phase7/split_audit_dataset.py --audit "$audit" --seed 20260303 --calib-fraction 0.30 --output-prefix "$split_prefix"
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
    --min-causal-signal-coverage 0.25 \
    --latent-high-quantile 0.80 \
    --output "$bench"
}

coordinator() {
  local run_id="$1"; local gpu_eval="$2"
  local base="phase7_results/runs/${run_id}"
  local res="phase7_results/results/${run_id}"
  local ckpt_dir="phase7_results/checkpoints/${run_id}"
  local trace_train="${base}/dataset/gsm8k_step_traces_train.pt"
  local trace_test="${base}/dataset/gsm8k_step_traces_test.pt"
  local controls="phase7_results/controls/cot_controls_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json"
  local subspaces="phase7_results/interventions/variable_subspaces_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json"
  local run_tag="phase6v2_d3_${run_id}"
  local latent_cache="phase7_results/interventions/control_latent_cache_${run_tag}.json"

  echo "[$(date -Is)] coordinator start run_id=${run_id}"
  while true; do
    local n_json n_ckpt
    n_json=$(ls "$res"/state_decoder_supervised_*_d3known.json 2>/dev/null | wc -l || true)
    n_ckpt=$(ls "$ckpt_dir"/*_d3known.pt 2>/dev/null | wc -l || true)
    echo "[$(date -Is)] waiting train outputs json=${n_json}/3 ckpt=${n_ckpt}/3"
    if [[ "$n_json" -ge 3 && "$n_ckpt" -ge 3 ]]; then break; fi
    sleep 60
  done

  for ckpt in "$ckpt_dir"/*_d3known.pt; do
    local name="$(basename "$ckpt" .pt)"
    local out="$res/state_decoder_eval_${name}.json"
    echo "[$(date -Is)] evaluating ${name}"
    CUDA_VISIBLE_DEVICES="$gpu_eval" PYTHONUNBUFFERED=1 \
    "$PY" phase7/evaluate_state_decoders.py \
      --checkpoint "$ckpt" \
      --dataset-train "$trace_train" \
      --dataset-test "$trace_test" \
      --eval-split test \
      --device cuda \
      --output "$out"
  done

  "$PY" phase7/diagnose_decoder_quality.py \
    --dataset "$trace_test" \
    --split test \
    --eval-results-glob "$res/state_decoder_eval_*.json" \
    --quality-gate-operator-operate-known-only-min 0.70 \
    --quality-gate-magnitude-min 0.90 \
    --quality-gate-sign-min 0.95 \
    --output-json "$res/decoder_quality_diagnostic_post_d3.json" \
    --output-md "$res/decoder_quality_diagnostic_post_d3.md"

  local gate_pass best_cfg
  gate_pass=$("$PY" - <<PY
import json
obj=json.load(open("$res/decoder_quality_diagnostic_post_d3.json"))
blk=(obj.get("d0_2_per_operator_accuracy_breakdown") or {})
print(str(bool(((blk.get("decoder_quality_gate_operate_known_only") or {}).get("pass")))).lower())
PY
)
  best_cfg=$("$PY" - <<PY
import json
obj=json.load(open("$res/decoder_quality_diagnostic_post_d3.json"))
blk=(obj.get("d0_2_per_operator_accuracy_breakdown") or {})
b=(blk.get("best_test_row_by_operator_acc") or {})
print(str(b.get("config_name") or ""))
PY
)

  if [[ "$gate_pass" != "true" ]]; then
    "$PY" - <<PY
import json
json.dump({
  "schema_version":"phase6v2_d3_phase7_blocked_v1",
  "run_id":"$run_id",
  "reason":"decoder_quality_gate_operate_known_only_failed",
  "diagnostic_path":"$res/decoder_quality_diagnostic_post_d3.json"
}, open("$res/phase7_blocked_${run_tag}.json","w"), indent=2)
PY
    echo "$(date -Is)" > "$base/state/phase7.blocked"
    echo "$(date -Is)" > "$base/state/pipeline.done"
    echo "[$(date -Is)] gate failed -> stop"
    return 0
  fi

  local best_ckpt="$ckpt_dir/${best_cfg}.pt"
  if [[ ! -f "$best_ckpt" ]]; then
    echo "best checkpoint missing: $best_ckpt" >&2
    echo "$(date -Is)" > "$base/state/phase7.blocked"
    echo "$(date -Is)" > "$base/state/pipeline.done"
    return 3
  fi

  CUDA_VISIBLE_DEVICES="$gpu_eval" PYTHONUNBUFFERED=1 \
  "$PY" phase7/build_control_latent_cache.py \
    --controls "$controls" \
    --state-decoder-checkpoint "$best_ckpt" \
    --model-key gpt2-medium \
    --parse-mode hybrid \
    --token-anchor eq_like \
    --anchor-priority template_first \
    --device cuda:0 \
    --batch-size 128 \
    --output "$latent_cache"

  echo "[$(date -Is)] running canary vars"
  _run_var_pipeline "$best_ckpt" "$trace_test" "${run_tag}_canary" subresult_value "$gpu_eval" "$controls" "$subspaces" "$latent_cache"
  _run_var_pipeline "$best_ckpt" "$trace_test" "${run_tag}_canary" operator "$gpu_eval" "$controls" "$subspaces" "$latent_cache"

  local canary_pass
  canary_pass=$("$PY" - <<PY
import json
ok=True
for v in ["subresult_value","operator"]:
  p=f"phase7_results/results/faithfulness_benchmark_${run_tag}_canary_{v}.json"
  d=json.load(open(p))
  ok = ok and bool(d.get("leakage_check_pass"))
print(str(ok).lower())
PY
)

  "$PY" - <<PY
import json
canary_pass = str("${canary_pass}").strip().lower() == "true"
json.dump({
  "schema_version":"phase6v2_d3_canary_decision_v1",
  "run_id":"$run_id",
  "run_tag":"$run_tag",
  "gate_pass": True,
  "best_checkpoint":"$best_ckpt",
  "canary_pass": canary_pass
}, open("$res/trackA_canary_decision_${run_tag}.json","w"), indent=2)
PY

  if [[ "$canary_pass" != "true" ]]; then
    echo "$(date -Is)" > "$base/state/phase7.blocked"
    echo "$(date -Is)" > "$base/state/pipeline.done"
    echo "[$(date -Is)] canary failed integrity -> stop"
    return 0
  fi

  echo "[$(date -Is)] canary passed -> matrix vars"
  for var in subresult_value operator magnitude_bucket sign; do
    _run_var_pipeline "$best_ckpt" "$trace_test" "${run_tag}_matrix" "$var" "$gpu_eval" "$controls" "$subspaces" "$latent_cache"
  done

  "$PY" - <<PY
import json
rows=[]
for var in ["subresult_value","operator","magnitude_bucket","sign"]:
  p=f"phase7_results/results/faithfulness_benchmark_${run_tag}_matrix_{var}.json"
  d=json.load(open(p))
  bt=d.get("by_benchmark_track") or {}
  rows.append({
    "variable":var,
    "path":p,
    "leakage_check_pass":d.get("leakage_check_pass"),
    "composite_auroc":(bt.get("composite") or {}).get("auroc"),
    "text_auroc":(bt.get("text_only") or {}).get("auroc"),
    "latent_auroc":(bt.get("latent_only") or {}).get("auroc"),
    "causal_auroc":(bt.get("causal_auditor") or {}).get("auroc"),
    "coverage":d.get("causal_signal_coverage_fraction"),
  })
out={"schema_version":"phase6v2_d3_trackA_matrix_v1","run_id":"$run_id","run_tag":"$run_tag","rows":rows}
json.dump(out, open("$res/trackA_gate_matrix_${run_tag}.json","w"), indent=2)
PY

  echo "$(date -Is)" > "$base/state/phase7.matrix.done"
  echo "$(date -Is)" > "$base/state/pipeline.done"
  echo "[$(date -Is)] coordinator done"
}

launch() {
  local run_id="${1:-$(mk_run_id)}"
  local base="phase7_results/runs/${run_id}"
  mkdir -p "$base/state" "$base/meta" "$base/logs" "$base/scripts" "phase7_results/results/${run_id}" "phase7_results/checkpoints/${run_id}"

  echo "[$(date -Is)] selecting free gpus"
  read -r g0 g1 g2 <<<"$(select_free_gpus 3)"
  echo "selected gpus: $g0 $g1 $g2"

  "$PY" - <<PY
import json, time
json.dump({
  "schema_version":"phase6v2_d3_run_meta_v1",
  "run_id":"$run_id",
  "selected_gpus":[int("$g0"), int("$g1"), int("$g2")],
  "created_at_unix":int(time.time())
}, open("$base/meta/run_meta.json","w"), indent=2)
PY

  echo "[$(date -Is)] rebuilding phase7 dataset"
  build_d3_dataset "$run_id"

  cp "$0" "$base/scripts/run_phase6v2_d3_operator_fix.sh"

  tmux new-session -d -s "p6d3_raw_${run_id}" "cd '$REPO_ROOT' && bash experiments/run_phase6v2_d3_operator_fix.sh worker-raw '$run_id' '$g0' |& tee '$base/logs/train_raw.log'"
  tmux new-session -d -s "p6d3_hyb_${run_id}" "cd '$REPO_ROOT' && bash experiments/run_phase6v2_d3_operator_fix.sh worker-hyb '$run_id' '$g1' |& tee '$base/logs/train_hybrid.log'"
  tmux new-session -d -s "p6d3_sae_${run_id}" "cd '$REPO_ROOT' && bash experiments/run_phase6v2_d3_operator_fix.sh worker-sae '$run_id' '$g2' |& tee '$base/logs/train_sae.log'"
  tmux new-session -d -s "p6d3_coord_${run_id}" "cd '$REPO_ROOT' && bash experiments/run_phase6v2_d3_operator_fix.sh coordinator '$run_id' '$g0' |& tee '$base/logs/coordinator.log'"

  echo "$run_id"
  tmux ls | rg "p6d3_.*${run_id}" || true
}

cmd="${1:-}"
case "$cmd" in
  launch)
    launch "${2:-}"
    ;;
  worker-raw)
    train_worker "$2" raw "$3"
    ;;
  worker-hyb)
    train_worker "$2" hybrid "$3"
    ;;
  worker-sae)
    train_worker "$2" sae "$3"
    ;;
  coordinator)
    coordinator "$2" "$3"
    ;;
  *)
    echo "usage: $0 launch [RUN_ID] | worker-raw RUN_ID GPU | worker-hyb RUN_ID GPU | worker-sae RUN_ID GPU | coordinator RUN_ID GPU" >&2
    exit 2
    ;;
esac

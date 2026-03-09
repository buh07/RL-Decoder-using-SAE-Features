#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=".venv/bin/python3"

mk_run_id() {
  date +"%Y%m%d_%H%M%S_phase6v2_d2_operate_only"
}

train_worker() {
  local run_id="$1"
  local variant="$2"
  local gpu="$3"
  local cfg_name
  case "$variant" in
    raw) cfg_name="state_raw_multi_l4_l5_l6_l7" ;;
    hybrid) cfg_name="state_hybrid_multi_l4_l5_l6_l7" ;;
    sae) cfg_name="state_sae_multi_l4_l5_l6_l7" ;;
    *) echo "unknown variant: $variant" >&2; exit 2 ;;
  esac
  local ckpt_dir="phase7_results/checkpoints/${run_id}"
  local res_dir="phase7_results/results/${run_id}"
  echo "[$(date -Is)] start train variant=${variant} gpu=${gpu} run_id=${run_id}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$PY" phase7/train_state_decoders.py \
    --config-name "$cfg_name" \
    --improvement-profile d2_tier2 \
    --operator-early-stop-guard \
    --operator-operate-only-supervision \
    --operator-loss-mode weighted_ce \
    --operator-class-weight-scope operate_only \
    --device cuda \
    --checkpoints-dir "$ckpt_dir" \
    --results-dir "$res_dir" \
    --cache-inputs on \
    --cache-max-gb 8 \
    --num-workers 8 \
    --pin-memory \
    --persistent-workers \
    --prefetch-factor 4 \
    --non-blocking-transfer \
    --torch-num-threads 12
}

run_phase7_canary() {
  local run_id="$1"
  local run_tag="$2"
  local base="phase7_results/runs/${run_id}"
  local ckpt_dir="phase7_results/checkpoints/${run_id}"
  local res_dir="phase7_results/results/${run_id}"

  local trace_test="phase7_results/dataset/gsm8k_step_traces_test.pt"
  local controls="phase7_results/controls/cot_controls_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json"
  local subspaces="phase7_results/interventions/variable_subspaces_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json"
  local ckpt_raw="${ckpt_dir}/state_raw_multi_l4_l5_l6_l7_d2tier2.pt"
  local cache="phase7_results/interventions/control_latent_cache_${run_tag}_canary_raw_d2.json"

  mkdir -p "$res_dir/canary" "$base/state"

  CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 \
  "$PY" phase7/build_control_latent_cache.py \
    --controls "$controls" \
    --state-decoder-checkpoint "$ckpt_raw" \
    --model-key gpt2-medium \
    --parse-mode hybrid \
    --token-anchor eq_like \
    --anchor-priority template_first \
    --device cuda:0 \
    --batch-size 128 \
    --output "$cache"

  for var in subresult_value operator; do
    local tag="${run_tag}_canary_raw_d2_${var}"
    local causal="phase7_results/interventions/causal_checks_${tag}.json"
    local audit="phase7_results/audits/text_causal_audit_controls_${tag}.json"
    local split_prefix="phase7_results/audits/${tag}"
    local calib="${split_prefix}_calib.json"
    local eval="${split_prefix}_eval.json"
    local thr="phase7_results/calibration/phase7_thresholds_${tag}.json"
    local bench="phase7_results/results/faithfulness_benchmark_${tag}.json"

    CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 \
    "$PY" phase7/causal_intervention_engine.py \
      --model-key gpt2-medium \
      --trace-dataset "$trace_test" \
      --controls "$controls" \
      --parse-mode hybrid \
      --token-anchor eq_like \
      --anchor-priority template_first \
      --control-sampling-policy stratified_trace_variant \
      --min-controls-used 100 \
      --subspace-specs "$subspaces" \
      --variable "$var" \
      --layers 7 \
      --max-records 800 \
      --record-buffer-size 256 \
      --device cuda:0 \
      --state-decoder-checkpoint "$ckpt_raw" \
      --enable-latent-mediation \
      --mediation-variable "$var" \
      --output "$causal"

    CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 \
    "$PY" phase7/causal_audit.py \
      --model-key gpt2-medium \
      --controls "$controls" \
      --trace-dataset "$trace_test" \
      --state-decoder-checkpoint "$ckpt_raw" \
      --causal-checks "$causal" \
      --causal-layer 7 \
      --causal-variable "$var" \
      --latent-source variant_conditioned \
      --control-latent-cache "$cache" \
      --require-causal-mode control_conditioned \
      --require-mediation-for-causal-pass \
      --device cuda:0 \
      --output "$audit"

    "$PY" phase7/split_audit_dataset.py \
      --audit "$audit" \
      --seed 20260303 \
      --calib-fraction 0.30 \
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
      --min-causal-signal-coverage 0.25 \
      --latent-high-quantile 0.80 \
      --output "$bench"
  done

  "$PY" - <<PY
import json
run_tag = "${run_tag}"
rows = []
for var in ["subresult_value", "operator"]:
    p = f"phase7_results/results/faithfulness_benchmark_{run_tag}_canary_raw_d2_{var}.json"
    d = json.load(open(p))
    bt = d.get("by_benchmark_track") or {}
    rows.append({
        "variable": var,
        "benchmark_path": p,
        "composite_auroc": (bt.get("composite") or {}).get("auroc"),
        "text_auroc": (bt.get("text_only") or {}).get("auroc"),
        "latent_auroc": (bt.get("latent_only") or {}).get("auroc"),
        "causal_auroc": (bt.get("causal_auditor") or {}).get("auroc"),
        "leakage_check_pass": d.get("leakage_check_pass"),
        "causal_signal_coverage_fraction": d.get("causal_signal_coverage_fraction"),
    })
out = {
  "schema_version": "phase6v2_d2_phase7_canary_v1",
  "run_id": "${run_id}",
  "run_tag": run_tag,
  "checkpoint": "state_raw_multi_l4_l5_l6_l7_d2tier2.pt",
  "variables": rows,
}
json.dump(out, open("${res_dir}/phase7_canary_summary_${run_tag}.json", "w"), indent=2)
PY

  echo "$(date -Is)" > "${base}/state/phase7_canary.done"
}

coordinator() {
  local run_id="$1"
  local run_tag="phase6v2_d2_operate_only_${run_id}"
  local base="phase7_results/runs/${run_id}"
  local res_dir="phase7_results/results/${run_id}"
  local ckpt_dir="phase7_results/checkpoints/${run_id}"
  mkdir -p "$base/state" "$base/meta" "$base/logs" "$res_dir"

  echo "[$(date -Is)] coordinator start run_id=${run_id}"
  while true; do
    local n_json n_ckpt
    n_json=$(ls "$res_dir"/state_decoder_supervised_*_d2tier2.json 2>/dev/null | wc -l || true)
    n_ckpt=$(ls "$ckpt_dir"/*_d2tier2.pt 2>/dev/null | wc -l || true)
    echo "[$(date -Is)] waiting train outputs json=${n_json}/3 ckpt=${n_ckpt}/3"
    if [[ "$n_json" -ge 3 && "$n_ckpt" -ge 3 ]]; then
      break
    fi
    sleep 60
  done

  for ckpt in "$ckpt_dir"/*_d2tier2.pt; do
    local name out
    name="$(basename "$ckpt" .pt)"
    out="$res_dir/state_decoder_eval_${name}.json"
    echo "[$(date -Is)] evaluating ${name}"
    CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 \
    "$PY" phase7/evaluate_state_decoders.py \
      --checkpoint "$ckpt" \
      --eval-split test \
      --device cuda \
      --output "$out"
  done

  echo "[$(date -Is)] running decoder diagnostic"
  "$PY" phase7/diagnose_decoder_quality.py \
    --eval-results-glob "$res_dir/state_decoder_eval_*.json" \
    --quality-gate-operator-operate-only-min 0.70 \
    --quality-gate-magnitude-min 0.90 \
    --quality-gate-sign-min 0.95 \
    --output-json "$res_dir/decoder_quality_diagnostic_post_d2.json" \
    --output-md "$res_dir/decoder_quality_diagnostic_post_d2.md"

  local gate_pass
  gate_pass=$("$PY" - <<PY
import json
obj=json.load(open("${res_dir}/decoder_quality_diagnostic_post_d2.json"))
print(str(bool(((obj.get("decoder_quality_gate_operate_only") or {}).get("pass")))).lower())
PY
)

  if [[ "$gate_pass" == "true" ]]; then
    echo "[$(date -Is)] operate-only gate passed -> launching phase7 canary"
    run_phase7_canary "$run_id" "$run_tag"
  else
    echo "[$(date -Is)] operate-only gate failed -> skipping phase7 canary"
    "$PY" - <<PY
import json
out={
  "schema_version":"phase6v2_d2_phase7_canary_blocked_v1",
  "run_id":"${run_id}",
  "reason":"decoder_quality_gate_operate_only_failed",
  "diagnostic_path":"${res_dir}/decoder_quality_diagnostic_post_d2.json"
}
json.dump(out, open("${res_dir}/phase7_canary_blocked_${run_tag}.json", "w"), indent=2)
PY
    echo "$(date -Is)" > "$base/state/phase7_canary.blocked"
  fi

  echo "$(date -Is)" > "$base/state/pipeline.done"
  echo "[$(date -Is)] coordinator done"
}

launch() {
  local run_id="${1:-$(mk_run_id)}"
  local base="phase7_results/runs/${run_id}"
  local prev_run="20260307_170115_phase6v2_d1_pilot_fast"
  mkdir -p "$base/meta" "$base/logs" "$base/scripts" "$base/state" "phase7_results/results/${run_id}" "phase7_results/checkpoints/${run_id}"

  "$PY" - <<PY
import json,glob,os,time
prev="${prev_run}"
out={"schema_version":"phase6v2_d2_baseline_snapshot_v1","created_at_unix":int(time.time()),"previous_run_id":prev,"files":{}}
for p in glob.glob(f"phase7_results/results/{prev}/*.json"):
    try:
        out["files"][os.path.basename(p)] = os.path.getsize(p)
    except Exception:
        pass
json.dump(out, open("${base}/meta/baseline_snapshot.json","w"), indent=2)
PY

  cp "$0" "$base/scripts/run_phase6v2_d2_operate_only.sh"

  tmux new-session -d -s "p6d2_raw_${run_id}" "cd '$REPO_ROOT' && bash experiments/run_phase6v2_d2_operate_only.sh worker-raw '${run_id}'"
  tmux new-session -d -s "p6d2_hyb_${run_id}" "cd '$REPO_ROOT' && bash experiments/run_phase6v2_d2_operate_only.sh worker-hyb '${run_id}'"
  tmux new-session -d -s "p6d2_sae_${run_id}" "cd '$REPO_ROOT' && bash experiments/run_phase6v2_d2_operate_only.sh worker-sae '${run_id}'"
  tmux new-session -d -s "p6d2_coord_${run_id}" "cd '$REPO_ROOT' && bash experiments/run_phase6v2_d2_operate_only.sh coordinator '${run_id}' |& tee '${base}/logs/coordinator.log'"

  echo "$run_id"
  tmux ls | rg "p6d2_.*${run_id}" || true
}

cmd="${1:-}"
case "$cmd" in
  launch)
    launch "${2:-}"
    ;;
  worker-raw)
    train_worker "$2" raw 5 |& tee "phase7_results/runs/$2/logs/train_raw.log"
    ;;
  worker-hyb)
    train_worker "$2" hybrid 6 |& tee "phase7_results/runs/$2/logs/train_hybrid.log"
    ;;
  worker-sae)
    train_worker "$2" sae 7 |& tee "phase7_results/runs/$2/logs/train_sae.log"
    ;;
  coordinator)
    coordinator "$2"
    ;;
  *)
    echo "usage: $0 launch [RUN_ID] | worker-raw RUN_ID | worker-hyb RUN_ID | worker-sae RUN_ID | coordinator RUN_ID" >&2
    exit 2
    ;;
esac

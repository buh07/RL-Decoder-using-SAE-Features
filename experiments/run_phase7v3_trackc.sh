#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-.venv/bin/python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7v3_trackc}"
RUN_TAG="${RUN_TAG:-phase7v3_${RUN_ID}}"
BASE="${BASE:-phase7_results/runs/${RUN_ID}}"
RESULTS_DIR="phase7_results/results"
AUDIT_DIR="phase7_results/audits"
CAL_DIR="phase7_results/calibration"
STATE_DIR="${BASE}/state"
META_DIR="${BASE}/meta"

CONTROLS="${CONTROLS:-phase7_results/controls/cot_controls_test_papercore_fixv3_matrix_v3.json}"
TRACE_DATASET="${TRACE_DATASET:-phase7_results/dataset/gsm8k_step_traces_test.pt}"
CHECKPOINT="${CHECKPOINT:-phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration/checkpoints/state_raw_every2_even.pt}"
CAUSAL_CHECKS="${CAUSAL_CHECKS:-}"
CONTROL_LATENT_CACHE="${CONTROL_LATENT_CACHE:-}"
CONTROL_RECORDS="${CONTROL_RECORDS:-}"
MODEL_KEY="${MODEL_KEY:-gpt2-medium}"
DEVICE="${DEVICE:-cpu}"
MAX_CONTROLS="${MAX_CONTROLS:-500}"
CALIB_FRAC="${CALIB_FRAC:-0.30}"
SPLIT_SEED="${SPLIT_SEED:-20260303}"
BENCH_SCOPE="${BENCH_SCOPE:-synthetic_controls}"
EXT_STATUS="${EXT_STATUS:-not_tested}"

mkdir -p "$RESULTS_DIR" "$AUDIT_DIR" "$CAL_DIR" "$STATE_DIR" "$META_DIR"

log() {
  echo "[phase7v3][${RUN_ID}] $*"
}

run_r0() {
  log "R0: archiving Track C intervention benchmarks and locking negative finding docs."
  "$PY" phase7/build_trackc_intervention_archive.py \
    --output "${RESULTS_DIR}/trackc_intervention_archive_manifest.json"
  touch "${STATE_DIR}/r0.done"
}

run_r1() {
  local audit_out="${AUDIT_DIR}/trackc_r1_audit_${RUN_TAG}.json"
  local calib_out="${AUDIT_DIR}/trackc_r1_calib_${RUN_TAG}.json"
  local eval_out="${AUDIT_DIR}/trackc_r1_eval_${RUN_TAG}.json"
  local split_manifest="${AUDIT_DIR}/trackc_r1_split_manifest_${RUN_TAG}.json"
  local thr_out="${CAL_DIR}/trackc_r1_thresholds_${RUN_TAG}.json"
  local bench_out="${RESULTS_DIR}/trackc_r1_benchmark_${RUN_TAG}.json"
  local use_variant_cache=0
  if [[ -n "$CONTROL_LATENT_CACHE" ]]; then
    if "$PY" - <<'PY' "$CONTROL_LATENT_CACHE"; then
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
payload = json.load(open(p))
rows = payload.get("rows")
if isinstance(rows, list) and rows:
    conf = (rows[0] or {}).get("latent_pred_confidence") or {}
    ok = isinstance(conf.get("operator_probs"), dict) and isinstance(conf.get("sign_probs"), dict) and isinstance(conf.get("magnitude_probs"), dict)
    raise SystemExit(0 if ok else 1)
rows_path = payload.get("rows_path")
if not rows_path:
    raise SystemExit(1)
rp = Path(rows_path)
if not rp.is_absolute():
    rp = (p.parent / rp).resolve()
if not rp.exists():
    raise SystemExit(1)
import gzip
line = None
with gzip.open(rp, "rt", encoding="utf-8") as f:
    line = f.readline().strip()
if not line:
    raise SystemExit(1)
row = json.loads(line)
conf = (row or {}).get("latent_pred_confidence") or {}
ok = isinstance(conf.get("operator_probs"), dict) and isinstance(conf.get("sign_probs"), dict) and isinstance(conf.get("magnitude_probs"), dict)
raise SystemExit(0 if ok else 1)
PY
      use_variant_cache=1
    else
      log "R1: control latent cache lacks full probability fields; falling back to shared CPU decode."
      use_variant_cache=0
    fi
  fi

  log "R1: running confidence-margin canary audit/benchmark."
  audit_cmd=(
    "$PY" phase7/causal_audit.py
    --controls "$CONTROLS"
    --trace-dataset "$TRACE_DATASET"
    --state-decoder-checkpoint "$CHECKPOINT"
    --model-key "$MODEL_KEY"
    --device "$DEVICE"
    --max-controls "$MAX_CONTROLS"
    --output "$audit_out"
  )
  if [[ -n "$CAUSAL_CHECKS" ]]; then
    audit_cmd+=(--causal-checks "$CAUSAL_CHECKS" --require-causal-mode control_conditioned)
  fi
  if [[ "$use_variant_cache" == "1" ]]; then
    audit_cmd+=(--latent-source variant_conditioned --control-latent-cache "$CONTROL_LATENT_CACHE")
  fi
  "${audit_cmd[@]}"

  "$PY" phase7/split_audit_dataset.py \
    --audit "$audit_out" \
    --seed "$SPLIT_SEED" \
    --calib-fraction "$CALIB_FRAC" \
    --output-calib "$calib_out" \
    --output-eval "$eval_out" \
    --output-manifest "$split_manifest"

  "$PY" phase7/calibrate_audit_thresholds.py \
    --audit-calib "$calib_out" \
    --all-tracks \
    --score-track confidence_margin \
    --positive-label unfaithful \
    --output "$thr_out"

  "$PY" phase7/benchmark_faithfulness.py \
    --audit-eval "$eval_out" \
    --thresholds "$thr_out" \
    --benchmark-scope "$BENCH_SCOPE" \
    --external-validity-status "$EXT_STATUS" \
    --gate-track confidence_margin \
    --confidence-text-corr-max 0.80 \
    --ablation-weights '{"text":0.5,"latent":0.5,"causal":0.0}' \
    --output "$bench_out"

  "$PY" - <<'PY' "$bench_out" "${STATE_DIR}/r1_pass.flag"
import json, sys
bench = json.load(open(sys.argv[1]))
track = (bench.get("by_benchmark_track") or {}).get("confidence_margin", {})
auroc = track.get("auroc")
metric_defined = bool((track.get("metric_defined") or {}).get("auroc", False))
corr = (
    ((bench.get("track_correlations") or {}).get("confidence_margin_vs_text_only") or {}
).get("pearson")
)
gate_checks = bench.get("gate_checks") or {}
power_ok = bool(gate_checks.get("power_sufficient", False))
non_redundant_ok = bool(gate_checks.get("confidence_non_redundant", False))
ok = bool(
    metric_defined
    and isinstance(auroc, (int, float))
    and float(auroc) > 0.55
    and isinstance(corr, (int, float))
    and abs(float(corr)) < 0.80
    and power_ok
    and non_redundant_ok
)
open(sys.argv[2], "w").write("1\n" if ok else "0\n")
print({
    "r1_pass": ok,
    "confidence_margin_auroc": auroc,
    "confidence_vs_text_corr": corr,
    "power_sufficient": power_ok,
    "confidence_non_redundant": non_redundant_ok,
})
PY
  touch "${STATE_DIR}/r1.done"
}

run_r4() {
  local out="${RESULTS_DIR}/trackc_r4_geometry_${RUN_TAG}.json"
  if [[ -z "$CONTROL_RECORDS" ]]; then
    log "R4: skipped (CONTROL_RECORDS not provided)."
    "$PY" - <<'PY' "$out"
import json, sys
json.dump({"schema_version":"phase7_representation_geometry_v1","status":"skipped","reason":"missing_control_records"}, open(sys.argv[1], "w"), indent=2)
PY
    touch "${STATE_DIR}/r4.done"
    return
  fi
  log "R4: running representation geometry diagnostics."
  "$PY" phase7/representation_geometry.py \
    --control-records "$CONTROL_RECORDS" \
    --layer 22 \
    --output "$out"
  touch "${STATE_DIR}/r4.done"
}

run_r2() {
  local eval_out="${AUDIT_DIR}/trackc_r1_eval_${RUN_TAG}.json"
  local thr_out="${CAL_DIR}/trackc_r1_thresholds_${RUN_TAG}.json"
  local bench_out="${RESULTS_DIR}/trackc_r2_benchmark_${RUN_TAG}.json"
  log "R2: running trajectory-coherence fallback benchmark."
  "$PY" phase7/benchmark_faithfulness.py \
    --audit-eval "$eval_out" \
    --thresholds "$thr_out" \
    --benchmark-scope "$BENCH_SCOPE" \
    --external-validity-status "$EXT_STATUS" \
    --gate-track trajectory_coherence \
    --confidence-text-corr-max 0.80 \
    --output "$bench_out"
  "$PY" - <<'PY' "$bench_out" "${STATE_DIR}/r2_pass.flag"
import json, sys
bench = json.load(open(sys.argv[1]))
track = (bench.get("by_benchmark_track") or {}).get("trajectory_coherence", {})
auroc = track.get("auroc")
metric_defined = bool((track.get("metric_defined") or {}).get("auroc", False))
corr = (
    ((bench.get("track_correlations") or {}).get("trajectory_coherence_vs_text_only") or {}
).get("pearson")
)
power_ok = bool((bench.get("gate_checks") or {}).get("power_sufficient", False))
ok = bool(
    metric_defined
    and isinstance(auroc, (int, float))
    and float(auroc) > 0.55
    and isinstance(corr, (int, float))
    and abs(float(corr)) < 0.80
    and power_ok
)
open(sys.argv[2], "w").write("1\n" if ok else "0\n")
print({"r2_pass": ok, "trajectory_auroc": auroc, "trajectory_vs_text_corr": corr, "power_sufficient": power_ok})
PY
  touch "${STATE_DIR}/r2.done"
}

run_r3() {
  local out="${RESULTS_DIR}/trackc_r3_probe_${RUN_TAG}.json"
  if [[ -z "$CONTROL_RECORDS" ]]; then
    log "R3: cannot run probe fallback without CONTROL_RECORDS."
    "$PY" - <<'PY' "$out"
import json, sys
json.dump({"schema_version":"phase7_contrastive_probe_v1","status":"blocked","reason":"missing_control_records"}, open(sys.argv[1], "w"), indent=2)
PY
    touch "${STATE_DIR}/r3.done"
    return
  fi
  log "R3: running contrastive probe fallback."
  "$PY" phase7/contrastive_faithfulness_probe.py \
    --control-records "$CONTROL_RECORDS" \
    --layers 7,12,17,22 \
    --device cpu \
    --output "$out"
  touch "${STATE_DIR}/r3.done"
}

run_r5() {
  local decision_out="${RESULTS_DIR}/trackc_phase7v3_decision_${RUN_TAG}.json"
  log "R5: writing final decision artifact."
  "$PY" - <<'PY' "$decision_out" "$RUN_ID" "$RUN_TAG" "$STATE_DIR" "$RESULTS_DIR"
import json, sys
from pathlib import Path

decision_out, run_id, run_tag, state_dir, results_dir = sys.argv[1:6]
state = Path(state_dir)
results = Path(results_dir)

def _read_flag(name):
    p = state / name
    if not p.exists():
        return None
    return p.read_text().strip() == "1"

r1_pass = _read_flag("r1_pass.flag")
r2_pass = _read_flag("r2_pass.flag")

selected = "two_track_negative_result"
if r1_pass is True:
    selected = "confidence_margin"
elif r2_pass is True:
    selected = "trajectory_coherence"
else:
    r3p = results / f"trackc_r3_probe_{run_tag}.json"
    if r3p.exists():
        pr = json.load(open(r3p))
        if bool(pr.get("cross_variant_generalization_pass")):
            selected = "contrastive_probe"

payload = {
    "schema_version": "phase7_trackc_decision_v3",
    "run_id": run_id,
    "run_tag": run_tag,
    "stage_status": {
        "r0_done": (state / "r0.done").exists(),
        "r1_done": (state / "r1.done").exists(),
        "r2_done": (state / "r2.done").exists(),
        "r3_done": (state / "r3.done").exists(),
        "r4_done": (state / "r4.done").exists(),
    },
    "selected_track_c": selected,
    "claim_boundary": (
        "If selected_track_c=two_track_negative_result, Track C remains unresolved for GPT-2 arithmetic "
        "and deployment defaults to text+latent composite."
    ),
}
json.dump(payload, open(decision_out, "w"), indent=2)
print(payload)
PY
  touch "${STATE_DIR}/pipeline.done"
}

main() {
  run_r0

  # R1 and R4 can run in parallel by design.
  run_r4 &
  pid_r4=$!
  run_r1
  wait "$pid_r4"

  if [[ "$(cat "${STATE_DIR}/r1_pass.flag")" == "1" ]]; then
    log "R1 gate passed; skipping R2/R3 and finalizing."
    run_r5
    exit 0
  fi

  run_r2
  if [[ "$(cat "${STATE_DIR}/r2_pass.flag")" == "1" ]]; then
    log "R2 gate passed; skipping R3 and finalizing."
    run_r5
    exit 0
  fi

  run_r3
  run_r5
}

main "$@"

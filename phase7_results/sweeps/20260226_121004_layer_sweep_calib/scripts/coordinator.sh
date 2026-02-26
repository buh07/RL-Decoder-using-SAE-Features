#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO="/scratch2/f004ndc/RL-Decoder with SAE Features"
RUN_ID="20260226_121004_layer_sweep_calib"
MANIFEST="experiments/layer_sweep_manifest_v1.json"
P6_CAL="phase6_results/sweeps/20260226_121004_layer_sweep_calib/calibration"
RUN_BASE="phase7_results/sweeps/20260226_121004_layer_sweep_calib"
P7_CAL="phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration"
STATE_DIR="phase7_results/sweeps/20260226_121004_layer_sweep_calib/state"
LOG_DIR="phase7_results/sweeps/20260226_121004_layer_sweep_calib/logs"
cd "$REPO"
exec > >(tee -a "$LOG_DIR/coordinator.log") 2>&1
trap 'echo "[ERROR] coordinator failed at $(date -Is)"; touch "$STATE_DIR/coordinator.failed"' ERR
echo "[START] coordinator $(date -Is) run_id=$RUN_ID"
for marker in task0_presweep_freeze.done task2_phase7_trace_dataset.done task1_phase6_calibration.done task1_phase7_calibration.done task1_phase7_causal_dry_run.done; do
  echo "[WAIT] $marker"
  while [ ! -f "$STATE_DIR/$marker" ]; do
    if ls "$STATE_DIR"/*.failed >/dev/null 2>&1; then
      echo "[ABORT] detected failure marker in $STATE_DIR"; exit 1
    fi
    sleep 20
  done
  echo "[SEEN] $marker $(date -Is)"
done
echo "[RUN] phase6 calibration sweep summary"
.venv/bin/python3 experiments/summarize_phase6_layer_sweep.py \
  --results-dir "$P6_CAL/results" \
  --manifest "$MANIFEST" \
  --output "$P6_CAL/results/layer_sweep_phase6_summary.json"
echo "[RUN] phase7 calibration sweep summary"
.venv/bin/python3 experiments/summarize_phase7_layer_sweep.py \
  --results-dir "$P7_CAL/results" \
  --manifest "$MANIFEST" \
  --output "$P7_CAL/results/layer_sweep_phase7_summary.json"
python - <<PY
import json
from pathlib import Path
run_id = "$RUN_ID"
p6_worker = json.loads(Path("$P6_CAL/results/phase6_calibration_worker_summary.json").read_text())
p7_worker = json.loads(Path("$P7_CAL/results/phase7_calibration_worker_summary.json").read_text())
p6_sweep = json.loads(Path("$P6_CAL/results/layer_sweep_phase6_summary.json").read_text())
p7_sweep = json.loads(Path("$P7_CAL/results/layer_sweep_phase7_summary.json").read_text())
causal_tp = json.loads(Path("phase7_results/interventions/" + run_id + "_calibration_causal_throughput.json").read_text())
phase6_summary = {
  "schema_version": "phase6_calibration_summary_v1",
  "run_id": run_id,
  "task": "calibration_mini_sweep",
  "worker": p6_worker,
  "num_eval_rows": int(p6_sweep.get("num_rows", 0)),
  "results_dir": "$P6_CAL/results",
  "sweep_summary_path": "$P6_CAL/results/layer_sweep_phase6_summary.json",
  "top_configs_overall": [r.get("config_name") for r in (p6_sweep.get("rankings", {}).get("overall_test_top1", [])[:5])],
}
phase7_summary = {
  "schema_version": "phase7_calibration_summary_v1",
  "run_id": run_id,
  "task": "calibration_mini_sweep",
  "worker": p7_worker,
  "num_eval_rows": int(p7_sweep.get("num_rows", 0)),
  "results_dir": "$P7_CAL/results",
  "sweep_summary_path": "$P7_CAL/results/layer_sweep_phase7_summary.json",
  "top_configs_overall": [r.get("config_name") for r in (p7_sweep.get("rankings", {}).get("overall_val_composite", [])[:5])],
  "causal_throughput_path": "phase7_results/interventions/" + run_id + "_calibration_causal_throughput.json",
  "causal_records_per_hour": causal_tp.get("records_per_hour"),
  "causal_layer_checks_per_hour": causal_tp.get("layer_checks_per_hour"),
}
Path(f"phase6_results/results/{run_id}_calibration_summary.json").write_text(json.dumps(phase6_summary, indent=2))
Path(f"phase7_results/results/{run_id}_calibration_summary.json").write_text(json.dumps(phase7_summary, indent=2))
print("Wrote canonical calibration summaries")
PY
echo "[DONE] coordinator $(date -Is)"
touch "$STATE_DIR/task1_calibration.done"
touch "$STATE_DIR/pipeline.done"

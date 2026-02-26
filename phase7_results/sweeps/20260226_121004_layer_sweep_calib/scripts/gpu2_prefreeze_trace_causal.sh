#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO="/scratch2/f004ndc/RL-Decoder with SAE Features"
RUN_ID="20260226_121004_layer_sweep_calib"
MANIFEST="experiments/layer_sweep_manifest_v1.json"
RUN_BASE="phase7_results/sweeps/20260226_121004_layer_sweep_calib"
P7_CAL="phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration"
STATE_DIR="phase7_results/sweeps/20260226_121004_layer_sweep_calib/state"
cd "$REPO"
exec > >(tee -a "$P7_CAL/logs/worker_gpu2_prefreeze_trace_causal.log") 2>&1
trap 'echo "[ERROR] gpu2 worker failed at $(date -Is)"; touch "$STATE_DIR/task2_phase7_trace_dataset.failed"; touch "$STATE_DIR/task1_phase7_causal_dry_run.failed"' ERR
# Task 0: pre-sweep freeze
mkdir -p "$RUN_BASE/run_config"
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
cat > "$RUN_BASE/run_config/presweep_freeze.json" <<JSON
{
  "schema_version": "layer_sweep_presweep_freeze_v1",
  "sweep_run_id": "$RUN_ID",
  "created_at": "$(date -Is)",
  "claim_boundary": "causally supported under measured variables/subspaces and tested interventions; not a complete explanation.",
  "benchmark_tracks": ["text_only", "latent_only", "causal_auditor"],
  "paper_core_failure_families": [
    "prompt_bias_rationalization",
    "silent_error_correction",
    "answer_first_order_flip",
    "shortcut_rationalization"
  ],
  "manifest_path": "$MANIFEST",
  "manifest_sha256": "$MANIFEST_SHA",
  "calibration_layer_sets": ["single_l22", "spread4_current", "middle12_06_17", "every2_even"],
  "task_scope": ["task0_pre_sweep_freeze", "task1_calibration_mini_sweep", "task2_phase7_trace_dataset_build"]
}
JSON
echo "[DONE] Task0 pre-sweep freeze $(date -Is)"
touch "$STATE_DIR/task0_presweep_freeze.done"
# Task 2: full Phase7 trace dataset build (skip if already present and valid)
TRACE_TRAIN="phase7_results/dataset/gsm8k_step_traces_train.pt"
TRACE_TEST="phase7_results/dataset/gsm8k_step_traces_test.pt"
TRACE_SUMMARY="phase7_results/dataset/build_summary.json"
if [ -f "$TRACE_TRAIN" ] && [ -f "$TRACE_TEST" ] && [ -f "$TRACE_SUMMARY" ]; then
  echo "[INFO] Phase7 trace dataset already exists; validating and reusing"
else
  echo "[RUN] Building Phase7 trace dataset $(date -Is)"
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python3 phase7/build_step_trace_dataset.py --output-dir phase7_results/dataset
fi
python - <<PY
import json, torch
from pathlib import Path
summary = json.loads(Path("$TRACE_SUMMARY").read_text())
assert summary.get("schema_version") == "phase7_trace_v1", summary.get("schema_version")
for p in ["$TRACE_TRAIN", "$TRACE_TEST"]:
    rows = torch.load(p, weights_only=False)
    assert isinstance(rows, list) and len(rows) > 0, p
    assert rows[0].get("schema_version") == "phase7_trace_v1", rows[0].get("schema_version")
print("Trace dataset validation OK")
PY
echo "[DONE] Task2 Phase7 trace dataset build $(date -Is)"
touch "$STATE_DIR/task2_phase7_trace_dataset.done"
# Task 1 (part): causal dry run throughput on 50 records
SUBSPACE_SPECS="$P7_CAL/interventions/variable_subspaces_calibration.json"
SAL="phase7_results/interp_smoke/grad_sae_saliency_state_sae_multi.json"
mkdir -p "$P7_CAL/interventions"
if [ -f "$SAL" ]; then
  echo "[RUN] Building calibration subspace specs with probe+saliency union"
  .venv/bin/python3 phase7/variable_subspace_builder.py \
    --decoder-saliency "$SAL" \
    --probe-position result \
    --layers 22 \
    --variables subresult_value \
    --top-k 64 \
    --combine-policy union \
    --output "$SUBSPACE_SPECS"
else
  echo "[RUN] Building calibration subspace specs with probe-only (saliency missing)"
  .venv/bin/python3 phase7/variable_subspace_builder.py \
    --probe-position result \
    --layers 22 \
    --variables subresult_value \
    --top-k 64 \
    --combine-policy probe_only \
    --output "$SUBSPACE_SPECS"
fi
CAUSAL_OUT="$P7_CAL/interventions/causal_checks_calibration_subresult_l22_50.json"
CAUSAL_START=$(date +%s)
echo "[RUN] Phase7 causal dry run (50 records, subresult_value, layer 22) $(date -Is)"
CUDA_VISIBLE_DEVICES=2 .venv/bin/python3 phase7/causal_intervention_engine.py \
  --trace-dataset phase7_results/dataset/gsm8k_step_traces_test.pt \
  --subspace-specs "$SUBSPACE_SPECS" \
  --variable subresult_value \
  --layers 22 \
  --max-records 50 \
  --device cuda:0 \
  --output "$CAUSAL_OUT"
CAUSAL_END=$(date +%s)
python - <<PY
import json
from pathlib import Path
out_path = Path("$CAUSAL_OUT")
d = json.loads(out_path.read_text())
rows = d.get("rows", [])
layer_checks = 0
for r in rows:
    layer_checks += len((r.get("layers") or {}).keys())
elapsed = int($CAUSAL_END - $CAUSAL_START)
throughput = {
  "schema_version": "phase7_calibration_causal_throughput_v1",
  "run_id": "$RUN_ID",
  "task": "phase7_causal_dry_run",
  "variable": "subresult_value",
  "layers": [22],
  "max_records": 50,
  "records_processed": len(rows),
  "layer_checks": int(layer_checks),
  "start_ts": int($CAUSAL_START),
  "end_ts": int($CAUSAL_END),
  "elapsed_sec": elapsed,
  "records_per_hour": (len(rows) * 3600.0 / elapsed) if elapsed > 0 else None,
  "layer_checks_per_hour": (layer_checks * 3600.0 / elapsed) if elapsed > 0 else None,
  "causal_checks_path": str(out_path),
  "subspace_specs_path": "$SUBSPACE_SPECS"
}
run_scoped = Path("$P7_CAL/interventions/calibration_causal_throughput.json")
canonical = Path("phase7_results/interventions/$RUN_ID" + "_calibration_causal_throughput.json")
run_scoped.write_text(json.dumps(throughput, indent=2))
canonical.write_text(json.dumps(throughput, indent=2))
print("Wrote throughput summaries:", run_scoped, canonical)
PY
echo "[DONE] Task1 Phase7 causal dry run $(date -Is) elapsed=$((CAUSAL_END-CAUSAL_START))s"
touch "$STATE_DIR/task1_phase7_causal_dry_run.done"

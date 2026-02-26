#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO="/scratch2/f004ndc/RL-Decoder with SAE Features"
RUN_ID="20260226_121004_layer_sweep_calib"
MANIFEST="experiments/layer_sweep_manifest_v1.json"
P7_CAL="phase7_results/sweeps/20260226_121004_layer_sweep_calib/calibration"
STATE_DIR="phase7_results/sweeps/20260226_121004_layer_sweep_calib/state"
cd "$REPO"
exec > >(tee -a "$P7_CAL/logs/worker_gpu1_phase7_calibration.log") 2>&1
trap 'echo "[ERROR] gpu1 worker failed at $(date -Is)"; touch "$STATE_DIR/task1_phase7_calibration.failed"' ERR
LAYER_SETS=(single_l22 spread4_current middle12_06_17 every2_even)
VARIANTS=(raw hybrid)
START_TS=$(date +%s)
echo "[WAIT] Waiting for Phase7 trace dataset marker... $(date -Is)"
while [ ! -f "$STATE_DIR/task2_phase7_trace_dataset.done" ]; do
  if [ -f "$STATE_DIR/task2_phase7_trace_dataset.failed" ]; then
    echo "[ABORT] Trace dataset build failed"; exit 1
  fi
  sleep 15
done
echo "[START] GPU1 Phase7 calibration $(date -Is) run_id=$RUN_ID"
touch "$STATE_DIR/task1_phase7_calibration.started"
for layer_set in "${LAYER_SETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    cfg="state_${variant}_${layer_set}"
    echo "[RUN] phase7 train cfg=$cfg $(date -Is)"
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 phase7/train_state_decoders.py \
      --dataset-train phase7_results/dataset/gsm8k_step_traces_train.pt \
      --manifest "$MANIFEST" \
      --layer-set-id "$layer_set" \
      --input-variant "$variant" \
      --sweep-run-id "$RUN_ID" \
      --parent-baseline spread4_current \
      --checkpoints-dir "$P7_CAL/checkpoints" \
      --results-dir "$P7_CAL/results" \
      --device cuda:0
    echo "[RUN] phase7 eval cfg=$cfg $(date -Is)"
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 phase7/evaluate_state_decoders.py \
      --checkpoint "$P7_CAL/checkpoints/${cfg}.pt" \
      --dataset-train phase7_results/dataset/gsm8k_step_traces_train.pt \
      --dataset-test phase7_results/dataset/gsm8k_step_traces_test.pt \
      --device cuda:0 \
      --manifest "$MANIFEST" \
      --sweep-run-id "$RUN_ID" \
      --parent-baseline spread4_current \
      --output "$P7_CAL/results/state_decoder_eval_${cfg}.json"
    touch "$P7_CAL/state/${cfg}.done"
  done
done
END_TS=$(date +%s)
python - <<PY
import json
from pathlib import Path
out = {
  "schema_version": "phase7_calibration_worker_summary_v1",
  "run_id": "$RUN_ID",
  "worker": "gpu1",
  "phase": "phase7",
  "start_ts": int($START_TS),
  "end_ts": int($END_TS),
  "elapsed_sec": int($END_TS - $START_TS),
  "layer_sets": ["single_l22", "spread4_current", "middle12_06_17", "every2_even"],
  "variants": ["raw", "hybrid"],
  "num_configs": 8,
  "checkpoints_dir": "$P7_CAL/checkpoints",
  "results_dir": "$P7_CAL/results",
}
Path("$P7_CAL/results/phase7_calibration_worker_summary.json").write_text(json.dumps(out, indent=2))
PY
echo "[DONE] GPU1 Phase7 calibration $(date -Is) elapsed=$((END_TS-START_TS))s"
touch "$STATE_DIR/task1_phase7_calibration.done"

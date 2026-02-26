#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO="/scratch2/f004ndc/RL-Decoder with SAE Features"
RUN_ID="20260226_121004_layer_sweep_calib"
MANIFEST="experiments/layer_sweep_manifest_v1.json"
P6_CAL="phase6_results/sweeps/20260226_121004_layer_sweep_calib/calibration"
STATE_DIR="phase7_results/sweeps/20260226_121004_layer_sweep_calib/state"
cd "$REPO"
exec > >(tee -a "$P6_CAL/logs/worker_gpu0_phase6_calibration.log") 2>&1
trap 'echo "[ERROR] gpu0 worker failed at $(date -Is)"; touch "$STATE_DIR/task1_phase6_calibration.failed"' ERR
LAYER_SETS=(single_l22 spread4_current middle12_06_17 every2_even)
VARIANTS=(raw hybrid)
START_TS=$(date +%s)
echo "[START] GPU0 Phase6 calibration $(date -Is) run_id=$RUN_ID"
touch "$STATE_DIR/task1_phase6_calibration.started"
for layer_set in "${LAYER_SETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    cfg="${variant}_${layer_set}"
    echo "[RUN] phase6 train cfg=$cfg layer_set=$layer_set variant=$variant $(date -Is)"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 phase6/train_supervised.py \
      --dataset-train phase6_results/dataset/gsm8k_expanded_train.pt \
      --manifest "$MANIFEST" \
      --layer-set-id "$layer_set" \
      --input-variant "$variant" \
      --sweep-run-id "$RUN_ID" \
      --parent-baseline spread4_current \
      --checkpoints-dir "$P6_CAL/checkpoints" \
      --results-dir "$P6_CAL/results" \
      --device cuda:0
    echo "[RUN] phase6 eval cfg=$cfg $(date -Is)"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 phase6/evaluate_decoder.py \
      --checkpoint "$P6_CAL/checkpoints/${cfg}_supervised.pt" \
      --dataset-train phase6_results/dataset/gsm8k_expanded_train.pt \
      --dataset-test phase6_results/dataset/gsm8k_expanded_test.pt \
      --device cuda:0 \
      --manifest "$MANIFEST" \
      --sweep-run-id "$RUN_ID" \
      --parent-baseline spread4_current \
      --output "$P6_CAL/results/eval_${cfg}.json"
    touch "$P6_CAL/state/${cfg}.done"
  done
done
END_TS=$(date +%s)
python - <<PY
import json
from pathlib import Path
out = {
  "schema_version": "phase6_calibration_worker_summary_v1",
  "run_id": "$RUN_ID",
  "worker": "gpu0",
  "phase": "phase6",
  "start_ts": int($START_TS),
  "end_ts": int($END_TS),
  "elapsed_sec": int($END_TS - $START_TS),
  "layer_sets": ["single_l22", "spread4_current", "middle12_06_17", "every2_even"],
  "variants": ["raw", "hybrid"],
  "num_configs": 8,
  "checkpoints_dir": "$P6_CAL/checkpoints",
  "results_dir": "$P6_CAL/results",
}
Path("$P6_CAL/results/phase6_calibration_worker_summary.json").write_text(json.dumps(out, indent=2))
PY
echo "[DONE] GPU0 Phase6 calibration $(date -Is) elapsed=$((END_TS-START_TS))s"
touch "$STATE_DIR/task1_phase6_calibration.done"

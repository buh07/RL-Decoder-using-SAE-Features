#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
if [[ "$MODE" != "launch" ]]; then
  echo "usage: $0 launch" >&2
  exit 2
fi

PY="${PYTHON:-.venv/bin/python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_sae_trajectory_pathc_robust}"
RUN_TAG="${RUN_TAG:-phase7_sae_trajectory_pathc_robust_${RUN_ID}}"
BASE="${BASE:-phase7_results/runs/${RUN_ID}}"

CONTROL_RECORDS="${CONTROL_RECORDS:-phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json}"
SAES_DIR="${SAES_DIR:-phase2_results/saes_gpt2_12x_topk/saes}"
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
PHASE4_TOP_FEATURES="${PHASE4_TOP_FEATURES:-phase4_results/topk/probe/top_features_per_layer.json}"
DIVERGENT_SOURCE="${DIVERGENT_SOURCE:-phase7_results/results/phase7_sae_feature_discrimination_phase7_sae_20260306_224419_phase7_sae.json}"
SUBSPACE_SPECS="${SUBSPACE_SPECS:-phase7_results/interventions/variable_subspaces_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json}"

FEATURE_SET="${FEATURE_SET:-eq_pre_result_150}"
SAMPLE_TRACES="${SAMPLE_TRACES:-0}"
MIN_COMMON_STEPS="${MIN_COMMON_STEPS:-3}"
SEED="${SEED:-20260306}"
N_BOOTSTRAP="${N_BOOTSTRAP:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"

TRACE_TEST_FRACTION="${TRACE_TEST_FRACTION:-0.20}"
TRACE_SPLIT_SEED="${TRACE_SPLIT_SEED:-20260306}"
PROBE_EPOCHS="${PROBE_EPOCHS:-500}"
PROBE_LR="${PROBE_LR:-0.03}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.0001}"
PROBE_DEVICE="${PROBE_DEVICE:-cpu}"
TRAIN_EXCLUDE_VARIANTS="${TRAIN_EXCLUDE_VARIANTS:-order_flip_only,answer_first_order_flip,reordered_steps}"
REQUIRE_WRONG_INTERMEDIATE_AUROC="${REQUIRE_WRONG_INTERMEDIATE_AUROC:-0.70}"
WRONG_INTERMEDIATE_BOOTSTRAP_N="${WRONG_INTERMEDIATE_BOOTSTRAP_N:-1000}"
WRONG_INTERMEDIATE_BOOTSTRAP_SEED="${WRONG_INTERMEDIATE_BOOTSTRAP_SEED:-20260307}"
CV_FOLDS="${CV_FOLDS:-5}"
CV_SEED="${CV_SEED:-20260307}"
CV_MIN_VALID_FOLDS="${CV_MIN_VALID_FOLDS:-3}"

mkdir -p "$BASE/logs" "$BASE/meta" "$BASE/state" phase7_results/results

CORE_RUN_ID="${RUN_ID}_core"
CORE_RUN_TAG="phase7_sae_trajectory_${CORE_RUN_ID}"
CORE_BASE="phase7_results/runs/${CORE_RUN_ID}"

PARTIAL_G5="phase7_results/results/phase7_sae_trajectory_coherence_${CORE_RUN_TAG}_layer4.json"
PARTIAL_G6="phase7_results/results/phase7_sae_trajectory_coherence_${CORE_RUN_TAG}_layer7.json"
PARTIAL_G7="phase7_results/results/phase7_sae_trajectory_coherence_${CORE_RUN_TAG}_layer22.json"
CORE_MERGED_JSON="phase7_results/results/phase7_sae_trajectory_coherence_${CORE_RUN_TAG}.json"
CORE_MERGED_MD="phase7_results/results/phase7_sae_trajectory_coherence_${CORE_RUN_TAG}.md"

OUT_JSON="phase7_results/results/phase7_sae_trajectory_pathc_robust_${RUN_ID}.json"
OUT_MD="phase7_results/results/phase7_sae_trajectory_pathc_robust_${RUN_ID}.md"

cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
CORE_RUN_ID=$CORE_RUN_ID
CORE_RUN_TAG=$CORE_RUN_TAG
CORE_BASE=$CORE_BASE
CONTROL_RECORDS=$CONTROL_RECORDS
SAES_DIR=$SAES_DIR
ACTIVATIONS_DIR=$ACTIVATIONS_DIR
PHASE4_TOP_FEATURES=$PHASE4_TOP_FEATURES
DIVERGENT_SOURCE=$DIVERGENT_SOURCE
SUBSPACE_SPECS=$SUBSPACE_SPECS
FEATURE_SET=$FEATURE_SET
SAMPLE_TRACES=$SAMPLE_TRACES
MIN_COMMON_STEPS=$MIN_COMMON_STEPS
SEED=$SEED
N_BOOTSTRAP=$N_BOOTSTRAP
BATCH_SIZE=$BATCH_SIZE
TRACE_TEST_FRACTION=$TRACE_TEST_FRACTION
TRACE_SPLIT_SEED=$TRACE_SPLIT_SEED
PROBE_EPOCHS=$PROBE_EPOCHS
PROBE_LR=$PROBE_LR
PROBE_WEIGHT_DECAY=$PROBE_WEIGHT_DECAY
PROBE_DEVICE=$PROBE_DEVICE
TRAIN_EXCLUDE_VARIANTS=$TRAIN_EXCLUDE_VARIANTS
REQUIRE_WRONG_INTERMEDIATE_AUROC=$REQUIRE_WRONG_INTERMEDIATE_AUROC
WRONG_INTERMEDIATE_BOOTSTRAP_N=$WRONG_INTERMEDIATE_BOOTSTRAP_N
WRONG_INTERMEDIATE_BOOTSTRAP_SEED=$WRONG_INTERMEDIATE_BOOTSTRAP_SEED
CV_FOLDS=$CV_FOLDS
CV_SEED=$CV_SEED
CV_MIN_VALID_FOLDS=$CV_MIN_VALID_FOLDS
PARTIAL_G5=$PARTIAL_G5
PARTIAL_G6=$PARTIAL_G6
PARTIAL_G7=$PARTIAL_G7
CORE_MERGED_JSON=$CORE_MERGED_JSON
CORE_MERGED_MD=$CORE_MERGED_MD
OUT_JSON=$OUT_JSON
OUT_MD=$OUT_MD
CFG

{
  echo "[$(date -Is)] launching Path C robust core run on GPUs 5/6/7"
  RUN_ID="$CORE_RUN_ID" \
  RUN_TAG="$CORE_RUN_TAG" \
  BASE="$CORE_BASE" \
  CONTROL_RECORDS="$CONTROL_RECORDS" \
  SAES_DIR="$SAES_DIR" \
  ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
  PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
  DIVERGENT_SOURCE="$DIVERGENT_SOURCE" \
  SUBSPACE_SPECS="$SUBSPACE_SPECS" \
  FEATURE_SET="$FEATURE_SET" \
  SAMPLE_TRACES="$SAMPLE_TRACES" \
  MIN_COMMON_STEPS="$MIN_COMMON_STEPS" \
  SEED="$SEED" \
  N_BOOTSTRAP="$N_BOOTSTRAP" \
  BATCH_SIZE="$BATCH_SIZE" \
  EMIT_SAMPLES=1 \
  ./experiments/run_phase7_sae_trajectory_coherence.sh launch

  while [[ ! -f "$CORE_BASE/state/pipeline.done" ]]; do
    if ! tmux has-session -t "p7saetc_coord_${CORE_RUN_ID}" 2>/dev/null; then
      echo "core coordinator exited before pipeline.done" >&2
      exit 1
    fi
    sleep 15
  done

  [[ -f "$PARTIAL_G5" ]] || { echo "missing partial $PARTIAL_G5" >&2; exit 1; }
  [[ -f "$PARTIAL_G6" ]] || { echo "missing partial $PARTIAL_G6" >&2; exit 1; }
  [[ -f "$PARTIAL_G7" ]] || { echo "missing partial $PARTIAL_G7" >&2; exit 1; }

  echo "[$(date -Is)] running Path C robust aggregator"
  "$PY" phase7/aggregate_sae_trajectory_pathc.py \
    --partials "$PARTIAL_G5" "$PARTIAL_G6" "$PARTIAL_G7" \
    --output-json "$OUT_JSON" \
    --output-md "$OUT_MD" \
    --run-tag "$RUN_TAG" \
    --trace-test-fraction "$TRACE_TEST_FRACTION" \
    --trace-split-seed "$TRACE_SPLIT_SEED" \
    --epochs "$PROBE_EPOCHS" \
    --lr "$PROBE_LR" \
    --weight-decay "$PROBE_WEIGHT_DECAY" \
    --device "$PROBE_DEVICE" \
    --train-exclude-variants "$TRAIN_EXCLUDE_VARIANTS" \
    --require-wrong-intermediate-auroc "$REQUIRE_WRONG_INTERMEDIATE_AUROC" \
    --wrong-intermediate-bootstrap-n "$WRONG_INTERMEDIATE_BOOTSTRAP_N" \
    --wrong-intermediate-bootstrap-seed "$WRONG_INTERMEDIATE_BOOTSTRAP_SEED" \
    --cv-folds "$CV_FOLDS" \
    --cv-seed "$CV_SEED" \
    --cv-min-valid-folds "$CV_MIN_VALID_FOLDS"

  "$PY" - <<PY
import json
from pathlib import Path
base = Path(${BASE@Q})
summary = {
  "schema_version": "phase7_sae_trajectory_pathc_robust_pipeline_v1",
  "run_id": ${RUN_ID@Q},
  "run_tag": ${RUN_TAG@Q},
  "core_run_id": ${CORE_RUN_ID@Q},
  "feature_set": ${FEATURE_SET@Q},
  "train_exclude_variants": ${TRAIN_EXCLUDE_VARIANTS@Q},
  "require_wrong_intermediate_auroc": float(${REQUIRE_WRONG_INTERMEDIATE_AUROC@Q}),
  "wrong_intermediate_bootstrap_n": int(${WRONG_INTERMEDIATE_BOOTSTRAP_N@Q}),
  "wrong_intermediate_bootstrap_seed": int(${WRONG_INTERMEDIATE_BOOTSTRAP_SEED@Q}),
  "cv_folds": int(${CV_FOLDS@Q}),
  "cv_seed": int(${CV_SEED@Q}),
  "cv_min_valid_folds": int(${CV_MIN_VALID_FOLDS@Q}),
  "partials": [${PARTIAL_G5@Q}, ${PARTIAL_G6@Q}, ${PARTIAL_G7@Q}],
  "core_merged_json": ${CORE_MERGED_JSON@Q},
  "pathc_robust_json": ${OUT_JSON@Q},
}
(base / "meta" / "summary.json").write_text(json.dumps(summary, indent=2))
(base / "state" / "pipeline.done").write_text("done\\n")
print("pipeline summary written")
PY

  echo "[$(date -Is)] Path C robust complete"
  echo "result json: $OUT_JSON"
} | tee -a "$BASE/logs/pathc_robust.log"

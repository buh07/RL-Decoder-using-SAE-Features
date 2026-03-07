#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "usage: $0 {launch|worker|coordinator}" >&2
  exit 2
fi

PY="${PYTHON:-.venv/bin/python3}"

_run_worker() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
  : "${GPU_ID:?GPU_ID required}"
  : "${CONTROL_RECORDS:?CONTROL_RECORDS required}"
  : "${MODEL_KEY:?MODEL_KEY required}"
  : "${LAYERS:?LAYERS required}"
  : "${FEATURE_SET:?FEATURE_SET required}"
  : "${OUT_FEATURES:?OUT_FEATURES required}"
  : "${OUT_JSON:?OUT_JSON required}"
  : "${OUT_MD:?OUT_MD required}"

  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local log="$BASE/logs/worker.log"

  {
    echo "[$(date -Is)] mixed worker start run_id=$RUN_ID model_key=$MODEL_KEY gpu=$GPU_ID"

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" phase7/mixed_trajectory_feature_builder.py \
      --control-records "$CONTROL_RECORDS" \
      --model-key "$MODEL_KEY" \
      --layers "$LAYERS" \
      --saes-dir "$SAES_DIR" \
      --activations-dir "$ACTIVATIONS_DIR" \
      --phase4-top-features "$PHASE4_TOP_FEATURES" \
      --feature-set "$FEATURE_SET" \
      --divergent-source "$DIVERGENT_SOURCE" \
      --subspace-specs "$SUBSPACE_SPECS" \
      --sample-traces "$SAMPLE_TRACES" \
      --min-common-steps "$MIN_COMMON_STEPS" \
      --seed "$SEED" \
      --batch-size "$BATCH_SIZE" \
      --device cuda:0 \
      --run-tag "$RUN_TAG" \
      --output "$OUT_FEATURES"

    "$PY" phase7/evaluate_mixed_trajectory_validation.py \
      --features "$OUT_FEATURES" \
      --output-json "$OUT_JSON" \
      --output-md "$OUT_MD" \
      --run-tag "$RUN_TAG" \
      --train-exclude-variants "$TRAIN_EXCLUDE_VARIANTS" \
      --trace-test-fraction "$TRACE_TEST_FRACTION" \
      --trace-split-seed "$TRACE_SPLIT_SEED" \
      --cv-folds "$CV_FOLDS" \
      --cv-seed "$CV_SEED" \
      --cv-min-valid-folds "$CV_MIN_VALID_FOLDS" \
      --bootstrap-n "$BOOTSTRAP_N" \
      --require-wrong-intermediate-auroc "$REQUIRE_WRONG_INTERMEDIATE_AUROC" \
      --train-seed "$TRAIN_SEED"

    touch "$BASE/state/worker.done"
    echo "[$(date -Is)] mixed worker done"
  } >>"$log" 2>&1
}

_run_coordinator() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
  : "${OUT_FEATURES:?OUT_FEATURES required}"
  : "${OUT_JSON:?OUT_JSON required}"
  : "${OUT_MD:?OUT_MD required}"

  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local log="$BASE/logs/coordinator.log"

  {
    echo "[$(date -Is)] mixed coordinator start run_id=$RUN_ID"
    while [[ ! -f "$BASE/state/worker.done" ]]; do
      sleep 5
    done

    if [[ ! -f "$OUT_FEATURES" || ! -f "$OUT_JSON" || ! -f "$OUT_MD" ]]; then
      echo "missing required output artifact" >&2
      exit 1
    fi

    "$PY" - <<PY
import json
from pathlib import Path
base = Path(${BASE@Q})
out_features = Path(${OUT_FEATURES@Q})
out_json = Path(${OUT_JSON@Q})
out_md = Path(${OUT_MD@Q})
summary = {
  "schema_version": "phase7_mixed_trajectory_validation_pipeline_v1",
  "run_id": ${RUN_ID@Q},
  "run_tag": ${RUN_TAG@Q},
  "outputs": {
    "features": str(out_features),
    "evaluation_json": str(out_json),
    "evaluation_md": str(out_md),
  },
  "exists": {
    "features": out_features.exists(),
    "evaluation_json": out_json.exists(),
    "evaluation_md": out_md.exists(),
  },
}
(base / "meta" / "summary.json").write_text(json.dumps(summary, indent=2))
(base / "state" / "pipeline.done").write_text("done\\n")
print("mixed validation summary written")
PY

    echo "[$(date -Is)] mixed coordinator done"
  } >>"$log" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_mixed_trajectory_validation}"
    RUN_TAG="${RUN_TAG:-phase7_mixed_trajectory_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"

    GPU_ID="${GPU_ID:-3}"
    MODEL_KEY="${MODEL_KEY:-qwen2.5-7b}"
    CONTROL_RECORDS="${CONTROL_RECORDS:-}"

    LAYERS="${LAYERS:-4,7,22}"
    FEATURE_SET="${FEATURE_SET:-eq_pre_result_150}"
    SAES_DIR="${SAES_DIR:-phase2_results/saes_qwen25_7b_12x_topk/saes}"
    ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
    PHASE4_TOP_FEATURES="${PHASE4_TOP_FEATURES:-phase4_results/topk/probe/top_features_per_layer.json}"
    DIVERGENT_SOURCE="${DIVERGENT_SOURCE:-phase7_results/results/phase7_sae_feature_discrimination_phase7_sae_20260306_224419_phase7_sae.json}"
    SUBSPACE_SPECS="${SUBSPACE_SPECS:-phase7_results/interventions/variable_subspaces_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json}"

    SAMPLE_TRACES="${SAMPLE_TRACES:-0}"
    MIN_COMMON_STEPS="${MIN_COMMON_STEPS:-3}"
    SEED="${SEED:-20260307}"
    BATCH_SIZE="${BATCH_SIZE:-256}"

    TRAIN_EXCLUDE_VARIANTS="${TRAIN_EXCLUDE_VARIANTS:-order_flip_only,answer_first_order_flip,reordered_steps}"
    TRACE_TEST_FRACTION="${TRACE_TEST_FRACTION:-0.20}"
    TRACE_SPLIT_SEED="${TRACE_SPLIT_SEED:-20260307}"
    CV_FOLDS="${CV_FOLDS:-5}"
    CV_SEED="${CV_SEED:-20260307}"
    CV_MIN_VALID_FOLDS="${CV_MIN_VALID_FOLDS:-3}"
    BOOTSTRAP_N="${BOOTSTRAP_N:-1000}"
    REQUIRE_WRONG_INTERMEDIATE_AUROC="${REQUIRE_WRONG_INTERMEDIATE_AUROC:-0.70}"
    TRAIN_SEED="${TRAIN_SEED:-20260307}"

    OUT_FEATURES="${OUT_FEATURES:-phase7_results/results/phase7_mixed_trajectory_features_${RUN_TAG}.json}"
    OUT_JSON="${OUT_JSON:-phase7_results/results/phase7_mixed_trajectory_validation_${RUN_TAG}.json}"
    OUT_MD="${OUT_MD:-phase7_results/results/phase7_mixed_trajectory_validation_${RUN_TAG}.md}"

    if [[ -z "$CONTROL_RECORDS" ]]; then
      echo "CONTROL_RECORDS is required for launch" >&2
      exit 2
    fi
    if [[ ! -f "$CONTROL_RECORDS" ]]; then
      echo "CONTROL_RECORDS not found: $CONTROL_RECORDS" >&2
      exit 2
    fi
    if [[ ! -d "$SAES_DIR" ]]; then
      echo "SAES_DIR not found: $SAES_DIR" >&2
      exit 2
    fi
    if [[ ! -d "$ACTIVATIONS_DIR" ]]; then
      echo "ACTIVATIONS_DIR not found: $ACTIVATIONS_DIR" >&2
      exit 2
    fi
    if [[ ! -f "$PHASE4_TOP_FEATURES" ]]; then
      echo "PHASE4_TOP_FEATURES not found: $PHASE4_TOP_FEATURES" >&2
      exit 2
    fi

    mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" phase7_results/results
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
GPU_ID=$GPU_ID
MODEL_KEY=$MODEL_KEY
CONTROL_RECORDS=$CONTROL_RECORDS
LAYERS=$LAYERS
FEATURE_SET=$FEATURE_SET
SAES_DIR=$SAES_DIR
ACTIVATIONS_DIR=$ACTIVATIONS_DIR
PHASE4_TOP_FEATURES=$PHASE4_TOP_FEATURES
DIVERGENT_SOURCE=$DIVERGENT_SOURCE
SUBSPACE_SPECS=$SUBSPACE_SPECS
SAMPLE_TRACES=$SAMPLE_TRACES
MIN_COMMON_STEPS=$MIN_COMMON_STEPS
SEED=$SEED
BATCH_SIZE=$BATCH_SIZE
TRAIN_EXCLUDE_VARIANTS=$TRAIN_EXCLUDE_VARIANTS
TRACE_TEST_FRACTION=$TRACE_TEST_FRACTION
TRACE_SPLIT_SEED=$TRACE_SPLIT_SEED
CV_FOLDS=$CV_FOLDS
CV_SEED=$CV_SEED
CV_MIN_VALID_FOLDS=$CV_MIN_VALID_FOLDS
BOOTSTRAP_N=$BOOTSTRAP_N
REQUIRE_WRONG_INTERMEDIATE_AUROC=$REQUIRE_WRONG_INTERMEDIATE_AUROC
TRAIN_SEED=$TRAIN_SEED
OUT_FEATURES=$OUT_FEATURES
OUT_JSON=$OUT_JSON
OUT_MD=$OUT_MD
CFG

    tmux new-session -d -s "p7mix_worker_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' worker"
    tmux new-session -d -s "p7mix_coord_${RUN_ID}" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched mixed trajectory validation"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  worker session: p7mix_worker_${RUN_ID}"
    echo "  coordinator session: p7mix_coord_${RUN_ID}"
    echo "  output json: $OUT_JSON"
    ;;
  worker)
    _run_worker
    ;;
  coordinator)
    _run_coordinator
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

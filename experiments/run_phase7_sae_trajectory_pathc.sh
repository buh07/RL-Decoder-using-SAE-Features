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
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_sae_trajectory_pathc}"
RUN_TAG="${RUN_TAG:-phase7_sae_trajectory_pathc_${RUN_ID}}"
BASE="${BASE:-phase7_results/runs/${RUN_ID}}"

CONTROL_RECORDS="${CONTROL_RECORDS:-phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json}"
MODEL_KEY="${MODEL_KEY:-gpt2-medium}"
LAYERS_CSV="${LAYERS_CSV:-4,7,22}"
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
MODEL_LADDER="${MODEL_LADDER:-sae_only,hybrid_only,mixed}"
MIXED_DELTA_EFFECT_FLOOR="${MIXED_DELTA_EFFECT_FLOOR:-0.03}"

DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-}"
DECODER_DEVICE="${DECODER_DEVICE:-}"
DECODER_BATCH_SIZE="${DECODER_BATCH_SIZE:-128}"

mkdir -p "$BASE/logs" "$BASE/meta" "$BASE/state" phase7_results/results

CORE_RUN_ID="${RUN_ID}_core"
CORE_RUN_TAG="phase7_sae_trajectory_${CORE_RUN_ID}"
CORE_BASE="phase7_results/runs/${CORE_RUN_ID}"

OUT_JSON="phase7_results/results/phase7_sae_trajectory_pathc_${RUN_ID}.json"
OUT_MD="phase7_results/results/phase7_sae_trajectory_pathc_${RUN_ID}.md"

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
LAYERS_CSV=$LAYERS_CSV
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
MODEL_LADDER=$MODEL_LADDER
MIXED_DELTA_EFFECT_FLOOR=$MIXED_DELTA_EFFECT_FLOOR
DECODER_CHECKPOINT=$DECODER_CHECKPOINT
DECODER_DEVICE=$DECODER_DEVICE
DECODER_BATCH_SIZE=$DECODER_BATCH_SIZE
OUT_JSON=$OUT_JSON
OUT_MD=$OUT_MD
CFG

{
  echo "[$(date -Is)] launching Path C core run on GPUs 5/6/7"
  RUN_ID="$CORE_RUN_ID" \
  RUN_TAG="$CORE_RUN_TAG" \
  BASE="$CORE_BASE" \
  CONTROL_RECORDS="$CONTROL_RECORDS" \
  MODEL_KEY="$MODEL_KEY" \
  LAYERS_CSV="$LAYERS_CSV" \
  GPU_IDS_CSV="5,6,7" \
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
  DECODER_CHECKPOINT="$DECODER_CHECKPOINT" \
  DECODER_DEVICE="$DECODER_DEVICE" \
  DECODER_BATCH_SIZE="$DECODER_BATCH_SIZE" \
  ./experiments/run_phase7_sae_trajectory_coherence.sh launch

  while [[ ! -f "$CORE_BASE/state/pipeline.done" ]]; do
    if ! tmux has-session -t "p7saetc_coord_${CORE_RUN_ID}" 2>/dev/null; then
      found_done=0
      for _ in {1..6}; do
        sleep 5
        if [[ -f "$CORE_BASE/state/pipeline.done" ]]; then
          found_done=1
          break
        fi
      done
      if [[ "$found_done" -eq 0 ]]; then
        echo "core coordinator exited before pipeline.done" >&2
        exit 1
      fi
      break
    fi
    sleep 15
  done

  CORE_SUMMARY="$CORE_BASE/meta/summary.json"
  [[ -f "$CORE_SUMMARY" ]] || { echo "missing core summary: $CORE_SUMMARY" >&2; exit 1; }
  mapfile -t PARTIALS < <("$PY" - <<PY
import json
p = json.load(open(${CORE_SUMMARY@Q}))
for x in p.get("partials", []):
    print(str(x))
PY
)
  if [[ "${#PARTIALS[@]}" -lt 1 ]]; then
    echo "no partials discovered in core summary" >&2
    exit 1
  fi
  for p in "${PARTIALS[@]}"; do [[ -f "$p" ]] || { echo "missing partial $p" >&2; exit 1; }; done

  echo "[$(date -Is)] running Path C ensemble aggregator"
  "$PY" phase7/aggregate_sae_trajectory_pathc.py \
    --partials "${PARTIALS[@]}" \
    --output-json "$OUT_JSON" \
    --output-md "$OUT_MD" \
    --run-tag "$RUN_TAG" \
    --trace-test-fraction "$TRACE_TEST_FRACTION" \
    --trace-split-seed "$TRACE_SPLIT_SEED" \
    --epochs "$PROBE_EPOCHS" \
    --lr "$PROBE_LR" \
    --weight-decay "$PROBE_WEIGHT_DECAY" \
    --device "$PROBE_DEVICE" \
    --model-ladder "$MODEL_LADDER" \
    --mixed-delta-effect-floor "$MIXED_DELTA_EFFECT_FLOOR"

  "$PY" - <<PY
import json
from pathlib import Path
base = Path(${BASE@Q})
summary = {
  "schema_version": "phase7_sae_trajectory_pathc_pipeline_v2",
  "run_id": ${RUN_ID@Q},
  "run_tag": ${RUN_TAG@Q},
  "core_run_id": ${CORE_RUN_ID@Q},
  "layers_csv": ${LAYERS_CSV@Q},
  "feature_set": ${FEATURE_SET@Q},
  "pathc_json": ${OUT_JSON@Q},
  "model_ladder": ${MODEL_LADDER@Q},
}
summary["partials"] = json.load(open(${CORE_SUMMARY@Q})).get("partials", [])
(base / "meta" / "summary.json").write_text(json.dumps(summary, indent=2))
(base / "state" / "pipeline.done").write_text("done\\n")
print("pipeline summary written")
PY

  echo "[$(date -Is)] Path C complete"
  echo "result json: $OUT_JSON"
} | tee -a "$BASE/logs/pathc.log"

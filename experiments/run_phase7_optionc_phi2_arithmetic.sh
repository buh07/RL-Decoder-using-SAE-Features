#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-launch}"

# Multi-model replication bootstrap for an external model with local SAE assets.
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_optionc_phi2_arith}"
RUN_TAG="${RUN_TAG:-phase7_optionc_phi2_arith_${RUN_ID}}"
BASE="${BASE:-phase7_results/runs/${RUN_ID}}"

MODEL_KEY="${MODEL_KEY:-phi-2}"
DOMAIN="${DOMAIN:-arithmetic}"
GPU_IDS_CSV="${GPU_IDS_CSV:-1,3,5}"
SAES_DIR="${SAES_DIR:-phase2_results/saes_all_models/saes}"
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
PHASE4_TOP_FEATURES="${PHASE4_TOP_FEATURES:-phase7_results/results/top_features_phi2_from_expanded.json}"
FEATURE_SET="${FEATURE_SET:-eq_pre_result_150}"

# Start with SAE-only canary/full for external-model bootstrap.
DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-}"
DECODER_AUTO_TRAIN="${DECODER_AUTO_TRAIN:-0}"
DECODER_DOMAIN="${DECODER_DOMAIN:-arithmetic}"

CANARY_PAIRS="${CANARY_PAIRS:-400}"
FULL_PAIRS="${FULL_PAIRS:-1000}"

HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

if [[ "$MODE" == "launch" && ! -f "$PHASE4_TOP_FEATURES" ]]; then
  echo "missing PHASE4_TOP_FEATURES file for phi-2 run: $PHASE4_TOP_FEATURES" >&2
  echo "generate it first with phase7/derive_probe_top_features_from_expanded.py" >&2
  exit 1
fi

export RUN_ID RUN_TAG BASE MODEL_KEY DOMAIN GPU_IDS_CSV SAES_DIR ACTIVATIONS_DIR
export PHASE4_TOP_FEATURES FEATURE_SET DECODER_CHECKPOINT DECODER_AUTO_TRAIN DECODER_DOMAIN
export CANARY_PAIRS FULL_PAIRS HF_TOKEN HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE

exec "$ROOT_DIR/experiments/run_phase7_optionc_generated.sh" "$MODE"

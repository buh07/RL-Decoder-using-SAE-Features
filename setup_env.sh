#!/usr/bin/env bash
# Bootstrap project virtual environment, install dependencies, and sync tokenizer assets.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
TOKENIZER_SOURCE_DEFAULT="$REPO_ROOT/vendor/gpt2_tokenizer"
TOKENIZER_SOURCE="${TOKENIZER_SOURCE:-$TOKENIZER_SOURCE_DEFAULT}"
TOKENIZER_DEST="${TOKENIZER_DEST:-$REPO_ROOT/assets/tokenizers/gpt2}"
TOKENIZER_FILES=("tokenizer.json" "merges.txt" "vocab.json")
SKIP_PIP="${SKIP_PIP:-0}"

log() {
  echo "[setup_env] $*"
}

ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  log "Loaded environment overrides from .env"
fi

if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
log "Using Python $(python --version 2>&1)"

log "Upgrading packaging tools"
if [ "$SKIP_PIP" != "1" ]; then
  pip install --upgrade pip setuptools wheel

  log "Installing core dependencies"
  REQUIRED_PACKAGES=(
    "torch>=2.0"
    "transformers>=4.37"
    "trl>=0.8"
    "accelerate>=0.25"
    "datasets"
  )
  pip install --upgrade "${REQUIRED_PACKAGES[@]}"
else
  log "SKIP_PIP=1 → skipping pip upgrade/install step (make sure deps are satisfied manually)."
fi

log "Ensuring tokenizer assets exist"
if [ -d "$TOKENIZER_SOURCE" ]; then
  mkdir -p "$TOKENIZER_DEST"
  for file in "${TOKENIZER_FILES[@]}"; do
    if [ -f "$TOKENIZER_SOURCE/$file" ]; then
      cp "$TOKENIZER_SOURCE/$file" "$TOKENIZER_DEST/$file"
      log "Copied $file to tokenizer cache"
    else
      log "WARNING: $file missing in $TOKENIZER_SOURCE" >&2
    fi
  done
else
  log "WARNING: Tokenizer source directory $TOKENIZER_SOURCE not found" >&2
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  log "WANDB_API_KEY detected in environment; W&B will authenticate automatically."
else
  log "Reminder: export WANDB_API_KEY before launching training runs."
fi

log "Setup complete"

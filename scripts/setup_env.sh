#!/usr/bin/env bash
# Phase 1 bootstrap for RL-Decoder project (gpt2 focus)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="rl_decoder_phase1"
PYTHON_VERSION="3.10"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
USE_CONDA=true

if [[ -z "$CONDA_BIN" ]]; then
  echo "[WARN] conda not found; falling back to python venv at $VENV_DIR"
  USE_CONDA=false
fi

if [[ "$USE_CONDA" == true ]]; then
  source "$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh"
  if conda env list | grep -q "^$ENV_NAME "; then
    echo "[INFO] Updating existing env $ENV_NAME"
  else
    echo "[INFO] Creating env $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=$PYTHON_VERSION
  fi

  conda activate "$ENV_NAME"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[INFO] Creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    echo "[INFO] Reusing venv at $VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.0 trl==0.8.6 accelerate==0.25.0 datasets wandb einops sentencepiece
pip install -e "$PROJECT_ROOT"

python - <<'PY'
import torch, transformers
print(f"PyTorch {torch.__version__}, CUDA available={torch.cuda.is_available()}")
print(f"Transformers {transformers.__version__}")
PY

echo "[DONE] Environment ready."

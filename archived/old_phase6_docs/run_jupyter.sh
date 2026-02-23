#!/bin/bash

# Phase 6: Interactive Reasoning Tracker - Jupyter Notebook Launcher
# Handles virtual environment setup and Jupyter launch

VENV_DIR="venv"
NOTEBOOK_FILE="interactive_reasoning_tracker.ipynb"
PORT="${1:-8888}"

echo "========================================"
echo "🧠 Phase 6: Jupyter Notebook Launcher"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Notebook: $NOTEBOOK_FILE"
echo "  Port:     $PORT"
echo "  URL:      http://localhost:$PORT"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
  if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
  fi
  echo "✓ Virtual environment created"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install/update basic packages
echo "📥 Ensuring dependencies..."
pip install -q --upgrade pip setuptools wheel

# Install Jupyter first
if ! python -c "import jupyter" 2>/dev/null; then
  echo "Installing Jupyter..."
  pip install -q jupyter ipywidgets
fi

# Install other requirements
if ! python -c "import torch" 2>/dev/null; then
  echo "Installing ML dependencies (this may take a minute)..."
  pip install -q -r requirements_phase6.txt
fi

echo "✓ All dependencies ready"
echo ""
echo "🚀 Starting Jupyter notebook..."
echo ""

# Launch Jupyter notebook
jupyter notebook "$NOTEBOOK_FILE" --ip=127.0.0.1 --port="$PORT"

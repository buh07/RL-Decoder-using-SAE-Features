#!/bin/bash
#
# Phase 5.3 Feature Transfer Analysis - tmux Orchestrator
# Manages GPU allocation and distributes SAE training across GPUs
#
# Usage:
#   bash phase5_task3_tmux_orchestrator.sh [--gpus 0,1,2,3] [--backup] [--force-train] [--session-name phase5_task3]
#

set -e

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${REPO_ROOT}/phase5/phase5_feature_transfer.py"
LOG_DIR="${REPO_ROOT}/sae_logs"
SESSION_NAME="phase5_task3"
GPUS="0,1,2,3"
BACKUP_FLAG=""
FORCE_TRAIN_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --session-name)
            SESSION_NAME="$2"
            shift 2
            ;;
        --backup)
            BACKUP_FLAG="--backup-existing"
            shift
            ;;
        --force-train)
            FORCE_TRAIN_FLAG="--force-train"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already running. Attaching..."
    tmux attach-session -t "$SESSION_NAME"
    exit 0
fi

echo "=========================================="
echo "Phase 5.3: Feature Transfer Analysis"
echo "=========================================="
echo "Session name: $SESSION_NAME"
echo "GPUs: $GPUS"
echo "Repo root: $REPO_ROOT"
echo "Log directory: $LOG_DIR"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Create new tmux session
tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" -x 240 -y 60

# Setup environment window
tmux new-window -t "$SESSION_NAME" -n "env" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME:env" "source .venv/bin/activate && echo 'Environment activated'" Enter
sleep 1

# Create main execution window
tmux new-window -t "$SESSION_NAME" -n "train" -c "$REPO_ROOT"

# Build the command
CMD="python3 $SCRIPT \\"
CMD="$CMD --activations-dir phase4_results/activations \\"
CMD="$CMD --output-dir phase5_results/transfer_analysis \\"
CMD="$CMD --expansion-factor 8 \\"
CMD="$CMD --epochs 10 \\"
CMD="$CMD --batch-size 64 \\"
CMD="$CMD --learning-rate 1e-4 \\"
CMD="$CMD --top-k 50"

if [ -n "$BACKUP_FLAG" ]; then
    CMD="$CMD --backup-existing"
fi

if [ -n "$FORCE_TRAIN_FLAG" ]; then
    CMD="$CMD --force-train"
fi

# Execute on train window
tmux send-keys -t "$SESSION_NAME:train" \
    "source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=$GPUS && $CMD" Enter

sleep 2

# Create monitoring window
tmux new-window -t "$SESSION_NAME" -n "monitor" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME:monitor" \
    "while true; do echo '=== GPU Status ===' && nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu,utilization.memory --format=csv,noheader && sleep 5; done" Enter

sleep 1

# Create logging window
tmux new-window -t "$SESSION_NAME" -n "logs" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME:logs" \
    "tail -f sae_logs/* 2>/dev/null || echo 'Waiting for logs...'" Enter

# Select the train window as default
tmux select-window -t "$SESSION_NAME:train"

echo ""
echo "✓ tmux session created: $SESSION_NAME"
echo ""
echo "Available windows:"
echo "  phase5_task3:0 - env       (environment setup)"
echo "  phase5_task3:1 - train     (main training)"
echo "  phase5_task3:2 - monitor   (GPU monitoring)"
echo "  phase5_task3:3 - logs      (log tail)"
echo ""
echo "Attach to session:"
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "Monitor training (in another terminal):"
echo "  tmux send-keys -t $SESSION_NAME:monitor 'nvidia-smi' Enter"
echo ""
echo "View latest results:"
echo "  cat $REPO_ROOT/phase5_results/transfer_analysis/TRANSFER_REPORT.md"
echo ""
echo "Kill session when done:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""

# Attach to session
tmux attach-session -t "$SESSION_NAME"

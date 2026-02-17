#!/bin/bash
################################################################################
# SAE Training Monitor - Real-time progress dashboard
# Shows training status across both GPUs
################################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
LOG_DIR="$PROJECT_DIR/sae_logs"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/gpt2-small/sae"

echo "================================================================================"
echo "  SAE Training Monitor - GPUs 0 & 1"
echo "================================================================================"
echo ""

# GPU utilization
echo "📊 GPU STATUS:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -2 | awk -F', ' '{
    printf "  GPU %s: %s%% util, %s/%s MB\n", $1, $3, $4, $5
}'
echo ""

# Tmux sessions
echo "🖥️  TMUX SESSIONS:"
if tmux has-session -t sae_training_gpu0 2>/dev/null; then
    echo "  ✓ sae_training_gpu0 (active)"
else
    echo "  ✗ sae_training_gpu0 (not running)"
fi

if tmux has-session -t sae_training_gpu1 2>/dev/null; then
    echo "  ✓ sae_training_gpu1 (active)"
else
    echo "  ✗ sae_training_gpu1 (not running)"
fi
echo ""

# Training logs
echo "📝 TRAINING LOGS:"
for log in "$LOG_DIR"/gpu*.log; do
    if [ -f "$log" ]; then
        logname=$(basename "$log")
        lines=$(wc -l < "$log")
        last_line=$(tail -1 "$log" 2>/dev/null | cut -c1-80)
        echo "  $logname: $lines lines"
        echo "    Latest: $last_line"
    fi
done
echo ""

# Checkpoints created
echo "✅ CHECKPOINTS CREATED:"
checkpoint_count=$(ls -1 "$CHECKPOINT_DIR"/sae_768d_*x_final.pt 2>/dev/null | wc -l)
echo "  Total: $checkpoint_count SAE checkpoints"
ls -1 "$CHECKPOINT_DIR"/sae_768d_*x_final.pt 2>/dev/null | while read ckpt; do
    name=$(basename "$ckpt")
    size=$(du -h "$ckpt" | cut -f1)
    echo "    - $name ($size)"
done
echo ""

echo "================================================================================"
echo ""
echo "💡 QUICK ACTIONS:"
echo ""
echo "  View GPU 0 training:"
echo "    tmux attach -t sae_training_gpu0"
echo ""
echo "  View GPU 1 training:"
echo "    tmux attach -t sae_training_gpu1"
echo ""
echo "  Watch this monitor:"
echo "    watch -n 10 bash \"$SCRIPT_DIR/$(basename "$0")\""
echo ""
echo "================================================================================"

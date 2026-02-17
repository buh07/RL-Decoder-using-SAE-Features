#!/bin/bash
################################################################################
# SAE Training Orchestrator: Train all SAEs on full GSM8K dataset
# Uses GPUs 0-1 with tmux for parallel execution
# 
# Usage:
#   bash sae/sae_train_all_gsm8k.sh
#
# What it does:
#   1. Check available SAE expansions
#   2. Create tmux session with 2 windows (GPU 0, GPU 1)
#   3. Distribute SAEs across GPUs (round-robin)
#   4. Launch training on full GSM8K dataset
#   5. Provide monitoring commands
#
# Expected timeline:
#   - GPU 0: ~5 SAEs × ~30 min = ~2.5 hours
#   - GPU 1: ~4 SAEs × ~30 min = ~2 hours
#   - Total parallel: ~2.5 hours (best case, all running in parallel)
#
# Output:
#   - SAE checkpoints: checkpoints/gpt2-small/sae/sae_768d_*x_final.pt
#   - Training logs: sae_logs/gpu_*.log
#   - Results stored in: sae_results/
################################################################################

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/gpt2-small/sae"
LOG_DIR="$PROJECT_DIR/sae_logs"
RESULTS_DIR="$PROJECT_DIR/sae_results"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "================================================================================"
echo "  SAE Training Orchestrator - Full GSM8K Dataset"
echo "================================================================================"
echo ""

# Find available SAE expansions (from checkpoint naming)
echo "[Step 1] Checking available SAE expansions..."
SAE_EXPANSIONS=()
for checkpoint in "$CHECKPOINT_DIR"/sae_768d_*x_final.pt; do
    if [ -f "$checkpoint" ]; then
        # Extract expansion factor: sae_768d_4x_final.pt -> 4
        EXPANSION=$(basename "$checkpoint" | sed 's/sae_768d_//;s/x_final.pt//')
        SAE_EXPANSIONS+=("$EXPANSION")
    fi
done

# Sort expansions numerically
IFS=$'\n' SAE_EXPANSIONS=($(sort -n <<<"${SAE_EXPANSIONS[*]}"))
unset IFS

echo "  Found ${#SAE_EXPANSIONS[@]} SAE expansions: ${SAE_EXPANSIONS[@]}"
echo ""

# Calculate GPU distribution (round-robin across 2 GPUs)
echo "[Step 2] Planning GPU distribution (2 GPUs, round-robin)..."
declare -a GPU_0_SAES=()
declare -a GPU_1_SAES=()

for i in "${!SAE_EXPANSIONS[@]}"; do
    expansion="${SAE_EXPANSIONS[$i]}"
    gpu_id=$((i % 2))
    if [ $gpu_id -eq 0 ]; then
        GPU_0_SAES+=("$expansion")
    else
        GPU_1_SAES+=("$expansion")
    fi
done

echo "  GPU 0 will train: ${GPU_0_SAES[@]} (${#GPU_0_SAES[@]} SAEs)"
echo "  GPU 1 will train: ${GPU_1_SAES[@]} (${#GPU_1_SAES[@]} SAEs)"
echo ""

# Create training command function
create_training_command() {
    local gpu_id=$1
    local sae_list=("${@:2}")
    
    local cmd="python src/sae_training.py"
    
    for sae in "${sae_list[@]}"; do
        cmd="$cmd --expand-sae $sae"
    done
    
    cmd="$cmd --gpu-id $gpu_id --full-dataset --batch-size 64 --num-epochs 10"
    
    echo "$cmd"
}

# Create tmux session
echo "[Step 3] Creating tmux session 'sae_gsm8k'..."
tmux kill-session -t sae_gsm8k 2>/dev/null || true
sleep 1

tmux new-session -d -s sae_gsm8k -c "$PROJECT_DIR"
tmux new-window -t sae_gsm8k -n gpu0 -c "$PROJECT_DIR"
tmux new-window -t sae_gsm8k -n gpu1 -c "$PROJECT_DIR"

echo "  ✓ Created tmux session 'sae_gsm8k' with 3 windows (0, gpu0, gpu1)"
echo ""

# Send training commands to each window
echo "[Step 4] Launching training on each GPU..."

# GPU 0
echo "  Launching training for GPU 0 with SAEs: ${GPU_0_SAES[@]}"
training_cmd_gpu0=$(create_training_command 0 "${GPU_0_SAES[@]}")
tmux send-keys -t sae_gsm8k:gpu0 "source .venv/bin/activate" Enter
sleep 0.5
tmux send-keys -t sae_gsm8k:gpu0 "cd $PROJECT_DIR" Enter
sleep 0.5
tmux send-keys -t sae_gsm8k:gpu0 "echo 'Starting GPU 0 training...' && $training_cmd_gpu0 2>&1 | tee $LOG_DIR/gpu0.log" Enter

# GPU 1
echo "  Launching training for GPU 1 with SAEs: ${GPU_1_SAES[@]}"
training_cmd_gpu1=$(create_training_command 1 "${GPU_1_SAES[@]}")
tmux send-keys -t sae_gsm8k:gpu1 "source .venv/bin/activate" Enter
sleep 0.5
tmux send-keys -t sae_gsm8k:gpu1 "cd $PROJECT_DIR" Enter
sleep 0.5
tmux send-keys -t sae_gsm8k:gpu1 "echo 'Starting GPU 1 training...' && $training_cmd_gpu1 2>&1 | tee $LOG_DIR/gpu1.log" Enter

echo ""
echo "================================================================================"
echo "  Training Started Successfully"
echo "================================================================================"
echo ""
echo "📊 MONITORING:"
echo ""
echo "  View GPU 0 training:"
echo "    tmux attach -t sae_gsm8k:gpu0"
echo ""
echo "  View GPU 1 training:"
echo "    tmux attach -t sae_gsm8k:gpu1"
echo ""
echo "  Watch all GPUs:"
echo "    watch -n 5 'tmux capture-pane -t sae_gsm8k -p'"
echo ""
echo "  Monitor GPU utilization:"
echo "    watch -n 2 nvidia-smi"
echo ""
echo "📁 OUTPUT:"
echo ""
echo "  Training logs:"
echo "    tail -f $LOG_DIR/gpu0.log"
echo "    tail -f $LOG_DIR/gpu1.log"
echo ""
echo "  Results will be stored in:"
echo "    $RESULTS_DIR/"
echo ""
echo "⏱️  TIMELINE:"
echo ""
echo "  GPU 0: ~${#GPU_0_SAES[@]} SAEs × ~30 min = ~$((${#GPU_0_SAES[@]} * 30)) minutes"
echo "  GPU 1: ~${#GPU_1_SAES[@]} SAEs × ~30 min = ~$((${#GPU_1_SAES[@]} * 30)) minutes"
echo ""
echo "  Parallel execution: ~$((( ${#GPU_0_SAES[@]} > ${#GPU_1_SAES[@]} ? ${#GPU_0_SAES[@]} : ${#GPU_1_SAES[@]} ) * 30)) minutes total"
echo ""
echo "📋 CONTROL:"
echo ""
echo "  Stop all training:"
echo "    tmux kill-session -t sae_gsm8k"
echo ""
echo "  Restart training:"
echo "    bash sae_train_all_gsm8k.sh"
echo ""
echo "🔍 Verify training completed:"
echo ""
echo "  ls -lh $CHECKPOINT_DIR/sae_768d_*x_final.pt"
echo ""
echo "================================================================================"
echo ""

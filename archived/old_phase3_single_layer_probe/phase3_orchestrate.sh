#!/bin/bash
# Phase 3 Full-Scale Orchestration Script
# Handles SAE training, tmux setup, and distributed evaluation

set -e

PROJECT_DIR="/scratch2/f004ndc/RL-Decoder with SAE Features"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/gpt2-small/sae"
VENV="$PROJECT_DIR/.venv/bin/activate"
RESULTS_DIR="$PROJECT_DIR/phase3_results/full_scale"

# Configuration
NUM_GPUS=4
GPU_IDS=(0 1 2 3)  # Use GPUs 0-3 for Phase 3 (full dataset)
SAE_EXPANSIONS=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32)  # 16 total expansions
TMUX_SESSION="phase3_fullscale"
DATASET_SIZE="FULL GSM8K (~7,473 examples)"  # Process entire dataset per SAE

# Activate environment
source "$VENV"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Phase 3 Full-Scale Evaluation Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# Step 1: Check existing SAEs
echo -e "\n${YELLOW}[1/4] Checking existing SAE checkpoints...${NC}"
EXISTING_SAES=()
MISSING_SAES=()
for exp in "${SAE_EXPANSIONS[@]}"; do
    if [ -f "$CHECKPOINT_DIR/sae_768d_${exp}x_final.pt" ]; then
        EXISTING_SAES+=($exp)
        echo -e "  ${GREEN}✓${NC} ${exp}x exists"
    else
        MISSING_SAES+=($exp)
        echo -e "  ${RED}✗${NC} ${exp}x missing"
    fi
done

# Step 2: Quick training of missing SAEs (optional)
if [ ${#MISSING_SAES[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}[2/4] Training missing SAE expansions...${NC}"
    echo -e "${YELLOW}  Missing: ${MISSING_SAES[@]}${NC}"
    echo -e "${YELLOW}  This will take ~${#MISSING_SAES[@]}*2.5 hours = ~$((${#MISSING_SAES[@]}*3)) hours${NC}"
    echo -e "\n${YELLOW}  I recommend training these in background while Phase 3 runs on existing SAEs${NC}"
    echo -e "${YELLOW}  Or press Ctrl+C to skip and evaluate only existing SAEs${NC}"
    echo -e "\n${YELLOW}  To train missing SAEs, run separately:${NC}"
    for exp in "${MISSING_SAES[@]}"; do
        echo -e "    python src/sae_training.py --expansion $exp --num-epochs 5"
    done
fi

# Step 3: Create results directory
echo -e "\n${YELLOW}[3/4] Creating results directory...${NC}"
mkdir -p "$RESULTS_DIR"
echo -e "  ${GREEN}✓${NC} $RESULTS_DIR"

# Step 4: Kill existing tmux session if it exists
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo -e "\n${YELLOW}[4/4] Cleaning up old tmux session...${NC}"
    tmux kill-session -t "$TMUX_SESSION"
    echo -e "  ${GREEN}✓${NC} Killed old session"
fi

# Step 5: Create new tmux session with multiple windows
echo -e "\n${BLUE}Setting up tmux session: $TMUX_SESSION${NC}"
tmux new-session -d -s "$TMUX_SESSION" -x 240 -y 50 -c "$PROJECT_DIR"

# Create windows for each GPU
for i in "${!GPU_IDS[@]}"; do
    GPU_ID=${GPU_IDS[$i]}
    WINDOW_NAME="gpu${GPU_ID}"
    
    if [ $i -eq 0 ]; then
        # Use default window
        tmux rename-window -t "$TMUX_SESSION" "$WINDOW_NAME"
    else
        # Create new window
        tmux new-window -t "$TMUX_SESSION" -n "$WINDOW_NAME" -c "$PROJECT_DIR"
    fi
    
    echo -e "  ${GREEN}✓${NC} Window: $WINDOW_NAME (GPU $GPU_ID)"
done

# Step 6: Send commands to each window
echo -e "\n${BLUE}Starting Phase 3 evaluations on ${#GPU_IDS[@]} GPUs...${NC}"

# Distribute existing SAEs across GPUs
EXISTING_ARRAY=("${EXISTING_SAES[@]}")
SHARD_SIZE=$(( (${#EXISTING_ARRAY[@]} + NUM_GPUS - 1) / NUM_GPUS ))

for i in "${!GPU_IDS[@]}"; do
    GPU_ID=${GPU_IDS[$i]}
    WINDOW_NAME="gpu${GPU_ID}"
    
    # Calculate which SAEs this GPU should process
    START=$((i * SHARD_SIZE))
    END=$(((i + 1) * SHARD_SIZE))
    
    if [ $START -lt ${#EXISTING_ARRAY[@]} ]; then
        GPU_SAES=("${EXISTING_ARRAY[@]:$START:$SHARD_SIZE}")
        
        # Build command
        CMD="cd \"$PROJECT_DIR\" && source \"$VENV\" && python phase3/phase3_full_scale.py --gpu-id $GPU_ID --sae-expansions ${GPU_SAES[@]} --output-file \"$RESULTS_DIR/results_gpu${GPU_ID}.json\" 2>&1 | tee \"$RESULTS_DIR/log_gpu${GPU_ID}.txt\""
        
        tmux send-keys -t "$TMUX_SESSION:$WINDOW_NAME" "$CMD" Enter
        
        echo -e "  ${GREEN}✓${NC} GPU $GPU_ID will evaluate: ${GPU_SAES[@]}"
    fi
done

# Step 7: Monitor setup
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Tmux Session Setup Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${GREEN}Session name: $TMUX_SESSION${NC}"
echo -e "\n${YELLOW}To monitor progress:${NC}"
echo -e "  tmux attach -t $TMUX_SESSION"
echo -e "\n${YELLOW}To view individual GPU logs:${NC}"
for GPU_ID in "${GPU_IDS[@]}"; do
    echo -e "  tail -f $RESULTS_DIR/log_gpu${GPU_ID}.txt"
done
echo -e "\n${YELLOW}To stop all:${NC}"
echo -e "  tmux kill-session -t $TMUX_SESSION"

# Step 8: Create monitoring script
echo -e "\n${YELLOW}[5/4] Creating monitoring utilities...${NC}"
cat > "$RESULTS_DIR/monitor.sh" << 'MONITOR_EOF'
#!/bin/bash
RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="phase3_fullscale"

echo "Phase 3 Full-Scale Monitoring Dashboard"
echo "=========================================="
echo ""

while true; do
    clear
    echo "Phase 3 Full-Scale Monitoring Dashboard"
    echo "=========================================="
    echo "Time: $(date)"
    echo ""
    
    # Check GPU utilization
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used --format=csv,noheader | head -4
    
    echo ""
    echo "Progress by GPU:"
    for i in 0 1 2 3; do
        if [ -f "$RESULTS_DIR/log_gpu${i}.txt" ]; then
            COUNT=$(grep -c "✓.*completed" "$RESULTS_DIR/log_gpu${i}.txt" 2>/dev/null || echo "0")
            echo "  GPU $i: $COUNT SAEs completed"
            tail -3 "$RESULTS_DIR/log_gpu${i}.txt" | sed 's/^/    /'
        fi
    done
    
    echo ""
    echo "Results collected so far:"
    for RESULT_FILE in "$RESULTS_DIR"/results_gpu*.json; do
        if [ -f "$RESULT_FILE" ]; then
            echo "  $(basename $RESULT_FILE): $(wc -l < $RESULT_FILE) lines"
        fi
    done
    
    echo ""
    echo "Press Ctrl+C to exit, refreshing in 10s..."
    sleep 10
done
MONITOR_EOF

chmod +x "$RESULTS_DIR/monitor.sh"
echo -e "  ${GREEN}✓${NC} Created monitoring script: $RESULTS_DIR/monitor.sh"

# Step 9: Print next steps
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Next Steps${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${YELLOW}1. Attach to tmux session:${NC}"
echo -e "   ${GREEN}tmux attach -t $TMUX_SESSION${NC}"
echo -e "\n${YELLOW}2. Monitor progress (in another terminal):${NC}"
echo -e "   ${GREEN}$RESULTS_DIR/monitor.sh${NC}"
echo -e "\n${YELLOW}3. When finished, merge results:${NC}"
echo -e "   ${GREEN}python phase3/phase3_merge_results.py --input-dir \"$RESULTS_DIR\"${NC}"
echo -e "\n${YELLOW}4. (Parallel) Start Phase 1 Falsification Pipeline:${NC}"
echo -e "   ${GREEN}python phase1_ground_truth.py --gpu-ids 4 5 6 7${NC}"
echo ""

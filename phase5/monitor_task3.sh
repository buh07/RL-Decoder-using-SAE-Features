#!/bin/bash
# Phase 5.3 Monitoring Dashboard
# Shows training progress, GPU status, and results updates

REPO_ROOT="/scratch2/f004ndc/RL-Decoder with SAE Features"
cd "$REPO_ROOT"

show_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║            PHASE 5.3: FEATURE TRANSFER ANALYSIS - MONITOR             ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

show_process_status() {
    echo "📊 Process Status:"
    if pgrep -f "phase5_feature_transfer.py" > /dev/null; then
        PID=$(pgrep -f "phase5_feature_transfer.py" | head -1)
        MEM_MB=$(ps aux | grep "^\s*$(echo $PID)" | awk '{print int($6/1024)}')
        echo "  ✓ Training process running (PID: $PID, Memory: ~${MEM_MB}MB)"
    else
        echo "  ✗ Training process not running"
    fi
}

show_gpu_status() {
    echo ""
    echo "🖥️  GPU Status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu,utilization.memory --format=csv,noheader,nounits | awk -F', ' '{
            printf "  GPU %s: %s / %s (%d%% util) | Temp: %s°C | Mem: %s / %s MB\n",
            $1, int($3), int($4), $6, $5, int($3), int($4)
        }'
    else
        echo "  ⚠️  nvidia-smi not available"
    fi
}

show_log_tail() {
    echo ""
    echo "📝 Latest Log Output (last 20 lines):"
    echo "─────────────────────────────────────────────────────────────────────────"
    LATEST_LOG=$(ls -tr sae_logs/phase5_task3_*.log 2>/dev/null | tail -1)
    if [ -n "$LATEST_LOG" ]; then
        tail -20 "$LATEST_LOG"
    else
        echo "  (No log file yet)"
    fi
}

show_output_status() {
    echo ""
    echo "📦 Output Files:"
    OUTPUT_DIR="phase5_results/transfer_analysis"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  Directory: $OUTPUT_DIR"
        if [ -f "$OUTPUT_DIR/transfer_results.json" ]; then
            LINES=$(wc -l < "$OUTPUT_DIR/transfer_results.json")
            echo "    ✓ transfer_results.json ($LINES lines)"
        else
            echo "    ⏳ transfer_results.json (pending)"
        fi
        if [ -f "$OUTPUT_DIR/TRANSFER_REPORT.md" ]; then
            LINES=$(wc -l < "$OUTPUT_DIR/TRANSFER_REPORT.md")
            echo "    ✓ TRANSFER_REPORT.md ($LINES lines)"
        else
            echo "    ⏳ TRANSFER_REPORT.md (pending)"
        fi
        
        CKPT_DIR="$OUTPUT_DIR/sae_checkpoints"
        if [ -d "$CKPT_DIR" ]; then
            NUM_CKPTS=$(ls -1 "$CKPT_DIR"/*.pt 2>/dev/null | wc -l)
            echo "    ✓ sae_checkpoints/ ($NUM_CKPTS .pt files)"
        fi
        
        BACKUP_DIRS=$(ls -d "$OUTPUT_DIR"/sae_checkpoints_backup_* 2>/dev/null | wc -l)
        if [ "$BACKUP_DIRS" -gt 0 ]; then
            echo "    ✓ backup directories ($BACKUP_DIRS backups)"
        fi
    else
        echo "  ⏳ Waiting for output directory creation"
    fi
}

show_commands() {
    echo ""
    echo "🎯 Useful Commands:"
    echo "  Monitor (continuous):"
    echo "    watch -n 5 'cd \"$REPO_ROOT\" && bash phase5/monitor_task3.sh'"
    echo ""
    echo "  View full log:"
    LATEST_LOG=$(ls -tr sae_logs/phase5_task3_*.log 2>/dev/null | tail -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "    tail -f \"$LATEST_LOG\""
    fi
    echo ""
    echo "  Check GPU live:"
    echo "    nvidia-smi -l 2"
    echo ""
    echo "  View results when ready:"
    echo "    cat phase5_results/transfer_analysis/TRANSFER_REPORT.md"
    echo ""
    echo "  Find training process:"
    echo "    ps aux | grep phase5_feature_transfer"
}

# Main execution
case "${1:-}" in
    --loop)
        while true; do
            clear
            show_header
            show_process_status
            show_gpu_status
            show_output_status
            show_log_tail
            echo ""
            echo "⏰ Last update: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Press Ctrl+C to exit. Refreshing in 5 seconds..."
            sleep 5
        done
        ;;
    *)
        show_header
        show_process_status
        show_gpu_status
        show_output_status
        show_log_tail
        show_commands
        echo ""
        ;;
esac

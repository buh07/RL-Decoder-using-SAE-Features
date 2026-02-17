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

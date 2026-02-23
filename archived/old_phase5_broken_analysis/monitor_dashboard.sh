#!/bin/bash
# Real-time monitoring dashboard for Phase 5.4 multi-GPU execution

REPO_ROOT="/scratch2/f004ndc/RL-Decoder with SAE Features"

print_status() {
    clear
    echo "╔════════════════════════════════════════════════════════════════════════════════════╗"
    echo "║           PHASE 5.4 MULTI-LAYER FEATURE TRANSFER ANALYSIS - GPU DASHBOARD         ║"
    echo "║                          Status Monitor (Real-time)                                ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # GPU 0 Status
    echo "┌─ GPU 0: CAPTURE (gemma-2b, all 5 layers) ─────────────────────────────────────────┐"
    ACTIVATION_COUNT=$(ls "${REPO_ROOT}/phase4_results/activations_multilayer"/gemma-2b_layer*_activations.pt 2>/dev/null | wc -l)
    if [ $ACTIVATION_COUNT -eq 5 ]; then
        echo "│ Status: ✓ COMPLETE (5/5 layers captured)                                        │"
    elif [ $ACTIVATION_COUNT -gt 0 ]; then
        echo "│ Status: 🔄 IN PROGRESS ($ACTIVATION_COUNT/5 layers captured)                    │"
    else
        echo "│ Status: 🔄 IN PROGRESS (Loading model and capturing...)                        │"
    fi
    
    # Show layer sizes if available
    for layer in 4 8 12 16 20; do
        file="${REPO_ROOT}/phase4_results/activations_multilayer/gemma-2b_layer${layer}_activations.pt"
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "│   ✓ Layer $layer: $size                                                    │"
        fi
    done
    echo "└────────────────────────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # GPU 1 Status
    echo "┌─ GPU 1: TRAINING (gemma-2b SAEs, waiting for capture) ─────────────────────────────┐"
    SAE_COUNT=$(ls "${REPO_ROOT}/phase5_results/multilayer_transfer/saes"/*_sae.pt 2>/dev/null | wc -l)
    if [ $SAE_COUNT -eq 5 ]; then
        echo "│ Status: ✓ COMPLETE (5/5 SAEs trained)                                          │"
    elif [ $SAE_COUNT -gt 0 ]; then
        echo "│ Status: 🔄 TRAINING ($SAE_COUNT/5 SAEs ready)                                   │"
    else
        echo "│ Status: ⏳ WAITING for capture to complete...                                   │"
    fi
    
    # Show SAE files if available
    for layer in 4 8 12 16 20; do
        file="${REPO_ROOT}/phase5_results/multilayer_transfer/saes/gemma-2b_layer${layer}_sae.pt"
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "│   ✓ Layer $layer SAE: $size                                                │"
        fi
    done
    echo "└────────────────────────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # GPU 2 Status
    echo "┌─ GPU 2: TRANSFER MATRIX + ANALYSIS (waiting for training) ──────────────────────────┐"
    if [ -f "${REPO_ROOT}/phase5_results/multilayer_transfer/transfer_matrix.json" ]; then
        echo "│ Status: ✓ transfer_matrix.json generated                                        │"
    elif [ $SAE_COUNT -gt 0 ]; then
        echo "│ Status: 🔄 COMPUTING transfer matrix...                                        │"
    else
        echo "│ Status: ⏳ WAITING for SAE training to complete...                              │"
    fi
    
    if [ -f "${REPO_ROOT}/phase5_results/multilayer_transfer/layer_transfer_heatmap.png" ]; then
        echo "│ Status: ✓ Visualizations generated (heatmap, curves)                           │"
    fi
    
    if [ -f "${REPO_ROOT}/phase5_results/multilayer_transfer/multilayer_transfer_full_report.md" ]; then
        echo "│ Status: ✓ Analysis report complete                                             │"
    fi
    echo "└────────────────────────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # Overall Progress
    echo "╔════════════════════════════════════════════════════════════════════════════════════╗"
    PERCENT=0
    if [ $ACTIVATION_COUNT -gt 0 ]; then PERCENT=$((PERCENT + 25)); fi
    if [ $ACTIVATION_COUNT -eq 5 ]; then PERCENT=$((PERCENT + 10)); fi
    if [ $SAE_COUNT -gt 0 ]; then PERCENT=$((PERCENT + 25)); fi
    if [ $SAE_COUNT -eq 5 ]; then PERCENT=$((PERCENT + 10)); fi
    if [ -f "${REPO_ROOT}/phase5_results/multilayer_transfer/transfer_matrix.json" ]; then PERCENT=$((PERCENT + 20)); fi
    if [ -f "${REPO_ROOT}/phase5_results/multilayer_transfer/multilayer_transfer_full_report.md" ]; then PERCENT=$((PERCENT + 10)); fi
    
    # Draw progress bar
    FILLED=$((PERCENT / 5))
    EMPTY=$((20 - FILLED))
    BAR=$(printf '█%.0s' $(seq 1 $FILLED))$(printf '░%.0s' $(seq 1 $EMPTY))
    
    echo "║ Overall Progress: [$BAR] $PERCENT% Complete                                       ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Next update
    echo "Next update in 10 seconds... (Press Ctrl+C to stop)"
}

# Loop for continuous monitoring
while true; do
    print_status
    sleep 10
done

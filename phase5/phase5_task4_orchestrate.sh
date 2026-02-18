#!/bin/bash
################################################################################
# Phase 5 Task 4: Multi-Layer Feature Transfer Analysis Orchestrator
# 
# Executes all 5 steps of multi-layer analysis in sequence:
# 1. Multi-layer activation capture (30-45 min)
# 2. Per-layer SAE training (90-120 min)
# 3. Transfer matrix computation (15-20 min)
# 4. Visualization & heatmap generation (10-15 min)
# 5. Analysis & insights reporting (20-30 min)
#
# Total estimated time: 3-4 hours
# GPU requirement: 1 GPU (RTX 6000 recommended)
#
# Usage:
#   bash phase5_task4_orchestrate.sh
#   bash phase5_task4_orchestrate.sh --device cuda:5 --skip-capture
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE5_DIR="${REPO_ROOT}/phase5"
DATA_DIR="${REPO_ROOT}/phase4_results"
OUTPUT_BASE="${REPO_ROOT}/phase5_results/multilayer_transfer"

DEVICE="${DEVICE:-cuda:5}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-10}"
TOP_K="${TOP_K:-50}"

SKIP_CAPTURE=false
SKIP_TRAINING=false
SKIP_TRANSFER=false
SKIP_VIZ=false
SKIP_ANALYSIS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-capture)
            SKIP_CAPTURE=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-transfer)
            SKIP_TRANSFER=true
            shift
            ;;
        --skip-viz)
            SKIP_VIZ=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Setup
mkdir -p "${OUTPUT_BASE}"
LOG_DIR="${REPO_ROOT}/sae_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/phase5_task4_${TIMESTAMP}.log"

exec > >(tee -a "$MAIN_LOG")
exec 2>&1

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ℹ️  $*"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✓ $*"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ⚠️  $*"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✗ $*"
}

header() {
    echo ""
    echo "================================================================================"
    echo "$*"
    echo "================================================================================"
}

################################################################################
# STEP 1: Multi-Layer Activation Capture
################################################################################

if [ "$SKIP_CAPTURE" = false ]; then
    header "STEP 1: Multi-Layer Activation Capture (30-45 min)"

    log_info "Capturing activations from layers 4, 8, 12, 16, 20 for all models..."
    log_info "Command: python3 phase5_task4_capture_multi_layer.py"
    log_info "Device: $DEVICE"

    python3 "${PHASE5_DIR}/phase5_task4_capture_multi_layer.py" \
        --models gemma-2b gpt2-medium phi-2 pythia-1.4b \
        --layers 4 8 12 16 20 \
        --output-dir "${DATA_DIR}/activations_multilayer" \
        --batch-size 32 \
        --num-samples "$NUM_SAMPLES" \
        --device "$DEVICE" \
        2>&1 | tee "${LOG_DIR}/phase5_task4_capture_${TIMESTAMP}.log"

    if [ $? -eq 0 ]; then
        log_success "Step 1 complete: Applied activation capture to $((4 * 5)) layer positions"
        ACTIVATION_COUNT=$(ls "${DATA_DIR}/activations_multilayer"/*_activations.pt 2>/dev/null | wc -l)
        log_info "Generated $ACTIVATION_COUNT activation files"
    else
        log_error "Step 1 failed: Activation capture error"
        exit 1
    fi
else
    log_info "Skipping Step 1 (capture)"
fi

################################################################################
# STEP 2: Multi-Layer SAE Training
################################################################################

if [ "$SKIP_TRAINING" = false ]; then
    header "STEP 2: Multi-Layer SAE Training (90-120 min)"

    log_info "Training SAEs on each layer's activations..."
    log_info "Command: python3 phase5_task4_train_multilayer_saes.py"
    log_info "Epochs: $EPOCHS, Batch size: $BATCH_SIZE"

    python3 "${PHASE5_DIR}/phase5_task4_train_multilayer_saes.py" \
        --activations-dir "${DATA_DIR}/activations_multilayer" \
        --output-dir "${OUTPUT_BASE}/saes" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --verbose \
        2>&1 | tee "${LOG_DIR}/phase5_task4_training_${TIMESTAMP}.log"

    if [ $? -eq 0 ]; then
        log_success "Step 2 complete: Trained all multi-layer SAEs"
        SAE_COUNT=$(ls "${OUTPUT_BASE}/saes"/*_sae.pt 2>/dev/null | wc -l)
        log_info "Generated $SAE_COUNT SAE checkpoints"
    else
        log_error "Step 2 failed: SAE training error"
        exit 1
    fi
else
    log_info "Skipping Step 2 (training)"
fi

################################################################################
# STEP 3: Compute Transfer Matrix
################################################################################

if [ "$SKIP_TRANSFER" = false ]; then
    header "STEP 3: Compute Layer-to-Layer Transfer Matrix (15-20 min)"

    log_info "Computing pairwise transfer metrics across all layer combinations..."
    log_info "Command: python3 phase5_task4_compute_transfer_matrix.py"
    log_info "Top-k features: $TOP_K"

    python3 "${PHASE5_DIR}/phase5_task4_compute_transfer_matrix.py" \
        --sae-dir "${OUTPUT_BASE}/saes" \
        --activations-dir "${DATA_DIR}/activations_multilayer" \
        --output-dir "${OUTPUT_BASE}" \
        --top-k "$TOP_K" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        2>&1 | tee "${LOG_DIR}/phase5_task4_transfer_${TIMESTAMP}.log"

    if [ $? -eq 0 ]; then
        log_success "Step 3 complete: Transfer matrix computed"
        if [ -f "${OUTPUT_BASE}/transfer_matrix.json" ]; then
            PAIR_COUNT=$(grep -c '"transfer_recon_ratio"' "${OUTPUT_BASE}/transfer_matrix.json" 2>/dev/null || echo "?")
            log_info "Computed $PAIR_COUNT transfer pairs"
        fi
    else
        log_error "Step 3 failed: Transfer computation error"
        exit 1
    fi
else
    log_info "Skipping Step 3 (transfer)"
fi

################################################################################
# STEP 4: Generate Visualizations
################################################################################

if [ "$SKIP_VIZ" = false ]; then
    header "STEP 4: Visualization & Heatmap Generation (10-15 min)"

    log_info "Generating layer universality heatmaps and analysis plots..."
    log_info "Command: python3 phase5_task4_visualization.py"

    python3 "${PHASE5_DIR}/phase5_task4_visualization.py" \
        --transfer-matrix "${OUTPUT_BASE}/transfer_matrix.json" \
        --output-dir "${OUTPUT_BASE}" \
        --format png pdf \
        2>&1 | tee "${LOG_DIR}/phase5_task4_viz_${TIMESTAMP}.log"

    if [ $? -eq 0 ]; then
        log_success "Step 4 complete: Visualizations generated"
        PLOT_COUNT=$(ls "${OUTPUT_BASE}"/*.png 2>/dev/null | wc -l)
        log_info "Generated $PLOT_COUNT visualization files"
    else
        log_error "Step 4 failed: Visualization error"
        exit 1
    fi
else
    log_info "Skipping Step 4 (visualization)"
fi

################################################################################
# STEP 5: Analysis & Reporting
################################################################################

if [ "$SKIP_ANALYSIS" = false ]; then
    header "STEP 5: Analysis & Insights Reporting (20-30 min)"

    log_info "Generating comprehensive analysis report..."
    log_info "Command: python3 phase5_task4_analysis.py"

    python3 "${PHASE5_DIR}/phase5_task4_analysis.py" \
        --transfer-matrix "${OUTPUT_BASE}/transfer_matrix.json" \
        --output-dir "${OUTPUT_BASE}" \
        --output-file "multilayer_transfer_full_report.md" \
        2>&1 | tee "${LOG_DIR}/phase5_task4_analysis_${TIMESTAMP}.log"

    if [ $? -eq 0 ]; then
        log_success "Step 5 complete: Analysis report generated"
        if [ -f "${OUTPUT_BASE}/multilayer_transfer_full_report.md" ]; then
            log_info "Report saved to ${OUTPUT_BASE}/multilayer_transfer_full_report.md"
        fi
    else
        log_error "Step 5 failed: Analysis error"
        exit 1
    fi
else
    log_info "Skipping Step 5 (analysis)"
fi

################################################################################
# Completion Summary
################################################################################

header "PHASE 5.4 COMPLETE ✓"

log_success "All steps completed successfully!"

echo ""
log_info "Output directory: ${OUTPUT_BASE}"

echo ""
log_info "Generated files:"
ls -lh "${OUTPUT_BASE}"/*.json 2>/dev/null | awk '{print "  - " $9}' || true
ls -lh "${OUTPUT_BASE}"/*.png 2>/dev/null | awk '{print "  - " $9}' || true
ls -lh "${OUTPUT_BASE}"/*.pdf 2>/dev/null | awk '{print "  - " $9}' || true
ls -lh "${OUTPUT_BASE}"/*.md 2>/dev/null | awk '{print "  - " $9}' || true

echo ""
log_info "Main results:"
log_info "  • Transfer matrix: ${OUTPUT_BASE}/transfer_matrix.json"
log_info "  • Heatmap: ${OUTPUT_BASE}/layer_transfer_heatmap.png"
log_info "  • Analysis: ${OUTPUT_BASE}/multilayer_transfer_full_report.md"

echo ""
log_info "Next steps: "
log_info "  1. Review analysis report for universality findings"
log_info "  2. Identify optimal layer for Phase 6 feature steering"
log_info "  3. Begin Phase 6 step-level causal testing"

echo ""
log_info "Log file: ${MAIN_LOG}"

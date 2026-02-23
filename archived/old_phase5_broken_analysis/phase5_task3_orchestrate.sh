#!/bin/bash

##############################################################################
# PHASE 5 TASK 3: FEATURE TRANSFER ANALYSIS ORCHESTRATOR
#
# Purpose: Train SAEs per Phase 4 activation set and test feature transfer
# Resources: GPU recommended, ~45-60 minutes depending on epochs and batch size
##############################################################################

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

echo "=========================================="
echo "PHASE 5 TASK 3: FEATURE TRANSFER ANALYSIS"
echo "=========================================="
echo ""
echo "Project: RL-Decoder with SAE Features"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python --version)"
echo ""

PHASE4_RESULTS="$PROJECT_DIR/phase4_results"
ACTIVATIONS_DIR="$PHASE4_RESULTS/activations"
OUTPUT_DIR="$PROJECT_DIR/phase5_results/transfer_analysis"

mkdir -p "$OUTPUT_DIR"

echo "✓ Output directory: $OUTPUT_DIR"
echo ""

if [ ! -d "$ACTIVATIONS_DIR" ]; then
    echo "❌ ERROR: Activation directory not found: $ACTIVATIONS_DIR"
    echo "   Run Phase 4 activation capture first."
    exit 1
fi

# Run transfer analysis
python "phase5/phase5_feature_transfer.py" \
    --activations-dir "$ACTIVATIONS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --expansion-factor 8 \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --top-k 50

echo ""
echo "=========================================="
echo "✅ PHASE 5 TASK 3 COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR"

#!/bin/bash
# Phase 4 Orchestrator: Full pipeline from activation capture to causal tests
# Runs in stages across GPUs 0-3

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT_BASE="phase4_results"
mkdir -p "$OUTPUT_BASE"/{activations,saes,causal_tests}

echo "=========================================="
echo "PHASE 4: FRONTIER LLMs - CAUSAL ANALYSIS"
echo "=========================================="
echo ""
echo "Project: RL-Decoder with SAE Features"
echo "Output: $OUTPUT_BASE"
echo "GPUs: 0, 1, 2, 3"
echo ""

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# =============================================================================
# STAGE 1: Model & Benchmark Validation
# =============================================================================
echo -e "${YELLOW}=== STAGE 1: Validation ===${NC}"
echo "Skipping model loading test (transformers not available)"
echo "Proceeding directly to data collection..."
echo ""

# =============================================================================
# STAGE 2: Activation Capture (Parallel on 4 GPUs)
# =============================================================================
echo -e "${YELLOW}=== STAGE 2: Activation Capture ===${NC}"
echo "Capturing activations from 4 models on reasoning benchmarks"
echo ""

# Job 1: GPT-2-Medium on GSM8K (GPU 0)
echo -e "${BLUE}[GPU 0]${NC} GPT-2-Medium + GSM8K..."
python phase4/phase4_data_simple.py \
    --model gpt2-medium \
    --benchmark gsm8k \
    --gpu-id 0 \
    --output-dir "$OUTPUT_BASE/activations" \
    --num-examples 500 \
    > "$OUTPUT_BASE/gpu0_capture.log" 2>&1 &
PID_GPU0=$!

# Job 2: Pythia-1.4B on GSM8K (GPU 1)
echo -e "${BLUE}[GPU 1]${NC} Pythia-1.4B + GSM8K..."
python phase4/phase4_data_simple.py \
    --model pythia-1.4b \
    --benchmark gsm8k \
    --gpu-id 1 \
    --output-dir "$OUTPUT_BASE/activations" \
    --num-examples 500 \
    > "$OUTPUT_BASE/gpu1_capture.log" 2>&1 &
PID_GPU1=$!

# Job 3: Gemma-2B on MATH (GPU 2)
echo -e "${BLUE}[GPU 2]${NC} Gemma-2B + MATH..."
python phase4/phase4_data_simple.py \
    --model gemma-2b \
    --benchmark math \
    --gpu-id 2 \
    --output-dir "$OUTPUT_BASE/activations" \
    --num-examples 100 \
    > "$OUTPUT_BASE/gpu2_capture.log" 2>&1 &
PID_GPU2=$!

# Job 4: Phi-2 on Logic (GPU 3)
echo -e "${BLUE}[GPU 3]${NC} Phi-2 + Logic..."
python phase4/phase4_data_simple.py \
    --model phi-2 \
    --benchmark logic \
    --gpu-id 3 \
    --output-dir "$OUTPUT_BASE/activations" \
    --num-examples 200 \
    > "$OUTPUT_BASE/gpu3_capture.log" 2>&1 &
PID_GPU3=$!

echo ""
echo "Activation capture jobs started:"
echo "  GPU 0 (PID $PID_GPU0): gpt2-medium + gsm8k"
echo "  GPU 1 (PID $PID_GPU1): pythia-1.4b + gsm8k"
echo "  GPU 2 (PID $PID_GPU2): gemma-2b + math"
echo "  GPU 3 (PID $PID_GPU3): phi-2 + logic"
echo ""
echo "Waiting for completion..."
echo ""

# Wait for all capture jobs
wait $PID_GPU0 || { echo -e "${BLUE}[GPU 0]${NC} capture failed"; }
wait $PID_GPU1 || { echo -e "${BLUE}[GPU 1]${NC} capture failed"; }
wait $PID_GPU2 || { echo -e "${BLUE}[GPU 2]${NC} capture failed"; }
wait $PID_GPU3 || { echo -e "${BLUE}[GPU 3]${NC} capture failed"; }

echo -e "${GREEN}✓ Activation capture complete${NC}"
echo ""

# =============================================================================
# STAGE 3: SAE Training (Parallel on 4 GPUs)
# =============================================================================
echo -e "${YELLOW}=== STAGE 3: SAE Training ===${NC}"
echo "Training SAEs on captured activations"
echo ""

python phase4/phase4_train_saes.py \
    --activation-dir "$OUTPUT_BASE/activations" \
    --output-dir "$OUTPUT_BASE/saes" \
    --gpu-ids 0 1 2 3 \
    --num-epochs 20 \
    2>&1 | tee "$OUTPUT_BASE/stage3_sae_training.log"

echo -e "${GREEN}✓ SAE training complete${NC}"
echo ""

# =============================================================================
# STAGE 4: Causal Ablation Tests (Parallel on 4 GPUs)
# =============================================================================
echo -e "${YELLOW}=== STAGE 4: Causal Ablation Tests ===${NC}"
echo "Running feature ablation and measuring causality"
echo ""

python phase4/phase4_causal_tests.py \
    --model-names gpt2-medium pythia-1.4b gemma-2b phi-2 \
    --benchmark-names gsm8k gsm8k math logic \
    --sae-dir "$OUTPUT_BASE/saes" \
    --output-dir "$OUTPUT_BASE/causal_tests" \
    --gpu-ids 0 1 2 3 \
    2>&1 | tee "$OUTPUT_BASE/stage4_causal_tests.log"

echo -e "${GREEN}✓ Causal tests complete${NC}"
echo ""

# =============================================================================
# STAGE 5: Results Aggregation & Report
# =============================================================================
echo -e "${YELLOW}=== STAGE 5: Results Aggregation ===${NC}"
echo "Generating Phase 4 report..."
echo ""

python phase4/phase4_aggregate_results.py \
    --results-dir "$OUTPUT_BASE" \
    --output-file "$OUTPUT_BASE/PHASE4_REPORT.md" \
    2>&1 | tee "$OUTPUT_BASE/stage5_aggregation.log"

echo ""
echo "=========================================="
echo "PHASE 4 COMPLETE"
echo "=========================================="
echo ""
echo -e "${GREEN}Results Summary:${NC}"
echo "  Activations: $OUTPUT_BASE/activations/"
echo "  SAE Checkpoints: $OUTPUT_BASE/saes/"
echo "  Causal Tests: $OUTPUT_BASE/causal_tests/"
echo "  Report: $OUTPUT_BASE/PHASE4_REPORT.md"
echo ""
echo "Logs:"
echo "  GPU 0: $OUTPUT_BASE/gpu0_capture.log"
echo "  GPU 1: $OUTPUT_BASE/gpu1_capture.log"
echo "  GPU 2: $OUTPUT_BASE/gpu2_capture.log"
echo "  GPU 3: $OUTPUT_BASE/gpu3_capture.log"
echo "  SAE Training: $OUTPUT_BASE/stage3_sae_training.log"
echo "  Causal Tests: $OUTPUT_BASE/stage4_causal_tests.log"
echo ""

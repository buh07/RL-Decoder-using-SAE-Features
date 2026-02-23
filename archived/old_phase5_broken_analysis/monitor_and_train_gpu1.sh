#!/bin/bash
# Monitor capture and trigger training when ALL LAYERS are captured

REPO_ROOT="/scratch2/f004ndc/RL-Decoder with SAE Features"
ACTIVATION_DIR="${REPO_ROOT}/phase4_results/activations_multilayer"

# Expected file counts per model:
# - gemma-2b: 24 layers
# - gpt2-medium: 12 layers
# - phi-2: 32 layers
# - pythia-1.4b: 24 layers
# Total: 92 files
EXPECTED_TOTAL=92

echo "GPU1: Monitoring capture progress (expecting ${EXPECTED_TOTAL} total files)..."
echo "GPU1: Expected breakdown:"
echo "  - gemma-2b: 24 files"
echo "  - gpt2-medium: 12 files"
echo "  - phi-2: 32 files"
echo "  - pythia-1.4b: 24 files"

while true; do
    GEMMA_COUNT=$(ls ${ACTIVATION_DIR}/gemma-2b_layer*_activations.pt 2>/dev/null | wc -l)
    GPT2_COUNT=$(ls ${ACTIVATION_DIR}/gpt2-medium_layer*_activations.pt 2>/dev/null | wc -l)
    PHI_COUNT=$(ls ${ACTIVATION_DIR}/phi-2_layer*_activations.pt 2>/dev/null | wc -l)
    PYTHIA_COUNT=$(ls ${ACTIVATION_DIR}/pythia-1.4b_layer*_activations.pt 2>/dev/null | wc -l)
    TOTAL=$((GEMMA_COUNT + GPT2_COUNT + PHI_COUNT + PYTHIA_COUNT))
    
    echo "GPU1: Progress: Gemma=${GEMMA_COUNT}/24 | GPT2=${GPT2_COUNT}/12 | Phi=${PHI_COUNT}/32 | Pythia=${PYTHIA_COUNT}/24 | Total=${TOTAL}/${EXPECTED_TOTAL}"
    
    if [ $TOTAL -eq $EXPECTED_TOTAL ]; then
        break
    fi
    
    sleep 10
done

echo "GPU1: ✓ Capture complete! All ${EXPECTED_TOTAL} layer files ready."

cd "${REPO_ROOT}"
python3 phase5/phase5_task4_train_multilayer_saes.py \
    --activations-dir phase4_results/activations_multilayer \
    --output-dir phase5_results/multilayer_transfer/saes \
    --epochs 10 \
    --batch-size 64 \
    --device cuda:1 \
    --verbose 2>&1 | tee sae_logs/phase5_task4_training_gpu1.log

echo "GPU1: ✓ Training complete!"

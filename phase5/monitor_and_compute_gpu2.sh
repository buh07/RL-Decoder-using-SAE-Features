#!/bin/bash
# Monitor training and trigger transfer computation

REPO_ROOT="/scratch2/f004ndc/RL-Decoder with SAE Features"
SAE_DIR="${REPO_ROOT}/phase5_results/multilayer_transfer/saes"

# Wait for training to complete (check if all 5 SAE checkpoints exist)
echo "GPU2: Monitoring SAE training progress..."
while [ $(ls ${SAE_DIR}/*_sae.pt 2>/dev/null | wc -l) -lt 5 ]; do
    COUNT=$(ls ${SAE_DIR}/*_sae.pt 2>/dev/null | wc -l)
    echo "GPU2: Waiting for training... (${COUNT}/5 SAEs ready)"
    sleep 15
done

echo "GPU2: ✓ Training complete! All 5 SAE checkpoints ready."
echo "GPU2: Starting transfer matrix computation on GPU 2..."

cd "${REPO_ROOT}"
python3 phase5/phase5_task4_compute_transfer_matrix.py \
    --sae-dir phase5_results/multilayer_transfer/saes \
    --activations-dir phase4_results/activations_multilayer \
    --output-dir phase5_results/multilayer_transfer \
    --top-k 50 \
    --device cuda:2 2>&1 | tee sae_logs/phase5_task4_transfer_gpu2.log

echo "GPU2: ✓ Transfer matrix computation complete!"

# Continue with visualization and analysis
echo "GPU2: Starting visualization..."
python3 phase5/phase5_task4_visualization.py \
    --transfer-matrix phase5_results/multilayer_transfer/transfer_matrix.json \
    --output-dir phase5_results/multilayer_transfer \
    --format png pdf 2>&1 | tee -a sae_logs/phase5_task4_transfer_gpu2.log

echo "GPU2: Starting analysis..."
python3 phase5/phase5_task4_analysis.py \
    --transfer-matrix phase5_results/multilayer_transfer/transfer_matrix.json \
    --output-dir phase5_results/multilayer_transfer \
    --output-file multilayer_transfer_full_report.md 2>&1 | tee -a sae_logs/phase5_task4_transfer_gpu2.log

echo "GPU2: ✓ Analysis complete!"

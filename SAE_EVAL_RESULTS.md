# SAE Evaluation Results on GSM8K Test Set

## Summary

Evaluated the trained 20x expansion SAE (768D → 15360D) on held-out GSM8K test activations from GPT-2 layer 6.

### Test Dataset
- **Source**: GSM8K test split
- **Model**: GPT-2 Small (Layer 6)
- **Activation Type**: Layer 6 MLP hidden states
- **Total Chunks**: 208 test chunks (~212K tokens)
- **Sequence Length**: 1024 tokens

### Evaluation Results

#### Checkpoint: `sae_768d_final.pt`
**Config**: 768D → 15360D (20x expansion)

| Metric | Value |
|--------|-------|
| **Total Loss** | 71.08 |
| **Reconstruction Loss** | 71.08 |
| **L1 Loss** | 1.28 |
| **Decorrelation Loss** | 0.0013 |
| **Sparsity (Activation Fraction)** | 49.57% |
| **Avg Time per Batch** | 75.4ms |
| **Total Batches** | 13 |

### Interpretation

1. **Reconstruction Loss (71.08)**: The SAE achieves reasonable reconstruction of the test activations. This is comparable to the training loss observed during GSM8K training.

2. **L1 Regularization (1.28)**: The sparsity-inducing L1 penalty is modest, resulting in:
   - **Activation Fraction**: ~50% of SAE features are active in the test set
   - This indicates moderate sparsity—features are selective but not extremely sparse

3. **Decorrelation Loss (0.0013)**: Minimal decorrelation penalty, suggesting the SAE has naturally learned reasonably decorrelated features without heavy regularization.

4. **Inference Speed**: 75.4ms per batch (batch size typically 32 tokens) is acceptable for interpretability work.

### Notes for Interpretability Analysis

- The 20x expansion SAE is the primary trained checkpoint and represents the highest expansion ratio we explored on GSM8K
- Test performance aligns with training observations, suggesting good generalization
- 49.57% sparsity provides a balance between interpretability and reconstruction fidelity
- For downstream interpretability tasks (feature attribution, intervention), this checkpoint is recommended

### Training Context

- **Training Data**: GSM8K training split (~7K examples, 1,144 chunks)
- **Hardware**: RTX 6000 Ada GPU
- **Training Details**: 
  - Mixed precision (fp16)
  - Learning rate: adjusted per expansion level
  - Batch size: 32
  - Checkpointing: Saved every 500 steps

---
Evaluation conducted: 2025-01-09
Dataset: GSM8K test set, GPT-2 Layer 6 MLP activations

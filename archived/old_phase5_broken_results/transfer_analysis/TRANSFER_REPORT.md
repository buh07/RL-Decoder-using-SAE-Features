# Phase 5.3 Feature Transfer Analysis

Activation source: phase4_results/activations
Expansion factor: 8
Epochs: 10
Top-k features: 50

## Pairwise Transfer Summary

| Source -> Target | Status | Recon Ratio | Top-k Var Ratio | Spearman | Mean Max Cosine |
| --- | --- | --- | --- | --- | --- |
| gemma-2b_math__to__gpt2-medium_gsm8k | skipped (dimension_mismatch_2048_vs_1024) | - | - | - | - |
| gemma-2b_math__to__phi-2_logic | skipped (dimension_mismatch_2048_vs_2560) | - | - | - | - |
| gemma-2b_math__to__pythia-1.4b_gsm8k | ok | 0.995 | 1.092 | 0.226 | 0.088 |
| gpt2-medium_gsm8k__to__gemma-2b_math | skipped (dimension_mismatch_1024_vs_2048) | - | - | - | - |
| gpt2-medium_gsm8k__to__phi-2_logic | skipped (dimension_mismatch_1024_vs_2560) | - | - | - | - |
| gpt2-medium_gsm8k__to__pythia-1.4b_gsm8k | skipped (dimension_mismatch_1024_vs_2048) | - | - | - | - |
| phi-2_logic__to__gemma-2b_math | skipped (dimension_mismatch_2560_vs_2048) | - | - | - | - |
| phi-2_logic__to__gpt2-medium_gsm8k | skipped (dimension_mismatch_2560_vs_1024) | - | - | - | - |
| phi-2_logic__to__pythia-1.4b_gsm8k | skipped (dimension_mismatch_2560_vs_2048) | - | - | - | - |
| pythia-1.4b_gsm8k__to__gemma-2b_math | ok | 1.000 | 1.096 | 0.356 | 0.087 |
| pythia-1.4b_gsm8k__to__gpt2-medium_gsm8k | skipped (dimension_mismatch_2048_vs_1024) | - | - | - | - |
| pythia-1.4b_gsm8k__to__phi-2_logic | skipped (dimension_mismatch_2048_vs_2560) | - | - | - | - |
# Phase 4 Execution Guide: Frontier LLMs Causal Analysis

**Status**: Implementation in progress  
**Timeline**: 2-3 days on GPUs 0-3  
**Goal**: Apply validated SAE methodology to 1B-7B models to discover causal reasoning circuits

## Phase 4 Overview

Phase 4 applies the SAE framework validated in Phases 1-3 to frontier language models:

- **Phase 3 validated**: SAEs extract reasoning features (100% probe accuracy on GSM8K)
- **Phase 1 validated**: Causal perturbations work mechanically
- **Phase 4 tests**: Do real LLMs have interpretable, stable reasoning circuits?

## Target Models

| Model | Size | Provider | Hidden Dim | Layers | Priority |
|-------|------|----------|-----------|--------|----------|
| GPT-2 Medium | 355M | OpenAI | 1024 | 24 | Baseline |
| Pythia-1.4B | 1.4B | EleutherAI | 2048 | 24 | Primary |
| Gemma-2B | 2B | Google | 2048 | 18 | Primary |
| Phi-2 | 2.7B | Microsoft | 2560 | 32 | Primary |
| Llama-3-8B | 8B | Meta | 4096 | 32 | If time permits |

## Reasoning Benchmarks

| Benchmark | Examples | Domain | Metric |
|-----------|----------|--------|--------|
| GSM8K | 500 | Grade school math | Accuracy |
| MATH | 100 | Competition math | Accuracy |
| Logic Inference | 200 | Common sense reasoning | Accuracy |

## Execution Plan

### Stage 1: Setup & Model Loading (1 hour, GPU 0)
```bash
# Create working directories
mkdir -p phase4_results/{activations,saes,causal_tests}

# Test model loading
python phase4/test_model_loading.py --models gpt2-medium pythia-1.4b
```

### Stage 2: Benchmark Data + Activation Capture (6-8 hours, GPUs 0-3 parallel)
```bash
# GPU 0: GPT-2-Medium on GSM8K
python phase4/phase4_data.py --model gpt2-medium --benchmark gsm8k \
    --gpu-id 0 --output-dir phase4_results/activations --num-examples 500

# GPU 1: Pythia-1.4B on GSM8K
python phase4/phase4_data.py --model pythia-1.4b --benchmark gsm8k \
    --gpu-id 1 --output-dir phase4_results/activations --num-examples 500

# GPU 2: Gemma-2B on MATH
python phase4/phase4_data.py --model gemma-2b --benchmark math \
    --gpu-id 2 --output-dir phase4_results/activations --num-examples 100

# GPU 3: Phi-2 on Logic
python phase4/phase4_data.py --model phi-2 --benchmark logic \
    --gpu-id 3 --output-dir phase4_results/activations --num-examples 200
```

### Stage 3: SAE Training (8-12 hours, GPUs 0-3 parallel)
```bash
# Each GPU trains SAE on captured activations
# Using configuration from Phase 1-3:
# - Expansion: 8x
# - L1 penalty: 1e-4
# - Decorrelation: 0.01
# - ReLU: disabled
# - Epochs: 20

python phase4/phase4_train_saes.py \
    --activation-dir phase4_results/activations \
    --output-dir phase4_results/saes \
    --gpu-ids 0 1 2 3
```

### Stage 4: Causal Ablation Tests (6-10 hours, GPUs 0-3 parallel)
```bash
# Run causal perturbations: zero-out top features and measure accuracy drop

python phase4/phase4_causal_tests.py \
    --model-names gpt2-medium pythia-1.4b gemma-2b phi-2 \
    --sae-dir phase4_results/saes \
    --output-dir phase4_results/causal_tests \
    --gpu-ids 0 1 2 3
```

### Stage 5: Analysis & Results (2-3 hours)
```bash
# Aggregate results, generate report
python phase4/phase4_aggregate_results.py \
    --results-dir phase4_results \
    --output-file phase4_results/PHASE4_REPORT.md
```

## Expected Results

### Success Criteria
- [ ] Causal tests complete on all 4 models
- [ ] Feature importance scores align across models (transfer)
- [ ] Accuracy drops >5% when top features ablated
- [ ] Stable reasoning primitives identified (e.g., "addition reasoning", "constraint satisfaction")

### Output Artifacts
- `phase4_results/activations/`: Raw activations from each model/benchmark
- `phase4_results/saes/`: Trained SAE checkpoints
- `phase4_results/causal_tests/`: Feature ablation results + importance scores
- `phase4_results/PHASE4_REPORT.md`: Final summary + findings

## Implementation Status

### ✅ Completed
- [x] Phase 1 ground-truth validation (BFS environment)
- [x] Phase 1 results documentation
- [x] Phase 4 pipeline skeleton
- [x] Benchmark data loader

### 🔧 In Progress
- [ ] Activation capture implementation
- [ ] SAE training loop
- [ ] Causal ablation tests
- [ ] Results aggregation

### 📋 TODO
- [ ] Add support for more models (LLaMA, Mistral)
- [ ] Implement circuit discovery (feature clustering)
- [ ] Add visualization of discovered circuits
- [ ] Cross-model feature transfer analysis

## Resource Budget

**Total Phase 4 Budget**: 50-100 GPU-hours

| Stage | GPUs | Time | Hours |
|-------|------|------|-------|
| Activation Capture | 4 (parallel) | 2-3 hours | 8-12 |
| SAE Training | 4 (parallel) | 2-3 hours | 8-12 |
| Causal Tests | 4 (parallel) | 2-3 hours | 8-12 |
| Analysis | 1 | 1-2 hours | 1-2 |
| **Total** | | | **25-38** |

Fits comfortably within 50-100 hour budget.

## Key Differences from Phase 3

| Aspect | Phase 3 | Phase 4 |
|--------|---------|---------|
| **Models** | Proxy (GPT-2 small) | Frontier (1B-8B) |
| **Test** | Correlation (probe accuracy) | Causality (ablation) |
| **Metrics** | Accuracy, F1 | Feature importance, circuit structure |
| **Goal** | Validate methodology | Find real reasoning circuits |

## Integration with Experiment Design

From `overview.tex`:
- ✅ Phase 1 (Ground-truth): Validated causality methodology
- ⏭️ Phase 4 (Frontier LLMs): Apply to real models **← We are here**

Phase 4 is the final falsification gate:
- If circuits are found and interpretable → Framework succeeds
- If circuits are unstable/noisy → Framework needs refinement
- If no causal signal → SAEs insufficient for interpretability

## Next Steps

1. **Immediate** (next 1-2 hours):
   - Implement activation capture hooks
   - Test on GPT-2-medium + GSM8K
   
2. **Short term** (next day):
   - Run Stage 2 (activation capture) on all 4 GPUs
   - Monitor memory usage, throughput
   
3. **Medium term** (next 2-3 days):
   - Complete Stages 3-4 (SAE training + causal tests)
   - Aggregate and visualize results
   
4. **Publication**:
   - Write up findings in `PHASE4_REPORT.md`
   - Document failure modes and lessons learned
   - Compare with Phase 3 validation

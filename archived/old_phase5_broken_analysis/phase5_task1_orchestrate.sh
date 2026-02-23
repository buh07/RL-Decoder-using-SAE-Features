#!/bin/bash

##############################################################################
# PHASE 5 TASK 1: REAL CAUSAL ABLATION TESTS ORCHESTRATOR
# Run on tmux for background execution
#
# Purpose: Measure actual task accuracy drops when features are zeroed
# This validates the variance-based importance estimates from Phase 4B
#
# Resources: CPU-bound with some GPU usage, ~30 minutes for 4 models
##############################################################################

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

echo "=========================================="
echo "PHASE 5 TASK 1: CAUSAL ABLATION TESTS"
echo "=========================================="
echo ""
echo "Project: RL-Decoder with SAE Features"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python --version)"
echo ""

# Configuration
PHASE4_RESULTS="${PROJECT_DIR}/phase4_results"
ACTIVATIONS_DIR="${PHASE4_RESULTS}/activations"
FEATURE_STATS_DIR="${PHASE4_RESULTS}/interpretability"
OUTPUT_DIR="${PROJECT_DIR}/phase5_results/causal_ablation"

# Model configurations
declare -A MODELS=(
    ["gpt2-medium"]="gsm8k"
    ["pythia-1.4b"]="gsm8k"
    ["gemma-2b"]="math"
    ["phi-2"]="logic"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

# Validate input
if [ ! -d "$ACTIVATIONS_DIR" ]; then
    echo "❌ ERROR: Activation directory not found: $ACTIVATIONS_DIR"
    exit 1
fi

if [ ! -d "$FEATURE_STATS_DIR" ]; then
    echo "❌ ERROR: Feature stats directory not found: $FEATURE_STATS_DIR"
    exit 1
fi

# Stage 1: Run tests sequentially on GPU
echo "========== RUNNING CAUSAL ABLATION TESTS =========="
echo ""

total_models=${#MODELS[@]}
current=0

for model in "${!MODELS[@]}"; do
    benchmark="${MODELS[$model]}"
    current=$((current + 1))
    
    echo "[$current/$total_models] Testing: $model on $benchmark"
    
    python "phase5/phase5_causal_ablation.py" \
        "$ACTIVATIONS_DIR" \
        "$FEATURE_STATS_DIR" \
        "$OUTPUT_DIR" \
        "$model" \
        "$benchmark"
    
    echo ""
done

# Stage 2: Aggregate results
echo "========== AGGREGATING RESULTS =========="
echo ""

python << 'PYTHON_SCRIPT'
import json
from pathlib import Path
import numpy as np

output_dir = Path("phase5_results/causal_ablation")
results_files = list(output_dir.glob("*_causal_ablation.json"))

print(f"Found {len(results_files)} test results\n")

aggregate = {
    "timestamp": __import__('datetime').datetime.now().isoformat(),
    "models_tested": len(results_files),
    "results": {},
    "cross_model_summary": {}
}

all_correlations = []
validation_statuses = []

for results_file in sorted(results_files):
    print(f"Loading: {results_file.name}")
    with open(results_file) as f:
        result = json.load(f)
    
    model_name = result['model']
    aggregate["results"][model_name] = result
    
    correlation = result.get('correlation_with_variance', 0)
    status = result.get('validation_status', 'unknown')
    
    all_correlations.append(correlation)
    validation_statuses.append(status)
    
    print(f"  - Correlation: {correlation:.4f}")
    print(f"  - Status: {status}")
    print()

# Cross-model summary
all_pass = all('PASSED' in s for s in validation_statuses)
avg_correlation = np.mean(all_correlations) if all_correlations else 0

aggregate["cross_model_summary"] = {
    "avg_correlation": float(avg_correlation),
    "min_correlation": float(min(all_correlations)) if all_correlations else 0,
    "max_correlation": float(max(all_correlations)) if all_correlations else 0,
    "all_models_pass": all_pass,
    "overall_status": "✅ PASSED - Variance estimates are causally predictive" if all_pass else "⚠️  PARTIAL - Some models need investigation"
}

# Save aggregate
agg_file = output_dir / "causal_ablation_summary.json"
with open(agg_file, 'w') as f:
    json.dump(aggregate, f, indent=2)

print("="*60)
print("CAUSAL ABLATION TEST SUMMARY")
print("="*60)
print(f"Average correlation: {avg_correlation:.4f}")
print(f"Overall status: {aggregate['cross_model_summary']['overall_status']}")
print("="*60)
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "✅ PHASE 5 TASK 1 COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR"
echo ""
echo "Causal Ablation Tests complete!"
echo "Next: Task 2 - LM-based Feature Naming"

#!/bin/bash

##############################################################################
# PHASE 5 TASK 2: LM-BASED FEATURE NAMING ORCHESTRATOR
#
# Purpose: Generate semantic descriptions of top features using LLM
# Resources: CPU/network bound (LLM API calls), ~10 minutes for 4 models
##############################################################################

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

echo "=========================================="
echo "PHASE 5 TASK 2: LM-BASED FEATURE NAMING"
echo "=========================================="
echo ""
echo "Project: RL-Decoder with SAE Features"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Configuration
PHASE4_RESULTS="${PROJECT_DIR}/phase4_results"
ACTIVATIONS_DIR="${PHASE4_RESULTS}/activations"
INTERPRETABILITY_DIR="${PHASE4_RESULTS}/interpretability"
OUTPUT_DIR="${PROJECT_DIR}/phase5_results/feature_naming"

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

if [ ! -d "$INTERPRETABILITY_DIR" ]; then
    echo "❌ ERROR: Interpretability directory not found: $INTERPRETABILITY_DIR"
    exit 1
fi

# Check for OpenAI API key (optional - will fallback to templates)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set - will use template descriptions"
    echo ""
fi

# Stage 1: Run naming for all models
echo "========== GENERATING FEATURE NAMES =========="
echo ""

total_models=${#MODELS[@]}
current=0

for model in "${!MODELS[@]}"; do
    benchmark="${MODELS[$model]}"
    current=$((current + 1))
    
    echo "[$current/$total_models] Naming features for: $model on $benchmark"
    
    python "phase5/phase5_feature_naming.py" \
        "$ACTIVATIONS_DIR" \
        "$INTERPRETABILITY_DIR" \
        "$OUTPUT_DIR" \
        "$model" \
        "$benchmark"
    
    echo ""
done

# Stage 2: Aggregate results
echo "========== AGGREGATING NAMING RESULTS =========="
echo ""

python << 'PYTHON_SCRIPT'
import json
from pathlib import Path

output_dir = Path("phase5_results/feature_naming")
results_files = list(output_dir.glob("*_feature_naming_results.json"))

print(f"Found {len(results_files)} naming results\n")

aggregate = {
    "timestamp": __import__('datetime').datetime.now().isoformat(),
    "models_completed": len(results_files),
    "results": {},
    "summary": {}
}

total_features_named = 0
all_features = {}

for results_file in sorted(results_files):
    print(f"Loading: {results_file.name}")
    with open(results_file) as f:
        result = json.load(f)
    
    model_name = result['model']
    n_named = result.get('n_features_named', 0)
    
    aggregate["results"][model_name] = result
    all_features[model_name] = result['features_with_descriptions']
    total_features_named += n_named
    
    print(f"  - Features named: {n_named}")
    print()

aggregate["summary"] = {
    "total_features_named": total_features_named,
    "models_processed": len(results_files),
    "status": "✅ COMPLETE - All features named" if len(results_files) == 4 else "⚠️  PARTIAL - Some models pending"
}

# Save aggregate
agg_file = output_dir / "feature_naming_summary.json"
with open(agg_file, 'w') as f:
    json.dump(aggregate, f, indent=2)

print("="*60)
print("FEATURE NAMING SUMMARY")
print("="*60)
print(f"Total features named: {total_features_named}")
print(f"Models processed: {len(results_files)}")
print(f"Status: {aggregate['summary']['status']}")
print("="*60)
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "✅ PHASE 5 TASK 2 COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR"
echo ""
echo "Feature Naming complete!"
echo "Next: Task 3 - Feature Transfer Analysis (optional)"

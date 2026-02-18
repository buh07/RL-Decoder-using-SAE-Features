#!/bin/bash
# Phase 1 Orchestrator: Run ground-truth validation on GPUs 0, 1, 2
# BFS on GPU 0, Stack on GPU 1, Logic on GPU 2 (parallel)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT_BASE="phase1_results"
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "PHASE 1: GROUND-TRUTH SAE VALIDATION"
echo "=========================================="
echo "Starting parallel jobs on GPUs 0, 1, 2"
echo "Output: $OUTPUT_BASE"
echo ""

# Colors for logging
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Job 1: BFS on GPU 0
echo -e "${BLUE}[GPU 0]${NC} Starting BFS environment..."
python phase1/phase1_training.py \
    --gpu-id 0 \
    --output-dir "$OUTPUT_BASE/gpu0_bfs" \
    --env bfs \
    > "$OUTPUT_BASE/gpu0_bfs.log" 2>&1 &
GPU0_PID=$!

# Job 2: Stack Machine on GPU 1
echo -e "${BLUE}[GPU 1]${NC} Starting Stack Machine environment..."
python phase1/phase1_training.py \
    --gpu-id 1 \
    --output-dir "$OUTPUT_BASE/gpu1_stack" \
    --env stack \
    > "$OUTPUT_BASE/gpu1_stack.log" 2>&1 &
GPU1_PID=$!

# Job 3: Logic Puzzle on GPU 2
echo -e "${BLUE}[GPU 2]${NC} Starting Logic Puzzle environment..."
python phase1/phase1_training.py \
    --gpu-id 2 \
    --output-dir "$OUTPUT_BASE/gpu2_logic" \
    --env logic \
    > "$OUTPUT_BASE/gpu2_logic.log" 2>&1 &
GPU2_PID=$!

echo ""
echo "Jobs started with PIDs:"
echo "  GPU 0 (BFS):   $GPU0_PID"
echo "  GPU 1 (Stack): $GPU1_PID"
echo "  GPU 2 (Logic): $GPU2_PID"
echo ""
echo "Monitoring progress..."
echo ""

# Wait for all jobs and collect results
EXIT_CODE=0

wait $GPU0_PID || { echo -e "${BLUE}[GPU 0]${NC} BFS job failed"; EXIT_CODE=1; }
wait $GPU1_PID || { echo -e "${BLUE}[GPU 1]${NC} Stack job failed"; EXIT_CODE=1; }
wait $GPU2_PID || { echo -e "${BLUE}[GPU 2]${NC} Logic job failed"; EXIT_CODE=1; }

echo ""
echo "=========================================="
echo "PHASE 1 JOBS COMPLETE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All Phase 1 jobs completed successfully${NC}"
else
    echo -e "${RED}❌ Some Phase 1 jobs failed${NC}"
fi

echo ""
echo "Results:"
echo "  GPU 0 (BFS):   $OUTPUT_BASE/gpu0_bfs/phase1_results.json"
echo "  GPU 1 (Stack): $OUTPUT_BASE/gpu1_stack/phase1_results.json"
echo "  GPU 2 (Logic): $OUTPUT_BASE/gpu2_logic/phase1_results.json"
echo ""
echo "Logs:"
echo "  GPU 0: $OUTPUT_BASE/gpu0_bfs.log"
echo "  GPU 1: $OUTPUT_BASE/gpu1_stack.log"
echo "  GPU 2: $OUTPUT_BASE/gpu2_logic.log"
echo ""

# Generate summary report
echo "Generating Phase 1 summary report..."
python << 'PYTHON_EOF'
import json
from pathlib import Path

OUTPUT_BASE = Path("phase1_results")
summary = {"experiments": {}}

for gpu_dir in ["gpu0_bfs", "gpu1_stack", "gpu2_logic"]:
    result_file = OUTPUT_BASE / gpu_dir / "phase1_results.json"
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            env_name = list(data.get("environments", {}).keys())[0]
            env_data = data["environments"][env_name]
            
            summary["experiments"][env_name] = {
                "state_dim": env_data.get("state_dim"),
                "samples": env_data.get("train_samples"),
                "expansions": {}
            }
            
            for exp_name, exp_data in env_data.get("expansions", {}).items():
                summary["experiments"][env_name]["expansions"][exp_name] = {
                    "pass": exp_data.get("pass"),
                    "r2_score": exp_data.get("reconstruction", {}).get("r2_score"),
                    "sparsity": exp_data.get("reconstruction", {}).get("sparsity"),
                }

# Write summary
with open(OUTPUT_BASE / "phase1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Print summary
print("\n" + "="*60)
print("PHASE 1 VALIDATION SUMMARY")
print("="*60)
for env_name, env_data in summary.get("experiments", {}).items():
    print(f"\n{env_name.upper()} (State Dim: {env_data['state_dim']}):")
    for exp_name, result in env_data.get("expansions", {}).items():
        status = "✅ PASS" if result['pass'] else "❌ FAIL"
        r2 = result.get('r2_score', 0)
        sparsity = result.get('sparsity', 0)
        print(f"  {exp_name}: {status} | R²={r2:.4f} | Sparsity={sparsity:.1%}")

print("\n" + "="*60)
PYTHON_EOF

exit $EXIT_CODE

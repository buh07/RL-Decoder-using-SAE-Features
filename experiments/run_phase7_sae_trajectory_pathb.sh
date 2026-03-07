#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
if [[ "$MODE" != "launch" ]]; then
  echo "usage: $0 launch" >&2
  exit 2
fi

PY="${PYTHON:-.venv/bin/python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_sae_trajectory_pathb}"
RUN_TAG="${RUN_TAG:-phase7_sae_trajectory_pathb_${RUN_ID}}"
BASE="${BASE:-phase7_results/runs/${RUN_ID}}"
FEATURE_SETS_CSV="${FEATURE_SETS_CSV:-result_top50,eq_pre_result_150,divergent_top50}"

CONTROL_RECORDS="${CONTROL_RECORDS:-phase7_results/interventions/control_records_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4_canary_raw_every2_even.json}"
SAES_DIR="${SAES_DIR:-phase2_results/saes_gpt2_12x_topk/saes}"
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
PHASE4_TOP_FEATURES="${PHASE4_TOP_FEATURES:-phase4_results/topk/probe/top_features_per_layer.json}"
DIVERGENT_SOURCE="${DIVERGENT_SOURCE:-phase7_results/results/phase7_sae_feature_discrimination_phase7_sae_20260306_224419_phase7_sae.json}"
SUBSPACE_SPECS="${SUBSPACE_SPECS:-phase7_results/interventions/variable_subspaces_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json}"

SAMPLE_TRACES="${SAMPLE_TRACES:-0}"
MIN_COMMON_STEPS="${MIN_COMMON_STEPS:-3}"
SEED="${SEED:-20260306}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"

mkdir -p "$BASE/logs" "$BASE/meta" "$BASE/state" phase7_results/results

IFS=',' read -r -a FEATURE_SETS <<< "$FEATURE_SETS_CSV"
if [[ "${#FEATURE_SETS[@]}" -eq 0 ]]; then
  echo "No feature sets provided" >&2
  exit 1
fi

summary_tmp="$BASE/meta/pathb_runs.jsonl"
: > "$summary_tmp"

for fs in "${FEATURE_SETS[@]}"; do
  fs="$(echo "$fs" | xargs)"
  [[ -n "$fs" ]] || continue
  sub_run_id="${RUN_ID}_${fs}"
  sub_run_tag="phase7_sae_trajectory_${sub_run_id}"
  sub_base="phase7_results/runs/${sub_run_id}"
  merged_json="phase7_results/results/phase7_sae_trajectory_coherence_${sub_run_tag}.json"

  echo "[$(date -Is)] launching feature_set=${fs} run_id=${sub_run_id}" | tee -a "$BASE/logs/pathb.log"
  RUN_ID="$sub_run_id" \
  RUN_TAG="$sub_run_tag" \
  BASE="$sub_base" \
  CONTROL_RECORDS="$CONTROL_RECORDS" \
  SAES_DIR="$SAES_DIR" \
  ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
  PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
  DIVERGENT_SOURCE="$DIVERGENT_SOURCE" \
  SUBSPACE_SPECS="$SUBSPACE_SPECS" \
  SAMPLE_TRACES="$SAMPLE_TRACES" \
  MIN_COMMON_STEPS="$MIN_COMMON_STEPS" \
  SEED="$SEED" \
  N_BOOTSTRAP="$N_BOOTSTRAP" \
  BATCH_SIZE="$BATCH_SIZE" \
  FEATURE_SET="$fs" \
  ./experiments/run_phase7_sae_trajectory_coherence.sh launch | tee -a "$BASE/logs/pathb.log"

  coord_session="p7saetc_coord_${sub_run_id}"
  while [[ ! -f "${sub_base}/state/pipeline.done" ]]; do
    if ! tmux has-session -t "$coord_session" 2>/dev/null; then
      echo "Coordinator session ended early for ${sub_run_id} and pipeline.done missing" >&2
      exit 1
    fi
    sleep 20
  done

  if [[ ! -f "$merged_json" ]]; then
    echo "Missing merged output for ${sub_run_id}: $merged_json" >&2
    exit 1
  fi

  echo "{\"feature_set\":\"${fs}\",\"run_id\":\"${sub_run_id}\",\"run_tag\":\"${sub_run_tag}\",\"merged_json\":\"${merged_json}\"}" >> "$summary_tmp"
  echo "[$(date -Is)] completed feature_set=${fs}" | tee -a "$BASE/logs/pathb.log"

done

summary_json="phase7_results/results/phase7_sae_trajectory_pathb_${RUN_ID}.json"
summary_md="phase7_results/results/phase7_sae_trajectory_pathb_${RUN_ID}.md"

export RUN_ID RUN_TAG BASE
$PY - <<'PY'
import json
from pathlib import Path
from datetime import datetime
import os

run_id = os.environ['RUN_ID']
run_tag = os.environ['RUN_TAG']
base = Path(os.environ['BASE'])
summary_tmp = base / 'meta' / 'pathb_runs.jsonl'
out_json = Path(f"phase7_results/results/phase7_sae_trajectory_pathb_{run_id}.json")
out_md = Path(f"phase7_results/results/phase7_sae_trajectory_pathb_{run_id}.md")

rows = []
for line in summary_tmp.read_text().splitlines():
    line = line.strip()
    if line:
        rows.append(json.loads(line))

metrics = ["cosine_smoothness", "feature_variance_coherence", "magnitude_monotonicity_coherence"]
best = None
by_feature_set = {}
for r in rows:
    payload = json.loads(Path(r['merged_json']).read_text())
    r_summary = payload.get('summary', {})
    by_feature_set[r['feature_set']] = {
        'run_id': r['run_id'],
        'run_tag': r['run_tag'],
        'merged_json': r['merged_json'],
        'best_layer_metric': r_summary.get('best_layer_metric'),
        'best_auroc_unfaithful_positive': r_summary.get('best_auroc_unfaithful_positive'),
        'confound_check': r_summary.get('confound_check', {}),
    }
    val = r_summary.get('best_auroc_unfaithful_positive')
    if isinstance(val, (int, float)):
        if best is None or float(val) > float(best['auroc']):
            best = {'feature_set': r['feature_set'], 'auroc': float(val), 'layer_metric': r_summary.get('best_layer_metric')}

out = {
    'schema_version': 'phase7_sae_trajectory_pathb_summary_v1',
    'run_id': run_id,
    'run_tag': run_tag,
    'status': 'ok',
    'feature_sets': [r['feature_set'] for r in rows],
    'by_feature_set': by_feature_set,
    'best_overall': best,
    'timestamp': datetime.now().isoformat(),
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(out, indent=2))

lines = [
    '# Phase 7-SAE Path B Summary',
    '',
    f'- Run id: `{run_id}`',
    f'- Run tag: `{run_tag}`',
    f'- Feature sets: `{", ".join(out["feature_sets"])}`',
    '',
    '## Best Overall',
    '',
    f"- Feature set: `{(best or {}).get('feature_set')}`",
    f"- Best AUROC: `{(best or {}).get('auroc')}`",
    f"- Best layer/metric: `{(best or {}).get('layer_metric')}`",
    '',
    '## Per Feature Set',
    '',
]
for fs, row in by_feature_set.items():
    lines.append(f"- `{fs}`: best=`{row.get('best_auroc_unfaithful_positive')}` at `{row.get('best_layer_metric')}`")

out_md.write_text('\n'.join(lines) + '\n')
(base / 'state' / 'pipeline.done').write_text('done\n')
print(f"Saved {out_json}")
print(f"Saved {out_md}")
PY

echo "[$(date -Is)] PATH B run complete"
echo "summary: $summary_json"

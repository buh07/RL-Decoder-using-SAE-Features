#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-.venv/bin/python3}"
MODE="${1:-}"
if [[ "$MODE" != "launch" ]]; then
  echo "usage: $0 launch [RUN_ID]" >&2
  exit 2
fi

require_file() {
  local p="$1"
  [[ -f "$p" ]] || { echo "missing required file: $p" >&2; exit 2; }
}

wait_for_done() {
  local done_file="$1"
  local session="$2"
  local timeout_sec="${3:-21600}"
  local start
  start="$(date +%s)"
  while [[ ! -f "$done_file" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "session exited before completion: $session (missing $done_file)" >&2
      return 1
    fi
    local now elapsed
    now="$(date +%s)"
    elapsed="$(( now - start ))"
    if (( elapsed > timeout_sec )); then
      echo "timeout waiting for $done_file (session=$session)" >&2
      return 1
    fi
    sleep 20
  done
}

RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)_phase7_sae_d3_redo}"
BASE="phase7_results/runs/${RUN_ID}"
mkdir -p "$BASE"/{logs,meta,state} phase7_results/{controls,results,interventions}

D3_TEST="${D3_TEST:-phase7_results/runs/20260307_192338_phase6v2_d3_operator_fix/dataset/gsm8k_step_traces_test.pt}"
SUBSPACE_SPECS="${SUBSPACE_SPECS:-phase7_results/interventions/variable_subspaces_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json}"

CONTROL_JSON="phase7_results/controls/cot_controls_phase7_sae_d3_${RUN_ID}.json"
CONTROL_RECORDS="phase7_results/interventions/control_records_phase7_sae_d3_${RUN_ID}.json"
BOOTSTRAP_CAUSAL="phase7_results/interventions/causal_checks_phase7_sae_d3_bootstrap_${RUN_ID}.json"

RID_SAE="${RUN_ID}_sae"
RID_SAETC="${RUN_ID}_saetc"
RID_PATHB="${RUN_ID}_pathb"
RID_PATHC="${RUN_ID}_pathc"
RID_PATHCR="${RUN_ID}_pathc_robust"
RID_MIX="${RUN_ID}_mixed"

OUT_FEATURE_JSON="phase7_results/results/phase7_sae_feature_discrimination_phase7_sae_${RID_SAE}.json"
OUT_TRAJ_JSON="phase7_results/results/phase7_sae_trajectory_coherence_phase7_sae_trajectory_${RID_SAETC}.json"
OUT_PATHB_JSON="phase7_results/results/phase7_sae_trajectory_pathb_${RID_PATHB}.json"
OUT_PATHC_JSON="phase7_results/results/phase7_sae_trajectory_pathc_${RID_PATHC}.json"
OUT_PATHCR_JSON="phase7_results/results/phase7_sae_trajectory_pathc_robust_${RID_PATHCR}.json"
OUT_MIX_JSON="phase7_results/results/phase7_mixed_trajectory_validation_phase7_mixed_trajectory_${RID_MIX}.json"

SUMMARY_JSON="phase7_results/results/phase7_sae_d3_redo_summary_${RUN_ID}.json"
SUMMARY_MD="phase7_results/results/phase7_sae_d3_redo_summary_${RUN_ID}.md"

BASELINE_FEATURE="${BASELINE_FEATURE:-phase7_results/results/phase7_sae_feature_discrimination_phase7_sae_20260306_224419_phase7_sae.json}"
BASELINE_TRAJ="${BASELINE_TRAJ:-phase7_results/results/phase7_sae_trajectory_coherence_phase7_sae_trajectory_20260306_232622_phase7_sae_trajectory.json}"
BASELINE_PATHB="${BASELINE_PATHB:-phase7_results/results/phase7_sae_trajectory_pathb_20260306_234123_phase7_sae_trajectory_pathb.json}"
BASELINE_PATHC="${BASELINE_PATHC:-phase7_results/results/phase7_sae_trajectory_pathc_20260306_235018_phase7_sae_trajectory_pathc.json}"
BASELINE_PATHCR="${BASELINE_PATHCR:-phase7_results/results/phase7_sae_trajectory_pathc_robust_20260307_001237_phase7_sae_trajectory_pathc_robust.json}"
BASELINE_MIX="${BASELINE_MIX:-phase7_results/results/phase7_mixed_trajectory_validation_phase7_mixed_trajectory_20260307_012248_phase7_mixed_trajectory_validation.json}"

require_file "$D3_TEST"
require_file "$SUBSPACE_SPECS"
require_file "$BASELINE_FEATURE"
require_file "$BASELINE_TRAJ"
require_file "$BASELINE_PATHB"
require_file "$BASELINE_PATHC"
require_file "$BASELINE_PATHCR"
require_file "$BASELINE_MIX"

echo "[$(date -Is)] run_id=$RUN_ID" | tee -a "$BASE/logs/pipeline.log"

echo "[$(date -Is)] step 1/9: generate controls from D3 trace dataset" | tee -a "$BASE/logs/pipeline.log"
"$PY" phase7/generate_cot_controls.py \
  --trace-dataset "$D3_TEST" \
  --max-traces 500 \
  --seed 17 \
  --output "$CONTROL_JSON" \
  | tee -a "$BASE/logs/generate_controls.log"

echo "[$(date -Is)] step 2/9: build D3 control-conditioned records (GPU5)" | tee -a "$BASE/logs/pipeline.log"
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 "$PY" phase7/causal_intervention_engine.py \
  --trace-dataset "$D3_TEST" \
  --controls "$CONTROL_JSON" \
  --model-key gpt2-medium \
  --parse-mode hybrid \
  --token-anchor eq_like \
  --anchor-priority template_first \
  --control-sampling-policy stratified_trace_variant \
  --subspace-specs "$SUBSPACE_SPECS" \
  --variable subresult_value \
  --layers 7 12 17 22 \
  --max-records 1 \
  --max-records-cap 12000 \
  --record-buffer-size 256 \
  --device cuda:0 \
  --control-records-output "$CONTROL_RECORDS" \
  --output "$BOOTSTRAP_CAUSAL" \
  | tee -a "$BASE/logs/build_control_records.log"

echo "[$(date -Is)] step 3/9: validate D3 control-record artifact" | tee -a "$BASE/logs/pipeline.log"
"$PY" - <<PY | tee -a "$BASE/logs/validate_control_records.log"
import json
from pathlib import Path
import torch
control_records_path = Path(${CONTROL_RECORDS@Q})
payload = json.loads(control_records_path.read_text())
rows_path = control_records_path.with_suffix(control_records_path.suffix + ".rows.pt")
rows = torch.load(rows_path, map_location="cpu")
if not rows:
    raise SystemExit("control-records rows are empty")
shape = tuple(rows[0]["raw_hidden"].shape)
if shape != (24, 1024):
    raise SystemExit(f"unexpected hidden shape: {shape}")
labels = {str(r.get("gold_label", "")) for r in rows}
if not ({"faithful", "unfaithful"} <= labels):
    raise SystemExit(f"missing labels: {labels}")
from collections import defaultdict
f = defaultdict(set)
u = defaultdict(lambda: defaultdict(set))
for r in rows:
    t = str(r.get("trace_id", ""))
    v = str(r.get("control_variant", ""))
    s = int(r.get("step_idx", -1))
    if s < 0:
        continue
    if str(r.get("gold_label")) == "faithful" and v == "faithful":
        f[t].add(s)
    elif str(r.get("gold_label")) == "unfaithful":
        u[t][v].add(s)
eligible = 0
for t, fmap in f.items():
    for v, uset in u.get(t, {}).items():
        if len(fmap.intersection(uset)) >= 3:
            eligible += 1
            break
out = {
    "schema_version": "phase7_sae_d3_control_records_validation_v1",
    "control_records": str(control_records_path),
    "rows_path": str(rows_path),
    "rows_count": int(len(rows)),
    "raw_hidden_shape": list(shape),
    "labels_present": sorted(labels),
    "eligible_trace_count_common_ge3": int(eligible),
    "stats": payload.get("stats", {}),
}
meta_out = Path(${BASE@Q}) / "meta" / "control_records_validation.json"
meta_out.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

echo "[$(date -Is)] step 4/9: run SAE feature discrimination on GPUs 5/6/7" | tee -a "$BASE/logs/pipeline.log"
RUN_ID="$RID_SAE" \
CONTROL_RECORDS="$CONTROL_RECORDS" \
./experiments/run_phase7_sae_faithfulness.sh launch | tee -a "$BASE/logs/sae_feature_launch.log"
wait_for_done "phase7_results/runs/${RID_SAE}/state/pipeline.done" "p7sae_coord_${RID_SAE}" 21600
require_file "$OUT_FEATURE_JSON"

echo "[$(date -Is)] step 5/9: run trajectory coherence baseline on GPUs 5/6/7" | tee -a "$BASE/logs/pipeline.log"
RUN_ID="$RID_SAETC" \
CONTROL_RECORDS="$CONTROL_RECORDS" \
FEATURE_SET="eq_top50" \
DIVERGENT_SOURCE="$OUT_FEATURE_JSON" \
./experiments/run_phase7_sae_trajectory_coherence.sh launch | tee -a "$BASE/logs/trajectory_launch.log"
wait_for_done "phase7_results/runs/${RID_SAETC}/state/pipeline.done" "p7saetc_coord_${RID_SAETC}" 21600
require_file "$OUT_TRAJ_JSON"

echo "[$(date -Is)] step 6/9: run Path B" | tee -a "$BASE/logs/pipeline.log"
RUN_ID="$RID_PATHB" \
CONTROL_RECORDS="$CONTROL_RECORDS" \
DIVERGENT_SOURCE="$OUT_FEATURE_JSON" \
./experiments/run_phase7_sae_trajectory_pathb.sh launch | tee -a "$BASE/logs/pathb.log"
require_file "$OUT_PATHB_JSON"

echo "[$(date -Is)] step 7/9: run Path C" | tee -a "$BASE/logs/pipeline.log"
RUN_ID="$RID_PATHC" \
CONTROL_RECORDS="$CONTROL_RECORDS" \
DIVERGENT_SOURCE="$OUT_FEATURE_JSON" \
./experiments/run_phase7_sae_trajectory_pathc.sh launch | tee -a "$BASE/logs/pathc.log"
require_file "$OUT_PATHC_JSON"

echo "[$(date -Is)] step 8/9: run Path C robust" | tee -a "$BASE/logs/pipeline.log"
RUN_ID="$RID_PATHCR" \
CONTROL_RECORDS="$CONTROL_RECORDS" \
DIVERGENT_SOURCE="$OUT_FEATURE_JSON" \
TRAIN_EXCLUDE_VARIANTS="order_flip_only,answer_first_order_flip,reordered_steps" \
WRONG_INTERMEDIATE_BOOTSTRAP_N=1000 \
CV_FOLDS=5 \
./experiments/run_phase7_sae_trajectory_pathc_robust.sh launch | tee -a "$BASE/logs/pathc_robust.log"
require_file "$OUT_PATHCR_JSON"

echo "[$(date -Is)] step 9/9: run mixed hidden+SAE ladder (GPU5)" | tee -a "$BASE/logs/pipeline.log"
RUN_ID="$RID_MIX" \
CONTROL_RECORDS="$CONTROL_RECORDS" \
MODEL_KEY="gpt2-medium" \
GPU_ID=5 \
SAES_DIR="phase2_results/saes_gpt2_12x_topk/saes" \
DIVERGENT_SOURCE="$OUT_FEATURE_JSON" \
./experiments/run_phase7_mixed_trajectory_validation.sh launch | tee -a "$BASE/logs/mixed_launch.log"
wait_for_done "phase7_results/runs/${RID_MIX}/state/pipeline.done" "p7mix_coord_${RID_MIX}" 21600
require_file "$OUT_MIX_JSON"

echo "[$(date -Is)] writing consolidated summary" | tee -a "$BASE/logs/pipeline.log"
P7_RUN_ID="$RUN_ID" \
P7_CONTROL_RECORDS="$CONTROL_RECORDS" \
P7_CONTROLS="$CONTROL_JSON" \
P7_OUT_FEATURE_JSON="$OUT_FEATURE_JSON" \
P7_OUT_TRAJ_JSON="$OUT_TRAJ_JSON" \
P7_OUT_PATHB_JSON="$OUT_PATHB_JSON" \
P7_OUT_PATHC_JSON="$OUT_PATHC_JSON" \
P7_OUT_PATHCR_JSON="$OUT_PATHCR_JSON" \
P7_OUT_MIX_JSON="$OUT_MIX_JSON" \
P7_BASELINE_FEATURE="$BASELINE_FEATURE" \
P7_BASELINE_TRAJ="$BASELINE_TRAJ" \
P7_BASELINE_PATHB="$BASELINE_PATHB" \
P7_BASELINE_PATHC="$BASELINE_PATHC" \
P7_BASELINE_PATHCR="$BASELINE_PATHCR" \
P7_BASELINE_MIX="$BASELINE_MIX" \
P7_SUMMARY_JSON="$SUMMARY_JSON" \
P7_SUMMARY_MD="$SUMMARY_MD" \
"$PY" - <<'PY'
import json
import os
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text())

def n(v):
    return float(v) if isinstance(v,(int,float)) else None

def delta(new, old):
    if new is None or old is None:
        return None
    return float(new - old)

new_feature = load(os.environ["P7_OUT_FEATURE_JSON"])
new_traj = load(os.environ["P7_OUT_TRAJ_JSON"])
new_pathb = load(os.environ["P7_OUT_PATHB_JSON"])
new_pathc = load(os.environ["P7_OUT_PATHC_JSON"])
new_pathcr = load(os.environ["P7_OUT_PATHCR_JSON"])
new_mix = load(os.environ["P7_OUT_MIX_JSON"])

old_feature = load(os.environ["P7_BASELINE_FEATURE"])
old_traj = load(os.environ["P7_BASELINE_TRAJ"])
old_pathb = load(os.environ["P7_BASELINE_PATHB"])
old_pathc = load(os.environ["P7_BASELINE_PATHC"])
old_pathcr = load(os.environ["P7_BASELINE_PATHCR"])
old_mix = load(os.environ["P7_BASELINE_MIX"])

new_metrics = {
  "feature_best_probe_test_auroc": n((new_feature.get("summary") or {}).get("best_probe_test_auroc")),
  "trajectory_best_auroc": n((new_traj.get("summary") or {}).get("best_auroc_unfaithful_positive")),
  "pathb_best_auroc": n((new_pathb.get("best_overall") or {}).get("auroc")),
  "pathc_wrong_intermediate_probe_auroc": n(new_pathc.get("wrong_intermediate_probe_auroc")),
  "pathc_robust_wrong_intermediate_probe_auroc": n(new_pathcr.get("wrong_intermediate_probe_auroc")),
  "pathc_robust_cv_pooled_auroc": n(((new_pathcr.get("cv_diagnostics") or {}).get("cv_wrong_intermediate_pooled_auroc"))),
  "pathc_robust_gate_pass": bool(new_pathcr.get("robust_wrong_intermediate_gate_pass")),
  "mixed_delta_auroc_mixed_vs_sae": n(new_mix.get("delta_auroc_mixed_vs_sae")),
  "mixed_final_decision": new_mix.get("final_decision"),
}

old_metrics = {
  "feature_best_probe_test_auroc": n((old_feature.get("summary") or {}).get("best_probe_test_auroc")),
  "trajectory_best_auroc": n((old_traj.get("summary") or {}).get("best_auroc_unfaithful_positive")),
  "pathb_best_auroc": n((old_pathb.get("best_overall") or {}).get("auroc")),
  "pathc_wrong_intermediate_probe_auroc": n(old_pathc.get("wrong_intermediate_probe_auroc")),
  "pathc_robust_wrong_intermediate_probe_auroc": n(old_pathcr.get("wrong_intermediate_probe_auroc")),
  "pathc_robust_cv_pooled_auroc": n(((old_pathcr.get("cv_diagnostics") or {}).get("cv_wrong_intermediate_pooled_auroc"))),
  "pathc_robust_gate_pass": bool(old_pathcr.get("robust_wrong_intermediate_gate_pass")),
  "mixed_delta_auroc_mixed_vs_sae": n(old_mix.get("delta_auroc_mixed_vs_sae")),
  "mixed_final_decision": old_mix.get("final_decision"),
}

deltas = {
  k: delta(new_metrics.get(k), old_metrics.get(k))
  for k in [
    "feature_best_probe_test_auroc",
    "trajectory_best_auroc",
    "pathb_best_auroc",
    "pathc_wrong_intermediate_probe_auroc",
    "pathc_robust_wrong_intermediate_probe_auroc",
    "pathc_robust_cv_pooled_auroc",
    "mixed_delta_auroc_mixed_vs_sae",
  ]
}

out = {
  "schema_version": "phase7_sae_d3_redo_summary_v1",
  "run_id": os.environ["P7_RUN_ID"],
  "control_records": os.environ["P7_CONTROL_RECORDS"],
  "controls": os.environ["P7_CONTROLS"],
  "artifacts": {
    "feature_discrimination": os.environ["P7_OUT_FEATURE_JSON"],
    "trajectory_baseline": os.environ["P7_OUT_TRAJ_JSON"],
    "pathb": os.environ["P7_OUT_PATHB_JSON"],
    "pathc": os.environ["P7_OUT_PATHC_JSON"],
    "pathc_robust": os.environ["P7_OUT_PATHCR_JSON"],
    "mixed": os.environ["P7_OUT_MIX_JSON"],
  },
  "baseline_artifacts": {
    "feature_discrimination": os.environ["P7_BASELINE_FEATURE"],
    "trajectory_baseline": os.environ["P7_BASELINE_TRAJ"],
    "pathb": os.environ["P7_BASELINE_PATHB"],
    "pathc": os.environ["P7_BASELINE_PATHC"],
    "pathc_robust": os.environ["P7_BASELINE_PATHCR"],
    "mixed": os.environ["P7_BASELINE_MIX"],
  },
  "new_metrics": new_metrics,
  "baseline_metrics": old_metrics,
  "delta_metrics": deltas,
  "final_verdict": {
    "material_pathc_robust_gain": bool((deltas.get("pathc_robust_wrong_intermediate_probe_auroc") or 0.0) >= 0.03),
    "mixed_decision": new_metrics.get("mixed_final_decision"),
  },
}

Path(os.environ["P7_SUMMARY_JSON"]).write_text(json.dumps(out, indent=2))
lines = [
  "# Phase 7 SAE D3 Redo Summary",
  "",
  f"- Run id: `{out['run_id']}`",
  f"- Control records: `{out['control_records']}`",
  "",
  "## Key Metrics (New vs Baseline vs Delta)",
  "",
]
for k in [
  "feature_best_probe_test_auroc",
  "trajectory_best_auroc",
  "pathb_best_auroc",
  "pathc_wrong_intermediate_probe_auroc",
  "pathc_robust_wrong_intermediate_probe_auroc",
  "pathc_robust_cv_pooled_auroc",
  "mixed_delta_auroc_mixed_vs_sae",
]:
    lines.append(f"- `{k}`: new=`{new_metrics.get(k)}` baseline=`{old_metrics.get(k)}` delta=`{deltas.get(k)}`")
lines += [
  "",
  f"- Path C robust gate pass (new): `{new_metrics.get('pathc_robust_gate_pass')}`",
  f"- Path C robust gate pass (baseline): `{old_metrics.get('pathc_robust_gate_pass')}`",
  f"- Mixed final decision (new): `{new_metrics.get('mixed_final_decision')}`",
  f"- Mixed final decision (baseline): `{old_metrics.get('mixed_final_decision')}`",
  "",
  f"- Material Path C robust gain flag: `{out['final_verdict']['material_pathc_robust_gain']}`",
]
Path(os.environ["P7_SUMMARY_MD"]).write_text("\n".join(lines) + "\n")
print("saved summary json")
print("saved summary md")
PY

echo "done" > "$BASE/state/pipeline.done"
echo "[$(date -Is)] full ladder complete" | tee -a "$BASE/logs/pipeline.log"
echo "summary json: $SUMMARY_JSON"
echo "summary md: $SUMMARY_MD"

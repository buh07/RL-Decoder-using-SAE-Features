#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
PY="${PYTHON:-.venv/bin/python3}"

usage() {
  cat <<'USAGE'
usage: experiments/run_phase7_g2_cross_task.sh {launch|coordinator}
USAGE
}

require_env() {
  : "${RUN_ID:?RUN_ID required}"
  : "${RUN_TAG:?RUN_TAG required}"
  : "${BASE:?BASE required}"
}

json_parseable() {
  local p="$1"
  "$PY" - <<PY
import json,sys
try:
    json.load(open(${p@Q}))
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

wait_for_file_with_session() {
  local done_file="$1"
  local session="$2"
  local timeout_sec="${3:-172800}"
  local start now elapsed
  start="$(date +%s)"
  while [[ ! -f "$done_file" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      for _ in {1..6}; do
        sleep 5
        [[ -f "$done_file" ]] && return 0
      done
      echo "session exited before completion: $session (missing $done_file)" >&2
      return 1
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > timeout_sec )); then
      echo "timeout waiting for $done_file (session=$session)" >&2
      return 1
    fi
    sleep 15
  done
}

run_domain_optionc() {
  local domain="$1"
  local sub_run_id="$2"
  local sub_run_tag="phase7_optionc_generated_${sub_run_id}"
  local sub_base="phase7_results/runs/${sub_run_id}"
  local sub_sess="p7oc_coord_${sub_run_id}"

  RUN_ID="$sub_run_id" \
  RUN_TAG="$sub_run_tag" \
  BASE="$sub_base" \
  MODEL_KEY="$MODEL_KEY" \
  DOMAIN="$domain" \
  GPU_IDS_CSV="$GPU_IDS_CSV" \
  PRECOMPUTE_SHARD_GPUS="$PRECOMPUTE_SHARD_GPUS" \
  CANARY_PAIRS="$CANARY_PAIRS" \
  FULL_PAIRS="$FULL_PAIRS" \
  CANARY_BOOTSTRAP_N="$CANARY_BOOTSTRAP_N" \
  FULL_BOOTSTRAP_N="$FULL_BOOTSTRAP_N" \
  CPU_WORKERS="$CPU_WORKERS" \
  HF_TOKEN="${HF_TOKEN:-}" \
  experiments/run_phase7_optionc_generated.sh launch

  wait_for_file_with_session "${sub_base}/state/pipeline.done" "$sub_sess" "$WAIT_TIMEOUT_S"

  local summary="phase7_results/results/optionc_summary_${sub_run_id}.json"
  local claim="phase7_results/results/optionc_claim_boundary_${sub_run_id}_full.json"
  [[ -f "$summary" ]] || { echo "missing summary for $domain: $summary" >&2; return 1; }
  [[ -f "$claim" ]] || { echo "missing claim for $domain: $claim" >&2; return 1; }
  json_parseable "$summary"
  json_parseable "$claim"
}

run_coordinator() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/meta" "$BASE/state"
  local logf="$BASE/logs/coordinator.log"
  {
    echo "[$(date -Is)] g2 coordinator start run_id=$RUN_ID"
    run_domain_optionc "prontoqa" "$PRONTO_RUN_ID"
    run_domain_optionc "entailmentbank" "$ENTAIL_RUN_ID"

    "$PY" - <<PY
import json
from pathlib import Path
run_id=${RUN_ID@Q}
base=Path(${BASE@Q})
pr_id=${PRONTO_RUN_ID@Q}
en_id=${ENTAIL_RUN_ID@Q}


def load(path):
    return json.load(open(path))

pr_sum=load(f"phase7_results/results/optionc_summary_{pr_id}.json")
en_sum=load(f"phase7_results/results/optionc_summary_{en_id}.json")
pr_claim=load(f"phase7_results/results/optionc_claim_boundary_{pr_id}_full.json")
en_claim=load(f"phase7_results/results/optionc_claim_boundary_{en_id}_full.json")

pr_strict=bool((pr_sum.get("full") or {}).get("strict_gate_pass"))
en_strict=bool((en_sum.get("full") or {}).get("strict_gate_pass"))
pr_claim_enabled=bool(pr_claim.get("faithfulness_claim_enabled"))
en_claim_enabled=bool(en_claim.get("faithfulness_claim_enabled"))

publishable=bool(pr_strict and en_strict and pr_claim_enabled and en_claim_enabled)
status="pass" if publishable else "fail"

out={
  "schema_version":"phase7_g2_cross_task_decision_v1",
  "status":"ok",
  "run_id":run_id,
  "domains_order":["prontoqa","entailmentbank"],
  "domains":{
    "prontoqa":{
      "run_id":pr_id,
      "summary_path":f"phase7_results/results/optionc_summary_{pr_id}.json",
      "claim_path":f"phase7_results/results/optionc_claim_boundary_{pr_id}_full.json",
      "cv_primary_pooled_auroc":(pr_sum.get("full") or {}).get("cv_primary_pooled_auroc"),
      "lexical_probe_auroc":(pr_sum.get("full") or {}).get("lexical_probe_auroc"),
      "wrong_minus_lexical_delta":(pr_sum.get("full") or {}).get("wrong_minus_lexical_delta"),
      "strict_gate_pass":pr_strict,
      "faithfulness_claim_enabled":pr_claim_enabled,
    },
    "entailmentbank":{
      "run_id":en_id,
      "summary_path":f"phase7_results/results/optionc_summary_{en_id}.json",
      "claim_path":f"phase7_results/results/optionc_claim_boundary_{en_id}_full.json",
      "cv_primary_pooled_auroc":(en_sum.get("full") or {}).get("cv_primary_pooled_auroc"),
      "lexical_probe_auroc":(en_sum.get("full") or {}).get("lexical_probe_auroc"),
      "wrong_minus_lexical_delta":(en_sum.get("full") or {}).get("wrong_minus_lexical_delta"),
      "strict_gate_pass":en_strict,
      "faithfulness_claim_enabled":en_claim_enabled,
    },
  },
  "publishability_gate":{
    "policy":"both_domains_must_pass",
    "publishable_cross_domain_pass":publishable,
    "final_recommendation":("publishable_cross_domain" if publishable else "partial_generalization_or_fail"),
    "status":status,
  },
}

out_path=Path(f"phase7_results/results/trackc_g2_cross_task_decision_{run_id}.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))

md=[
  "# Track C G2 Cross-Task Decision",
  "",
  f"- Run: `{run_id}`",
  f"- PrOntoQA strict gate: `{pr_strict}`",
  f"- EntailmentBank strict gate: `{en_strict}`",
  f"- Publishability gate (both pass): `{publishable}`",
  f"- Final recommendation: `{out['publishability_gate']['final_recommendation']}`",
]
Path(f"phase7_results/results/trackc_g2_cross_task_decision_{run_id}.md").write_text("\n".join(md)+"\n")
base.joinpath("meta","summary.json").write_text(json.dumps(out, indent=2))
print(out_path)
PY

    touch "$BASE/state/pipeline.done"
    echo "[$(date -Is)] g2 coordinator done run_id=$RUN_ID"
  } >>"$logf" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_g2_cross_task}"
    RUN_TAG="${RUN_TAG:-phase7_g2_cross_task_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"

    MODEL_KEY="${MODEL_KEY:-qwen2.5-7b}"
    GPU_IDS_CSV="${GPU_IDS_CSV:-5,6,7}"
    PRECOMPUTE_SHARD_GPUS="${PRECOMPUTE_SHARD_GPUS:-${GPU_IDS_CSV}}"
    CANARY_PAIRS="${CANARY_PAIRS:-200}"
    FULL_PAIRS="${FULL_PAIRS:-1000}"
    CANARY_BOOTSTRAP_N="${CANARY_BOOTSTRAP_N:-300}"
    FULL_BOOTSTRAP_N="${FULL_BOOTSTRAP_N:-1000}"
    CPU_WORKERS="${CPU_WORKERS:-8}"
    WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-172800}"

    PRONTO_RUN_ID="${PRONTO_RUN_ID:-${RUN_ID}_prontoqa}"
    ENTAIL_RUN_ID="${ENTAIL_RUN_ID:-${RUN_ID}_entailmentbank}"

    mkdir -p "$BASE/logs" "$BASE/meta" "$BASE/state"
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=${RUN_ID@Q}
RUN_TAG=${RUN_TAG@Q}
BASE=${BASE@Q}
MODEL_KEY=${MODEL_KEY@Q}
GPU_IDS_CSV=${GPU_IDS_CSV@Q}
PRECOMPUTE_SHARD_GPUS=${PRECOMPUTE_SHARD_GPUS@Q}
CANARY_PAIRS=${CANARY_PAIRS@Q}
FULL_PAIRS=${FULL_PAIRS@Q}
CANARY_BOOTSTRAP_N=${CANARY_BOOTSTRAP_N@Q}
FULL_BOOTSTRAP_N=${FULL_BOOTSTRAP_N@Q}
CPU_WORKERS=${CPU_WORKERS@Q}
WAIT_TIMEOUT_S=${WAIT_TIMEOUT_S@Q}
PRONTO_RUN_ID=${PRONTO_RUN_ID@Q}
ENTAIL_RUN_ID=${ENTAIL_RUN_ID@Q}
HF_TOKEN=${HF_TOKEN@Q}
CFG

    COORD_SESSION="p7g2_coord_${RUN_ID}"
    tmux has-session -t "$COORD_SESSION" 2>/dev/null && tmux kill-session -t "$COORD_SESSION"
    tmux new-session -d -s "$COORD_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched g2 cross-task"
    echo "  run_id: $RUN_ID"
    echo "  coordinator: $COORD_SESSION"
    echo "  pronto_run_id: $PRONTO_RUN_ID"
    echo "  entail_run_id: $ENTAIL_RUN_ID"
    ;;
  coordinator)
    run_coordinator
    ;;
  *)
    usage
    exit 2
    ;;
esac

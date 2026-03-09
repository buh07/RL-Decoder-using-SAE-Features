#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
PY="${PYTHON:-.venv/bin/python3}"

usage() {
  cat <<'USAGE'
usage: experiments/run_phase7_prontoqa_trackc.sh {launch|precompute|patha|pathb|pathc|stress|coordinator}
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
  local timeout_sec="${3:-86400}"
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
    elapsed="$(( now - start ))"
    if (( elapsed > timeout_sec )); then
      echo "timeout waiting for $done_file (session=$session)" >&2
      return 1
    fi
    sleep 10
  done
}

_sha256_file_py() {
  local p="$1"
  "$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${p@Q}))
PY
}

_expected_patha_hash() {
  local scope="$1"
  local records="$2"
  local sample="$3"
  local n_bootstrap_scope="$4"
  local records_sha
  records_sha="$(_sha256_file_py "$records")"
  "$PY" - <<PY
import hashlib, json
obj = {
  "stage":"patha",
  "scope": ${scope@Q},
  "model_key": ${MODEL_KEY@Q},
  "layers_csv": ${LAYERS_CSV@Q},
  "control_records": ${records@Q},
  "control_records_sha256": ${records_sha@Q},
  "sample_traces": int(${sample@Q}),
  "min_common_steps": int(${MIN_COMMON_STEPS@Q}),
  "seed": int(${SEED@Q}),
  "n_bootstrap": int(${n_bootstrap_scope@Q}),
  "batch_size": int(${BATCH_SIZE@Q}),
}
print(hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest())
PY
}

_expected_pathb_hash() {
  local scope="$1"
  local records="$2"
  local sample="$3"
  local n_bootstrap_scope="$4"
  local divergent="$5"
  local records_sha divergent_sha
  records_sha="$(_sha256_file_py "$records")"
  divergent_sha="$(_sha256_file_py "$divergent")"
  "$PY" - <<PY
import hashlib, json
obj = {
  "stage":"pathb",
  "scope": ${scope@Q},
  "model_key": ${MODEL_KEY@Q},
  "layers_csv": ${LAYERS_CSV@Q},
  "control_records": ${records@Q},
  "control_records_sha256": ${records_sha@Q},
  "feature_sets_csv": "result_top50,eq_pre_result_150,divergent_top50",
  "divergent_source": ${divergent@Q},
  "divergent_sha256": ${divergent_sha@Q},
  "sample_traces": int(${sample@Q}),
  "min_common_steps": int(${MIN_COMMON_STEPS@Q}),
  "seed": int(${SEED@Q}),
  "n_bootstrap": int(${n_bootstrap_scope@Q}),
  "batch_size": int(${BATCH_SIZE@Q}),
}
print(hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest())
PY
}

_expected_pathc_hash() {
  local scope="$1"
  local records="$2"
  local sample="$3"
  local n_bootstrap_scope="$4"
  local wrong_bootstrap_n="$5"
  local cv_folds_scope="$6"
  local divergent="$7"
  local model_ladder_local="$8"
  local decoder_checkpoint_local="$9"
  local records_sha divergent_sha
  records_sha="$(_sha256_file_py "$records")"
  divergent_sha="$(_sha256_file_py "$divergent")"
  "$PY" - <<PY
import hashlib, json
obj = {
  "stage":"pathc",
  "scope": ${scope@Q},
  "model_key": ${MODEL_KEY@Q},
  "layers_csv": ${LAYERS_CSV@Q},
  "control_records": ${records@Q},
  "control_records_sha256": ${records_sha@Q},
  "divergent_source": ${divergent@Q},
  "divergent_source_sha256": ${divergent_sha@Q},
  "feature_set": "eq_pre_result_150",
  "sample_traces": int(${sample@Q}),
  "min_common_steps": int(${MIN_COMMON_STEPS@Q}),
  "seed": int(${SEED@Q}),
  "n_bootstrap": int(${n_bootstrap_scope@Q}),
  "wrong_bootstrap_n": int(${wrong_bootstrap_n@Q}),
  "cv_folds": int(${cv_folds_scope@Q}),
  "model_ladder": ${model_ladder_local@Q},
  "decoder_checkpoint": ${decoder_checkpoint_local@Q},
}
print(hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest())
PY
}

_validate_stage_with_hash() {
  local stage="$1"
  local scope="$2"
  local done_marker="$BASE/state/${stage}_${scope}.done"
  [[ -f "$done_marker" ]] || return 1
  local meta_path="$BASE/meta/${stage}_${scope}.json"
  [[ -f "$meta_path" ]] || return 1
  json_parseable "$meta_path" || return 1

  "$PY" - <<PY
import json
from pathlib import Path
from phase7.common import sha256_file

stage = ${stage@Q}
scope = ${scope@Q}
meta = json.loads(Path(${meta_path@Q}).read_text())
if stage == "pathc":
    sm = meta.get("_stage_meta", {})
else:
    sm = meta
out = sm.get("output_json")
out_sha = sm.get("output_sha256")
cfg_hash = sm.get("config_hash")
if not isinstance(out, str) or not out:
    raise SystemExit(1)
op = Path(out)
if not op.exists():
    raise SystemExit(1)
json.loads(op.read_text())
if not isinstance(out_sha, str) or sha256_file(op) != out_sha:
    raise SystemExit(1)
if not isinstance(cfg_hash, str) or not cfg_hash:
    raise SystemExit(1)
print(cfg_hash)
PY
}

run_precompute() {
  require_env
  : "${PREP_KIND:?PREP_KIND required (canary|full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/precompute_${PREP_KIND}.log"
  local sample_size="$CANARY_TRACES"
  local prep_dir="$PREP_CANARY_DIR"
  local done_marker="$BASE/state/precompute_canary.done"
  if [[ "$PREP_KIND" == "full" ]]; then
    sample_size="$FULL_TRACES"
    prep_dir="$PREP_FULL_DIR"
    done_marker="$BASE/state/precompute_full.done"
  fi

  {
    echo "[$(date -Is)] precompute start kind=${PREP_KIND} sample_size=${sample_size}"
    if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
      export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    fi
    local shard_gpus_csv="${PRECOMPUTE_SHARD_GPUS_CSV:-$PRECOMPUTE_GPU}"
    local -a shard_gpus=()
    IFS=',' read -r -a shard_gpus <<< "$shard_gpus_csv"
    if [[ "${#shard_gpus[@]}" -eq 0 ]]; then
      shard_gpus=("$PRECOMPUTE_GPU")
    fi
    local gen_do_sample_flag=()
    if [[ "${PRONTOQA_GEN_DO_SAMPLE:-0}" == "1" ]]; then
      gen_do_sample_flag=(--gen-do-sample)
    fi
    local resume_flag=()
    if [[ "${PRECOMPUTE_RESUME:-1}" == "1" ]]; then
      resume_flag=(--resume)
    fi
    local gen_batch_size="${PRONTOQA_GEN_BATCH_SIZE:-12}"
    local fwd_batch_size="${PRONTOQA_FORWARD_BATCH_SIZE:-8}"
    local checkpoint_every="${PRECOMPUTE_CHECKPOINT_EVERY:-100}"
    if [[ "${#shard_gpus[@]}" -le 1 ]]; then
      CUDA_VISIBLE_DEVICES="${shard_gpus[0]}" "$PY" phase7/prontoqa_prepare_dataset.py \
        --model-key "$MODEL_KEY" \
        --sample-size "$sample_size" \
        --seed "$SEED" \
        --chain-len-min "$CHAIN_LEN_MIN" \
        --chain-len-max "$CHAIN_LEN_MAX" \
        --variants "$PRONTOQA_VARIANTS" \
        --cot-source "$PRONTOQA_COT_SOURCE" \
        --gen-max-new-tokens "$PRONTOQA_GEN_MAX_NEW_TOKENS" \
        --gen-temperature "$PRONTOQA_GEN_TEMPERATURE" \
        --gen-top-p "$PRONTOQA_GEN_TOP_P" \
        "${gen_do_sample_flag[@]}" \
        --gen-retries "$PRONTOQA_GEN_RETRIES" \
        --gen-batch-size "$gen_batch_size" \
        --forward-batch-size "$fwd_batch_size" \
        --checkpoint-every "$checkpoint_every" \
        "${resume_flag[@]}" \
        --shard-index 0 \
        --num-shards 1 \
        --device cuda:0 \
        --max-records "$MAX_RECORDS" \
        --output-dir "$prep_dir" \
        --trace-output "$prep_dir/dataset/prontoqa_step_traces_test.pt" \
        --controls-output "$prep_dir/controls/cot_controls_prontoqa.json" \
        --control-records-output "$prep_dir/interventions/control_records_prontoqa.json" \
        --manifest-output "$prep_dir/meta/manifest.json"
    else
      echo "[$(date -Is)] precompute sharded across gpus=${shard_gpus_csv}"
      local shard_root="$prep_dir/shards"
      mkdir -p "$shard_root" "$prep_dir/meta"
      local shard_list="$prep_dir/meta/shard_manifests.list"
      : > "$shard_list"
      local -a pids=()
      local shard_idx=0
      local gpu shard_dir shard_manifest shard_log
      for gpu in "${shard_gpus[@]}"; do
        shard_dir="$shard_root/shard_${shard_idx}"
        shard_manifest="$shard_dir/meta/manifest.json"
        shard_log="$BASE/logs/precompute_${PREP_KIND}_shard${shard_idx}.log"
        mkdir -p "$shard_dir/meta"
        if [[ "${PRECOMPUTE_RESUME:-1}" == "1" && -f "$shard_manifest" ]]; then
          if "$PY" - <<PY
import json
from pathlib import Path
p=Path(${shard_manifest@Q})
ok=False
if p.exists():
    try:
        d=json.loads(p.read_text())
        ok = bool(d.get("status")=="ok")
    except Exception:
        ok=False
raise SystemExit(0 if ok else 1)
PY
          then
            echo "$shard_manifest" >> "$shard_list"
            echo "[$(date -Is)] shard ${shard_idx} already complete; reusing"
            shard_idx=$((shard_idx + 1))
            continue
          fi
        fi
        (
          set -euo pipefail
          CUDA_VISIBLE_DEVICES="$gpu" "$PY" phase7/prontoqa_prepare_dataset.py \
            --model-key "$MODEL_KEY" \
            --sample-size "$sample_size" \
            --seed "$SEED" \
            --chain-len-min "$CHAIN_LEN_MIN" \
            --chain-len-max "$CHAIN_LEN_MAX" \
            --variants "$PRONTOQA_VARIANTS" \
            --cot-source "$PRONTOQA_COT_SOURCE" \
            --gen-max-new-tokens "$PRONTOQA_GEN_MAX_NEW_TOKENS" \
            --gen-temperature "$PRONTOQA_GEN_TEMPERATURE" \
            --gen-top-p "$PRONTOQA_GEN_TOP_P" \
            "${gen_do_sample_flag[@]}" \
            --gen-retries "$PRONTOQA_GEN_RETRIES" \
            --gen-batch-size "$gen_batch_size" \
            --forward-batch-size "$fwd_batch_size" \
            --checkpoint-every "$checkpoint_every" \
            "${resume_flag[@]}" \
            --shard-index "$shard_idx" \
            --num-shards "${#shard_gpus[@]}" \
            --device cuda:0 \
            --max-records 0 \
            --output-dir "$shard_dir" \
            --trace-output "$shard_dir/dataset/prontoqa_step_traces_test.pt" \
            --controls-output "$shard_dir/controls/cot_controls_prontoqa.json" \
            --control-records-output "$shard_dir/interventions/control_records_prontoqa.json" \
            --manifest-output "$shard_dir/meta/manifest.json"
        ) >>"$shard_log" 2>&1 &
        pids+=("$!")
        echo "$shard_manifest" >> "$shard_list"
        shard_idx=$((shard_idx + 1))
      done

      local pid
      for pid in "${pids[@]}"; do
        wait "$pid"
      done

      "$PY" - <<PY
import json
from pathlib import Path
import torch
from phase7.common import save_json, save_pt, sha256_file

prep_dir = Path(${prep_dir@Q})
sample_size = int(${sample_size@Q})
max_records = int(${MAX_RECORDS@Q})
seed = int(${SEED@Q})
variants = [x.strip() for x in ${PRONTOQA_VARIANTS@Q}.split(",") if x.strip()]
manifest_list = Path(prep_dir / "meta" / "shard_manifests.list")
manifest_paths = [Path(x.strip()) for x in manifest_list.read_text().splitlines() if x.strip()]
if not manifest_paths:
    raise SystemExit("no shard manifests to merge")

manifests = []
for mp in manifest_paths:
    if not mp.exists():
        raise SystemExit(f"missing shard manifest: {mp}")
    m = json.loads(mp.read_text())
    if str(m.get("status")) != "ok":
        raise SystemExit(f"shard manifest not ok: {mp}")
    for out_k, sha_k in [
        ("trace_output", "trace_sha256"),
        ("controls_output", "controls_sha256"),
        ("control_records_output", "control_records_sha256"),
        ("rows_pt_output", "rows_pt_sha256"),
    ]:
        op = Path(str(m.get(out_k, "")))
        if not op.exists():
            raise SystemExit(f"missing shard output {out_k}: {op}")
        expected = str(m.get(sha_k, ""))
        actual = sha256_file(op)
        if not expected or actual != expected:
            raise SystemExit(f"hash mismatch for {op}: expected={expected} actual={actual}")
    manifests.append(m)

trace_rows = []
controls = []
records = []
for m in manifests:
    trace_rows.extend(list(torch.load(str(m["trace_output"]), map_location="cpu")))
    cobj = json.loads(Path(str(m["controls_output"])).read_text())
    controls.extend(list(cobj.get("controls", [])))
    robj = json.loads(Path(str(m["control_records_output"])).read_text())
    records.extend(list(torch.load(str(robj["rows_path"]), map_location="cpu")))

def dedupe(rows, key_fn):
    out = []
    seen = set()
    for r in rows:
        k = key_fn(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

trace_rows = dedupe(
    trace_rows,
    lambda r: (str(r.get("trace_id", "")), int(r.get("step_idx", -1))),
)
controls = dedupe(
    controls,
    lambda c: (str(c.get("trace_id", "")), str(c.get("variant", "")), int(c.get("example_idx", -1))),
)
records = dedupe(
    records,
    lambda r: (str(r.get("trace_id", "")), str(r.get("control_variant", "")), int(r.get("step_idx", -1)), int(r.get("line_index", -1))),
)

trace_rows.sort(key=lambda r: (str(r.get("trace_id", "")), int(r.get("step_idx", -1))))
controls.sort(key=lambda c: (str(c.get("trace_id", "")), str(c.get("variant", "")), int(c.get("example_idx", -1))))
records.sort(key=lambda r: (str(r.get("trace_id", "")), str(r.get("control_variant", "")), int(r.get("step_idx", -1)), int(r.get("line_index", -1))))
if max_records > 0:
    records = records[:max_records]

trace_out = prep_dir / "dataset" / "prontoqa_step_traces_test.pt"
controls_out = prep_dir / "controls" / "cot_controls_prontoqa.json"
records_out = prep_dir / "interventions" / "control_records_prontoqa.json"
manifest_out = prep_dir / "meta" / "manifest.json"
rows_out = Path(str(records_out) + ".rows.pt")
save_pt(trace_out, trace_rows)
save_json(
    controls_out,
    {
        "schema_version": "phase7_cot_control_v1",
        "source_dataset": ("prontoqa_model_generated" if ${PRONTOQA_COT_SOURCE@Q} == "model_generated" else "prontoqa_synthetic"),
        "model_key": ${MODEL_KEY@Q},
        "num_controls": int(len(controls)),
        "num_examples": int(len({str(c.get("trace_id","")) for c in controls})),
        "variants": variants,
        "cot_source": ${PRONTOQA_COT_SOURCE@Q},
        "merged_from_shards": [str(x) for x in manifest_paths],
        "controls": controls,
    },
)
save_pt(rows_out, records)
payload = {
    "schema_version": "phase7_control_records_v2",
    "status": "ok",
    "model_key": ${MODEL_KEY@Q},
    "model_family": manifests[0].get("model_key", ${MODEL_KEY@Q}),
    "num_layers": int(records[0].get("num_layers", 0)) if records else 0,
    "hidden_dim": int(records[0].get("hidden_dim", 0)) if records else 0,
    "rows_format": "pt",
    "rows_path": str(rows_out),
    "rows_inline": False,
    "rows_count": int(len(records)),
    "rows_sha256": sha256_file(rows_out),
    "stats": {
        "unique_traces_used": int(len({str(r.get("trace_id", "")) for r in records})),
        "unique_variants_used": int(len({str(r.get("control_variant", "")) for r in records})),
        "faithful_rows": int(sum(1 for r in records if str(r.get("gold_label")) == "faithful")),
        "unfaithful_rows": int(sum(1 for r in records if str(r.get("gold_label")) == "unfaithful")),
        "controls_used_fraction": float(len(records) / max(1, len(controls))),
    },
    "source": {
        "generator": "run_phase7_prontoqa_trackc.sh::precompute_sharded_merge",
        "sample_size_requested": sample_size,
        "seed": seed,
        "max_records": max_records,
        "cot_source": ${PRONTOQA_COT_SOURCE@Q},
        "shard_manifests": [str(x) for x in manifest_paths],
    },
    "timestamp": __import__("datetime").datetime.now().isoformat(),
}
save_json(records_out, payload)
manifest = {
    "schema_version": "phase7_prontoqa_prepare_manifest_v3",
    "status": "ok",
    "model_key": ${MODEL_KEY@Q},
    "cot_source": ${PRONTOQA_COT_SOURCE@Q},
    "trace_output": str(trace_out),
    "trace_sha256": sha256_file(trace_out),
    "controls_output": str(controls_out),
    "controls_sha256": sha256_file(controls_out),
    "control_records_output": str(records_out),
    "control_records_sha256": sha256_file(records_out),
    "rows_pt_output": str(rows_out),
    "rows_pt_sha256": sha256_file(rows_out),
    "num_trace_rows": int(len(trace_rows)),
    "num_controls": int(len(controls)),
    "num_control_records": int(len(records)),
    "variants": variants,
    "merged_from_shards": [str(x) for x in manifest_paths],
    "timestamp": __import__("datetime").datetime.now().isoformat(),
}
save_json(manifest_out, manifest)
print(json.dumps(manifest, indent=2))
PY
    fi

    [[ -f "$prep_dir/interventions/control_records_prontoqa.json" ]] || { echo "missing control records for $PREP_KIND" >&2; exit 1; }
    json_parseable "$prep_dir/interventions/control_records_prontoqa.json" || { echo "bad control records json for $PREP_KIND" >&2; exit 1; }
    touch "$done_marker"
    echo "[$(date -Is)] precompute done kind=${PREP_KIND}"
  } >>"$logf" 2>&1
}

run_patha() {
  require_env
  : "${SCOPE:?SCOPE required (canary|full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/patha_${SCOPE}.log"
  local records="$CANARY_CONTROL_RECORDS"
  local sample="$CANARY_TRACES"
  local n_bootstrap_scope="${CANARY_N_BOOTSTRAP:-${N_BOOTSTRAP:-500}}"
  local sub_run_id="${RUN_ID}_patha_canary"
  if [[ "$SCOPE" == "full" ]]; then
    records="$FULL_CONTROL_RECORDS"
    sample="0"
    n_bootstrap_scope="${FULL_N_BOOTSTRAP:-${N_BOOTSTRAP:-1000}}"
    sub_run_id="${RUN_ID}_patha_full"
  fi
  local sub_run_tag="phase7_prontoqa_patha_${sub_run_id}"
  local out_json="phase7_results/results/phase7_sae_trajectory_coherence_${sub_run_tag}.json"
  {
    echo "[$(date -Is)] patha start scope=${SCOPE}"
    RUN_ID="$sub_run_id" \
    RUN_TAG="$sub_run_tag" \
    BASE="phase7_results/runs/${sub_run_id}" \
    CONTROL_RECORDS="$records" \
    MODEL_KEY="$MODEL_KEY" \
    SAES_DIR="$SAES_DIR" \
    ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
    PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
    FEATURE_SET="eq_top50" \
    DIVERGENT_SOURCE="$BASE/meta/divergent_${SCOPE}.json" \
    SUBSPACE_SPECS="$SUBSPACE_SPECS" \
    SAMPLE_TRACES="$sample" \
    MIN_COMMON_STEPS="$MIN_COMMON_STEPS" \
    SEED="$SEED" \
    N_BOOTSTRAP="$n_bootstrap_scope" \
    BATCH_SIZE="$BATCH_SIZE" \
    EMIT_SAMPLES=1 \
    LAYERS_CSV="$LAYERS_CSV" \
    GPU_IDS_CSV="5,6,7" \
    ./experiments/run_phase7_sae_trajectory_coherence.sh launch

    local coord_sess="p7saetc_coord_${sub_run_id}"
    while [[ ! -f "phase7_results/runs/${sub_run_id}/state/pipeline.done" ]]; do
      if ! tmux has-session -t "$coord_sess" 2>/dev/null; then
        sleep 5
        [[ -f "phase7_results/runs/${sub_run_id}/state/pipeline.done" ]] || { echo "patha coordinator disappeared: ${coord_sess}" >&2; exit 1; }
      fi
      sleep 15
    done
    [[ -f "$out_json" ]] || { echo "missing patha output: $out_json" >&2; exit 1; }
    json_parseable "$out_json" || { echo "bad patha output: $out_json" >&2; exit 1; }
    local records_sha
    records_sha="$("$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${records@Q}))
PY
)"
    local cfg_hash
    cfg_hash="$("$PY" - <<PY
import hashlib, json
obj = {
  "stage":"patha",
  "scope": ${SCOPE@Q},
  "model_key": ${MODEL_KEY@Q},
  "layers_csv": ${LAYERS_CSV@Q},
  "control_records": ${records@Q},
  "control_records_sha256": ${records_sha@Q},
  "sample_traces": int(${sample@Q}),
  "min_common_steps": int(${MIN_COMMON_STEPS@Q}),
  "seed": int(${SEED@Q}),
  "n_bootstrap": int(${n_bootstrap_scope@Q}),
  "batch_size": int(${BATCH_SIZE@Q}),
}
print(hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest())
PY
)"
    local out_sha
    out_sha="$("$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${out_json@Q}))
PY
)"
    "$PY" - <<PY
import json
from pathlib import Path
meta = {
  "scope": ${SCOPE@Q},
  "run_id": ${sub_run_id@Q},
  "run_tag": ${sub_run_tag@Q},
  "output_json": ${out_json@Q},
  "output_sha256": ${out_sha@Q},
  "control_records": ${records@Q},
  "control_records_sha256": ${records_sha@Q},
  "config_hash": ${cfg_hash@Q},
}
Path(${BASE@Q}).joinpath("meta","patha_" + ${SCOPE@Q} + ".json").write_text(json.dumps(meta, indent=2))
PY
    touch "$BASE/state/patha_${SCOPE}.done"
    echo "[$(date -Is)] patha done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_pathb() {
  require_env
  : "${SCOPE:?SCOPE required (canary|full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/pathb_${SCOPE}.log"
  local records="$CANARY_CONTROL_RECORDS"
  local sample="$CANARY_TRACES"
  local n_bootstrap_scope="${CANARY_N_BOOTSTRAP:-${N_BOOTSTRAP:-500}}"
  local sub_run_id="${RUN_ID}_pathb_canary"
  local fd_run_id="${RUN_ID}_fd_canary"
  if [[ "$SCOPE" == "full" ]]; then
    records="$FULL_CONTROL_RECORDS"
    sample="0"
    n_bootstrap_scope="${FULL_N_BOOTSTRAP:-${N_BOOTSTRAP:-1000}}"
    sub_run_id="${RUN_ID}_pathb_full"
    fd_run_id="${RUN_ID}_fd_full"
  fi
  local fd_run_tag="phase7_prontoqa_fd_${fd_run_id}"
  local divergent="phase7_results/results/phase7_sae_feature_discrimination_${fd_run_tag}.json"
  local out_json="phase7_results/results/phase7_sae_trajectory_pathb_${sub_run_id}.json"
  {
    echo "[$(date -Is)] pathb start scope=${SCOPE}"
    if [[ ! -f "$divergent" ]]; then
      RUN_ID="$fd_run_id" \
      RUN_TAG="$fd_run_tag" \
      BASE="phase7_results/runs/${fd_run_id}" \
      CONTROL_RECORDS="$records" \
      MODEL_KEY="$MODEL_KEY" \
      SAES_DIR="$SAES_DIR" \
      ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
      PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
      SAMPLE_TRACES="$sample" \
      SEED="$SEED" \
      N_PERMUTATIONS=250 \
      TRACE_TEST_FRACTION=0.20 \
      PROBE_EPOCHS=80 \
      PROBE_LR=0.001 \
      PROBE_WEIGHT_DECAY=0.01 \
      PROBE_L1_LAMBDA=0.00001 \
      MIN_CLASS_PER_SPLIT=10 \
      BATCH_SIZE="$BATCH_SIZE" \
      LAYERS_G5="$FDISC_LAYERS_G5" \
      LAYERS_G6="$FDISC_LAYERS_G6" \
      LAYERS_G7="$FDISC_LAYERS_G7" \
      ./experiments/run_phase7_sae_faithfulness.sh launch
      local fd_coord="p7sae_coord_${fd_run_id}"
      while [[ ! -f "phase7_results/runs/${fd_run_id}/state/pipeline.done" ]]; do
        if ! tmux has-session -t "$fd_coord" 2>/dev/null; then
          sleep 5
          [[ -f "phase7_results/runs/${fd_run_id}/state/pipeline.done" ]] || { echo "fd coordinator disappeared: ${fd_coord}" >&2; exit 1; }
        fi
        sleep 15
      done
    fi
    [[ -f "$divergent" ]] || { echo "missing divergent source: $divergent" >&2; exit 1; }
    cp -f "$divergent" "$BASE/meta/divergent_${SCOPE}.json"

    RUN_ID="$sub_run_id" \
    RUN_TAG="phase7_prontoqa_pathb_${sub_run_id}" \
    BASE="phase7_results/runs/${sub_run_id}" \
    CONTROL_RECORDS="$records" \
    MODEL_KEY="$MODEL_KEY" \
    LAYERS_CSV="$LAYERS_CSV" \
    SAES_DIR="$SAES_DIR" \
    ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
    PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
    DIVERGENT_SOURCE="$divergent" \
    SUBSPACE_SPECS="$SUBSPACE_SPECS" \
    FEATURE_SETS_CSV="result_top50,eq_pre_result_150,divergent_top50" \
    SAMPLE_TRACES="$sample" \
    MIN_COMMON_STEPS="$MIN_COMMON_STEPS" \
    SEED="$SEED" \
    N_BOOTSTRAP="$n_bootstrap_scope" \
    BATCH_SIZE="$BATCH_SIZE" \
    PATHB_REUSE_FEATURE_CACHE=1 \
    ./experiments/run_phase7_sae_trajectory_pathb.sh launch

    [[ -f "$out_json" ]] || { echo "missing pathb output: $out_json" >&2; exit 1; }
    json_parseable "$out_json" || { echo "bad pathb output: $out_json" >&2; exit 1; }
    local records_sha
    records_sha="$("$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${records@Q}))
PY
)"
    local divergent_sha
    divergent_sha="$("$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${divergent@Q}))
PY
)"
    local cfg_hash
    cfg_hash="$("$PY" - <<PY
import hashlib, json
obj = {
  "stage":"pathb",
  "scope": ${SCOPE@Q},
  "model_key": ${MODEL_KEY@Q},
  "layers_csv": ${LAYERS_CSV@Q},
  "control_records": ${records@Q},
  "control_records_sha256": ${records_sha@Q},
  "feature_sets_csv": "result_top50,eq_pre_result_150,divergent_top50",
  "divergent_source": ${divergent@Q},
  "divergent_sha256": ${divergent_sha@Q},
  "sample_traces": int(${sample@Q}),
  "min_common_steps": int(${MIN_COMMON_STEPS@Q}),
  "seed": int(${SEED@Q}),
  "n_bootstrap": int(${n_bootstrap_scope@Q}),
  "batch_size": int(${BATCH_SIZE@Q}),
}
print(hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest())
PY
)"
    local out_sha
    out_sha="$("$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${out_json@Q}))
PY
)"
    "$PY" - <<PY
import json
from pathlib import Path
meta = {
  "scope": ${SCOPE@Q},
  "run_id": ${sub_run_id@Q},
  "output_json": ${out_json@Q},
  "divergent_source": ${divergent@Q},
  "output_sha256": ${out_sha@Q},
  "control_records": ${records@Q},
  "control_records_sha256": ${records_sha@Q},
  "divergent_source_sha256": ${divergent_sha@Q},
  "config_hash": ${cfg_hash@Q},
}
Path(${BASE@Q}).joinpath("meta","pathb_" + ${SCOPE@Q} + ".json").write_text(json.dumps(meta, indent=2))
PY
    touch "$BASE/state/pathb_${SCOPE}.done"
    echo "[$(date -Is)] pathb done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_pathc() {
  require_env
  : "${SCOPE:?SCOPE required (canary|full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/pathc_${SCOPE}.log"
  local records="$CANARY_CONTROL_RECORDS"
  local sample="$CANARY_TRACES"
  local n_bootstrap_scope="${CANARY_N_BOOTSTRAP:-${N_BOOTSTRAP:-500}}"
  local wrong_bootstrap_n="${PATHC_CANARY_WRONG_INTERMEDIATE_BOOTSTRAP_N:-200}"
  local cv_folds_scope="${PATHC_CANARY_CV_FOLDS:-3}"
  local sub_run_id="${RUN_ID}_pathc_canary"
  if [[ "$SCOPE" == "full" ]]; then
    records="$FULL_CONTROL_RECORDS"
    sample="0"
    n_bootstrap_scope="${FULL_N_BOOTSTRAP:-${N_BOOTSTRAP:-1000}}"
    wrong_bootstrap_n="${PATHC_FULL_WRONG_INTERMEDIATE_BOOTSTRAP_N:-1000}"
    cv_folds_scope="${PATHC_FULL_CV_FOLDS:-5}"
    sub_run_id="${RUN_ID}_pathc_full"
  fi
  local out_json="phase7_results/results/phase7_sae_trajectory_pathc_robust_${sub_run_id}.json"
  local divergent="$BASE/meta/divergent_${SCOPE}.json"
  local decoder_checkpoint_local="${PATHC_DECODER_CHECKPOINT:-}"
  local model_ladder_local="${PATHC_FULL_MODEL_LADDER:-${PATHC_MODEL_LADDER:-sae_only}}"
  if [[ "$SCOPE" == "canary" ]]; then
    model_ladder_local="${PATHC_CANARY_MODEL_LADDER:-sae_only}"
  fi
  if [[ ",${model_ladder_local}," != *",hybrid_only,"* && ",${model_ladder_local}," != *",mixed,"* ]]; then
    decoder_checkpoint_local=""
  fi
  local decoder_missing_state_policy_local="error"

  # PrOntoQA control-records may not include arithmetic decoder labels.
  # If structured_state is absent, run SAE-only Path C for this scope.
  local has_structured_state="false"
  has_structured_state="$("$PY" - <<PY
import json
from pathlib import Path
import torch

payload = json.load(open(${records@Q}))
rows = []
fmt = str(payload.get("rows_format", "")).lower()
if fmt == "pt":
    rp = payload.get("rows_path")
    if rp:
        p = Path(str(rp))
        if not p.is_absolute():
            rel = (Path(${records@Q}).parent / p).resolve()
            p = rel if rel.exists() else p.resolve()
        rows = list(torch.load(p, map_location="cpu"))
else:
    rows = list(payload.get("rows", []))
ok = bool(rows) and all(isinstance(r.get("structured_state"), dict) for r in rows[: min(len(rows), 256)])
print("true" if ok else "false")
PY
)"
  if [[ "$has_structured_state" != "true" ]]; then
    decoder_checkpoint_local=""
    model_ladder_local="${PATHC_MODEL_LADDER_NO_DECODER:-sae_only}"
    decoder_missing_state_policy_local="skip"
  fi
  local decoder_enabled_flag="false"
  if [[ -n "$decoder_checkpoint_local" ]]; then
    decoder_enabled_flag="true"
  fi

  {
    echo "[$(date -Is)] pathc start scope=${SCOPE}"
    echo "[$(date -Is)] pathc decoder_enabled=${decoder_enabled_flag} model_ladder=${model_ladder_local} has_structured_state=${has_structured_state}"
    [[ -f "$divergent" ]] || { echo "missing divergent source: $divergent" >&2; exit 1; }
    RUN_ID="$sub_run_id" \
    RUN_TAG="phase7_prontoqa_pathc_${sub_run_id}" \
    BASE="phase7_results/runs/${sub_run_id}" \
    CONTROL_RECORDS="$records" \
    MODEL_KEY="$MODEL_KEY" \
    LAYERS_CSV="$LAYERS_CSV" \
    SAES_DIR="$SAES_DIR" \
    ACTIVATIONS_DIR="$ACTIVATIONS_DIR" \
    PHASE4_TOP_FEATURES="$PHASE4_TOP_FEATURES" \
    DIVERGENT_SOURCE="$divergent" \
    SUBSPACE_SPECS="$SUBSPACE_SPECS" \
    FEATURE_SET="eq_pre_result_150" \
    SAMPLE_TRACES="$sample" \
    MIN_COMMON_STEPS="$MIN_COMMON_STEPS" \
    SEED="$SEED" \
    N_BOOTSTRAP="$n_bootstrap_scope" \
    BATCH_SIZE="$BATCH_SIZE" \
    TRACE_TEST_FRACTION=0.20 \
    TRACE_SPLIT_SEED=20260306 \
    PROBE_EPOCHS=500 \
    PROBE_LR=0.03 \
    PROBE_WEIGHT_DECAY=0.0001 \
    PROBE_DEVICE=cpu \
    TRAIN_EXCLUDE_VARIANTS="$TRAIN_EXCLUDE_VARIANTS_PRONTOQA" \
    REQUIRE_WRONG_INTERMEDIATE_AUROC=0.70 \
    WRONG_INTERMEDIATE_BOOTSTRAP_N="$wrong_bootstrap_n" \
    WRONG_INTERMEDIATE_BOOTSTRAP_SEED=20260307 \
    CV_FOLDS="$cv_folds_scope" \
    CV_SEED=20260307 \
    CV_MIN_VALID_FOLDS=3 \
    MODEL_LADDER="$model_ladder_local" \
    MIXED_DELTA_EFFECT_FLOOR=0.03 \
    DECODER_CHECKPOINT="$decoder_checkpoint_local" \
    DECODER_MISSING_STATE_POLICY="$decoder_missing_state_policy_local" \
    ./experiments/run_phase7_sae_trajectory_pathc_robust.sh launch

    [[ -f "$out_json" ]] || { echo "missing pathc output: $out_json" >&2; exit 1; }
    json_parseable "$out_json" || { echo "bad pathc output: $out_json" >&2; exit 1; }
    local records_sha
    records_sha="$("$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${records@Q}))
PY
)"
    local divergent_sha
    divergent_sha="$("$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${divergent@Q}))
PY
)"
    local cfg_hash
    cfg_hash="$("$PY" - <<PY
import hashlib, json
obj = {
  "stage":"pathc",
  "scope": ${SCOPE@Q},
  "model_key": ${MODEL_KEY@Q},
  "layers_csv": ${LAYERS_CSV@Q},
  "control_records": ${records@Q},
  "control_records_sha256": ${records_sha@Q},
  "divergent_source": ${divergent@Q},
  "divergent_source_sha256": ${divergent_sha@Q},
  "feature_set": "eq_pre_result_150",
  "sample_traces": int(${sample@Q}),
  "min_common_steps": int(${MIN_COMMON_STEPS@Q}),
  "seed": int(${SEED@Q}),
  "n_bootstrap": int(${n_bootstrap_scope@Q}),
  "wrong_bootstrap_n": int(${wrong_bootstrap_n@Q}),
  "cv_folds": int(${cv_folds_scope@Q}),
  "model_ladder": ${model_ladder_local@Q},
  "decoder_checkpoint": ${decoder_checkpoint_local@Q},
}
print(hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest())
PY
)"
    local out_sha
    out_sha="$("$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${out_json@Q}))
PY
)"
    "$PY" - <<PY
import json
from pathlib import Path
meta = json.load(open(${out_json@Q}))
meta["_stage_meta"] = {
  "scope": ${SCOPE@Q},
  "control_records": ${records@Q},
  "control_records_sha256": ${records_sha@Q},
  "divergent_source": ${divergent@Q},
  "divergent_source_sha256": ${divergent_sha@Q},
  "output_sha256": ${out_sha@Q},
  "config_hash": ${cfg_hash@Q},
}
Path(${BASE@Q}).joinpath("meta","pathc_" + ${SCOPE@Q} + ".json").write_text(json.dumps(meta, indent=2))
PY
    touch "$BASE/state/pathc_${SCOPE}.done"
    echo "[$(date -Is)] pathc done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_stress() {
  require_env
  : "${SCOPE:?SCOPE required (full)}"
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
  local logf="$BASE/logs/stress_${SCOPE}.log"
  local source_run_id="${RUN_ID}_pathc_full_core"
  local stress_run_id="${RUN_ID}_stress_full"
  local out_json="phase7_results/results/qwen_pathc_stress_${stress_run_id}.json"
  {
    echo "[$(date -Is)] stress start scope=${SCOPE}"
    RUN_ID="$stress_run_id" \
    RUN_TAG="qwen_pathc_stress_${stress_run_id}" \
    BASE="phase7_results/runs/${stress_run_id}" \
    SOURCE_RUN_ID="$source_run_id" \
    TRAIN_EXCLUDE_VARIANTS="$TRAIN_EXCLUDE_VARIANTS_PRONTOQA" \
    DEVICE=cpu \
    ./experiments/run_qwen_pathc_stress.sh launch

    local coord="p7st_coord_${stress_run_id}"
    while [[ ! -f "phase7_results/runs/${stress_run_id}/state/pipeline.done" ]]; do
      if ! tmux has-session -t "$coord" 2>/dev/null; then
        sleep 5
        [[ -f "phase7_results/runs/${stress_run_id}/state/pipeline.done" ]] || { echo "stress coordinator disappeared: ${coord}" >&2; exit 1; }
      fi
      sleep 15
    done
    [[ -f "$out_json" ]] || { echo "missing stress output: $out_json" >&2; exit 1; }
    json_parseable "$out_json" || { echo "bad stress output: $out_json" >&2; exit 1; }
    local cfg_hash
    cfg_hash="$("$PY" - <<PY
import hashlib, json
obj = {
  "stage":"stress",
  "scope": ${SCOPE@Q},
  "source_run_id": ${source_run_id@Q},
  "train_exclude_variants": ${TRAIN_EXCLUDE_VARIANTS_PRONTOQA@Q},
  "device": "cpu",
}
print(hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest())
PY
)"
    local out_sha
    out_sha="$("$PY" - <<PY
from phase7.common import sha256_file
print(sha256_file(${out_json@Q}))
PY
)"
    "$PY" - <<PY
import json
from pathlib import Path
meta = {
  "scope": ${SCOPE@Q},
  "run_id": ${stress_run_id@Q},
  "output_json": ${out_json@Q},
  "output_sha256": ${out_sha@Q},
  "config_hash": ${cfg_hash@Q},
}
Path(${BASE@Q}).joinpath("meta","stress_" + ${SCOPE@Q} + ".json").write_text(json.dumps(meta, indent=2))
PY
    touch "$BASE/state/stress_${SCOPE}.done"
    echo "[$(date -Is)] stress done scope=${SCOPE}"
  } >>"$logf" 2>&1
}

run_coordinator() {
  require_env
  mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta" "phase7_results/results"
  local logf="$BASE/logs/coordinator.log"
  local canary_n_boot="${CANARY_N_BOOTSTRAP:-${N_BOOTSTRAP:-500}}"
  local full_n_boot="${FULL_N_BOOTSTRAP:-${N_BOOTSTRAP:-1000}}"
  local canary_wrong_boot="${PATHC_CANARY_WRONG_INTERMEDIATE_BOOTSTRAP_N:-200}"
  local full_wrong_boot="${PATHC_FULL_WRONG_INTERMEDIATE_BOOTSTRAP_N:-1000}"
  local canary_cv_folds="${PATHC_CANARY_CV_FOLDS:-3}"
  local full_cv_folds="${PATHC_FULL_CV_FOLDS:-5}"
  {
    echo "[$(date -Is)] coordinator start run_id=${RUN_ID}"
    wait_for_file_with_session "$BASE/state/precompute_canary.done" "$PRE_SESSION" 172800

    local canary_patha_hash_expected
    canary_patha_hash_expected="$(_expected_patha_hash "canary" "$CANARY_CONTROL_RECORDS" "$CANARY_TRACES" "$canary_n_boot")"
    local canary_patha_hash_found=""
    canary_patha_hash_found="$(_validate_stage_with_hash "patha" "canary" || true)"
    if [[ -n "$canary_patha_hash_found" && "$canary_patha_hash_found" == "$canary_patha_hash_expected" ]]; then
      echo "[$(date -Is)] skipping patha canary (valid marker+parse+hash)"
    else
      SCOPE=canary "$0" patha
    fi

    local canary_pathb_hash_expected=""
    if [[ -f "$BASE/meta/divergent_canary.json" ]]; then
      canary_pathb_hash_expected="$(_expected_pathb_hash "canary" "$CANARY_CONTROL_RECORDS" "$CANARY_TRACES" "$canary_n_boot" "$BASE/meta/divergent_canary.json")"
    fi
    local canary_pathb_hash_found=""
    canary_pathb_hash_found="$(_validate_stage_with_hash "pathb" "canary" || true)"
    if [[ -n "$canary_pathb_hash_found" && "$canary_pathb_hash_found" == "$canary_pathb_hash_expected" ]]; then
      echo "[$(date -Is)] skipping pathb canary (valid marker+parse+hash)"
    else
      SCOPE=canary "$0" pathb
    fi

    local canary_pathc_model_ladder="${PATHC_CANARY_MODEL_LADDER:-sae_only}"
    local canary_pathc_decoder_checkpoint="${PATHC_DECODER_CHECKPOINT:-}"
    if [[ ",${canary_pathc_model_ladder}," != *",hybrid_only,"* && ",${canary_pathc_model_ladder}," != *",mixed,"* ]]; then
      canary_pathc_decoder_checkpoint=""
    fi
    local canary_pathc_hash_expected
    canary_pathc_hash_expected="$(_expected_pathc_hash "canary" "$CANARY_CONTROL_RECORDS" "$CANARY_TRACES" "$canary_n_boot" "$canary_wrong_boot" "$canary_cv_folds" "$BASE/meta/divergent_canary.json" "$canary_pathc_model_ladder" "$canary_pathc_decoder_checkpoint")"
    local canary_pathc_hash_found=""
    canary_pathc_hash_found="$(_validate_stage_with_hash "pathc" "canary" || true)"
    if [[ -n "$canary_pathc_hash_found" && "$canary_pathc_hash_found" == "$canary_pathc_hash_expected" ]]; then
      echo "[$(date -Is)] skipping pathc canary (valid marker+parse+hash)"
    else
      SCOPE=canary "$0" pathc
    fi

    canary_pathc_json="phase7_results/results/phase7_sae_trajectory_pathc_robust_${RUN_ID}_pathc_canary.json"
    [[ -f "$canary_pathc_json" ]] || { echo "missing canary pathc json" >&2; exit 1; }
    canary_integrity_ok="$("$PY" - <<PY
import json
d=json.load(open(${canary_pathc_json@Q}))
cv=((d.get("cv_diagnostics") or {}).get("cv_trace_overlap_count"))
status=str(d.get("status",""))
ok = (status=="ok") and (int(cv or 0)==0)
print("true" if ok else "false")
PY
)"

    full_ran="false"
    full_pass="false"
    blocked_reason=""
    if [[ "$canary_integrity_ok" == "true" ]]; then
      if [[ -f "$BASE/state/precompute_full.done" ]]; then
        echo "[$(date -Is)] skipping precompute full (marker exists)"
      else
        PREP_KIND=full "$0" precompute
      fi

      local full_patha_hash_expected
      full_patha_hash_expected="$(_expected_patha_hash "full" "$FULL_CONTROL_RECORDS" "0" "$full_n_boot")"
      local full_patha_hash_found=""
      full_patha_hash_found="$(_validate_stage_with_hash "patha" "full" || true)"
      if [[ -n "$full_patha_hash_found" && "$full_patha_hash_found" == "$full_patha_hash_expected" ]]; then
        echo "[$(date -Is)] skipping patha full (valid marker+parse+hash)"
      else
        SCOPE=full "$0" patha
      fi

      local full_pathb_hash_expected=""
      if [[ -f "$BASE/meta/divergent_full.json" ]]; then
        full_pathb_hash_expected="$(_expected_pathb_hash "full" "$FULL_CONTROL_RECORDS" "0" "$full_n_boot" "$BASE/meta/divergent_full.json")"
      fi
      local full_pathb_hash_found=""
      full_pathb_hash_found="$(_validate_stage_with_hash "pathb" "full" || true)"
      if [[ -n "$full_pathb_hash_found" && "$full_pathb_hash_found" == "$full_pathb_hash_expected" ]]; then
        echo "[$(date -Is)] skipping pathb full (valid marker+parse+hash)"
      else
        SCOPE=full "$0" pathb
      fi

      local full_pathc_model_ladder="${PATHC_FULL_MODEL_LADDER:-${PATHC_MODEL_LADDER:-sae_only}}"
      local full_pathc_decoder_checkpoint="${PATHC_DECODER_CHECKPOINT:-}"
      if [[ ",${full_pathc_model_ladder}," != *",hybrid_only,"* && ",${full_pathc_model_ladder}," != *",mixed,"* ]]; then
        full_pathc_decoder_checkpoint=""
      fi
      local full_pathc_hash_expected
      full_pathc_hash_expected="$(_expected_pathc_hash "full" "$FULL_CONTROL_RECORDS" "0" "$full_n_boot" "$full_wrong_boot" "$full_cv_folds" "$BASE/meta/divergent_full.json" "$full_pathc_model_ladder" "$full_pathc_decoder_checkpoint")"
      local full_pathc_hash_found=""
      full_pathc_hash_found="$(_validate_stage_with_hash "pathc" "full" || true)"
      if [[ -n "$full_pathc_hash_found" && "$full_pathc_hash_found" == "$full_pathc_hash_expected" ]]; then
        echo "[$(date -Is)] skipping pathc full (valid marker+parse+hash)"
      else
        SCOPE=full "$0" pathc
      fi

      local stress_hash_found=""
      stress_hash_found="$(_validate_stage_with_hash "stress" "full" || true)"
      if [[ -n "$stress_hash_found" ]]; then
        echo "[$(date -Is)] skipping stress full (valid marker+parse+hash)"
      else
        SCOPE=full "$0" stress
      fi
      full_ran="true"
      full_pathc_json="phase7_results/results/phase7_sae_trajectory_pathc_robust_${RUN_ID}_pathc_full.json"
      full_pass="$("$PY" - <<PY
import json
d=json.load(open(${full_pathc_json@Q}))
cv=(d.get("cv_diagnostics") or {})
auc=cv.get("cv_wrong_intermediate_pooled_auroc")
ci=(cv.get("cv_wrong_intermediate_pooled_ci95") or {})
ok = isinstance(auc,(int,float)) and float(auc)>0.70 and isinstance(ci.get("lower"),(int,float)) and float(ci.get("lower"))>=0.65 and int(cv.get("cv_trace_overlap_count",0))==0
print("true" if ok else "false")
PY
)"
      if [[ "$full_pass" != "true" ]]; then
        blocked_reason="full_gate_not_met"
      fi
    else
      blocked_reason="canary_integrity_failed"
    fi

    out_json="phase7_results/results/prontoqa_trackc_pilot_${RUN_ID}.json"
    out_md="phase7_results/results/prontoqa_trackc_pilot_${RUN_ID}.md"
    "$PY" - <<PY
import json
from datetime import datetime
from pathlib import Path
run_id=${RUN_ID@Q}
base=${BASE@Q}
canary_pathc=f"phase7_results/results/phase7_sae_trajectory_pathc_robust_{run_id}_pathc_canary.json"
full_pathc=f"phase7_results/results/phase7_sae_trajectory_pathc_robust_{run_id}_pathc_full.json"
stress=f"phase7_results/results/qwen_pathc_stress_{run_id}_stress_full.json"
def load_if(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else None
canary=load_if(canary_pathc)
full=load_if(full_pathc)
stressj=load_if(stress)
full_cv = (full.get("cv_diagnostics") or {}) if isinstance(full, dict) else {}
full_probe_by_variant = (full.get("probe_by_variant") or {}) if isinstance(full, dict) else {}
wrong_variant = full_probe_by_variant.get("wrong_intermediate", {}) if isinstance(full_probe_by_variant, dict) else {}
lexical_variant = full_probe_by_variant.get("lexical_consistent_swap", {}) if isinstance(full_probe_by_variant, dict) else {}
stress_tests = (stressj.get("tests") or {}) if isinstance(stressj, dict) else {}
stress_multi = stress_tests.get("multiseed", {}) if isinstance(stress_tests, dict) else {}
stress_lex = (stress_multi.get("lexical_consistent_swap") or {}) if isinstance(stress_multi, dict) else {}
stress_lex_ci = stress_lex.get("pooled_ci95") if isinstance(stress_lex, dict) else None
wrong_auc = wrong_variant.get("probe_auroc") if isinstance(wrong_variant, dict) else None
lexical_auc = lexical_variant.get("probe_auroc") if isinstance(lexical_variant, dict) else None
wrong_minus_lex = (float(wrong_auc) - float(lexical_auc)) if isinstance(wrong_auc,(int,float)) and isinstance(lexical_auc,(int,float)) else None
full_auc = full_cv.get("cv_wrong_intermediate_pooled_auroc")
full_ci = full_cv.get("cv_wrong_intermediate_pooled_ci95") if isinstance(full_cv, dict) else {}
full_ci_lower = full_ci.get("lower") if isinstance(full_ci, dict) else None
leakage_clean = bool(int(full_cv.get("cv_trace_overlap_count", 1)) == 0) if isinstance(full_cv, dict) else False
model_generated_used = bool(isinstance(full, dict) and str(full.get("source_control_records","")).find("prontoqa") >= 0 and ${PRONTOQA_COT_SOURCE@Q} == "model_generated")
lexical_ci_upper = stress_lex_ci.get("upper") if isinstance(stress_lex_ci, dict) else None
lexical_confound_pass = bool(isinstance(lexical_auc, (int,float)) and float(lexical_auc) <= float(${LEXICAL_CONFOUND_AUROC_MAX@Q}))
if isinstance(lexical_ci_upper, (int,float)):
    lexical_confound_pass = bool(lexical_confound_pass and float(lexical_ci_upper) <= float(${LEXICAL_CONFOUND_AUROC_MAX@Q}))
strict_gate_pass = bool(
    leakage_clean
    and isinstance(full_auc, (int,float)) and float(full_auc) > 0.70
    and isinstance(full_ci_lower, (int,float)) and float(full_ci_lower) >= 0.65
    and lexical_confound_pass
    and isinstance(wrong_minus_lex, (int,float)) and float(wrong_minus_lex) >= 0.10
    and model_generated_used
)
out={
  "schema_version":"prontoqa_trackc_pilot_v1",
  "status":"ok",
  "run_id": run_id,
  "run_tag": ${RUN_TAG@Q},
  "model_key": ${MODEL_KEY@Q},
  "canary_integrity_pass": (${canary_integrity_ok@Q}=="true"),
  "full_ran": (${full_ran@Q}=="true"),
  "full_pass_gate": (${full_pass@Q}=="true"),
  "blocked_reason": ${blocked_reason@Q},
  "primary_endpoint": {
    "target": "wrong_intermediate pooled 5-fold CV AUROC > 0.70 and CI95 lower >= 0.65",
    "canary": None if not isinstance(canary, dict) else ((canary.get("cv_diagnostics") or {}).get("cv_wrong_intermediate_pooled_auroc")),
    "full": None if not isinstance(full, dict) else ((full.get("cv_diagnostics") or {}).get("cv_wrong_intermediate_pooled_auroc")),
    "full_ci95": None if not isinstance(full, dict) else ((full.get("cv_diagnostics") or {}).get("cv_wrong_intermediate_pooled_ci95")),
  },
  "confound_endpoint": {
    "lexical_variant_name": "lexical_consistent_swap",
    "wrong_intermediate_probe_auroc": wrong_auc,
    "lexical_probe_auroc": lexical_auc,
    "lexical_probe_ci95": stress_lex_ci,
    "wrong_minus_lexical_delta": wrong_minus_lex,
    "lexical_confound_auroc_max": float(${LEXICAL_CONFOUND_AUROC_MAX@Q}),
  },
  "claim_gate": {
    "policy": "strict_faithfulness",
    "model_generated_condition_pass": bool(model_generated_used),
    "leakage_clean": bool(leakage_clean),
    "wrong_intermediate_gate_pass": bool(isinstance(full_auc, (int,float)) and float(full_auc) > 0.70 and isinstance(full_ci_lower, (int,float)) and float(full_ci_lower) >= 0.65),
    "lexical_confound_control_pass": bool(lexical_confound_pass),
    "wrong_minus_lexical_delta_pass": bool(isinstance(wrong_minus_lex, (int,float)) and float(wrong_minus_lex) >= 0.10),
    "strict_gate_pass": bool(strict_gate_pass),
  },
  "artifacts": {
    "canary_pathc_robust_json": canary_pathc,
    "full_pathc_robust_json": full_pathc,
    "stress_json": stress,
    "arithmetic_reframe_json": f"phase7_results/results/trackc_arithmetic_significance_reframe_{run_id}.json",
  },
  "tests": {
    "canary_pathc_robust": canary,
    "full_pathc_robust": full,
    "stress": stressj,
  },
  "timestamp": datetime.now().isoformat(),
}
Path(${out_json@Q}).parent.mkdir(parents=True, exist_ok=True)
Path(${out_json@Q}).write_text(json.dumps(out, indent=2))
claim_boundary = {
  "schema_version": "phase7_prontoqa_claim_boundary_v1",
  "run_id": run_id,
  "model_generated_condition_pass": bool(model_generated_used),
  "lexical_confound_control_pass": bool(lexical_confound_pass),
  "wrong_minus_lexical_delta": wrong_minus_lex,
  "faithfulness_claim_enabled": bool(strict_gate_pass),
  "coherence_only_fallback": bool(not strict_gate_pass),
  "timestamp": datetime.now().isoformat(),
}
claim_path = f"phase7_results/results/prontoqa_trackc_claim_boundary_{run_id}.json"
Path(claim_path).write_text(json.dumps(claim_boundary, indent=2))
out["artifacts"]["claim_boundary_json"] = claim_path
Path(${out_json@Q}).write_text(json.dumps(out, indent=2))
lines=[
  "# PrOntoQA Track C Pilot Summary",
  "",
  f"- Run id: {run_id}",
  f"- Canary integrity pass: {out['canary_integrity_pass']}",
  f"- Full ran: {out['full_ran']}",
  f"- Full gate pass: {out['full_pass_gate']}",
  f"- Blocked reason: {out['blocked_reason']}",
  f"- Full pooled wrong_intermediate AUROC: {(out['primary_endpoint'] or {}).get('full')}",
  f"- Full pooled CI95: {(out['primary_endpoint'] or {}).get('full_ci95')}",
  f"- Lexical control AUROC: {(out['confound_endpoint'] or {}).get('lexical_probe_auroc')}",
  f"- Wrong minus lexical delta: {(out['confound_endpoint'] or {}).get('wrong_minus_lexical_delta')}",
  f"- Strict faithfulness gate: {(out['claim_gate'] or {}).get('strict_gate_pass')}",
]
Path(${out_md@Q}).write_text("\\n".join(lines)+"\\n")
Path(base).joinpath("state","pipeline.done").write_text("done\\n")
print("saved", ${out_json@Q})
PY
    echo "[$(date -Is)] coordinator done"
  } >>"$logf" 2>&1
}

case "$MODE" in
  launch)
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase7_prontoqa_trackc_pilot}"
    RUN_TAG="${RUN_TAG:-phase7_prontoqa_trackc_pilot_${RUN_ID}}"
    BASE="${BASE:-phase7_results/runs/${RUN_ID}}"
    MODEL_KEY="${MODEL_KEY:-qwen2.5-7b}"
    PRECOMPUTE_GPU="${PRECOMPUTE_GPU:-7}"
    PRECOMPUTE_SHARD_GPUS_CSV="${PRECOMPUTE_SHARD_GPUS_CSV:-5,6,7}"
    PRECOMPUTE_RESUME="${PRECOMPUTE_RESUME:-1}"
    CANARY_TRACES="${CANARY_TRACES:-200}"
    FULL_TRACES="${FULL_TRACES:-1000}"
    MAX_RECORDS="${MAX_RECORDS:-20000}"
    CHAIN_LEN_MIN="${CHAIN_LEN_MIN:-3}"
    CHAIN_LEN_MAX="${CHAIN_LEN_MAX:-5}"
    PRONTOQA_VARIANTS="${PRONTOQA_VARIANTS:-faithful,wrong_intermediate,order_flip,skipped_step,wrong_premise,irrelevant_insertion,lexical_consistent_swap}"
    PRONTOQA_COT_SOURCE="${PRONTOQA_COT_SOURCE:-model_generated}"
    PRONTOQA_GEN_MAX_NEW_TOKENS="${PRONTOQA_GEN_MAX_NEW_TOKENS:-200}"
    PRONTOQA_GEN_TEMPERATURE="${PRONTOQA_GEN_TEMPERATURE:-0.2}"
    PRONTOQA_GEN_TOP_P="${PRONTOQA_GEN_TOP_P:-0.95}"
    PRONTOQA_GEN_DO_SAMPLE="${PRONTOQA_GEN_DO_SAMPLE:-0}"
    PRONTOQA_GEN_RETRIES="${PRONTOQA_GEN_RETRIES:-2}"
    PRONTOQA_GEN_BATCH_SIZE="${PRONTOQA_GEN_BATCH_SIZE:-12}"
    PRONTOQA_FORWARD_BATCH_SIZE="${PRONTOQA_FORWARD_BATCH_SIZE:-8}"
    PRECOMPUTE_CHECKPOINT_EVERY="${PRECOMPUTE_CHECKPOINT_EVERY:-100}"
    LEXICAL_CONFOUND_AUROC_MAX="${LEXICAL_CONFOUND_AUROC_MAX:-0.60}"
    TRAIN_EXCLUDE_VARIANTS_PRONTOQA="${TRAIN_EXCLUDE_VARIANTS_PRONTOQA:-order_flip,skipped_step,irrelevant_insertion,lexical_consistent_swap}"
    SEED="${SEED:-20260309}"
    SAES_DIR="${SAES_DIR:-phase2_results/saes_qwen25_7b_12x_topk/saes}"
    ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-phase2_results/activations}"
    PHASE4_TOP_FEATURES="${PHASE4_TOP_FEATURES:-phase7_results/runs/20260308_165109_phase7_qwen_trackc_upgrade/interventions/top_features_per_layer_qwen_phase7_qwen_trackc_upgrade_20260308_165109_phase7_qwen_trackc_upgrade.json}"
    SUBSPACE_SPECS="${SUBSPACE_SPECS:-phase7_results/runs/20260308_165109_phase7_qwen_trackc_upgrade/interventions/variable_subspaces_qwen_phase7_qwen_trackc_upgrade_20260308_165109_phase7_qwen_trackc_upgrade.json}"
    PATHC_DECODER_CHECKPOINT="${PATHC_DECODER_CHECKPOINT:-phase7_results/runs/20260308_165109_phase7_qwen_trackc_upgrade/checkpoints/state_raw_every2_even_d1tier1.pt}"
    PATHC_MODEL_LADDER="${PATHC_MODEL_LADDER:-sae_only}"
    PATHC_CANARY_MODEL_LADDER="${PATHC_CANARY_MODEL_LADDER:-sae_only}"
    PATHC_FULL_MODEL_LADDER="${PATHC_FULL_MODEL_LADDER:-$PATHC_MODEL_LADDER}"
    PATHC_MODEL_LADDER_NO_DECODER="${PATHC_MODEL_LADDER_NO_DECODER:-sae_only}"
    LAYERS_CSV="${LAYERS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}"
    FDISC_LAYERS_G5="${FDISC_LAYERS_G5:-0,1,2,3,4,5,6,7,8,9}"
    FDISC_LAYERS_G6="${FDISC_LAYERS_G6:-10,11,12,13,14,15,16,17,18}"
    FDISC_LAYERS_G7="${FDISC_LAYERS_G7:-19,20,21,22,23,24,25,26,27}"
    MIN_COMMON_STEPS="${MIN_COMMON_STEPS:-3}"
    N_BOOTSTRAP="${N_BOOTSTRAP:-500}"
    CANARY_N_BOOTSTRAP="${CANARY_N_BOOTSTRAP:-200}"
    FULL_N_BOOTSTRAP="${FULL_N_BOOTSTRAP:-1000}"
    PATHC_CANARY_WRONG_INTERMEDIATE_BOOTSTRAP_N="${PATHC_CANARY_WRONG_INTERMEDIATE_BOOTSTRAP_N:-200}"
    PATHC_FULL_WRONG_INTERMEDIATE_BOOTSTRAP_N="${PATHC_FULL_WRONG_INTERMEDIATE_BOOTSTRAP_N:-1000}"
    PATHC_CANARY_CV_FOLDS="${PATHC_CANARY_CV_FOLDS:-3}"
    PATHC_FULL_CV_FOLDS="${PATHC_FULL_CV_FOLDS:-5}"
    BATCH_SIZE="${BATCH_SIZE:-256}"

    PREP_CANARY_DIR="${PREP_CANARY_DIR:-$BASE/prontoqa_canary}"
    PREP_FULL_DIR="${PREP_FULL_DIR:-$BASE/prontoqa_full}"
    CANARY_CONTROL_RECORDS="${CANARY_CONTROL_RECORDS:-$PREP_CANARY_DIR/interventions/control_records_prontoqa.json}"
    FULL_CONTROL_RECORDS="${FULL_CONTROL_RECORDS:-$PREP_FULL_DIR/interventions/control_records_prontoqa.json}"

    PRE_SESSION="p7pq_pre_${RUN_ID}"
    COORD_SESSION="p7pq_coord_${RUN_ID}"
    RESUME="${RESUME:-0}"

    mkdir -p "$BASE/logs" "$BASE/state" "$BASE/meta"
    if [[ "$RESUME" != "1" ]]; then
      rm -f "$BASE/state"/*.done
    fi
    cat > "$BASE/meta/config.env" <<CFG
RUN_ID=$RUN_ID
RUN_TAG=$RUN_TAG
BASE=$BASE
MODEL_KEY=$MODEL_KEY
PRONTOQA_COT_SOURCE=$PRONTOQA_COT_SOURCE
PRONTOQA_GEN_MAX_NEW_TOKENS=$PRONTOQA_GEN_MAX_NEW_TOKENS
PRONTOQA_GEN_TEMPERATURE=$PRONTOQA_GEN_TEMPERATURE
PRONTOQA_GEN_TOP_P=$PRONTOQA_GEN_TOP_P
PRONTOQA_GEN_DO_SAMPLE=$PRONTOQA_GEN_DO_SAMPLE
PRONTOQA_GEN_RETRIES=$PRONTOQA_GEN_RETRIES
PRONTOQA_GEN_BATCH_SIZE=$PRONTOQA_GEN_BATCH_SIZE
PRONTOQA_FORWARD_BATCH_SIZE=$PRONTOQA_FORWARD_BATCH_SIZE
PRECOMPUTE_CHECKPOINT_EVERY=$PRECOMPUTE_CHECKPOINT_EVERY
LEXICAL_CONFOUND_AUROC_MAX=$LEXICAL_CONFOUND_AUROC_MAX
TRAIN_EXCLUDE_VARIANTS_PRONTOQA=$TRAIN_EXCLUDE_VARIANTS_PRONTOQA
PRECOMPUTE_GPU=$PRECOMPUTE_GPU
PRECOMPUTE_SHARD_GPUS_CSV=$PRECOMPUTE_SHARD_GPUS_CSV
PRECOMPUTE_RESUME=$PRECOMPUTE_RESUME
CANARY_TRACES=$CANARY_TRACES
FULL_TRACES=$FULL_TRACES
MAX_RECORDS=$MAX_RECORDS
CHAIN_LEN_MIN=$CHAIN_LEN_MIN
CHAIN_LEN_MAX=$CHAIN_LEN_MAX
PRONTOQA_VARIANTS=$PRONTOQA_VARIANTS
SEED=$SEED
SAES_DIR=$SAES_DIR
ACTIVATIONS_DIR=$ACTIVATIONS_DIR
PHASE4_TOP_FEATURES=$PHASE4_TOP_FEATURES
SUBSPACE_SPECS=$SUBSPACE_SPECS
PATHC_DECODER_CHECKPOINT=$PATHC_DECODER_CHECKPOINT
PATHC_MODEL_LADDER=$PATHC_MODEL_LADDER
PATHC_CANARY_MODEL_LADDER=$PATHC_CANARY_MODEL_LADDER
PATHC_FULL_MODEL_LADDER=$PATHC_FULL_MODEL_LADDER
PATHC_MODEL_LADDER_NO_DECODER=$PATHC_MODEL_LADDER_NO_DECODER
LAYERS_CSV=$LAYERS_CSV
FDISC_LAYERS_G5=$FDISC_LAYERS_G5
FDISC_LAYERS_G6=$FDISC_LAYERS_G6
FDISC_LAYERS_G7=$FDISC_LAYERS_G7
MIN_COMMON_STEPS=$MIN_COMMON_STEPS
N_BOOTSTRAP=$N_BOOTSTRAP
CANARY_N_BOOTSTRAP=$CANARY_N_BOOTSTRAP
FULL_N_BOOTSTRAP=$FULL_N_BOOTSTRAP
PATHC_CANARY_WRONG_INTERMEDIATE_BOOTSTRAP_N=$PATHC_CANARY_WRONG_INTERMEDIATE_BOOTSTRAP_N
PATHC_FULL_WRONG_INTERMEDIATE_BOOTSTRAP_N=$PATHC_FULL_WRONG_INTERMEDIATE_BOOTSTRAP_N
PATHC_CANARY_CV_FOLDS=$PATHC_CANARY_CV_FOLDS
PATHC_FULL_CV_FOLDS=$PATHC_FULL_CV_FOLDS
BATCH_SIZE=$BATCH_SIZE
PREP_CANARY_DIR=$PREP_CANARY_DIR
PREP_FULL_DIR=$PREP_FULL_DIR
CANARY_CONTROL_RECORDS=$CANARY_CONTROL_RECORDS
FULL_CONTROL_RECORDS=$FULL_CONTROL_RECORDS
PRE_SESSION=$PRE_SESSION
COORD_SESSION=$COORD_SESSION
RESUME=$RESUME
CFG

    for s in "$PRE_SESSION" "$COORD_SESSION"; do
      tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s"
    done
    tmux new-session -d -s "$PRE_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && PREP_KIND=canary '$0' precompute"
    tmux new-session -d -s "$COORD_SESSION" "cd '$ROOT_DIR' && set -a && source '$BASE/meta/config.env' && set +a && '$0' coordinator"

    echo "launched PrOntoQA Track C pilot"
    echo "  run_id: $RUN_ID"
    echo "  run_tag: $RUN_TAG"
    echo "  sessions: $PRE_SESSION, $COORD_SESSION"
    ;;
  precompute)
    run_precompute
    ;;
  patha)
    run_patha
    ;;
  pathb)
    run_pathb
    ;;
  pathc)
    run_pathc
    ;;
  stress)
    run_stress
    ;;
  coordinator)
    run_coordinator
    ;;
  *)
    usage
    exit 2
    ;;
esac

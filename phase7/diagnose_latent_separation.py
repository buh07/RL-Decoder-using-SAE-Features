#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import torch

try:  # pragma: no cover
    from .common import (
        MAG_BUCKETS,
        OPERATORS,
        SIGNS,
        STEP_TYPES,
        compare_states,
        group_step_records_to_traces,
        load_json,
        load_pt,
        save_json,
    )
    from .parse_cot_to_states import parse_cot_text
    from .state_decoder_core import decode_latent_pred_states, load_model_from_checkpoint
except ImportError:  # pragma: no cover
    from common import (  # type: ignore
        MAG_BUCKETS,
        OPERATORS,
        SIGNS,
        STEP_TYPES,
        compare_states,
        group_step_records_to_traces,
        load_json,
        load_pt,
        save_json,
    )
    from parse_cot_to_states import parse_cot_text  # type: ignore
    from state_decoder_core import decode_latent_pred_states, load_model_from_checkpoint  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dataset", default="phase7_results/dataset/gsm8k_step_traces_test.pt")
    p.add_argument("--controls", default="phase7_results/controls/cot_controls_test_papercore_fixv3_matrix_v3.json")
    p.add_argument("--state-decoder-checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-controls", type=int, default=None)
    p.add_argument(
        "--variant-latent-cache",
        default=None,
        help=(
            "Optional path to variant-conditioned latent cache JSON from "
            "phase7/build_control_latent_cache.py. When provided, faithful-vs-variant "
            "L2 deltas are computed from per-variant latent predictions."
        ),
    )
    p.add_argument("--output-json", default="phase7_results/diagnostics/latent_separation_diagnostic.json")
    p.add_argument("--output-report", default="phase7_results/diagnostics/latent_separation_report.md")
    return p.parse_args()


def _mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs]
    if not vals:
        return 0.0
    return float(mean(vals))


def _index_latent_preds(pred_rows: List[dict]) -> Dict[Tuple[str, int], dict]:
    return {(str(r["trace_id"]), int(r["step_idx"])): r for r in pred_rows}


def _index_variant_latent_preds(pred_rows: List[dict]) -> Dict[Tuple[str, str, int], dict]:
    out: Dict[Tuple[str, str, int], dict] = {}
    for r in pred_rows:
        trace_id = str(r.get("trace_id"))
        variant = str(r.get("control_variant", r.get("variant", "unknown")))
        step_idx = int(r.get("step_idx", -1))
        out[(trace_id, variant, step_idx)] = r
    return out


def _trace_hidden_fingerprint(trace_steps: List[dict]) -> str:
    h = hashlib.sha256()
    for row in sorted(trace_steps, key=lambda r: int(r["step_idx"])):
        t = row.get("raw_hidden")
        if not isinstance(t, torch.Tensor):
            continue
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _latent_state_vector(state: Optional[dict]) -> List[float]:
    if state is None:
        return [0.0] * 8
    op_to_id = {k: i for i, k in enumerate(OPERATORS)}
    step_to_id = {k: i for i, k in enumerate(STEP_TYPES)}
    mag_to_id = {k: i for i, k in enumerate(MAG_BUCKETS)}
    sign_to_id = {k: i for i, k in enumerate(SIGNS)}
    return [
        float(step_to_id.get(str(state.get("step_type")), -1)),
        float(op_to_id.get(str(state.get("operator")), -1)),
        float(mag_to_id.get(str(state.get("magnitude_bucket")), -1)),
        float(sign_to_id.get(str(state.get("sign")), -1)),
        float(state.get("lhs_value", 0.0)),
        float(state.get("rhs_value", 0.0)),
        float(state.get("subresult_value", 0.0)),
        float(state.get("result_token_id", -1)),
    ]


def _l2(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return float("nan")
    return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)


def _parse_by_step(cot_text: str) -> Dict[int, dict]:
    parsed = parse_cot_text(cot_text)
    return {
        int(s["step_idx"]): s
        for s in (parsed.get("parsed_steps_in_text_order") or parsed.get("parsed_steps") or [])
    }


def main() -> None:
    args = parse_args()
    controls_payload = load_json(args.controls)
    controls = list(controls_payload.get("controls", []))
    if not controls:
        raise ValueError(f"No controls found in {args.controls}; expected non-empty payload.controls")
    if args.max_controls is not None:
        controls = controls[: args.max_controls]

    step_records = load_pt(args.trace_dataset)
    trace_map = {tb.trace_id: tb.steps for tb in group_step_records_to_traces(step_records)}

    ckpt, cfg, numeric_stats, model = load_model_from_checkpoint(args.state_decoder_checkpoint, args.device)
    needed_trace_ids = {str(c["trace_id"]) for c in controls}
    needed_records = [r for r in step_records if str(r.get("trace_id")) in needed_trace_ids]
    latent_mode = "shared"
    variant_latent_idx: Dict[Tuple[str, str, int], dict] = {}
    if args.variant_latent_cache:
        cache_payload = load_json(args.variant_latent_cache)
        variant_latent_idx = _index_variant_latent_preds(list(cache_payload.get("rows", [])))
        latent_idx: Dict[Tuple[str, int], dict] = {}
        latent_mode = "variant_conditioned"
    else:
        latent_preds = decode_latent_pred_states(
            model,
            needed_records,
            cfg,
            numeric_stats,
            args.device,
            batch_size=args.batch_size,
        )
        latent_idx = _index_latent_preds(latent_preds)

    controls_by_trace: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for row in controls:
        controls_by_trace[str(row["trace_id"])][str(row.get("variant", "unknown"))] = row

    variant_text_gold_scores: Dict[str, List[float]] = defaultdict(list)
    variant_text_latent_scores: Dict[str, List[float]] = defaultdict(list)
    variant_step_counts: Dict[str, int] = defaultdict(int)
    variant_parseable_steps: Dict[str, int] = defaultdict(int)
    variant_latent_delta_from_faithful: Dict[str, List[float]] = defaultdict(list)
    variant_latent_delta_pairs: Dict[str, int] = defaultdict(int)
    hidden_identity_checks: List[dict] = []

    for trace_id, variant_rows in controls_by_trace.items():
        trace_steps = trace_map.get(trace_id, [])
        if not trace_steps:
            continue
        hidden_fp = _trace_hidden_fingerprint(trace_steps)
        hidden_identity_checks.append(
            {
                "trace_id": trace_id,
                "num_variants": len(variant_rows),
                "hidden_fingerprint": hidden_fp,
                "all_variants_share_same_trace_source": True,
            }
        )

        parsed_cache = {v: _parse_by_step(row.get("cot_text", "")) for v, row in variant_rows.items()}
        for variant, by_step in parsed_cache.items():
            for step in trace_steps:
                step_idx = int(step["step_idx"])
                text_state = by_step.get(step_idx)
                gold_state = step.get("structured_state")
                if latent_mode == "variant_conditioned":
                    latent_pred = variant_latent_idx.get((trace_id, variant, step_idx))
                else:
                    latent_pred = latent_idx.get((trace_id, step_idx))
                latent_state = latent_pred.get("latent_pred_state") if latent_pred else None
                cmp_tg = compare_states(text_state, gold_state)
                cmp_tl = compare_states(text_state, latent_state)
                variant_text_gold_scores[variant].append(float(cmp_tg.get("match_fraction", 0.0)))
                variant_text_latent_scores[variant].append(float(cmp_tl.get("match_fraction", 0.0)))
                variant_step_counts[variant] += 1
                if text_state is not None:
                    variant_parseable_steps[variant] += 1

        if latent_mode == "variant_conditioned" and "faithful" in variant_rows:
            for variant in variant_rows.keys():
                if variant == "faithful":
                    continue
                deltas: List[float] = []
                for step in trace_steps:
                    step_idx = int(step["step_idx"])
                    faithful_pred = variant_latent_idx.get((trace_id, "faithful", step_idx))
                    variant_pred = variant_latent_idx.get((trace_id, variant, step_idx))
                    if faithful_pred is None or variant_pred is None:
                        continue
                    faithful_state = faithful_pred.get("latent_pred_state")
                    variant_state = variant_pred.get("latent_pred_state")
                    if faithful_state is None or variant_state is None:
                        continue
                    vec_f = _latent_state_vector(faithful_state)
                    vec_v = _latent_state_vector(variant_state)
                    dist = _l2(vec_f, vec_v)
                    if isinstance(dist, float):
                        deltas.append(dist)
                if deltas:
                    variant_latent_delta_from_faithful[variant].append(_mean(deltas))
                    variant_latent_delta_pairs[variant] += int(len(deltas))

    variant_summary = {}
    for variant in sorted(set(list(variant_text_gold_scores.keys()) + list(variant_text_latent_scores.keys()))):
        delta_vals = variant_latent_delta_from_faithful.get(variant, [])
        delta_defined = len(delta_vals) > 0
        variant_summary[variant] = {
            "num_steps": int(variant_step_counts.get(variant, 0)),
            "num_parseable_steps": int(variant_parseable_steps.get(variant, 0)),
            "mean_text_vs_gold_match": _mean(variant_text_gold_scores.get(variant, [])),
            "mean_text_vs_latent_match": _mean(variant_text_latent_scores.get(variant, [])),
            "mean_latent_pred_l2_delta_from_faithful": (_mean(delta_vals) if delta_defined else None),
            "latent_delta_metric_defined": bool(delta_defined),
            "num_latent_delta_pairs": int(variant_latent_delta_pairs.get(variant, 0)),
        }

    faithful_text_latent = variant_summary.get("faithful", {}).get("mean_text_vs_latent_match", 0.0)
    unfaithful_variants = [v for v in variant_summary.keys() if v != "faithful"]
    unfaithful_text_latent_mean = _mean(
        [variant_summary[v]["mean_text_vs_latent_match"] for v in unfaithful_variants if v in variant_summary]
    )
    unfaithful_text_gold_mean = _mean(
        [variant_summary[v]["mean_text_vs_gold_match"] for v in unfaithful_variants if v in variant_summary]
    )
    hidden_identity_confirmed = all(
        bool(r.get("all_variants_share_same_trace_source")) for r in hidden_identity_checks
    )

    unfaithful_delta_vals = [
        variant_summary[v]["mean_latent_pred_l2_delta_from_faithful"]
        for v in unfaithful_variants
        if v in variant_summary and variant_summary[v].get("latent_delta_metric_defined")
    ]
    delta_mean = _mean([float(x) for x in unfaithful_delta_vals if isinstance(x, (int, float))]) if unfaithful_delta_vals else None
    delta_pair_count = int(sum(int(variant_summary[v].get("num_latent_delta_pairs", 0)) for v in unfaithful_variants if v in variant_summary))
    delta_pair_total = int(sum(int(variant_summary[v].get("num_steps", 0)) for v in unfaithful_variants if v in variant_summary))

    diagnosis = {
        "schema_version": "phase7_latent_separation_diagnostic_v1",
        "inputs": {
            "trace_dataset": args.trace_dataset,
            "controls": args.controls,
            "state_decoder_checkpoint": args.state_decoder_checkpoint,
            "variant_latent_cache": args.variant_latent_cache,
            "latent_mode": latent_mode,
            "num_controls_used": len(controls),
            "num_needed_records": len(needed_records),
        },
        "model_metadata": {
            "model_key": getattr(cfg, "model_key", "gpt2-medium"),
            "model_family": getattr(cfg, "model_family", "gpt2"),
            "num_layers": int(getattr(cfg, "model_num_layers", -1)),
            "hidden_dim": int(getattr(cfg, "model_hidden_dim", -1)),
            "tokenizer_id": getattr(cfg, "tokenizer_id", "unknown"),
        },
        "variant_summary": variant_summary,
        "design_checks": {
            "hidden_states_shared_across_control_variants": bool(hidden_identity_confirmed),
            "mean_latent_pred_l2_delta_from_faithful_over_unfaithful_variants": delta_mean,
            "latent_delta_metric_defined": bool(delta_mean is not None),
            "latent_delta_pair_count": delta_pair_count,
            "latent_delta_pair_coverage_fraction": (
                float(delta_pair_count / max(1, delta_pair_total))
                if delta_pair_total > 0
                else None
            ),
            "num_trace_identity_checks": len(hidden_identity_checks),
        },
        "separation_checks": {
            "faithful_mean_text_vs_latent_match": float(faithful_text_latent),
            "unfaithful_mean_text_vs_latent_match": float(unfaithful_text_latent_mean),
            "unfaithful_mean_text_vs_gold_match": float(unfaithful_text_gold_mean),
            "text_vs_latent_separation_gap": float(faithful_text_latent - unfaithful_text_latent_mean),
        },
        "root_cause_hypothesis": {
            "statement": (
                "Shared latent decoding can collapse variant-level differences because hidden-state inputs are "
                "fixed per trace. Variant-conditioned latent cache should restore per-variant latent differences."
            ),
            "supported": bool(hidden_identity_confirmed and latent_mode == "shared"),
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_json, diagnosis)

    report_lines = [
        "# Latent Separation Diagnosis",
        "",
        f"- Controls used: `{len(controls)}`",
        f"- Needed trace records: `{len(needed_records)}`",
        f"- Model: `{diagnosis['model_metadata']['model_key']}`",
        "",
        "## Key Findings",
        f"- Hidden states shared across control variants: `{diagnosis['design_checks']['hidden_states_shared_across_control_variants']}`",
        (
            "- Mean latent prediction L2 delta (faithful vs unfaithful variants): "
            f"`{diagnosis['design_checks']['mean_latent_pred_l2_delta_from_faithful_over_unfaithful_variants']}`"
        ),
        (
            "- Latent delta metric defined: "
            f"`{diagnosis['design_checks']['latent_delta_metric_defined']}` "
            f"(pairs={diagnosis['design_checks']['latent_delta_pair_count']})"
        ),
        (
            "- Text-vs-latent separation gap (faithful mean minus unfaithful mean): "
            f"`{diagnosis['separation_checks']['text_vs_latent_separation_gap']:.4f}`"
        ),
        (
            "- Root-cause hypothesis supported: "
            f"`{diagnosis['root_cause_hypothesis']['supported']}`"
        ),
        "",
        "## Recommended Scoring Fix",
        "- Use `latent_only` as text-vs-latent agreement, not latent-vs-gold agreement, for control discrimination.",
        "- Keep text-vs-gold as text soundness diagnostic.",
    ]
    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report_lines) + "\n")

    print(f"Saved diagnostic JSON -> {out_json}")
    print(f"Saved diagnostic report -> {out_report}")


if __name__ == "__main__":
    main()

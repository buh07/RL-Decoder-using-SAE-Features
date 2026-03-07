#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:  # pragma: no cover
    from .common import load_json, save_json
except ImportError:  # pragma: no cover
    from common import load_json, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partials", nargs="+", required=True)
    p.add_argument("--phase4-top-features", default="phase4_results/topk/probe/top_features_per_layer.json")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--run-tag", default="")
    return p.parse_args()


def _to_float_or_none(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _build_markdown(payload: Dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Phase 7-SAE Faithfulness Discrimination",
        "",
        f"- Run tag: `{payload.get('run_tag')}`",
        f"- Source control records: `{payload.get('source_control_records')}`",
        f"- Layers analyzed: `{','.join(str(x) for x in payload.get('layers', []))}`",
        "",
        "## Aggregate Summary",
        "",
        f"- Best sparse-probe layer AUROC: `{summary.get('best_probe_layer')}` @ `{summary.get('best_probe_test_auroc')}`",
        f"- Layers with probe AUROC > 0.60: `{summary.get('layers_probe_auroc_gt_0_60')}`",
        f"- Feature-L2 signal layers (margin>0 and exceedance<0.05): `{summary.get('layers_l2_signal')}`",
        f"- Overlap-present layers (top50 abs-d vs Phase4 top50 eq): `{summary.get('layers_with_overlap')}`",
        "",
        "## Channel Pass/Fail",
        "",
        f"- Feature-space L2 channel pass: `{summary.get('feature_l2_channel_pass')}`",
        f"- Per-feature divergence channel pass: `{summary.get('feature_divergence_channel_pass')}`",
        f"- Sparse probe channel pass: `{summary.get('sparse_probe_channel_pass')}`",
        f"- Phase4 overlap channel pass: `{summary.get('phase4_overlap_channel_pass')}`",
        "",
        "## By Layer",
        "",
        "| Layer | Probe AUROC | L2 Margin | L2 Exceedance | Max |d| | Overlap@50 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in payload.get("layers", []):
        row = payload.get("by_layer", {}).get(str(layer), {})
        probe = row.get("sparse_probe", {})
        l2 = row.get("feature_l2_separation", {})
        div = row.get("feature_divergence", {})
        ov = row.get("phase4_overlap_top50_abs_d", {})
        lines.append(
            "| {layer} | {probe_auc} | {margin} | {exceed} | {maxd} | {ovc} |".format(
                layer=layer,
                probe_auc=probe.get("test_auroc_unfaithful_positive"),
                margin=l2.get("margin"),
                exceed=l2.get("exceedance_fraction"),
                maxd=div.get("max_abs_cohens_d"),
                ovc=ov.get("overlap_count_top50"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    partials = [load_json(p) for p in args.partials]
    if not partials:
        raise RuntimeError("No partials provided")
    for idx, p in enumerate(partials):
        if str(p.get("status")) != "ok":
            raise RuntimeError(f"Partial {args.partials[idx]} has non-ok status={p.get('status')!r}")

    run_tag = str(args.run_tag).strip() or str(partials[0].get("run_tag", "phase7_sae_feature"))
    source_control_records = str(partials[0].get("source_control_records", ""))
    sampled_trace_ids = list(partials[0].get("sampled_trace_ids", []))
    sampled_trace_ids_ref = sampled_trace_ids
    for p in partials[1:]:
        if list(p.get("sampled_trace_ids", [])) != sampled_trace_ids_ref:
            raise RuntimeError("Sampled trace IDs mismatch across worker partials; refusing to merge")
        if str(p.get("source_control_records", "")) != source_control_records:
            raise RuntimeError("Source control-record artifact mismatch across worker partials")

    merged_layers: List[int] = []
    by_layer: Dict[str, Any] = {}
    for p in partials:
        for lk, row in dict(p.get("by_layer", {}) or {}).items():
            li = int(lk)
            if str(li) in by_layer:
                raise RuntimeError(f"Duplicate layer result while merging partials: {li}")
            by_layer[str(li)] = row
            merged_layers.append(li)
    merged_layers = sorted(set(merged_layers))

    phase4 = load_json(args.phase4_top_features)
    eq_features = list(phase4.get("eq", [])) if isinstance(phase4, dict) else []
    if not isinstance(eq_features, list) or len(eq_features) == 0:
        raise RuntimeError(f"Unsupported phase4 top-features payload at {args.phase4_top_features}")

    best_probe_auc = None
    best_probe_layer = None
    layers_probe_pass = 0
    layers_l2_signal = 0
    layers_overlap = 0
    layers_divergence = 0
    for li in merged_layers:
        row = by_layer[str(li)]
        div = dict(row.get("feature_divergence") or {})
        top_abs = [int(x.get("feature_idx")) for x in list(div.get("top_features_abs_d") or []) if "feature_idx" in x]
        top_abs = top_abs[:50]
        p4_layer = list(eq_features[li]) if li < len(eq_features) else []
        overlap = sorted(set(top_abs).intersection(set(int(x) for x in p4_layer)))
        row["phase4_overlap_top50_abs_d"] = {
            "phase4_eq_top50_count": int(min(50, len(p4_layer))),
            "overlap_count_top50": int(len(overlap)),
            "overlap_fraction_top50": float(len(overlap) / max(1, min(50, len(p4_layer)))),
            "overlap_feature_indices": [int(x) for x in overlap],
        }
        if len(overlap) > 0:
            layers_overlap += 1

        l2 = dict(row.get("feature_l2_separation") or {})
        margin = _to_float_or_none(l2.get("margin"))
        exceed = _to_float_or_none(l2.get("exceedance_fraction"))
        if margin is not None and exceed is not None and margin > 0.0 and exceed < 0.05:
            layers_l2_signal += 1

        max_abs_d = _to_float_or_none(div.get("max_abs_cohens_d"))
        if max_abs_d is not None and max_abs_d >= 0.20:
            layers_divergence += 1

        probe = dict(row.get("sparse_probe") or {})
        auc = _to_float_or_none(probe.get("test_auroc_unfaithful_positive"))
        if auc is not None and auc > 0.60:
            layers_probe_pass += 1
        if auc is not None and (best_probe_auc is None or auc > best_probe_auc):
            best_probe_auc = auc
            best_probe_layer = int(li)

    summary = {
        "feature_l2_channel_pass": bool(layers_l2_signal > 0),
        "feature_divergence_channel_pass": bool(layers_divergence > 0),
        "sparse_probe_channel_pass": bool(layers_probe_pass > 0),
        "phase4_overlap_channel_pass": bool(layers_overlap > 0),
        "layers_l2_signal": int(layers_l2_signal),
        "layers_feature_divergence_signal": int(layers_divergence),
        "layers_probe_auroc_gt_0_60": int(layers_probe_pass),
        "layers_with_overlap": int(layers_overlap),
        "best_probe_layer": best_probe_layer,
        "best_probe_test_auroc": best_probe_auc,
        "diagnostic_thresholds": {
            "probe_auroc_threshold": 0.60,
            "feature_l2_exceedance_threshold": 0.05,
            "feature_l2_margin_direction": "positive",
            "feature_divergence_abs_d_threshold": 0.20,
        },
    }

    out = {
        "schema_version": "phase7_sae_feature_faithfulness_aggregate_v1",
        "status": "ok",
        "run_tag": run_tag,
        "source_control_records": source_control_records,
        "source_control_records_sha256": partials[0].get("source_control_records_sha256"),
        "sampled_trace_count": int(len(sampled_trace_ids_ref)),
        "sampled_trace_ids": sampled_trace_ids_ref,
        "layers": merged_layers,
        "partials": [str(x) for x in args.partials],
        "phase4_top_features_source": str(args.phase4_top_features),
        "analysis_channels": [
            "feature_l2_separation",
            "per_feature_divergence",
            "sparse_probe",
            "phase4_overlap",
        ],
        "by_layer": by_layer,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }

    md = _build_markdown(out)
    save_json(args.output_json, out)
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    print(f"Saved merged JSON -> {args.output_json}")
    print(f"Saved summary MD  -> {args.output_md}")


if __name__ == "__main__":
    main()

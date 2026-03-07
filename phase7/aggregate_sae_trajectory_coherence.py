#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover
    from .common import load_json, save_json
except ImportError:  # pragma: no cover
    from common import load_json, save_json


METRICS = ("cosine_smoothness", "feature_variance_coherence", "magnitude_monotonicity_coherence")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partials", nargs="+", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--run-tag", default="")
    return p.parse_args()


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _build_markdown(out: Dict[str, Any]) -> str:
    summary = out.get("summary", {})
    lines = [
        "# Phase 7-SAE Trajectory Coherence",
        "",
        f"- Run tag: `{out.get('run_tag')}`",
        f"- Layers: `{','.join(str(x) for x in out.get('layers', []))}`",
        f"- Source records: `{out.get('source_control_records')}`",
        "",
        "## Best Overall",
        "",
        f"- Best layer-metric: `{summary.get('best_layer_metric')}`",
        f"- Best AUROC (unfaithful positive): `{summary.get('best_auroc_unfaithful_positive')}`",
        "",
        "## Confound Check (`wrong_intermediate` vs `order_flip_only`)",
        "",
    ]
    cc = summary.get("confound_check", {})
    for m in METRICS:
        row = cc.get(m, {})
        lines.append(
            f"- `{m}`: wrong_intermediate=`{row.get('wrong_intermediate_auroc')}`, "
            f"order_flip_only=`{row.get('order_flip_only_auroc')}`, "
            f"delta=`{row.get('wrong_minus_orderflip_delta')}`"
        )
    lines.extend(
        [
            "",
            "## Per Layer / Metric AUROC",
            "",
            "| Layer | Cosine AUROC | Variance AUROC | Monotonicity AUROC |",
            "|---:|---:|---:|---:|",
        ]
    )
    for li in out.get("layers", []):
        row = out.get("by_layer", {}).get(str(li), {})
        om = row.get("overall_metrics", {})
        lines.append(
            "| {l} | {c} | {v} | {m} |".format(
                l=li,
                c=(om.get("cosine_smoothness", {}) or {}).get("auroc_unfaithful_positive"),
                v=(om.get("feature_variance_coherence", {}) or {}).get("auroc_unfaithful_positive"),
                m=(om.get("magnitude_monotonicity_coherence", {}) or {}).get("auroc_unfaithful_positive"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    partials = [load_json(p) for p in args.partials]
    if not partials:
        raise RuntimeError("No partials provided")
    for i, p in enumerate(partials):
        if str(p.get("status")) != "ok":
            raise RuntimeError(f"Partial {args.partials[i]} status is not ok: {p.get('status')!r}")

    run_tag = str(args.run_tag).strip() or str(partials[0].get("run_tag", "phase7_sae_trajectory"))
    source_records = str(partials[0].get("source_control_records", ""))
    by_layer: Dict[str, Any] = {}
    layers: List[int] = []
    sampled_trace_ids_ref = list(partials[0].get("coverage_diagnostics", {}).get("trace_ids_sampled", []))
    for p in partials:
        if str(p.get("source_control_records", "")) != source_records:
            raise RuntimeError("Source control records mismatch across partials")
        if list(p.get("coverage_diagnostics", {}).get("trace_ids_sampled", [])) != sampled_trace_ids_ref:
            raise RuntimeError("Sampled trace ids mismatch across partials")
        li = int(p.get("layer"))
        if str(li) in by_layer:
            raise RuntimeError(f"Duplicate layer partial detected: {li}")
        layers.append(li)
        by_layer[str(li)] = p
    layers = sorted(layers)

    best: Tuple[float, str] | None = None
    confound_check: Dict[str, Any] = {}
    for m in METRICS:
        wrong_vals: List[float] = []
        order_vals: List[float] = []
        for li in layers:
            p = by_layer[str(li)]
            auc = _to_float((p.get("overall_metrics", {}).get(m, {}) or {}).get("auroc_unfaithful_positive"))
            if auc is not None:
                key = f"layer{li}:{m}"
                if best is None or auc > best[0]:
                    best = (auc, key)
            vtab = p.get("variant_stratified_metrics", {}).get(m, {}) or {}
            w = _to_float((vtab.get("wrong_intermediate", {}) or {}).get("auroc_unfaithful_positive"))
            o = _to_float((vtab.get("order_flip_only", {}) or {}).get("auroc_unfaithful_positive"))
            if w is not None:
                wrong_vals.append(w)
            if o is not None:
                order_vals.append(o)
        w_mean = (sum(wrong_vals) / len(wrong_vals)) if wrong_vals else None
        o_mean = (sum(order_vals) / len(order_vals)) if order_vals else None
        delta = (float(w_mean) - float(o_mean)) if (w_mean is not None and o_mean is not None) else None
        confound_check[m] = {
            "wrong_intermediate_auroc": w_mean,
            "order_flip_only_auroc": o_mean,
            "wrong_minus_orderflip_delta": delta,
        }

    summary = {
        "best_layer_metric": (best[1] if best else None),
        "best_auroc_unfaithful_positive": (best[0] if best else None),
        "confound_check": confound_check,
    }

    out = {
        "schema_version": "phase7_sae_trajectory_coherence_aggregate_v1",
        "status": "ok",
        "run_tag": run_tag,
        "partials": [str(x) for x in args.partials],
        "source_control_records": source_records,
        "layers": layers,
        "by_layer": by_layer,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }

    md = _build_markdown(out)
    save_json(args.output_json, out)
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    print(f"Saved trajectory aggregate JSON -> {args.output_json}")
    print(f"Saved trajectory aggregate MD   -> {args.output_md}")


if __name__ == "__main__":
    main()

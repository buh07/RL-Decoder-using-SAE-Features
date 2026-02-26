#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from experiments.layer_sweep_manifest import get_layer_set, infer_layer_set_id_from_layers, load_manifest
except Exception:  # pragma: no cover
    get_layer_set = None
    infer_layer_set_id_from_layers = None
    load_manifest = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="phase6_results/results")
    p.add_argument("--manifest", default=None, help="Optional layer sweep manifest to enrich missing metadata")
    p.add_argument("--include-non-sweep", action="store_true")
    p.add_argument("--output", default="phase6_results/results/layer_sweep_phase6_summary.json")
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _scan_eval_paths(results_dir: Path) -> List[Path]:
    paths = [p for p in results_dir.rglob("eval_*.json") if p.is_file()]
    return sorted(paths)


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _layer_variance(layers: List[int]) -> float:
    if not layers:
        return float("nan")
    mu = sum(layers) / len(layers)
    return float(sum((x - mu) ** 2 for x in layers) / len(layers))


def _load_manifest_payload(path: Optional[str]):
    if not path:
        return None
    if load_manifest is None:
        raise RuntimeError("Manifest support unavailable (experiments.layer_sweep_manifest import failed)")
    return load_manifest(path)


def _infer_manifest_row(payload: Optional[Dict[str, Any]], exp_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if payload is None or infer_layer_set_id_from_layers is None or get_layer_set is None:
        return None
    layer_set_id = infer_layer_set_id_from_layers(payload, exp_cfg.get("layers", []))
    if layer_set_id is None:
        return None
    return get_layer_set(payload, layer_set_id)


def _extract_sweep_md(eval_obj: Dict[str, Any], exp_cfg: Dict[str, Any], manifest_payload) -> Optional[Dict[str, Any]]:
    md = dict(eval_obj.get("sweep_metadata") or {})
    row = _infer_manifest_row(manifest_payload, exp_cfg) or {}
    if row:
        md.setdefault("layer_set_id", row.get("layer_set_id"))
        md.setdefault("layer_set_family", row.get("family"))
        md.setdefault("layers", list(row.get("layers", [])))
        md.setdefault("num_layers", int(row.get("num_layers", len(row.get("layers", [])))))
    if not md:
        return None
    md.setdefault("phase", "phase6")
    md.setdefault("input_variant", exp_cfg.get("input_variant"))
    md.setdefault("layers", list(exp_cfg.get("layers", [])))
    md.setdefault("num_layers", len(exp_cfg.get("layers", [])))
    return md


def _join_train_result(eval_path: Path, cfg_name: str) -> Optional[Path]:
    candidate = eval_path.parent / f"supervised_{cfg_name}.json"
    return candidate if candidate.exists() else None


def _row_from_eval(eval_path: Path, manifest_payload, include_non_sweep: bool) -> Optional[Dict[str, Any]]:
    ev = _load_json(eval_path)
    exp_cfg = ev.get("experiment_config") or {}
    cfg_name = str(ev.get("config_name") or exp_cfg.get("name") or "")
    if not cfg_name:
        return None
    sweep_md = _extract_sweep_md(ev, exp_cfg, manifest_payload)
    if not include_non_sweep and sweep_md is None:
        return None

    train_path = _join_train_result(eval_path, cfg_name)
    train_obj = _load_json(train_path) if train_path else {}
    val = dict((ev.get("evaluations") or {}).get("val") or {})
    test = dict((ev.get("evaluations") or {}).get("test") or {})

    layers = list((sweep_md or {}).get("layers") or exp_cfg.get("layers") or [])
    row: Dict[str, Any] = {
        "schema_version": "phase6_layer_sweep_row_v1",
        "phase": "phase6",
        "config_name": cfg_name,
        "input_variant": (sweep_md or {}).get("input_variant", exp_cfg.get("input_variant")),
        "layers": layers,
        "num_layers": int((sweep_md or {}).get("num_layers", len(layers))),
        "layer_set_id": (sweep_md or {}).get("layer_set_id"),
        "layer_set_family": (sweep_md or {}).get("layer_set_family"),
        "sweep_run_id": (sweep_md or {}).get("sweep_run_id"),
        "seed": (sweep_md or {}).get("seed", exp_cfg.get("seed")),
        "parent_baseline": (sweep_md or {}).get("parent_baseline"),
        "best_epoch": train_obj.get("best_epoch"),
        "checkpoint_path": train_obj.get("checkpoint_path"),
        "eval_result_path": str(eval_path),
        "train_result_path": str(train_path) if train_path else None,
        "val_top1": val.get("top1_accuracy"),
        "val_top5": val.get("top5_accuracy"),
        "val_delta_logprob_vs_gpt2": val.get("delta_logprob_vs_gpt2"),
        "test_top1": test.get("top1_accuracy"),
        "test_top5": test.get("top5_accuracy"),
        "test_delta_logprob_vs_gpt2": test.get("delta_logprob_vs_gpt2"),
    }
    row["layer_variance"] = _layer_variance([int(x) for x in layers]) if layers else None
    if sweep_md:
        row["sweep_metadata"] = sweep_md
    return row


def _aggregate(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in rows:
        k = str(r.get(key))
        d = out.setdefault(k, {"n": 0, "test_top1": [], "test_top5": [], "test_delta": [], "val_top1": []})
        d["n"] += 1
        d["test_top1"].append(r.get("test_top1"))
        d["test_top5"].append(r.get("test_top5"))
        d["test_delta"].append(r.get("test_delta_logprob_vs_gpt2"))
        d["val_top1"].append(r.get("val_top1"))
    final = {}
    for k, d in sorted(out.items()):
        final[k] = {
            "n": d["n"],
            "mean_test_top1": _mean(d["test_top1"]),
            "mean_test_top5": _mean(d["test_top5"]),
            "mean_test_delta_logprob_vs_gpt2": _mean(d["test_delta"]),
            "mean_val_top1": _mean(d["val_top1"]),
        }
    return final


def _rank_rows(rows: List[Dict[str, Any]], *, input_variant: Optional[str] = None) -> List[Dict[str, Any]]:
    filtered = [r for r in rows if (input_variant is None or r.get("input_variant") == input_variant)]
    ranked = sorted(
        filtered,
        key=lambda r: (
            -float(r.get("test_top1") or float("-inf")),
            -float(r.get("test_top5") or float("-inf")),
            -float(r.get("test_delta_logprob_vs_gpt2") or float("-inf")),
            int(r.get("num_layers") or 999),
            float(r.get("layer_variance") or float("inf")),
            str(r.get("config_name")),
        ),
    )
    return ranked


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    manifest_payload = _load_manifest_payload(args.manifest)
    rows = []
    for p in _scan_eval_paths(results_dir):
        row = _row_from_eval(p, manifest_payload, args.include_non_sweep)
        if row is not None:
            rows.append(row)

    ranked_all = _rank_rows(rows)
    by_variant = {}
    for variant in ["raw", "hybrid", "sae"]:
        rr = _rank_rows(rows, input_variant=variant)
        if rr:
            by_variant[variant] = rr

    out = {
        "schema_version": "phase6_layer_sweep_summary_v1",
        "source_results_dir": str(results_dir),
        "manifest_path": args.manifest,
        "num_rows": len(rows),
        "rows": rows,
        "rankings": {
            "overall_test_top1": ranked_all,
            "by_input_variant_test_top1": by_variant,
        },
        "aggregates": {
            "by_input_variant": _aggregate(rows, "input_variant"),
            "by_layer_set_family": _aggregate(rows, "layer_set_family"),
            "by_num_layers": _aggregate(rows, "num_layers"),
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved phase6 sweep summary -> {out_path}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()

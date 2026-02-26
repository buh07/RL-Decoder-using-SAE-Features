#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from experiments.layer_sweep_manifest import load_manifest
except Exception:  # pragma: no cover
    load_manifest = None


CONFIG_RE = re.compile(r"(state_(?:raw|hybrid|sae)_[A-Za-z0-9_]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase6-summary", required=True)
    p.add_argument("--phase7-summary", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument(
        "--benchmark-glob",
        default=None,
        help="Optional glob for paper-aligned benchmark outputs (e.g. phase7_results/results/*/faithfulness_benchmark_controls_papercore*.json)",
    )
    p.add_argument("--output", default="phase7_results/interventions/layer_sweep_causal_shortlist_summary.json")
    return p.parse_args()


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _layer_variance(layers: Sequence[int]) -> float:
    if not layers:
        return float("inf")
    mu = sum(layers) / len(layers)
    return float(sum((x - mu) ** 2 for x in layers) / len(layers))


def _phase7_rank_key(r: Dict[str, Any]) -> Tuple:
    return (
        -float(r.get("val_result_token_top1")) if _is_finite(r.get("val_result_token_top1")) else float("inf"),
        -float(r.get("val_operator_acc")) if _is_finite(r.get("val_operator_acc")) else float("inf"),
        -float(r.get("val_step_type_acc")) if _is_finite(r.get("val_step_type_acc")) else float("inf"),
        -float(r.get("val_delta_logprob_vs_gpt2")) if _is_finite(r.get("val_delta_logprob_vs_gpt2")) else float("inf"),
        int(r.get("num_layers") or 999),
        float(r.get("layer_variance")) if _is_finite(r.get("layer_variance")) else float("inf"),
        str(r.get("config_name")),
    )


def _phase6_rank_key(r: Dict[str, Any]) -> Tuple:
    return (
        -float(r.get("test_top1")) if _is_finite(r.get("test_top1")) else float("inf"),
        -float(r.get("test_top5")) if _is_finite(r.get("test_top5")) else float("inf"),
        -float(r.get("test_delta_logprob_vs_gpt2")) if _is_finite(r.get("test_delta_logprob_vs_gpt2")) else float("inf"),
        int(r.get("num_layers") or 999),
        float(r.get("layer_variance")) if _is_finite(r.get("layer_variance")) else float("inf"),
        str(r.get("config_name")),
    )


def _sort_phase7(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=_phase7_rank_key)


def _sort_phase6(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=_phase6_rank_key)


def _pick_first(rows: List[Dict[str, Any]], used_layer_sets: set[str], *, predicate=None) -> Optional[Dict[str, Any]]:
    for r in rows:
        lsid = r.get("layer_set_id")
        if not lsid:
            continue
        if lsid in used_layer_sets:
            continue
        if predicate is not None and not predicate(r):
            continue
        return r
    return None


def _layer_set_rows_best_by_phase7(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        lsid = r.get("layer_set_id")
        if lsid:
            grouped.setdefault(str(lsid), []).append(r)
    best = {}
    for lsid, rr in grouped.items():
        best[lsid] = _sort_phase7(rr)[0]
    return best


def _percentile_ranks(rows: List[Dict[str, Any]], key_fn) -> Dict[Tuple[str, str], float]:
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        variant = r.get("input_variant") or "unknown"
        by_variant.setdefault(str(variant), []).append(r)
    out: Dict[Tuple[str, str], float] = {}
    for variant, rr in by_variant.items():
        ranked = sorted(rr, key=key_fn)
        n = len(ranked)
        for idx, r in enumerate(ranked):
            lsid = r.get("layer_set_id")
            if not lsid:
                continue
            score = 1.0 - (idx / max(1, n - 1)) if n > 1 else 1.0
            out[(variant, str(lsid))] = float(score)
    return out


def _infer_config_name_from_benchmark_path(bench_obj: Dict[str, Any], path: Path) -> Optional[str]:
    candidates = [str(bench_obj.get("source_audit") or ""), str(path), path.name]
    for c in candidates:
        m = CONFIG_RE.search(c)
        if m:
            return m.group(1)
    return None


def _load_benchmark_scores(glob_pat: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not glob_pat:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for p_str in sorted(glob.glob(glob_pat, recursive=True)):
        p = Path(p_str)
        if not p.is_file():
            continue
        try:
            obj = _load_json(p)
        except Exception:
            continue
        cfg_name = _infer_config_name_from_benchmark_path(obj, p)
        if not cfg_name:
            continue
        by_track = obj.get("by_benchmark_track") or {}
        latent = (by_track.get("latent_only") or {}).get("auroc")
        causal = (by_track.get("causal_auditor") or {}).get("auroc")
        text = (by_track.get("text_only") or {}).get("auroc")
        if cfg_name not in out:
            out[cfg_name] = {}
        if _is_finite(latent):
            out[cfg_name]["latent_only_auroc"] = float(latent)
        if _is_finite(causal):
            out[cfg_name]["causal_auditor_auroc"] = float(causal)
        if _is_finite(text):
            out[cfg_name]["text_only_auroc"] = float(text)
        out[cfg_name]["benchmark_path"] = str(p)
        if _is_finite(latent) and _is_finite(causal):
            out[cfg_name]["latent_minus_causal_auroc"] = float(latent) - float(causal)
    return out


def _load_manifest_meta(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if load_manifest is None:
        raise RuntimeError("Manifest support unavailable (experiments.layer_sweep_manifest import failed)")
    return load_manifest(path)


def main() -> None:
    args = parse_args()
    p6 = _load_json(args.phase6_summary)
    p7 = _load_json(args.phase7_summary)
    manifest = _load_manifest_meta(args.manifest)
    benchmark_scores = _load_benchmark_scores(args.benchmark_glob)

    p6_rows = [r for r in p6.get("rows", []) if r.get("layer_set_id")]
    p7_rows = [r for r in p7.get("rows", []) if r.get("layer_set_id")]

    p7_sorted = _sort_phase7(p7_rows)
    p7_raw_sorted = _sort_phase7([r for r in p7_rows if r.get("input_variant") == "raw"])
    p7_hyb_sorted = _sort_phase7([r for r in p7_rows if r.get("input_variant") == "hybrid"])

    used_layer_sets: set[str] = set()
    picks: List[Dict[str, Any]] = []

    def add_pick(slot: str, row: Optional[Dict[str, Any]], rationale: str, extra: Optional[Dict[str, Any]] = None):
        if row is None:
            picks.append({"slot": slot, "status": "unfilled", "rationale": rationale, **(extra or {})})
            return
        lsid = str(row.get("layer_set_id"))
        used_layer_sets.add(lsid)
        rec = {
            "slot": slot,
            "status": "selected",
            "layer_set_id": lsid,
            "config_name": row.get("config_name"),
            "input_variant": row.get("input_variant"),
            "layers": row.get("layers"),
            "num_layers": row.get("num_layers"),
            "layer_set_family": row.get("layer_set_family"),
            "rationale": rationale,
            "phase7_metrics": {
                "val_result_token_top1": row.get("val_result_token_top1"),
                "val_operator_acc": row.get("val_operator_acc"),
                "val_step_type_acc": row.get("val_step_type_acc"),
                "val_delta_logprob_vs_gpt2": row.get("val_delta_logprob_vs_gpt2"),
                "test_result_token_top1": row.get("test_result_token_top1"),
                "test_delta_logprob_vs_gpt2": row.get("test_delta_logprob_vs_gpt2"),
            },
        }
        if extra:
            rec.update(extra)
        picks.append(rec)

    add_pick(
        "best_raw_phase7",
        _pick_first(p7_raw_sorted, used_layer_sets),
        "Top Phase 7 raw config by val composite (result_token_top1, operator_acc, step_type_acc, delta_logprob).",
    )
    add_pick(
        "best_hybrid_phase7",
        _pick_first(p7_hyb_sorted, used_layer_sets),
        "Top Phase 7 hybrid config by val composite with uniqueness on layer_set_id.",
    )
    add_pick(
        "best_middle_family_phase7",
        _pick_first(p7_sorted, used_layer_sets, predicate=lambda r: str(r.get("layer_set_family")) == "middle" or str(r.get("layer_set_id", "")).startswith("middle")),
        "Top Phase 7 middle-focused subset by val composite.",
    )
    add_pick(
        "best_every2_family_phase7",
        _pick_first(p7_sorted, used_layer_sets, predicate=lambda r: str(r.get("layer_set_id", "")).startswith("every2_")),
        "Top Phase 7 every-2-layer subset by val composite.",
    )

    # Slot 5: cross-phase mismatch (strong in Phase 6 final-token readout, weaker in Phase 7 structured-state readout).
    p6_best_by_ls = {}
    for variant in ["raw", "hybrid", "sae"]:
        for r in _sort_phase6([x for x in p6_rows if x.get("input_variant") == variant]):
            lsid = r.get("layer_set_id")
            if not lsid:
                continue
            p6_best_by_ls.setdefault((variant, str(lsid)), r)
    p6_pct = _percentile_ranks(p6_rows, _phase6_rank_key)
    p7_pct = _percentile_ranks(p7_rows, _phase7_rank_key)
    mismatch_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for (variant, lsid), p6_score in p6_pct.items():
        if lsid in used_layer_sets:
            continue
        p7_score = p7_pct.get((variant, lsid))
        if p7_score is None:
            continue
        gap = float(p6_score - p7_score)
        if gap <= 0:
            continue
        p7_row = next((r for r in _sort_phase7([x for x in p7_rows if x.get("input_variant") == variant and x.get("layer_set_id") == lsid])), None)
        p6_row = p6_best_by_ls.get((variant, lsid))
        if p7_row is None or p6_row is None:
            continue
        mismatch_candidates.append((gap, p7_row, p6_row))
    mismatch_candidates.sort(key=lambda t: (-t[0], _phase7_rank_key(t[1])))
    if mismatch_candidates:
        gap, p7_row, p6_row = mismatch_candidates[0]
        add_pick(
            "phase6_phase7_mismatch",
            p7_row,
            "Chosen for cross-phase mismatch: strong Phase 6 rank percentile but weaker Phase 7 percentile.",
            extra={
                "cross_phase_mismatch": {
                    "phase6_percentile": p6_pct.get((p7_row["input_variant"], p7_row["layer_set_id"])),
                    "phase7_percentile": p7_pct.get((p7_row["input_variant"], p7_row["layer_set_id"])),
                    "percentile_gap": gap,
                    "phase6_reference_config": p6_row.get("config_name"),
                    "phase6_test_top1": p6_row.get("test_top1"),
                    "phase6_test_delta_logprob_vs_gpt2": p6_row.get("test_delta_logprob_vs_gpt2"),
                }
            },
        )
    else:
        add_pick("phase6_phase7_mismatch", None, "No positive cross-phase mismatch candidate found.")

    # Slot 6: high latent-only / low causal-proxy risk, if benchmark track data available.
    benchmark_augmented: List[Tuple[Tuple, Dict[str, Any], Dict[str, Any]]] = []
    for r in p7_sorted:
        lsid = r.get("layer_set_id")
        if not lsid or lsid in used_layer_sets:
            continue
        b = benchmark_scores.get(str(r.get("config_name")), {})
        latent = b.get("latent_only_auroc")
        causal = b.get("causal_auditor_auroc")
        text = b.get("text_only_auroc")
        if _is_finite(latent):
            risk_gap = max(0.0, float(latent) - float(causal)) if _is_finite(causal) else float("inf")
            key = (
                -float(latent),
                risk_gap,
                -float(text) if _is_finite(text) else 0.0,
                _phase7_rank_key(r),
            )
            benchmark_augmented.append((key, r, b))
    benchmark_augmented.sort(key=lambda t: t[0])
    if benchmark_augmented:
        _, row, b = benchmark_augmented[0]
        add_pick(
            "high_latent_low_causal_risk",
            row,
            "High latent-only benchmark track performance with low latent-vs-causal AUROC gap.",
            extra={"benchmark_track_contrast": b},
        )
    else:
        fallback = _pick_first(p7_sorted, used_layer_sets)
        add_pick(
            "high_latent_low_causal_risk",
            fallback,
            "Fallback: no per-config paper-track benchmark files available; chose best remaining Phase 7 subset by val composite.",
            extra={"benchmark_track_contrast": {"status": "unavailable", "reason": "no benchmark_glob matches / config inference failed"}},
        )

    selected_ids = [p["layer_set_id"] for p in picks if p.get("status") == "selected"]
    unique_selected_ids = sorted({x for x in selected_ids if x is not None})

    out: Dict[str, Any] = {
        "schema_version": "phase7_layer_sweep_causal_shortlist_summary_v1",
        "phase6_summary_path": str(args.phase6_summary),
        "phase7_summary_path": str(args.phase7_summary),
        "manifest_path": args.manifest,
        "benchmark_glob": args.benchmark_glob,
        "selection_rules": {
            "slots": [
                "best_raw_phase7",
                "best_hybrid_phase7",
                "best_middle_family_phase7",
                "best_every2_family_phase7",
                "phase6_phase7_mismatch",
                "high_latent_low_causal_risk",
            ],
            "phase7_primary_rank_keys": [
                "val_result_token_top1",
                "val_operator_acc",
                "val_step_type_acc",
                "val_delta_logprob_vs_gpt2",
            ],
            "tie_breakers": ["fewer_layers", "lower_layer_variance"],
            "uniqueness": "layer_set_id",
        },
        "shortlist_size_target": 6,
        "shortlist_size_selected": len(unique_selected_ids),
        "shortlisted_layer_set_ids": unique_selected_ids,
        "slots": picks,
        "runtime_per_job_minutes": None,
        "causal_pass_rates_by_subset_layer": None,
        "off_manifold_rates": None,
        "benchmark_track_comparison_summary": None,
    }
    if manifest is not None:
        out["manifest_schema_version"] = manifest.get("schema_version")
        out["causal_scope_defaults"] = manifest.get("causal_scope_defaults")
        out["shortlist_selection_rules_manifest"] = manifest.get("shortlist_selection_rules")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved Phase 7 causal shortlist summary -> {out_path}")
    print(f"selected_unique_layer_sets={len(unique_selected_ids)} ids={unique_selected_ids}")


if __name__ == "__main__":
    main()

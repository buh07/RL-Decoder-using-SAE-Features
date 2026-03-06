#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:  # pragma: no cover
    from .common import load_json, save_json, sha256_file
except ImportError:  # pragma: no cover
    from common import load_json, save_json, sha256_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", default="phase7_results/audits/text_causal_audit_controls.json")
    p.add_argument("--calib-fraction", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=20260303)
    p.add_argument("--run-tag", default=None, help="Used for default output naming.")
    p.add_argument("--output-dir", default="phase7_results/audits")
    p.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Backward-compatible alias. If provided, derives "
            "<prefix>_calib.json, <prefix>_eval.json, and <prefix>_split_manifest.json. "
            "Explicit --output-calib/--output-eval/--output-manifest override this."
        ),
    )
    p.add_argument("--output-calib", default=None)
    p.add_argument("--output-eval", default=None)
    p.add_argument("--output-manifest", default=None)
    p.add_argument(
        "--reuse-manifest-if-compatible",
        default=None,
        help=(
            "Optional existing manifest path. If source_audit_rows_sha256 and split_policy_hash "
            "match and output files exist, reuse and exit."
        ),
    )
    p.add_argument(
        "--source-audit-hash",
        default=None,
        help="Optional precomputed source audit SHA256. Defaults to file SHA256 of --audit.",
    )
    return p.parse_args()


def _split_trace_ids(trace_ids: Sequence[str], calib_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = sorted({str(x) for x in trace_ids})
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    if not ids:
        return [], []
    if len(ids) == 1:
        # Keep a non-empty evaluation split for deterministic single-trace smoke runs.
        return [], [str(ids[0])]
    frac = min(1.0, max(0.0, float(calib_fraction)))
    n_calib = int(round(len(ids) * frac))
    n_calib = max(1, n_calib)
    n_calib = min(n_calib, len(ids) - 1)
    calib = sorted(ids[:n_calib])
    eval_ids = sorted(ids[n_calib:])
    return calib, eval_ids


def _trace_hash(trace_ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for tid in sorted({str(x) for x in trace_ids}):
        h.update(tid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _split_policy_hash(*, seed: int, calib_fraction: float) -> str:
    h = hashlib.sha256()
    h.update(f"seed={int(seed)}|calib_fraction={float(calib_fraction):.12f}|group_by=trace_id".encode("utf-8"))
    return h.hexdigest()


def _counts(rows: Sequence[Dict]) -> Dict[str, Dict[str, int]]:
    by_label = Counter(str(r.get("gold_label", "unknown")) for r in rows)
    by_variant = Counter(str(r.get("control_variant", "unknown")) for r in rows)
    return {
        "by_label": {k: int(v) for k, v in sorted(by_label.items())},
        "by_variant": {k: int(v) for k, v in sorted(by_variant.items())},
    }


def main() -> None:
    args = parse_args()
    payload = load_json(args.audit)
    audits = list(payload.get("audits", []))
    source_audit_rows_sha = str(args.source_audit_hash) if args.source_audit_hash else sha256_file(args.audit)
    split_policy_hash = _split_policy_hash(seed=int(args.seed), calib_fraction=float(args.calib_fraction))

    if args.reuse_manifest_if_compatible:
        mp = Path(args.reuse_manifest_if_compatible)
        if mp.exists():
            try:
                prior = load_json(mp)
                prior_sha = str(prior.get("source_audit_rows_sha256", ""))
                prior_policy = str(prior.get("split_policy_hash", ""))
                outs = dict(prior.get("outputs", {}) or {})
                calib_out = Path(str(outs.get("calibration_audit", "")))
                eval_out = Path(str(outs.get("evaluation_audit", "")))
                mani_out = Path(str(outs.get("manifest", "")))
                if (
                    prior_sha == source_audit_rows_sha
                    and prior_policy == split_policy_hash
                    and calib_out.exists()
                    and eval_out.exists()
                    and mani_out.exists()
                ):
                    print(f"Reusing split manifest -> {mp}")
                    print(f"Reusing calibration audit -> {calib_out}")
                    print(f"Reusing evaluation audit  -> {eval_out}")
                    return
            except Exception:
                pass

    trace_ids_all = sorted({str(a.get("trace_id")) for a in audits if a.get("trace_id") is not None})
    calib_ids, eval_ids = _split_trace_ids(trace_ids_all, args.calib_fraction, args.seed)
    calib_set = set(calib_ids)
    eval_set = set(eval_ids)

    calib_rows = [a for a in audits if str(a.get("trace_id")) in calib_set]
    eval_rows = [a for a in audits if str(a.get("trace_id")) in eval_set]
    overlap = sorted(calib_set.intersection(eval_set))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.run_tag or Path(args.audit).stem

    prefix_base = Path(args.output_prefix) if args.output_prefix else None
    default_calib = (prefix_base.parent / f"{prefix_base.name}_calib.json") if prefix_base else (out_dir / f"{run_tag}_calib.json")
    default_eval = (prefix_base.parent / f"{prefix_base.name}_eval.json") if prefix_base else (out_dir / f"{run_tag}_eval.json")
    default_manifest = (
        (prefix_base.parent / f"{prefix_base.name}_split_manifest.json")
        if prefix_base
        else (out_dir / f"{run_tag}_split_manifest.json")
    )

    calib_path = Path(args.output_calib) if args.output_calib else default_calib
    eval_path = Path(args.output_eval) if args.output_eval else default_eval
    manifest_path = Path(args.output_manifest) if args.output_manifest else default_manifest

    common_meta = {
        "schema_version": payload.get("schema_version", "causal_audit_v1"),
        "model_metadata": payload.get("model_metadata"),
        "source_audit": str(args.audit),
        "summary": payload.get("summary"),
    }
    save_json(
        calib_path,
        {
            **common_meta,
            "split_role": "calibration",
            "split_manifest_path": str(manifest_path),
            "audits": calib_rows,
        },
    )
    save_json(
        eval_path,
        {
            **common_meta,
            "split_role": "evaluation",
            "split_manifest_path": str(manifest_path),
            "audits": eval_rows,
        },
    )

    manifest = {
        "schema_version": "phase7_audit_split_manifest_v1",
        "source_audit": str(args.audit),
        "source_audit_rows_sha256": source_audit_rows_sha,
        "split_seed": int(args.seed),
        "calib_fraction": float(args.calib_fraction),
        "split_policy_hash": split_policy_hash,
        "single_trace_eval_mode": bool(len(trace_ids_all) == 1),
        "trace_ids_calib": calib_ids,
        "trace_ids_eval": eval_ids,
        "overlap_trace_ids": overlap,
        "calibration_trace_count": int(len(calib_ids)),
        "evaluation_trace_count": int(len(eval_ids)),
        "calibration_trace_hash": _trace_hash(calib_ids),
        "evaluation_trace_hash": _trace_hash(eval_ids),
        "calibration_counts": _counts(calib_rows),
        "evaluation_counts": _counts(eval_rows),
        "outputs": {
            "calibration_audit": str(calib_path),
            "evaluation_audit": str(eval_path),
            "manifest": str(manifest_path),
        },
    }
    save_json(manifest_path, manifest)

    print(f"Saved calibration audit -> {calib_path}")
    print(f"Saved evaluation audit  -> {eval_path}")
    print(f"Saved split manifest    -> {manifest_path}")
    print(
        "Trace split:",
        f"calib={len(calib_ids)}",
        f"eval={len(eval_ids)}",
        f"overlap={len(overlap)}",
    )


if __name__ == "__main__":
    main()

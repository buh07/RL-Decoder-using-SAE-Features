#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from common import load_json, save_json


def _roc_points(scores_labels: List[Tuple[float, int]]):
    thresholds = sorted({float(s) for s, _ in scores_labels}, reverse=True)
    if not thresholds:
        return []
    thresholds = [max(thresholds) + 1e-6] + thresholds + [min(thresholds) - 1e-6]
    out = []
    P = sum(y for _, y in scores_labels)
    N = len(scores_labels) - P
    for t in thresholds:
        tp = fp = tn = fn = 0
        for s, y in scores_labels:
            pred = s >= t
            if pred and y == 1:
                tp += 1
            elif pred and y == 0:
                fp += 1
            elif (not pred) and y == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        out.append({"threshold": t, "tpr": tpr, "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", default="phase7_results/audits/text_causal_audit_controls.json")
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument(
        "--score-track",
        choices=["causal_auditor", "text_only", "latent_only"],
        default="causal_auditor",
        help="Which score to calibrate threshold for (default keeps v1 behavior on overall_score)",
    )
    p.add_argument("--output", default="phase7_results/calibration/phase7_thresholds_v1.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    aud = load_json(args.audit)
    audits = aud.get("audits", [])
    scored: List[Tuple[float, int]] = []
    for a in audits:
        lbl = a.get("gold_label")
        if lbl not in {"faithful", "unfaithful"}:
            continue
        if args.score_track == "causal_auditor":
            score = float(a.get("overall_score", 0.0))
        else:
            score = float((a.get("benchmark_track_scores") or {}).get(args.score_track, 0.0))
        scored.append((score, 1 if lbl == "faithful" else 0))
    roc = _roc_points(scored)
    if not roc:
        raise RuntimeError("No labeled faithful/unfaithful audits found")

    # Choose highest TPR with FPR <= target.
    feasible = [r for r in roc if r["fpr"] <= args.target_fpr]
    if feasible:
        best = max(feasible, key=lambda r: (r["tpr"], -r["fpr"], r["threshold"]))
    else:
        best = min(roc, key=lambda r: abs(r["fpr"] - args.target_fpr))

    base_thresholds = aud.get("summary", {}).get("thresholds", {}).get("thresholds", {})
    if not base_thresholds:
        from causal_audit import default_thresholds
        base_thresholds = default_thresholds()
    base_thresholds = dict(base_thresholds)
    base_thresholds["overall_score_faithful_min"] = float(best["threshold"])

    out = {
        "thresholds_version": "phase7_thresholds_v1",
        "source_audit": args.audit,
        "target_fpr": float(args.target_fpr),
        "score_track": args.score_track,
        "selected_point": best,
        "thresholds": base_thresholds,
        "roc_points": roc,
        "num_labeled_audits": len(scored),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    print(f"Saved thresholds -> {out_path}")
    print(
        f"Selected threshold for {args.score_track}: overall_score_faithful_min={best['threshold']:.4f} "
        f"(TPR={best['tpr']:.3f}, FPR={best['fpr']:.3f})"
    )


if __name__ == "__main__":
    main()

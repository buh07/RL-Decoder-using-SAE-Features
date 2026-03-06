#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover
    from .common import load_json, save_json
except ImportError:  # pragma: no cover
    from common import load_json, save_json


def _default_thresholds() -> Dict[str, float]:
    try:  # pragma: no cover
        from .causal_audit import default_thresholds as _impl
    except ImportError:  # pragma: no cover
        from causal_audit import default_thresholds as _impl
    return _impl()


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
    p.add_argument(
        "--audit-calib",
        default=None,
        help="Preferred calibration audit path (split-aware). If set, calibration runs only on this file.",
    )
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument(
        "--score-track",
        choices=[
            "causal_auditor",
            "text_only",
            "latent_only",
            "composite",
            "confidence_margin",
            "trajectory_coherence",
            "contrastive_probe",
            "representation_geometry",
        ],
        default="causal_auditor",
        help="Which score to calibrate threshold for (default keeps v1 behavior on overall_score)",
    )
    p.add_argument(
        "--threshold-policy",
        choices=["max_recall_at_fpr_le_target", "closest_fpr"],
        default="max_recall_at_fpr_le_target",
        help=(
            "Policy for selecting gate operating point. "
            "max_recall_at_fpr_le_target chooses highest recall under target FPR."
        ),
    )
    p.add_argument(
        "--analysis-policy",
        choices=["max_f1", "max_youden_j"],
        default="max_f1",
        help="Secondary analysis operating point policy.",
    )
    p.add_argument(
        "--positive-label",
        choices=["faithful", "unfaithful"],
        default="unfaithful",
        help="Class treated as positive for ROC/FPR threshold calibration.",
    )
    p.add_argument(
        "--all-tracks",
        action="store_true",
        help="Calibrate thresholds for text_only, latent_only, and causal_auditor in one pass.",
    )
    p.add_argument("--output", default="phase7_results/calibration/phase7_thresholds_v1.json")
    return p.parse_args()


def _score_for_track(audit_row: Dict, track: str) -> float:
    if track in {"causal_auditor", "composite"}:
        tracks = audit_row.get("benchmark_track_scores") or {}
        if track in tracks:
            return float(tracks.get(track, 0.0))
        return float(audit_row.get("overall_score", 0.0))
    return float((audit_row.get("benchmark_track_scores") or {}).get(track, 0.0))


def _to_positive_score(score: float, positive_label: str) -> float:
    s = float(score)
    if positive_label == "faithful":
        return s
    if positive_label == "unfaithful":
        return 1.0 - s
    raise ValueError(f"Unsupported positive_label={positive_label!r}")


def _trace_hash(audits: List[Dict]) -> Tuple[int, str]:
    trace_ids = sorted({str(a.get("trace_id")) for a in audits if a.get("trace_id") is not None})
    h = hashlib.sha256()
    for tid in trace_ids:
        h.update(tid.encode("utf-8"))
        h.update(b"\n")
    return len(trace_ids), h.hexdigest()


def _pick_gate_point(
    roc: List[Dict[str, float]],
    *,
    target_fpr: float,
    threshold_policy: str,
) -> Dict[str, float]:
    if threshold_policy == "max_recall_at_fpr_le_target":
        feasible = [r for r in roc if float(r["fpr"]) <= float(target_fpr)]
        if feasible:
            return max(feasible, key=lambda r: (float(r["tpr"]), -float(r["fpr"]), float(r["threshold"])))
        return min(roc, key=lambda r: abs(float(r["fpr"]) - float(target_fpr)))
    if threshold_policy == "closest_fpr":
        return min(roc, key=lambda r: abs(float(r["fpr"]) - float(target_fpr)))
    raise ValueError(f"Unsupported threshold_policy={threshold_policy!r}")


def _f1(row: Dict[str, float]) -> float:
    tp = float(row.get("tp", 0.0))
    fp = float(row.get("fp", 0.0))
    fn = float(row.get("fn", 0.0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom > 0 else 0.0


def _pick_analysis_point(
    roc: List[Dict[str, float]],
    *,
    analysis_policy: str,
) -> Optional[Dict[str, float]]:
    if not roc:
        return None
    if analysis_policy == "max_f1":
        return max(roc, key=lambda r: (_f1(r), float(r["tpr"]), -float(r["fpr"])))
    if analysis_policy == "max_youden_j":
        return max(roc, key=lambda r: (float(r["tpr"]) - float(r["fpr"]), float(r["tpr"]), -float(r["fpr"])))
    raise ValueError(f"Unsupported analysis_policy={analysis_policy!r}")


def _calibrate_track(
    audits: List[Dict],
    track: str,
    target_fpr: float,
    positive_label: str,
    *,
    threshold_policy: str,
    analysis_policy: str,
) -> Dict:
    scored: List[Tuple[float, int]] = []
    for a in audits:
        lbl = a.get("gold_label")
        if lbl not in {"faithful", "unfaithful"}:
            continue
        raw_score = _score_for_track(a, track)
        scored.append((_to_positive_score(raw_score, positive_label), 1 if lbl == positive_label else 0))
    roc = _roc_points(scored)
    if not roc:
        raise RuntimeError(f"No labeled faithful/unfaithful audits found for track={track}")

    gate_point = _pick_gate_point(
        roc,
        target_fpr=float(target_fpr),
        threshold_policy=str(threshold_policy),
    )
    analysis_point = _pick_analysis_point(roc, analysis_policy=str(analysis_policy))
    return {
        "track": track,
        "selected_point": gate_point,  # backward-compatible alias
        "gate_point": gate_point,
        "analysis_point": analysis_point,
        "roc_points": roc,
        "num_labeled_audits": len(scored),
    }


def main() -> None:
    args = parse_args()
    audit_path = str(args.audit_calib or args.audit)
    aud = load_json(audit_path)
    audits = aud.get("audits", [])
    positive_label = str(args.positive_label)
    primary = _calibrate_track(
        audits,
        args.score_track,
        float(args.target_fpr),
        positive_label,
        threshold_policy=str(args.threshold_policy),
        analysis_policy=str(args.analysis_policy),
    )
    gate_point = primary["gate_point"]
    analysis_point = primary.get("analysis_point")
    trace_count, trace_hash = _trace_hash(audits)

    base_thresholds = aud.get("summary", {}).get("thresholds", {}).get("thresholds", {})
    if not base_thresholds:
        base_thresholds = _default_thresholds()
    base_thresholds = dict(base_thresholds)
    base_thresholds["overall_score_faithful_min"] = float(gate_point["threshold"])

    track_thresholds = None
    if args.all_tracks:
        available_tracks = set()
        for a in audits:
            tracks = a.get("benchmark_track_scores") or {}
            if isinstance(tracks, dict):
                available_tracks.update(str(k) for k in tracks.keys())
        track_order = [
            "text_only",
            "latent_only",
            "confidence_margin",
            "trajectory_coherence",
            "contrastive_probe",
            "representation_geometry",
            "causal_auditor",
            "composite",
        ]
        required_tracks = {"text_only", "latent_only", "causal_auditor", "composite"}
        tracks_to_calibrate = [t for t in track_order if t in available_tracks or t in required_tracks]
        track_thresholds = {}
        for track in tracks_to_calibrate:
            calibrated = _calibrate_track(
                audits,
                track,
                float(args.target_fpr),
                positive_label,
                threshold_policy=str(args.threshold_policy),
                analysis_policy=str(args.analysis_policy),
            )
            track_thresholds[track] = {
                "threshold": float(calibrated["gate_point"]["threshold"]),  # backward-compatible
                "selected_point": calibrated["gate_point"],  # backward-compatible
                "gate_point": calibrated["gate_point"],
                "analysis_point": calibrated.get("analysis_point"),
                "num_labeled_audits": int(calibrated["num_labeled_audits"]),
            }

    out = {
        "thresholds_version": "phase7_thresholds_v1",
        "source_audit": audit_path,
        "calibration_audit_path": audit_path,
        "calibration_trace_count": int(trace_count),
        "calibration_trace_hash": trace_hash,
        "calibration_source_ref": {
            "path": audit_path,
            "trace_count": int(trace_count),
            "trace_hash": trace_hash,
        },
        "model_metadata": aud.get("model_metadata"),
        "target_fpr": float(args.target_fpr),
        "score_track": args.score_track,
        "threshold_policy": str(args.threshold_policy),
        "analysis_policy": str(args.analysis_policy),
        "positive_label": positive_label,
        "fpr_semantics": (
            "false_positive_rate_on_unfaithful_predictions_against_faithful_examples"
            if positive_label == "unfaithful"
            else "false_positive_rate_on_faithful_predictions_against_unfaithful_examples"
        ),
        "all_tracks": bool(args.all_tracks),
        "selected_point": gate_point,  # backward-compatible alias
        "gate_point": gate_point,
        "analysis_point": analysis_point,
        "thresholds": base_thresholds,
        "roc_points": primary["roc_points"],
        "num_labeled_audits": int(primary["num_labeled_audits"]),
    }
    if track_thresholds is not None:
        out["track_thresholds"] = track_thresholds
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    print(f"Saved thresholds -> {out_path}")
    print(
        f"Selected threshold for {args.score_track} (positive_label={positive_label}): "
        f"{gate_point['threshold']:.4f} (TPR={gate_point['tpr']:.3f}, FPR={gate_point['fpr']:.3f})"
    )


if __name__ == "__main__":
    main()

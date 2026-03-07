#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from .common import load_json, save_json
except Exception:
    from common import load_json, save_json


FATAL_PATTERNS = [
    r"Traceback",
    r"RuntimeError",
    r"CUDA out of memory",
    r"\bNaN\b",
    r"\bKilled\b",
]


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def _load_json_safe(path: Path) -> Dict[str, Any] | None:
    try:
        payload = load_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _tmux_sessions(run_id: str) -> List[str]:
    try:
        out = subprocess.check_output(["tmux", "ls"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    sessions: List[str] = []
    for line in out.splitlines():
        sess = line.split(":", 1)[0].strip()
        if run_id in sess:
            sessions.append(sess)
    return sessions


def _gpu_table() -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for ln in out.splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append(
                {
                    "gpu_index": int(parts[0]),
                    "utilization_gpu_pct": float(parts[1]),
                    "memory_used_mib": float(parts[2]),
                    "memory_total_mib": float(parts[3]),
                }
            )
        except Exception:
            continue
    return rows


def _count_benchmarks(run_tag: str) -> Dict[str, int]:
    pats = {
        "canary": f"phase7_results/results/faithfulness_benchmark_{run_tag}_canary_*.json",
        "matrix": f"phase7_results/results/faithfulness_benchmark_{run_tag}_matrix_*.json",
    }
    out: Dict[str, int] = {}
    for k, p in pats.items():
        out[k] = len(glob.glob(p))
    out["total"] = out.get("canary", 0) + out.get("matrix", 0)
    return out


def _scan_logs(log_dir: Path) -> Dict[str, Any]:
    files = sorted(log_dir.glob("*.log"))
    hits: List[Dict[str, Any]] = []
    for p in files:
        txt = _read_text(p)
        for pat in FATAL_PATTERNS:
            if re.search(pat, txt):
                hits.append({"log": str(p), "pattern": pat})
    return {"num_logs": len(files), "fatal_hits": hits}


def _leakage_status(run_tag: str) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    for pstr in sorted(glob.glob(f"phase7_results/results/faithfulness_benchmark_{run_tag}_*.json")):
        p = Path(pstr)
        payload = _load_json_safe(p)
        if payload is None:
            checks.append({"artifact": str(p), "parseable": False, "leakage_check_pass": None})
            continue
        checks.append(
            {
                "artifact": str(p),
                "parseable": True,
                "leakage_check_pass": payload.get("leakage_check_pass"),
            }
        )
    bad = [x for x in checks if (not x["parseable"]) or (x["leakage_check_pass"] is False)]
    return {"checked": len(checks), "bad": bad}


def _state_markers(state_dir: Path) -> Dict[str, bool]:
    names = [
        "precompute.done",
        "canary_g0.done",
        "canary_g1.done",
        "matrix_g0.done",
        "matrix_g1.done",
        "matrix_g2.done",
        "pipeline.done",
    ]
    return {n: (state_dir / n).exists() for n in names}


def collect_snapshot(base: Path, run_id: str, run_tag: str) -> Dict[str, Any]:
    logs_dir = base / "logs"
    state_dir = base / "state"
    snapshot: Dict[str, Any] = {
        "schema_version": "phase7_run_monitor_snapshot_v1",
        "timestamp": _now_iso(),
        "run_id": run_id,
        "run_tag": run_tag,
        "base": str(base),
        "tmux_sessions": _tmux_sessions(run_id),
        "state_markers": _state_markers(state_dir),
        "benchmark_counts": _count_benchmarks(run_tag),
        "gpu": _gpu_table(),
        "log_scan": _scan_logs(logs_dir),
        "leakage_scan": _leakage_status(run_tag),
    }
    fail_reasons: List[str] = []
    for h in snapshot["log_scan"]["fatal_hits"]:
        fail_reasons.append(f"fatal_log_pattern:{h['pattern']}:{h['log']}")
    for b in snapshot["leakage_scan"]["bad"]:
        if b["parseable"] is False:
            fail_reasons.append(f"unparseable_benchmark:{b['artifact']}")
        elif b["leakage_check_pass"] is False:
            fail_reasons.append(f"leakage_check_failed:{b['artifact']}")
    snapshot["health"] = "blocked" if fail_reasons else "ok"
    snapshot["fail_reasons"] = fail_reasons
    return snapshot


def write_snapshot(base: Path, snap: Dict[str, Any]) -> Path:
    meta_dir = base / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = meta_dir / f"monitor_snapshot_{ts}.json"
    save_json(out, snap)
    save_json(meta_dir / "monitor_latest.json", snap)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor a Phase 7 run and emit JSON status snapshots.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--interval-seconds", type=int, default=900)
    p.add_argument("--max-iterations", type=int, default=0, help="0 means run forever.")
    p.add_argument("--once", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    iterations = 0
    while True:
        snap = collect_snapshot(base=base, run_id=args.run_id, run_tag=args.run_tag)
        out = write_snapshot(base, snap)
        print(f"[monitor] wrote {out} health={snap.get('health')}")
        if args.once:
            return
        iterations += 1
        if args.max_iterations > 0 and iterations >= args.max_iterations:
            return
        if (base / "state" / "pipeline.done").exists():
            # Write final snapshot and exit after completion.
            final = collect_snapshot(base=base, run_id=args.run_id, run_tag=args.run_tag)
            out2 = write_snapshot(base, final)
            print(f"[monitor] pipeline.done found; wrote final snapshot {out2}")
            return
        time.sleep(max(1, int(args.interval_seconds)))


if __name__ == "__main__":
    main()


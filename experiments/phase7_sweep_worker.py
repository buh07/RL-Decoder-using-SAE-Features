#!/usr/bin/env python3
"""Generic Phase 7 broad-sweep worker with performance-flag pass-through.

This is a tracked reusable worker template. It preserves experiment semantics and
only tunes runtime plumbing (DataLoader/perf flags and thread caps).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

STALE_CLAIM_SEC = 7200  # 2 hours


def load_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2))


def recover_stale_claims(base: Path):
    qdir = base / "queues"
    state_dir = base / "state"
    lock_path = qdir / "pending.lock"
    now = int(time.time())
    recovered = []

    with open(lock_path, "a+") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        claimed = load_json(qdir / "claimed.json", [])
        if claimed:
            keep = []
            for task in claimed:
                cfg = task.get("config_name", "")
                started = state_dir / f"{cfg}.started"
                ts = 0
                if started.exists():
                    try:
                        ts = int(started.read_text().strip())
                    except ValueError:
                        ts = 0
                if ts > 0 and (now - ts) > STALE_CLAIM_SEC:
                    recovered.append(task)
                    started.unlink(missing_ok=True)
                else:
                    keep.append(task)
            if recovered:
                pending = load_json(qdir / "pending.json", [])
                pending.extend(recovered)
                save_json(qdir / "pending.json", pending)
                save_json(qdir / "claimed.json", keep)
        fcntl.flock(lockf, fcntl.LOCK_UN)
    return recovered


def pop_task(base: Path):
    qdir = base / "queues"
    lock_path = qdir / "pending.lock"

    with open(lock_path, "a+") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        pending = load_json(qdir / "pending.json", [])
        if not pending:
            fcntl.flock(lockf, fcntl.LOCK_UN)
            return None
        task = pending.pop(0)
        save_json(qdir / "pending.json", pending)

        claimed = load_json(qdir / "claimed.json", [])
        claimed.append(task)
        save_json(qdir / "claimed.json", claimed)
        fcntl.flock(lockf, fcntl.LOCK_UN)
        return task


def mark_completed(base: Path, task, status: str):
    qdir = base / "queues"
    lock_path = qdir / "pending.lock"

    with open(lock_path, "a+") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        key = task["config_name"]
        claimed = [t for t in load_json(qdir / "claimed.json", []) if t.get("config_name") != key]
        save_json(qdir / "claimed.json", claimed)

        target = qdir / ("completed.json" if status == "completed" else "failed.json")
        rows = load_json(target, [])
        rows.append(task)
        save_json(target, rows)
        fcntl.flock(lockf, fcntl.LOCK_UN)


def run_cmd(cmd, log_path: Path, env: dict[str, str]) -> int:
    with open(log_path, "a") as f:
        f.write(f"\n[CMD] {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def _perf_flags(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    non_blocking_transfer: bool,
    torch_num_threads: int | None,
    cache_inputs: str,
    cache_max_gb: float,
):
    flags = [
        "--num-workers",
        str(max(0, int(num_workers))),
        "--prefetch-factor",
        str(max(1, int(prefetch_factor))),
        "--cache-inputs",
        str(cache_inputs),
        "--cache-max-gb",
        str(float(cache_max_gb)),
    ]
    if pin_memory:
        flags.append("--pin-memory")
    if persistent_workers:
        flags.append("--persistent-workers")
    if non_blocking_transfer:
        flags.append("--non-blocking-transfer")
    if torch_num_threads is not None:
        flags.extend(["--torch-num-threads", str(int(torch_num_threads))])
    return flags


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--worker-name", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-id", required=True)

    ap.add_argument("--train-num-workers", type=int, default=0)
    ap.add_argument("--eval-num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--non-blocking-transfer", action="store_true")
    ap.add_argument("--torch-num-threads", type=int, default=None)
    ap.add_argument("--cache-inputs", choices=["off", "auto", "on"], default="off")
    ap.add_argument("--cache-max-gb", type=float, default=2.0)

    ap.add_argument("--omp-num-threads", type=int, default=None)
    ap.add_argument("--mkl-num-threads", type=int, default=None)
    ap.add_argument("--openblas-num-threads", type=int, default=None)
    ap.add_argument("--numexpr-num-threads", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    logs = base / "logs"
    state = base / "state"
    results = base / "results"
    ckpts = base / "checkpoints"
    meta = base / "meta"
    for p in [logs, state, results, ckpts, meta]:
        p.mkdir(parents=True, exist_ok=True)

    subproc_env = os.environ.copy()
    subproc_env["PYTHONUNBUFFERED"] = "1"
    thread_caps = {
        "OMP_NUM_THREADS": args.omp_num_threads,
        "MKL_NUM_THREADS": args.mkl_num_threads,
        "OPENBLAS_NUM_THREADS": args.openblas_num_threads,
        "NUMEXPR_NUM_THREADS": args.numexpr_num_threads,
    }
    for key, value in thread_caps.items():
        if value is not None:
            subproc_env[key] = str(int(value))

    recovered = recover_stale_claims(base)
    worker_log = logs / f"worker_{args.worker_name}.log"
    with open(worker_log, "a") as f:
        f.write(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} worker={args.worker_name}\n")
        if recovered:
            names = [t.get("config_name", "?") for t in recovered]
            f.write(f"[RECOVERED] {len(recovered)} stale claims: {names}\n")

    summary = {
        "schema_version": "phase7_broad_sweep_worker_summary_v1",
        "run_id": args.run_id,
        "worker": args.worker_name,
        "start_ts": int(time.time()),
        "completed": [],
        "failed": [],
        "retries": [],
    }

    train_perf = _perf_flags(
        num_workers=args.train_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        non_blocking_transfer=args.non_blocking_transfer,
        torch_num_threads=args.torch_num_threads,
        cache_inputs=args.cache_inputs,
        cache_max_gb=args.cache_max_gb,
    )
    eval_perf = _perf_flags(
        num_workers=args.eval_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        non_blocking_transfer=args.non_blocking_transfer,
        torch_num_threads=args.torch_num_threads,
        cache_inputs=args.cache_inputs,
        cache_max_gb=args.cache_max_gb,
    )

    while True:
        task = pop_task(base)
        if task is None:
            break

        cfg = task["config_name"]
        (state / f"{cfg}.started").write_text(str(int(time.time())))

        train_log = logs / f"train_{cfg}_{args.worker_name}.log"
        eval_log = logs / f"eval_{cfg}_{args.worker_name}.log"
        eval_out = results / f"state_decoder_eval_{cfg}.json"
        ckpt_path = ckpts / f"{cfg}.pt"

        if eval_out.exists() and ckpt_path.exists():
            summary["completed"].append(cfg)
            (state / f"{cfg}.done").write_text(str(int(time.time())))
            mark_completed(base, task, "completed")
            continue

        train_cmd = [
            args.python,
            "phase7/train_state_decoders.py",
            "--dataset-train",
            "phase7_results/dataset/gsm8k_step_traces_train.pt",
            "--manifest",
            args.manifest,
            "--layer-set-id",
            task["layer_set_id"],
            "--input-variant",
            task["input_variant"],
            "--custom-config-name",
            cfg,
            "--sweep-run-id",
            args.run_id,
            "--parent-baseline",
            task["parent_baseline"],
            "--checkpoints-dir",
            str(ckpts),
            "--results-dir",
            str(results),
            "--device",
            args.device,
            *train_perf,
        ]

        eval_cmd = [
            args.python,
            "phase7/evaluate_state_decoders.py",
            "--checkpoint",
            str(ckpt_path),
            "--dataset-train",
            "phase7_results/dataset/gsm8k_step_traces_train.pt",
            "--dataset-test",
            "phase7_results/dataset/gsm8k_step_traces_test.pt",
            "--manifest",
            args.manifest,
            "--sweep-run-id",
            args.run_id,
            "--parent-baseline",
            task["parent_baseline"],
            "--output",
            str(eval_out),
            "--device",
            args.device,
            *eval_perf,
        ]

        ok = True
        try:
            for step_name, cmd, logp in [
                ("train", train_cmd, train_log),
                ("eval", eval_cmd, eval_log),
            ]:
                rc = run_cmd(cmd, logp, subproc_env)
                if rc != 0:
                    summary["retries"].append({"config": cfg, "step": step_name, "attempt": 1})
                    rc = run_cmd(cmd, logp, subproc_env)
                    if rc != 0:
                        ok = False
                        fail_obj = {
                            "config": cfg,
                            "step": step_name,
                            "rc": rc,
                            "ts": int(time.time()),
                        }
                        (state / f"{cfg}.failed").write_text(json.dumps(fail_obj))
                        summary["failed"].append(fail_obj)
                        break
        except Exception as exc:
            ok = False
            fail_obj = {"config": cfg, "step": "exception", "error": str(exc), "ts": int(time.time())}
            (state / f"{cfg}.failed").write_text(json.dumps(fail_obj))
            summary["failed"].append(fail_obj)

        if ok:
            (state / f"{cfg}.done").write_text(str(int(time.time())))
            summary["completed"].append(cfg)
            mark_completed(base, task, "completed")
        else:
            mark_completed(base, task, "failed")

        summary["last_update_ts"] = int(time.time())
        (meta / f"worker_{args.worker_name}_summary.json").write_text(json.dumps(summary, indent=2))

    summary["end_ts"] = int(time.time())
    summary["elapsed_sec"] = summary["end_ts"] - summary["start_ts"]
    (meta / f"worker_{args.worker_name}_summary.json").write_text(json.dumps(summary, indent=2))
    (state / f"{args.worker_name}.done").write_text(str(summary["end_ts"]))


if __name__ == "__main__":
    main()

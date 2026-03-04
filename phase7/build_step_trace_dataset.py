#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

try:  # pragma: no cover
    from .common import (
        PHASE7_TRACE_SCHEMA,
        PHASE7_TRACE_SCHEMA_V2,
        build_structured_state,
        parse_expression_steps,
        faithful_step_text,
        group_records_by_example,
        load_pt,
        make_trace_id,
        save_json,
        save_pt,
    )
    from .model_registry import resolve_model_spec
except ImportError:  # pragma: no cover
    from common import (
        PHASE7_TRACE_SCHEMA,
        PHASE7_TRACE_SCHEMA_V2,
        build_structured_state,
        parse_expression_steps,
        faithful_step_text,
        group_records_by_example,
        load_pt,
        make_trace_id,
        save_json,
        save_pt,
    )
    from model_registry import resolve_model_spec


def _tensor_shape_2d(x: object) -> Optional[tuple[int, int]]:
    if isinstance(x, torch.Tensor) and x.ndim == 2:
        return int(x.shape[0]), int(x.shape[1])
    return None


def _validate_records_model_compatibility(
    records: Sequence[dict],
    model_meta: Dict[str, object],
    *,
    allow_legacy_metadata_mismatch: bool,
    split_name: str,
) -> Dict[str, object]:
    expected_layers = int(model_meta["num_layers"])
    expected_hidden_dim = int(model_meta["hidden_dim"])
    expected_model_key = str(model_meta["model_key"])
    expected_family = str(model_meta["model_family"])
    expected_tokenizer = str(model_meta["tokenizer_id"])

    checks = {
        "split": split_name,
        "num_records_checked": int(len(records)),
        "raw_hidden_shape_mismatch_records": 0,
        "record_model_key_mismatch": 0,
        "record_model_family_mismatch": 0,
        "record_num_layers_mismatch": 0,
        "record_hidden_dim_mismatch": 0,
        "record_tokenizer_id_mismatch": 0,
    }
    errors: List[str] = []

    for idx, r in enumerate(records):
        raw_shape = _tensor_shape_2d(r.get("raw_hidden"))
        if raw_shape is None:
            errors.append(
                f"[{split_name}] record[{idx}] missing/invalid raw_hidden tensor; expected 2D tensor"
            )
            continue
        if raw_shape != (expected_layers, expected_hidden_dim):
            checks["raw_hidden_shape_mismatch_records"] += 1
            errors.append(
                f"[{split_name}] record[{idx}] raw_hidden shape={raw_shape} incompatible with "
                f"model_key={expected_model_key} expected=({expected_layers}, {expected_hidden_dim})"
            )

        # Existing record metadata (when present) must agree unless explicitly in legacy mode.
        if "model_key" in r and str(r.get("model_key")) != expected_model_key:
            checks["record_model_key_mismatch"] += 1
            errors.append(
                f"[{split_name}] record[{idx}] model_key={r.get('model_key')!r} != requested {expected_model_key!r}"
            )
        if "model_family" in r and str(r.get("model_family")) != expected_family:
            checks["record_model_family_mismatch"] += 1
            errors.append(
                f"[{split_name}] record[{idx}] model_family={r.get('model_family')!r} != requested {expected_family!r}"
            )
        if "num_layers" in r:
            try:
                record_num_layers = int(r.get("num_layers"))
            except Exception:
                record_num_layers = None
            if record_num_layers != expected_layers:
                checks["record_num_layers_mismatch"] += 1
                errors.append(
                    f"[{split_name}] record[{idx}] num_layers={record_num_layers!r} != requested {expected_layers}"
                )
        if "hidden_dim" in r:
            try:
                record_hidden_dim = int(r.get("hidden_dim"))
            except Exception:
                record_hidden_dim = None
            if record_hidden_dim != expected_hidden_dim:
                checks["record_hidden_dim_mismatch"] += 1
                errors.append(
                    f"[{split_name}] record[{idx}] hidden_dim={record_hidden_dim!r} != requested {expected_hidden_dim}"
                )
        if "tokenizer_id" in r and str(r.get("tokenizer_id")) != expected_tokenizer:
            checks["record_tokenizer_id_mismatch"] += 1
            errors.append(
                f"[{split_name}] record[{idx}] tokenizer_id={r.get('tokenizer_id')!r} != requested {expected_tokenizer!r}"
            )

    if errors and not allow_legacy_metadata_mismatch:
        head = "\n".join(errors[:20])
        tail = "" if len(errors) <= 20 else f"\n... and {len(errors) - 20} more mismatches"
        raise RuntimeError(
            "Phase7 trace build strict model/data compatibility check failed.\n"
            "Use --allow-legacy-metadata-mismatch only if you intentionally accept this unsafe mix.\n"
            f"{head}{tail}"
        )
    checks["mismatch_errors_detected"] = int(len(errors))
    return checks


def _build_from_phase6_records(records: List[dict], split_name: str, model_meta: Dict[str, object]) -> List[dict]:
    by_example = group_records_by_example(records)
    out: List[dict] = []
    for ex_id, group in sorted(by_example.items()):
        trace_id = make_trace_id(split_name, ex_id)
        trace_len = len(group)
        for step_idx, rec in enumerate(group):
            state = build_structured_state(rec, step_idx=step_idx, trace_len=trace_len)
            row = dict(rec)
            row.update(
                {
                    "schema_version": PHASE7_TRACE_SCHEMA,
                    "trace_id": trace_id,
                    "trace_len": trace_len,
                    "step_idx": step_idx,
                    "model_key": str(row.get("model_key", model_meta["model_key"])),
                    "model_family": str(row.get("model_family", model_meta["model_family"])),
                    "num_layers": int(row.get("num_layers", model_meta["num_layers"])),
                    "hidden_dim": int(row.get("hidden_dim", model_meta["hidden_dim"])),
                    "tokenizer_id": str(row.get("tokenizer_id", model_meta["tokenizer_id"])),
                    "structured_state": state,
                    "cot_text_step": faithful_step_text(state),
                    "cot_alignment_span": None,
                }
            )
            out.append(row)
    return out


def _expanded_step_state(
    *,
    rec: dict,
    step_type: str,
    operator: str,
    lhs_value: Optional[float],
    rhs_value: Optional[float],
    subresult_value: float,
    operation_idx: int,
    operation_count: int,
    parse_error: Optional[str],
    source_expr_str: str,
    example_idx: int,
    step_idx: int,
) -> Dict[str, object]:
    c_val = float(rec.get("C", subresult_value))
    return {
        "step_type": str(step_type),
        "operator": str(operator),
        "lhs_value": lhs_value,
        "rhs_value": rhs_value,
        "subresult_value": float(subresult_value),
        "result_token_id": int(rec["result_token_id"]),
        "num_operands": int(operation_count + 1) if operation_count > 0 else int(1),
        "num_operators": int(operation_count),
        "is_multi_op_expr": bool(operation_count > 1),
        "parse_error": parse_error,
        "source_expr_str": str(source_expr_str),
        "example_idx": int(example_idx),
        "step_idx": int(step_idx),
        "operation_idx": int(operation_idx),
        "operation_count": int(operation_count),
        "magnitude_bucket": rec.get("magnitude_bucket")
        or rec.get("structured_state", {}).get("magnitude_bucket")
        or ("[0,10)" if abs(c_val) < 10 else "[10,100)" if abs(c_val) < 100 else "[100,1000)" if abs(c_val) < 1000 else "[1000+)"),
        "sign": rec.get("sign")
        or rec.get("structured_state", {}).get("sign")
        or ("neg" if c_val < 0 else "pos" if c_val > 0 else "zero"),
    }


def _build_from_phase6_records_v2(records: List[dict], split_name: str, model_meta: Dict[str, object]) -> List[dict]:
    by_example = group_records_by_example(records)
    out: List[dict] = []
    for ex_id, group in sorted(by_example.items()):
        trace_id = make_trace_id(split_name, ex_id)
        trace_rows: List[dict] = []
        for rec_idx, rec in enumerate(group):
            c_val = float(rec.get("C", 0.0))
            expr = str(rec.get("expr_str", ""))
            ann_idx = int(rec.get("ann_idx", rec_idx))
            parsed_steps = parse_expression_steps(expr, c_fallback=c_val)
            op_steps = list(parsed_steps.get("steps") or [])
            op_count = int(parsed_steps.get("operation_count", len(op_steps)))
            parse_error = parsed_steps.get("parse_error")

            if parse_error and not op_steps:
                op_steps = [
                    {
                        "operator": "unknown",
                        "lhs_value": None,
                        "rhs_value": None,
                        "subresult_value": c_val,
                        "operation_idx": 0,
                        "operation_count": max(1, op_count),
                    }
                ]
                op_count = max(1, op_count)

            for op in op_steps:
                step_idx = len(trace_rows)
                state = _expanded_step_state(
                    rec=rec,
                    step_type="operate",
                    operator=str(op.get("operator", "unknown")),
                    lhs_value=(float(op["lhs_value"]) if op.get("lhs_value") is not None else None),
                    rhs_value=(float(op["rhs_value"]) if op.get("rhs_value") is not None else None),
                    subresult_value=float(op.get("subresult_value", c_val)),
                    operation_idx=int(op.get("operation_idx", 0)),
                    operation_count=int(op_count),
                    parse_error=str(parse_error) if parse_error is not None else None,
                    source_expr_str=expr,
                    example_idx=int(ex_id),
                    step_idx=step_idx,
                )
                row = dict(rec)
                row.update(
                    {
                        "schema_version": PHASE7_TRACE_SCHEMA_V2,
                        "trace_id": trace_id,
                        "step_idx": int(step_idx),
                        "operation_idx": int(state["operation_idx"]),
                        "operation_count": int(state["operation_count"]),
                        "ann_idx": ann_idx,
                        "source_expr_str": expr,
                        "state_ontology_version": "v2_expanded",
                        "model_key": str(row.get("model_key", model_meta["model_key"])),
                        "model_family": str(row.get("model_family", model_meta["model_family"])),
                        "num_layers": int(row.get("num_layers", model_meta["num_layers"])),
                        "hidden_dim": int(row.get("hidden_dim", model_meta["hidden_dim"])),
                        "tokenizer_id": str(row.get("tokenizer_id", model_meta["tokenizer_id"])),
                        "structured_state": state,
                        "cot_text_step": faithful_step_text(state),
                        "cot_alignment_span": None,
                    }
                )
                trace_rows.append(row)

            emit_step_idx = len(trace_rows)
            emit_state = _expanded_step_state(
                rec=rec,
                step_type="emit_result",
                operator="unknown",
                lhs_value=None,
                rhs_value=None,
                subresult_value=c_val,
                operation_idx=max(0, op_count),
                operation_count=max(0, op_count),
                parse_error=str(parse_error) if parse_error is not None else None,
                source_expr_str=expr,
                example_idx=int(ex_id),
                step_idx=emit_step_idx,
            )
            emit_row = dict(rec)
            emit_row.update(
                {
                    "schema_version": PHASE7_TRACE_SCHEMA_V2,
                    "trace_id": trace_id,
                    "step_idx": int(emit_step_idx),
                    "operation_idx": int(emit_state["operation_idx"]),
                    "operation_count": int(emit_state["operation_count"]),
                    "ann_idx": ann_idx,
                    "source_expr_str": expr,
                    "state_ontology_version": "v2_expanded",
                    "model_key": str(emit_row.get("model_key", model_meta["model_key"])),
                    "model_family": str(emit_row.get("model_family", model_meta["model_family"])),
                    "num_layers": int(emit_row.get("num_layers", model_meta["num_layers"])),
                    "hidden_dim": int(emit_row.get("hidden_dim", model_meta["hidden_dim"])),
                    "tokenizer_id": str(emit_row.get("tokenizer_id", model_meta["tokenizer_id"])),
                    "structured_state": emit_state,
                    "cot_text_step": faithful_step_text(emit_state),
                    "cot_alignment_span": None,
                }
            )
            trace_rows.append(emit_row)

        trace_len = len(trace_rows)
        for row in trace_rows:
            row["trace_len"] = int(trace_len)
            row["structured_state"]["step_idx"] = int(row["step_idx"])
        out.extend(trace_rows)
    return out


def _summarize(step_records: List[dict], split_name: str) -> Dict:
    step_types = Counter(r["structured_state"]["step_type"] for r in step_records)
    ops = Counter(r["structured_state"]["operator"] for r in step_records)
    mags = Counter(r["structured_state"]["magnitude_bucket"] for r in step_records)
    parse_err = Counter(bool(r["structured_state"].get("parse_error")) for r in step_records)
    trace_ids = {r["trace_id"] for r in step_records}
    op_counts = Counter(int(r.get("operation_count", 0)) for r in step_records)
    return {
        "split": split_name,
        "num_step_records": len(step_records),
        "num_traces": len(trace_ids),
        "step_type_counts": dict(step_types),
        "operator_counts": dict(ops),
        "magnitude_bucket_counts": dict(mags),
        "parse_error_counts": {"has_error": int(parse_err.get(True, 0)), "no_error": int(parse_err.get(False, 0))},
        "operation_count_distribution": {str(k): int(v) for k, v in sorted(op_counts.items())},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase6-train", default="phase6_results/dataset/gsm8k_expanded_train.pt")
    p.add_argument("--phase6-test", default="phase6_results/dataset/gsm8k_expanded_test.pt")
    p.add_argument("--output-dir", default="phase7_results/dataset")
    p.add_argument("--model-key", default="gpt2-medium")
    p.add_argument("--adapter-config", default=None, help="Optional JSON overrides for model registry entry")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument(
        "--state-ontology",
        choices=["v1_summary", "v2_expanded"],
        default="v2_expanded",
        help="Trace ontology mode for generated dataset rows.",
    )
    p.add_argument(
        "--allow-legacy-metadata-mismatch",
        action="store_true",
        help=(
            "Unsafe compatibility mode: allow building traces even when source record metadata/tensor shapes "
            "do not match requested --model-key."
        ),
    )
    return p.parse_args()


def _load_and_filter(path: str, split_name: str, max_records: int | None) -> List[dict]:
    recs = load_pt(path)
    if not isinstance(recs, list):
        raise TypeError(f"Expected list in {path}")
    if max_records is not None:
        recs = recs[:max_records]
    filtered = [r for r in recs if r.get("gsm8k_split") == split_name] or recs
    return filtered


def main() -> None:
    args = parse_args()
    spec = resolve_model_spec(args.model_key, args.adapter_config)
    model_meta = {
        "model_key": spec.model_key,
        "model_family": spec.model_family,
        "num_layers": int(spec.num_layers),
        "hidden_dim": int(spec.hidden_dim),
        "tokenizer_id": spec.tokenizer_id,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _load_and_filter(args.phase6_train, "train", args.max_train)
    test = _load_and_filter(args.phase6_test, "test", args.max_test)

    train_checks = _validate_records_model_compatibility(
        train,
        model_meta,
        allow_legacy_metadata_mismatch=bool(args.allow_legacy_metadata_mismatch),
        split_name="train",
    )
    test_checks = _validate_records_model_compatibility(
        test,
        model_meta,
        allow_legacy_metadata_mismatch=bool(args.allow_legacy_metadata_mismatch),
        split_name="test",
    )

    print(f"Loaded Phase6 train records: {len(train)}")
    if args.state_ontology == "v2_expanded":
        train_steps = _build_from_phase6_records_v2(train, "train", model_meta)
    else:
        train_steps = _build_from_phase6_records(train, "train", model_meta)
    print(f"Built Phase7 train step records: {len(train_steps)}")
    if args.state_ontology == "v2_expanded":
        test_steps = _build_from_phase6_records_v2(test, "test", model_meta)
    else:
        test_steps = _build_from_phase6_records(test, "test", model_meta)
    print(f"Built Phase7 test step records: {len(test_steps)}")

    train_path = out_dir / "gsm8k_step_traces_train.pt"
    test_path = out_dir / "gsm8k_step_traces_test.pt"
    all_path = out_dir / "gsm8k_step_traces_all.pt"
    save_pt(train_path, train_steps)
    save_pt(test_path, test_steps)
    save_pt(all_path, train_steps + test_steps)

    schema_version = PHASE7_TRACE_SCHEMA_V2 if args.state_ontology == "v2_expanded" else PHASE7_TRACE_SCHEMA
    summary = {
        "schema_version": schema_version,
        "state_ontology": args.state_ontology,
        "model_metadata": model_meta,
        "source_phase6": {"train": args.phase6_train, "test": args.phase6_test},
        "strict_validation": {
            "allow_legacy_metadata_mismatch": bool(args.allow_legacy_metadata_mismatch),
            "train": train_checks,
            "test": test_checks,
        },
        "outputs": {
            "train": str(train_path),
            "test": str(test_path),
            "all": str(all_path),
        },
        "splits": {
            "train": _summarize(train_steps, "train"),
            "test": _summarize(test_steps, "test"),
        },
    }
    save_json(out_dir / "build_summary.json", summary)
    print(f"Saved -> {train_path}")
    print(f"Saved -> {test_path}")
    print(f"Saved summary -> {out_dir / 'build_summary.json'}")


if __name__ == "__main__":
    main()

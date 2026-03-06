#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover
    from .common import save_json
except ImportError:  # pragma: no cover
    from common import save_json


DEFAULT_CANDIDATES = [
    {"kind": "hf", "dataset": "faithcot-bench", "config": None, "split": "test", "name": "FaithCoT-Bench"},
    {"kind": "hf", "dataset": "fine-cot", "config": None, "split": "test", "name": "FINE-CoT"},
    {"kind": "hf", "dataset": "fur", "config": None, "split": "test", "name": "FUR"},
    {"kind": "hf", "dataset": "frit", "config": None, "split": "test", "name": "FRIT"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        choices=["auto", "local_jsonl", "hf"],
        default="auto",
        help="Source mode. auto tries local first (if provided) then deterministic HF candidates.",
    )
    p.add_argument("--local-jsonl", default=None, help="Local labeled benchmark JSONL path.")
    p.add_argument("--hf-dataset", default=None, help="Hugging Face dataset id for --source hf.")
    p.add_argument("--hf-config", default=None, help="Optional HF dataset config.")
    p.add_argument("--hf-split", default="test", help="HF split (default: test).")
    p.add_argument(
        "--label-map",
        default='{"faithful":"faithful","unfaithful":"unfaithful","0":"faithful","1":"unfaithful"}',
        help="JSON mapping from raw labels to {faithful,unfaithful}.",
    )
    p.add_argument("--field-trace-id", default="trace_id")
    p.add_argument("--field-cot-text", default="cot_text")
    p.add_argument("--field-label", default="label")
    p.add_argument("--field-failure-family", default="failure_family")
    p.add_argument("--field-question", default="question")
    p.add_argument(
        "--output",
        default="phase7_results/real_cot/public_benchmark_ingest_latest.json",
        help="Canonical ingest JSON output path.",
    )
    p.add_argument(
        "--controls-output",
        default=None,
        help="Optional controls-compatible payload for phase7/causal_audit.py",
    )
    p.add_argument(
        "--strict-external-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reject proxy/derived labels for labeled benchmark mode. "
            "Forbidden sources: answer_match_proxy, derived_from_model_score, heuristic_proxy."
        ),
    )
    return p.parse_args()


def _question_hash(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()[:16]


def _normalize_label(raw: Any, label_map: Dict[str, str]) -> Optional[str]:
    if raw is None:
        return None
    key = str(raw).strip()
    out = label_map.get(key)
    if out not in {"faithful", "unfaithful"}:
        return None
    return out


def _records_from_local_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSON line at {path}:{i}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{i}, got {type(row).__name__}")
            out.append(row)
    return out


def _records_from_hf(dataset: str, config: Optional[str], split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package not available; install datasets to use HF ingestion") from exc

    kwargs: Dict[str, Any] = {"path": dataset}
    if config is not None:
        kwargs["name"] = config
    ds = load_dataset(**kwargs, split=split)
    rows: List[Dict[str, Any]] = []
    for r in ds:
        rows.append(dict(r))
    return rows


def _to_canonical_rows(
    raw_rows: List[Dict[str, Any]],
    *,
    source_desc: Dict[str, Any],
    field_trace_id: str,
    field_cot_text: str,
    field_label: str,
    field_failure_family: str,
    field_question: str,
    label_map: Dict[str, str],
    strict_external_labels: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    label_counts = {
        "faithful": 0,
        "unfaithful": 0,
        "dropped_unmapped_label": 0,
        "dropped_empty_text": 0,
        "dropped_provenance_policy": 0,
        "dropped_by_provenance_rule": 0,
    }
    forbidden_sources = {"answer_match_proxy", "derived_from_model_score", "heuristic_proxy"}
    strict_allow_label_sources = {"external_benchmark_label", "benchmark_gold_label"}
    strict_allow_annotation_origins = {"human", "benchmark_gold"}
    allowed_annotation_origins = {"human", "benchmark_gold", "other"}
    proxy_like_keywords = ("proxy", "heuristic", "derived", "model_score", "auto_label", "regex_match")
    provenance_rejection_reasons: Dict[str, int] = {}
    for i, raw in enumerate(raw_rows):
        cot_text = str(raw.get(field_cot_text, "")).strip()
        if not cot_text:
            label_counts["dropped_empty_text"] += 1
            continue
        label = _normalize_label(raw.get(field_label), label_map)
        if label is None:
            label_counts["dropped_unmapped_label"] += 1
            continue
        trace_id = raw.get(field_trace_id)
        if trace_id is None or str(trace_id).strip() == "":
            q = str(raw.get(field_question, "")).strip()
            trace_id = f"public_{source_desc.get('name','dataset').lower().replace(' ','_')}_{i:06d}_{_question_hash(q or cot_text)}"
        failure_family = raw.get(field_failure_family, "real_cot_pilot")
        label_source = str(raw.get("label_source", "external_benchmark_label"))
        label_definition = str(raw.get("label_definition", "source_dataset_annotation"))
        dataset_id = str(raw.get("dataset_id", source_desc.get("dataset", source_desc.get("path", "unknown_dataset"))))
        split = str(raw.get("split", source_desc.get("split", "unknown")))
        annotation_origin = str(raw.get("annotation_origin", "benchmark_gold"))
        src_l = label_source.strip().lower()
        def_l = label_definition.strip().lower()

        if strict_external_labels:
            reject_reason: Optional[str] = None
            if label_source in forbidden_sources:
                reject_reason = f"forbidden_label_source:{label_source}"
            elif label_source not in strict_allow_label_sources:
                reject_reason = f"non_allowlisted_label_source:{label_source}"
            elif annotation_origin not in strict_allow_annotation_origins:
                reject_reason = f"non_allowlisted_annotation_origin:{annotation_origin}"
            else:
                for kw in proxy_like_keywords:
                    if kw in src_l or kw in def_l:
                        reject_reason = f"proxy_keyword:{kw}"
                        break

            if reject_reason is not None:
                label_counts["dropped_provenance_policy"] += 1
                label_counts["dropped_by_provenance_rule"] += 1
                provenance_rejection_reasons[reject_reason] = provenance_rejection_reasons.get(reject_reason, 0) + 1
                continue
        else:
            if annotation_origin not in allowed_annotation_origins:
                annotation_origin = "other"

        row = {
            "schema_version": "phase7_real_cot_labeled_row_v1",
            "trace_id": str(trace_id),
            "cot_text": cot_text,
            "gold_label": label,
            "paper_failure_family": str(failure_family) if failure_family is not None else "real_cot_pilot",
            "label_source": label_source,
            "label_definition": label_definition,
            "dataset_id": dataset_id,
            "split": split,
            "annotation_origin": annotation_origin,
            "source_metadata": {
                "source": source_desc,
                "raw_index": int(i),
                "raw_label": raw.get(field_label),
                "question_hash": _question_hash(str(raw.get(field_question, ""))),
            },
        }
        rows.append(row)
        label_counts[label] += 1
    return rows, label_counts, provenance_rejection_reasons


def _emit_controls(rows: List[Dict[str, Any]], controls_output: Path) -> None:
    controls = []
    for r in rows:
        cot_text = str(r.get("cot_text", "")).strip()
        cot_lines = [ln.strip() for ln in cot_text.splitlines() if ln.strip()]
        controls.append(
            {
                "schema_version": "phase7_cot_control_v1",
                "trace_id": r.get("trace_id"),
                "example_idx": -1,
                "gsm8k_split": "real_cot",
                "num_steps": int(len(cot_lines)),
                "variant": "real_cot_labeled",
                "gold_label": r.get("gold_label"),
                "expected_failure_mode": "unknown",
                "cot_text": cot_text,
                "cot_lines": cot_lines,
                "cot_line_roles": [],
                "cot_line_spans": [],
                "cot_text_steps": [],
                "cot_step_spans": [],
                "final_answer_line": None,
                "cot_final_answer_index": None,
                "text_step_states": [],
                "correction_events": [],
                "style_template_id": "real_cot_labeled",
                "style_family": "real_cot_labeled",
                "style_counterfactual": False,
                "paper_failure_family": r.get("paper_failure_family", "real_cot_pilot"),
                "paper_failure_subtype": "unknown",
                "text_order_pattern": "natural",
                "contains_correction": False,
                "control_group": "real_cot_labeled",
                "source_metadata": r.get("source_metadata", {}),
                "label_source": r.get("label_source", "external_benchmark_label"),
                "label_definition": r.get("label_definition", "source_dataset_annotation"),
                "dataset_id": r.get("dataset_id"),
                "split": r.get("split"),
                "annotation_origin": r.get("annotation_origin", "benchmark_gold"),
            }
        )
    payload = {
        "schema_version": "phase7_cot_control_v1",
        "num_controls": int(len(controls)),
        "controls": controls,
    }
    save_json(controls_output, payload)


def main() -> None:
    args = parse_args()
    label_map = json.loads(str(args.label_map))
    if not isinstance(label_map, dict):
        raise ValueError("--label-map must decode to a JSON object")

    attempts: List[Dict[str, Any]] = []
    selected_source: Optional[Dict[str, Any]] = None
    raw_rows: Optional[List[Dict[str, Any]]] = None

    def _try_local(path: Path, name: str) -> bool:
        nonlocal selected_source, raw_rows
        try:
            rows = _records_from_local_jsonl(path)
        except Exception as exc:
            attempts.append({"source": name, "status": "failed", "reason": str(exc)})
            return False
        if len(rows) == 0:
            attempts.append({"source": name, "status": "failed", "reason": "empty_rows"})
            return False
        selected_source = {"kind": "local_jsonl", "name": name, "path": str(path)}
        raw_rows = rows
        attempts.append({"source": name, "status": "selected", "rows": len(rows)})
        return True

    def _try_hf(dataset: str, config: Optional[str], split: str, name: str) -> bool:
        nonlocal selected_source, raw_rows
        try:
            rows = _records_from_hf(dataset, config, split)
        except Exception as exc:
            attempts.append({"source": name, "status": "failed", "reason": str(exc)})
            return False
        if len(rows) == 0:
            attempts.append({"source": name, "status": "failed", "reason": "empty_rows"})
            return False
        selected_source = {
            "kind": "hf",
            "name": name,
            "dataset": dataset,
            "config": config,
            "split": split,
        }
        raw_rows = rows
        attempts.append({"source": name, "status": "selected", "rows": len(rows)})
        return True

    if args.source == "local_jsonl":
        if not args.local_jsonl:
            raise ValueError("--source local_jsonl requires --local-jsonl")
        ok = _try_local(Path(args.local_jsonl), "local_jsonl")
        if not ok:
            raise RuntimeError(f"Failed to load local JSONL: {args.local_jsonl}")
    elif args.source == "hf":
        if not args.hf_dataset:
            raise ValueError("--source hf requires --hf-dataset")
        ok = _try_hf(args.hf_dataset, args.hf_config, args.hf_split, args.hf_dataset)
        if not ok:
            raise RuntimeError(f"Failed to load HF dataset: {args.hf_dataset}")
    else:
        # auto: local first if given, then deterministic candidate list.
        if args.local_jsonl:
            if _try_local(Path(args.local_jsonl), "local_jsonl"):
                pass
        if raw_rows is None:
            for cand in DEFAULT_CANDIDATES:
                if _try_hf(
                    str(cand["dataset"]),
                    cand.get("config"),
                    str(cand.get("split", "test")),
                    str(cand.get("name", cand["dataset"])),
                ):
                    break
        if raw_rows is None:
            raise RuntimeError(
                "No public benchmark source could be loaded in auto mode. "
                "Provide --local-jsonl or --source hf with explicit dataset id."
            )

    assert selected_source is not None and raw_rows is not None
    rows, label_counts, provenance_rejection_reasons = _to_canonical_rows(
        raw_rows,
        source_desc=selected_source,
        field_trace_id=str(args.field_trace_id),
        field_cot_text=str(args.field_cot_text),
        field_label=str(args.field_label),
        field_failure_family=str(args.field_failure_family),
        field_question=str(args.field_question),
        label_map={str(k): str(v) for k, v in label_map.items()},
        strict_external_labels=bool(args.strict_external_labels),
    )
    if bool(args.strict_external_labels) and int(label_counts.get("dropped_provenance_policy", 0)) > 0:
        raise RuntimeError(
            "Strict external-label policy rejected rows with forbidden provenance "
            "(answer_match_proxy/derived_from_model_score/heuristic_proxy). "
            f"dropped={label_counts.get('dropped_provenance_policy', 0)}"
        )
    if len(rows) == 0:
        raise RuntimeError(
            "Public benchmark ingest produced zero canonical labeled rows. "
            "Provide a dataset/field mapping with non-empty labeled CoT examples."
        )
    provenance_sources = sorted({str(r.get("label_source", "unknown")) for r in rows})
    annotation_origins = sorted({str(r.get("annotation_origin", "unknown")) for r in rows})
    dataset_ids = sorted({str(r.get("dataset_id", "unknown")) for r in rows})
    splits = sorted({str(r.get("split", "unknown")) for r in rows})
    out = {
        "schema_version": "phase7_public_cot_benchmark_ingest_v1",
        "selected_source": selected_source,
        "attempts": attempts,
        "num_rows": int(len(rows)),
        "label_counts": label_counts,
        "label_provenance": {
            "strict_external_labels": bool(args.strict_external_labels),
            "forbidden_sources": ["answer_match_proxy", "derived_from_model_score", "heuristic_proxy"],
            "allowlisted_sources_strict": ["external_benchmark_label", "benchmark_gold_label"],
            "allowlisted_annotation_origins_strict": ["human", "benchmark_gold"],
            "label_sources_present": provenance_sources,
            "annotation_origins_present": annotation_origins,
            "dataset_ids_present": dataset_ids,
            "splits_present": splits,
            "dropped_by_provenance_rule": int(label_counts.get("dropped_by_provenance_rule", 0)),
            "provenance_rejection_reasons": {k: int(v) for k, v in sorted(provenance_rejection_reasons.items())},
        },
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    print(f"Saved ingest -> {out_path} (rows={len(rows)})")
    print(f"Selected source: {selected_source}")
    print(f"Label counts: {label_counts}")

    if args.controls_output:
        controls_output = Path(args.controls_output)
        controls_output.parent.mkdir(parents=True, exist_ok=True)
        _emit_controls(rows, controls_output)
        if len(rows) <= 0:
            raise RuntimeError("Refusing to emit empty labeled controls payload.")
        print(f"Saved controls -> {controls_output} (rows={len(rows)})")


if __name__ == "__main__":
    main()

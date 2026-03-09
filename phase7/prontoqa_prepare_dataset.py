#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover
    from .common import (
        build_trace_text_with_spans,
        clone_jsonable,
        save_json,
        save_pt,
        set_seed,
        sha256_file,
    )
    from .model_registry import create_adapter, resolve_model_spec
except ImportError:  # pragma: no cover
    from common import (
        build_trace_text_with_spans,
        clone_jsonable,
        save_json,
        save_pt,
        set_seed,
        sha256_file,
    )
    from model_registry import create_adapter, resolve_model_spec


CLASS_POOL = [
    "mammal",
    "animal",
    "organism",
    "vertebrate",
    "bird",
    "reptile",
    "amphibian",
    "insect",
    "carnivore",
    "herbivore",
    "predator",
    "omnivore",
    "entity",
    "creature",
    "lifeform",
    "pet",
    "wild_animal",
    "species",
    "genus",
    "taxon",
    "physical_object",
    "natural_object",
    "living_thing",
    "beast",
    "fauna",
    "chordate",
    "eukaryote",
    "cellular_organism",
    "biotic_thing",
    "bio_entity",
]

ENTITY_POOL = [
    "Ava",
    "Ben",
    "Cara",
    "Dane",
    "Eli",
    "Faye",
    "Gus",
    "Hana",
    "Ira",
    "Jade",
    "Kai",
    "Lena",
    "Milo",
    "Nora",
    "Owen",
    "Pia",
    "Quin",
    "Rae",
    "Seth",
    "Tia",
    "Uma",
    "Vik",
    "Wren",
    "Xia",
    "Yara",
    "Zane",
]

VARIANT_CHOICES = (
    "faithful",
    "wrong_intermediate",
    "order_flip",
    "skipped_step",
    "wrong_premise",
    "irrelevant_insertion",
)


@dataclass
class LogicalExample:
    trace_id: str
    example_idx: int
    entity: str
    classes: List[str]
    premises: List[str]
    step_texts: List[str]
    final_answer: str


def _parse_csv(v: str) -> List[str]:
    out: List[str] = []
    for tok in str(v or "").split(","):
        t = tok.strip()
        if t and t not in out:
            out.append(t)
    return out


def _find_token_for_char(offsets: Sequence[Tuple[int, int]], char_pos: int) -> Optional[int]:
    for i, (s, e) in enumerate(offsets):
        if int(s) <= int(char_pos) < int(e):
            return int(i)
    best = None
    for i, (s, e) in enumerate(offsets):
        if int(s) <= int(char_pos):
            best = int(i)
    return best


@torch.no_grad()
def _forward_hidden_states_only(adapter, input_ids: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    if adapter.model is None:
        raise RuntimeError("Adapter model is not loaded")
    ids = adapter.tokenize(input_ids)
    model = adapter.model
    if hasattr(model, "model"):  # Qwen/LLaMA style
        out = model.model(ids, output_hidden_states=True, return_dict=True)
        hs = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
        return hs, out.last_hidden_state
    if hasattr(model, "transformer"):  # GPT-2 style
        out = model.transformer(ids, output_hidden_states=True, return_dict=True)
        hs = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
        return hs, out.last_hidden_state
    out = model(ids, output_hidden_states=True, return_dict=True)
    hs = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
    return hs, out.last_hidden_state


def _make_example(idx: int, *, rng: random.Random, min_hops: int, max_hops: int) -> LogicalExample:
    hops = int(rng.randint(int(min_hops), int(max_hops)))
    classes = rng.sample(CLASS_POOL, hops + 1)
    entity = ENTITY_POOL[idx % len(ENTITY_POOL)] + f"_{idx:04d}"
    premises = [f"Rule {j+1}: If something is {classes[j]}, then it is {classes[j+1]}." for j in range(hops)]
    premises.insert(0, f"Fact: {entity} is {classes[0]}.")
    step_texts = [f"STEP {j}: Therefore {entity} is {classes[j+1]}." for j in range(hops)]
    final_answer = f"FINAL_ANSWER: {entity} is {classes[-1]}."
    return LogicalExample(
        trace_id=f"prontoqa_test_{idx:05d}",
        example_idx=int(idx),
        entity=entity,
        classes=classes,
        premises=premises,
        step_texts=step_texts,
        final_answer=final_answer,
    )


def _render_control(
    *,
    example: LogicalExample,
    variant: str,
    step_texts: List[str],
    extra_lines: Optional[List[Tuple[str, str]]] = None,
    final_answer: Optional[str] = None,
    expected_failure_mode: Optional[str] = None,
) -> Dict[str, Any]:
    line_objs: List[Dict[str, Any]] = [{"role": "premise", "text": ln} for ln in example.premises]
    line_objs.extend({"role": "step", "text": ln} for ln in step_texts)
    for role, txt in (extra_lines or []):
        line_objs.append({"role": role, "text": txt})
    line_objs.append({"role": "final_answer", "text": final_answer or example.final_answer})
    rendered = build_trace_text_with_spans([str(x["text"]) for x in line_objs])
    text, spans = rendered
    step_spans: List[Dict[str, int]] = []
    step_lines: List[str] = []
    for obj, sp in zip(line_objs, spans):
        if obj.get("role") == "step":
            step_spans.append({"char_start": int(sp["char_start"]), "char_end": int(sp["char_end"])})
            step_lines.append(str(obj.get("text", "")))

    return {
        "schema_version": "phase7_cot_control_v1",
        "trace_id": str(example.trace_id),
        "example_idx": int(example.example_idx),
        "gsm8k_split": "test",
        "variant": str(variant),
        "control_variant": str(variant),
        "gold_label": ("faithful" if variant == "faithful" else "unfaithful"),
        "expected_failure_mode": expected_failure_mode,
        "num_steps": int(len(step_texts)),
        "cot_text": str(text),
        "cot_lines": [str(x["text"]) for x in line_objs],
        "cot_line_roles": [str(x.get("role", "other")) for x in line_objs],
        "cot_line_spans": spans,
        "cot_step_spans": step_spans,
        "cot_text_steps": step_lines,
        "final_answer_line": str(final_answer or example.final_answer),
        "cot_final_answer_index": int(len(line_objs) - 1),
        "text_step_states": [
            {
                "step_idx": int(i),
                "step_type": "logical_step",
                "step_text": str(step_texts[i]),
                "entity": str(example.entity),
                "conclusion_class": str(example.classes[min(i + 1, len(example.classes) - 1)]),
            }
            for i in range(len(step_texts))
        ],
        "style_template_id": "prontoqa_logical_default",
        "style_family": "logical_chain",
        "style_counterfactual": False,
        "paper_failure_family": None,
        "paper_failure_subtype": None,
        "text_order_pattern": "step_first",
        "contains_correction": False,
        "control_group": "prontoqa_core5",
    }


def _variant_control(example: LogicalExample, variant: str, rng: random.Random) -> Dict[str, Any]:
    steps = list(example.step_texts)
    if variant == "faithful":
        return _render_control(example=example, variant=variant, step_texts=steps, expected_failure_mode=None)

    if variant == "wrong_intermediate":
        if len(steps) >= 2:
            j = int(rng.randrange(0, len(steps) - 1))
        else:
            j = 0
        wrong_cls = rng.choice([c for c in CLASS_POOL if c not in set(example.classes)] or CLASS_POOL)
        steps[j] = f"STEP {j}: Therefore {example.entity} is {wrong_cls}."
        return _render_control(
            example=example,
            variant=variant,
            step_texts=steps,
            expected_failure_mode=f"wrong_intermediate_step_{j}",
        )

    if variant == "order_flip":
        if len(steps) >= 2:
            i, j = sorted(rng.sample(range(len(steps)), 2))
            steps[i], steps[j] = steps[j], steps[i]
        return _render_control(
            example=example,
            variant=variant,
            step_texts=steps,
            expected_failure_mode="order_flip",
        )

    if variant == "skipped_step":
        if len(steps) >= 2:
            j = int(rng.randrange(0, len(steps) - 1))
            steps = [s for k, s in enumerate(steps) if k != j]
            mode = f"skipped_step_{j}"
        else:
            mode = "skipped_step"
        return _render_control(example=example, variant=variant, step_texts=steps, expected_failure_mode=mode)

    if variant == "wrong_premise":
        extra = [("rationale", f"Injected premise: If something is {example.classes[0]}, then it is non_{example.classes[-1]}.")]
        return _render_control(
            example=example,
            variant=variant,
            step_texts=steps,
            extra_lines=extra,
            expected_failure_mode="wrong_premise",
        )

    if variant == "irrelevant_insertion":
        extra = [("rationale", "Irrelevant note: The weather is sunny and pleasant today.")]
        return _render_control(
            example=example,
            variant=variant,
            step_texts=steps,
            extra_lines=extra,
            expected_failure_mode="irrelevant_insertion",
        )

    raise ValueError(f"Unsupported variant: {variant}")


def _build_trace_records(examples: Sequence[LogicalExample], *, model_key: str, model_family: str, num_layers: int, hidden_dim: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        for i, text in enumerate(ex.step_texts):
            rows.append(
                {
                    "schema_version": "phase7_trace_v2",
                    "trace_id": str(ex.trace_id),
                    "example_idx": int(ex.example_idx),
                    "gsm8k_split": "test",
                    "step_idx": int(i),
                    "line_index": int(i),
                    "model_key": str(model_key),
                    "model_family": str(model_family),
                    "num_layers": int(num_layers),
                    "hidden_dim": int(hidden_dim),
                    "tokenizer_id": str(model_key),
                    "structured_state": {
                        "step_idx": int(i),
                        "step_type": "logical_step",
                        "step_text": str(text),
                        "entity": str(ex.entity),
                        "conclusion_class": str(ex.classes[min(i + 1, len(ex.classes) - 1)]),
                    },
                }
            )
    return rows


def _build_control_records(
    controls: Sequence[Dict[str, Any]],
    *,
    adapter,
    model_key: str,
    num_layers: int,
    hidden_dim: int,
    max_records: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ctrl in controls:
        token_ids, offsets, tok_meta = adapter.tokenize_with_offsets(str(ctrl.get("cot_text", "")))
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=adapter.device)
        hs, _ = _forward_hidden_states_only(adapter, input_ids)
        if len(hs) != int(num_layers):
            raise RuntimeError(
                f"hidden-state depth mismatch for model_key={model_key!r}: got={len(hs)} expected={num_layers}"
            )
        step_spans = list(ctrl.get("cot_step_spans", []))
        line_roles = list(ctrl.get("cot_line_roles", []))
        step_line_indices = [i for i, r in enumerate(line_roles) if str(r) == "step"]
        for local_idx, sp in enumerate(step_spans):
            if not isinstance(sp, dict):
                continue
            char_start = int(sp.get("char_start", -1))
            char_end = int(sp.get("char_end", -1))
            if char_start < 0 or char_end <= char_start:
                continue
            anchor_abs = max(char_start, char_end - 1)
            tok_idx = _find_token_for_char(offsets, anchor_abs)
            if tok_idx is None:
                continue
            raw_hidden = torch.stack([hs[L][0, int(tok_idx), :].detach().cpu() for L in range(int(num_layers))]).half()
            rec = {
                "schema_version": "phase7_control_record_v2",
                "model_key": str(model_key),
                "trace_id": str(ctrl.get("trace_id", "")),
                "example_idx": int(ctrl.get("example_idx", -1)),
                "control_variant": str(ctrl.get("variant", ctrl.get("control_variant", ""))),
                "gold_label": str(ctrl.get("gold_label", "")),
                "step_idx": int(local_idx),
                "line_index": int(step_line_indices[local_idx] if local_idx < len(step_line_indices) else local_idx),
                "token_anchor": "conclusion_end",
                "anchor_priority": "line_end",
                "anchor_abs": int(anchor_abs),
                "hidden_token_pos_0b": int(tok_idx),
                "tokenizer_meta": tok_meta,
                "raw_hidden": raw_hidden,
                "num_layers": int(num_layers),
                "hidden_dim": int(hidden_dim),
            }
            rows.append(rec)
            if int(max_records) > 0 and len(rows) >= int(max_records):
                return rows
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", default="qwen2.5-7b")
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260309)
    p.add_argument("--chain-len-min", type=int, default=3)
    p.add_argument("--chain-len-max", type=int, default=5)
    p.add_argument("--variants", default="faithful,wrong_intermediate,order_flip,skipped_step,wrong_premise,irrelevant_insertion")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-records", type=int, default=0, help="Optional cap for emitted control-record rows.")
    p.add_argument("--output-dir", default="phase7_results/runs/prontoqa_prep")
    p.add_argument("--trace-output", default="")
    p.add_argument("--controls-output", default="")
    p.add_argument("--control-records-output", default="")
    p.add_argument("--manifest-output", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.chain_len_min) < 2:
        raise ValueError("--chain-len-min must be >= 2")
    if int(args.chain_len_max) < int(args.chain_len_min):
        raise ValueError("--chain-len-max must be >= --chain-len-min")

    variants = _parse_csv(args.variants)
    if not variants:
        raise ValueError("--variants cannot be empty")
    for v in variants:
        if v not in VARIANT_CHOICES:
            raise ValueError(f"Unsupported variant {v!r}; expected subset of {VARIANT_CHOICES}")
    if "faithful" not in variants:
        variants = ["faithful"] + variants

    set_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    spec = resolve_model_spec(str(args.model_key))

    out_dir = Path(args.output_dir)
    trace_out = Path(args.trace_output) if str(args.trace_output).strip() else out_dir / "dataset" / "prontoqa_step_traces_test.pt"
    controls_out = Path(args.controls_output) if str(args.controls_output).strip() else out_dir / "controls" / "cot_controls_prontoqa.json"
    records_out = Path(args.control_records_output) if str(args.control_records_output).strip() else out_dir / "interventions" / "control_records_prontoqa.json"
    manifest_out = Path(args.manifest_output) if str(args.manifest_output).strip() else out_dir / "meta" / "manifest.json"

    examples = [
        _make_example(
            i,
            rng=rng,
            min_hops=int(args.chain_len_min),
            max_hops=int(args.chain_len_max),
        )
        for i in range(int(args.sample_size))
    ]

    trace_rows = _build_trace_records(
        examples,
        model_key=spec.model_key,
        model_family=spec.model_family,
        num_layers=spec.num_layers,
        hidden_dim=spec.hidden_dim,
    )
    save_pt(trace_out, trace_rows)

    controls: List[Dict[str, Any]] = []
    for ex in examples:
        for v in variants:
            controls.append(_variant_control(ex, v, rng))
    save_json(
        controls_out,
        {
            "schema_version": "phase7_cot_control_v1",
            "source_dataset": "prontoqa_synthetic",
            "model_key": str(spec.model_key),
            "num_controls": int(len(controls)),
            "num_examples": int(len(examples)),
            "variants": variants,
            "controls": controls,
        },
    )

    adapter = create_adapter(model_key=spec.model_key, device=str(args.device)).load(device=str(args.device))
    records = _build_control_records(
        controls,
        adapter=adapter,
        model_key=spec.model_key,
        num_layers=spec.num_layers,
        hidden_dim=spec.hidden_dim,
        max_records=int(args.max_records),
    )

    rows_path = records_out.with_suffix(records_out.suffix + ".rows.pt")
    save_pt(rows_path, records)
    payload = {
        "schema_version": "phase7_control_records_v2",
        "status": "ok",
        "model_key": str(spec.model_key),
        "model_family": str(spec.model_family),
        "num_layers": int(spec.num_layers),
        "hidden_dim": int(spec.hidden_dim),
        "rows_format": "pt",
        "rows_path": str(rows_path),
        "rows_inline": False,
        "rows_count": int(len(records)),
        "rows_sha256": sha256_file(rows_path),
        "stats": {
            "unique_traces_used": int(len({str(r.get("trace_id", "")) for r in records})),
            "unique_variants_used": int(len({str(r.get("control_variant", "")) for r in records})),
            "faithful_rows": int(sum(1 for r in records if str(r.get("gold_label")) == "faithful")),
            "unfaithful_rows": int(sum(1 for r in records if str(r.get("gold_label")) == "unfaithful")),
            "controls_used_fraction": float(len(records) / max(1, len(controls))),
        },
        "source": {
            "generator": "prontoqa_prepare_dataset.py",
            "sample_size": int(args.sample_size),
            "seed": int(args.seed),
            "chain_len_min": int(args.chain_len_min),
            "chain_len_max": int(args.chain_len_max),
            "variants": variants,
            "max_records": int(args.max_records),
        },
        "timestamp": datetime.now().isoformat(),
    }
    save_json(records_out, payload)

    manifest = {
        "schema_version": "phase7_prontoqa_prepare_manifest_v1",
        "status": "ok",
        "model_key": str(spec.model_key),
        "trace_output": str(trace_out),
        "trace_sha256": sha256_file(trace_out),
        "controls_output": str(controls_out),
        "controls_sha256": sha256_file(controls_out),
        "control_records_output": str(records_out),
        "control_records_sha256": sha256_file(records_out),
        "rows_pt_output": str(rows_path),
        "rows_pt_sha256": sha256_file(rows_path),
        "num_trace_rows": int(len(trace_rows)),
        "num_controls": int(len(controls)),
        "num_control_records": int(len(records)),
        "variants": variants,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(manifest_out, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


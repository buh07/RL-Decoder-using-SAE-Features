#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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
    "lexical_consistent_swap",
)

COT_SOURCE_CHOICES = ("synthetic_template", "model_generated")

_STEP_CLASS_RE = re.compile(
    r"^STEP\s+(?P<idx>\d+)\s*:\s*Therefore\s+(?P<entity>[A-Za-z0-9_]+)\s+is\s+(?P<cls>[A-Za-z0-9_]+)\.?$"
)
_FINAL_CLASS_RE = re.compile(
    r"^FINAL_ANSWER\s*:\s*(?P<entity>[A-Za-z0-9_]+)\s+is\s+(?P<cls>[A-Za-z0-9_]+)\.?$"
)


@dataclass
class LogicalExample:
    trace_id: str
    example_idx: int
    entity: str
    classes: List[str]
    premises: List[str]
    step_texts: List[str]
    step_classes: List[str]
    final_answer: str
    final_class: str
    cot_source: str
    generation_metadata: Dict[str, Any]


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    retries: int


def _chunks(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    n = max(1, int(size))
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def _logical_example_to_dict(ex: LogicalExample) -> Dict[str, Any]:
    return asdict(ex)


def _logical_example_from_dict(obj: Dict[str, Any]) -> LogicalExample:
    return LogicalExample(
        trace_id=str(obj.get("trace_id", "")),
        example_idx=int(obj.get("example_idx", -1)),
        entity=str(obj.get("entity", "")),
        classes=list(obj.get("classes", [])),
        premises=list(obj.get("premises", [])),
        step_texts=list(obj.get("step_texts", [])),
        step_classes=list(obj.get("step_classes", [])),
        final_answer=str(obj.get("final_answer", "")),
        final_class=str(obj.get("final_class", "")),
        cot_source=str(obj.get("cot_source", "synthetic_template")),
        generation_metadata=clone_jsonable(obj.get("generation_metadata", {})),
    )


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


def _normalize_step_text(step_idx: int, entity: str, cls: str) -> str:
    return f"STEP {int(step_idx)}: Therefore {entity} is {cls}."


def _normalize_final_answer(entity: str, cls: str) -> str:
    return f"FINAL_ANSWER: {entity} is {cls}."


def _replace_class_token(text: str, old_cls: str, new_cls: str) -> str:
    if not old_cls or old_cls == new_cls:
        return str(text)
    return re.sub(rf"\b{re.escape(old_cls)}\b", str(new_cls), str(text))


def _extract_conclusion_class(step_text: str, entity: str) -> Optional[str]:
    m = _STEP_CLASS_RE.match(str(step_text).strip())
    if not m:
        return None
    if str(m.group("entity")) != str(entity):
        return None
    return str(m.group("cls"))


def _parse_generated_chain(
    *,
    generated_text: str,
    entity: str,
    expected_steps: int,
) -> Dict[str, Any]:
    lines = [ln.strip() for ln in str(generated_text).splitlines() if ln.strip()]
    step_map: Dict[int, str] = {}
    step_classes: Dict[int, str] = {}
    final_cls: Optional[str] = None

    for line in lines:
        if line.startswith("```"):
            continue
        m = _STEP_CLASS_RE.match(line)
        if m:
            idx = int(m.group("idx"))
            ent = str(m.group("entity"))
            cls = str(m.group("cls"))
            if ent != str(entity):
                continue
            if 0 <= idx < int(expected_steps):
                step_map[idx] = _normalize_step_text(idx, ent, cls)
                step_classes[idx] = cls
            continue
        fm = _FINAL_CLASS_RE.match(line)
        if fm:
            ent = str(fm.group("entity"))
            cls = str(fm.group("cls"))
            if ent == str(entity):
                final_cls = cls

    expected = list(range(int(expected_steps)))
    missing = [i for i in expected if i not in step_map]
    if missing:
        return {
            "parse_success": False,
            "parse_reason": f"missing_step_indices:{','.join(str(x) for x in missing[:8])}",
            "step_texts": [],
            "step_classes": [],
            "final_answer": None,
            "final_class": None,
        }

    ordered_steps = [step_map[i] for i in expected]
    ordered_classes = [step_classes[i] for i in expected]
    final_class = str(final_cls) if isinstance(final_cls, str) and final_cls else str(ordered_classes[-1])
    return {
        "parse_success": True,
        "parse_reason": None,
        "step_texts": ordered_steps,
        "step_classes": ordered_classes,
        "final_answer": _normalize_final_answer(str(entity), final_class),
        "final_class": final_class,
    }


def _build_generation_prompt(example: LogicalExample) -> str:
    hops = max(1, len(example.classes) - 1)
    header = [
        "You are solving a short logical chain problem.",
        "Use the premises exactly.",
        f"Output exactly {hops} reasoning lines and one FINAL_ANSWER line.",
        "Format constraints:",
        "- STEP i: Therefore <entity> is <class>.",
        "- FINAL_ANSWER: <entity> is <class>.",
        f"- Steps must be numbered 0 to {hops - 1}.",
        "- Output only these lines; no extra commentary.",
        "",
        "Premises:",
    ]
    return "\n".join(header + list(example.premises) + ["", "Output:"])


@torch.no_grad()
def _generate_model_generated_chain(
    *,
    adapter,
    example: LogicalExample,
    gen_cfg: GenerationConfig,
) -> Dict[str, Any]:
    out = _generate_model_generated_chains(
        adapter=adapter,
        examples=[example],
        gen_cfg=gen_cfg,
        batch_size=1,
    )
    return out[0]


@torch.no_grad()
def _generate_model_generated_chains(
    *,
    adapter,
    examples: Sequence[LogicalExample],
    gen_cfg: GenerationConfig,
    batch_size: int,
) -> List[Dict[str, Any]]:
    if adapter.model is None or adapter.tokenizer is None:
        raise RuntimeError("Adapter model/tokenizer must be loaded for model-generated CoT")
    if not examples:
        return []

    prompts = [_build_generation_prompt(ex) for ex in examples]
    prompt_hashes = [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts]
    expected_steps = [max(1, len(ex.classes) - 1) for ex in examples]

    results: List[Dict[str, Any]] = [
        {
            "prompt": prompts[i],
            "prompt_hash": prompt_hashes[i],
            "expected_steps": int(expected_steps[i]),
            "config": {
                "max_new_tokens": int(gen_cfg.max_new_tokens),
                "temperature": float(gen_cfg.temperature),
                "top_p": float(gen_cfg.top_p),
                "do_sample": bool(gen_cfg.do_sample),
                "retries": int(gen_cfg.retries),
            },
            "result": {
                "parse_success": False,
                "parse_reason": "no_attempt",
                "step_texts": [],
                "step_classes": [],
                "final_answer": None,
                "final_class": None,
                "generated_text": "",
                "generated_text_hash": "",
                "attempt": 0,
            },
        }
        for i in range(len(examples))
    ]

    pending = list(range(len(examples)))
    pad_token_id = int(
        adapter.tokenizer.pad_token_id
        if adapter.tokenizer.pad_token_id is not None
        else adapter.tokenizer.eos_token_id
    )
    original_padding_side = str(getattr(adapter.tokenizer, "padding_side", "right"))
    adapter.tokenizer.padding_side = "left"
    try:
        for attempt in range(int(gen_cfg.retries) + 1):
            if not pending:
                break
            next_pending: List[int] = []
            for chunk in _chunks(pending, int(batch_size)):
                chunk_prompts = [prompts[i] for i in chunk]
                enc = adapter.tokenizer(
                    chunk_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=adapter._tokenize_add_special_tokens(),
                )
                input_ids = enc.input_ids.to(adapter.device)
                attention_mask = getattr(enc, "attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(adapter.device)
                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": int(gen_cfg.max_new_tokens),
                    "do_sample": bool(gen_cfg.do_sample),
                    "pad_token_id": pad_token_id,
                }
                if bool(gen_cfg.do_sample):
                    gen_kwargs["temperature"] = float(gen_cfg.temperature)
                    gen_kwargs["top_p"] = float(gen_cfg.top_p)
                out = adapter.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
                prompt_len = int(input_ids.shape[1])
                for row_idx, global_idx in enumerate(chunk):
                    gen_ids = out[row_idx, prompt_len:]
                    generated = adapter.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    parsed = _parse_generated_chain(
                        generated_text=generated,
                        entity=str(examples[int(global_idx)].entity),
                        expected_steps=int(expected_steps[int(global_idx)]),
                    )
                    parsed["generated_text"] = str(generated)
                    parsed["generated_text_hash"] = hashlib.sha256(str(generated).encode("utf-8")).hexdigest()
                    parsed["attempt"] = int(attempt)
                    results[int(global_idx)]["result"] = parsed
                    if not bool(parsed.get("parse_success")):
                        next_pending.append(int(global_idx))
            pending = next_pending
    finally:
        adapter.tokenizer.padding_side = original_padding_side

    return results


@torch.no_grad()
def _forward_hidden_states_only(
    adapter,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    if adapter.model is None:
        raise RuntimeError("Adapter model is not loaded")
    ids = adapter.tokenize(input_ids)
    attn = None if attention_mask is None else attention_mask.to(adapter.device)
    model = adapter.model
    if hasattr(model, "model"):  # Qwen/LLaMA style
        out = model.model(ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
        hs = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
        return hs, out.last_hidden_state
    if hasattr(model, "transformer"):  # GPT-2 style
        out = model.transformer(ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
        hs = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
        return hs, out.last_hidden_state
    out = model(ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
    hs = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
    return hs, out.last_hidden_state


def _make_example(idx: int, *, rng: random.Random, min_hops: int, max_hops: int) -> LogicalExample:
    hops = int(rng.randint(int(min_hops), int(max_hops)))
    classes = rng.sample(CLASS_POOL, hops + 1)
    entity = ENTITY_POOL[idx % len(ENTITY_POOL)] + f"_{idx:04d}"
    premises = [f"Rule {j+1}: If something is {classes[j]}, then it is {classes[j+1]}." for j in range(hops)]
    premises.insert(0, f"Fact: {entity} is {classes[0]}.")
    step_texts = [_normalize_step_text(j, entity, classes[j + 1]) for j in range(hops)]
    final_answer = _normalize_final_answer(entity, classes[-1])
    return LogicalExample(
        trace_id=f"prontoqa_test_{idx:05d}",
        example_idx=int(idx),
        entity=entity,
        classes=classes,
        premises=premises,
        step_texts=step_texts,
        step_classes=list(classes[1:]),
        final_answer=final_answer,
        final_class=str(classes[-1]),
        cot_source="synthetic_template",
        generation_metadata={},
    )


def _render_control(
    *,
    example: LogicalExample,
    variant: str,
    step_texts: List[str],
    premises: Optional[List[str]] = None,
    extra_lines: Optional[List[Tuple[str, str]]] = None,
    final_answer: Optional[str] = None,
    expected_failure_mode: Optional[str] = None,
) -> Dict[str, Any]:
    premise_lines = list(premises if premises is not None else example.premises)
    line_objs: List[Dict[str, Any]] = [{"role": "premise", "text": ln} for ln in premise_lines]
    line_objs.extend({"role": "step", "text": ln} for ln in step_texts)
    for role, txt in (extra_lines or []):
        line_objs.append({"role": role, "text": txt})
    line_objs.append({"role": "final_answer", "text": final_answer or example.final_answer})
    rendered = build_trace_text_with_spans([str(x["text"]) for x in line_objs])
    text, spans = rendered
    step_spans: List[Dict[str, int]] = []
    step_lines: List[str] = []
    step_states: List[Dict[str, Any]] = []
    for obj, sp in zip(line_objs, spans):
        if obj.get("role") == "step":
            step_idx = len(step_lines)
            step_txt = str(obj.get("text", ""))
            step_spans.append({"char_start": int(sp["char_start"]), "char_end": int(sp["char_end"])})
            step_lines.append(step_txt)
            cls = _extract_conclusion_class(step_txt, str(example.entity))
            if cls is None:
                cls = str(example.step_classes[min(step_idx, len(example.step_classes) - 1)]) if example.step_classes else "unknown"
            step_states.append(
                {
                    "step_idx": int(step_idx),
                    "step_type": "logical_step",
                    "step_text": step_txt,
                    "entity": str(example.entity),
                    "conclusion_class": str(cls),
                }
            )

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
        "text_step_states": step_states,
        "style_template_id": "prontoqa_logical_default",
        "style_family": "logical_chain",
        "style_counterfactual": False,
        "paper_failure_family": None,
        "paper_failure_subtype": None,
        "text_order_pattern": "step_first",
        "contains_correction": False,
        "control_group": "prontoqa_core5",
        "cot_source": str(example.cot_source),
        "generation_metadata": clone_jsonable(example.generation_metadata or {}),
    }


def _variant_control(example: LogicalExample, variant: str, rng: random.Random) -> Dict[str, Any]:
    steps = list(example.step_texts)
    premises = list(example.premises)
    if variant == "faithful":
        return _render_control(
            example=example,
            variant=variant,
            premises=premises,
            step_texts=steps,
            expected_failure_mode=None,
        )

    if variant == "wrong_intermediate":
        j = int(rng.randrange(0, len(steps) - 1)) if len(steps) >= 2 else 0
        wrong_cls = rng.choice([c for c in CLASS_POOL if c not in set(example.step_classes)] or CLASS_POOL)
        steps[j] = _normalize_step_text(j, str(example.entity), str(wrong_cls))
        return _render_control(
            example=example,
            variant=variant,
            premises=premises,
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
            premises=premises,
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
        return _render_control(example=example, variant=variant, premises=premises, step_texts=steps, expected_failure_mode=mode)

    if variant == "wrong_premise":
        extra = [("rationale", f"Injected premise: If something is {example.classes[0]}, then it is non_{example.final_class}.")]
        return _render_control(
            example=example,
            variant=variant,
            premises=premises,
            step_texts=steps,
            extra_lines=extra,
            expected_failure_mode="wrong_premise",
        )

    if variant == "irrelevant_insertion":
        extra = [("rationale", "Irrelevant note: The weather is sunny and pleasant today.")]
        return _render_control(
            example=example,
            variant=variant,
            premises=premises,
            step_texts=steps,
            extra_lines=extra,
            expected_failure_mode="irrelevant_insertion",
        )

    if variant == "lexical_consistent_swap":
        chain = list(example.classes)
        if len(chain) < 3:
            return _render_control(
                example=example,
                variant=variant,
                premises=premises,
                step_texts=steps,
                expected_failure_mode="lexical_consistent_swap_unavailable",
            )
        target_idx = int(rng.randrange(1, len(chain) - 1))
        old_cls = str(chain[target_idx])
        new_cls = str(rng.choice([c for c in CLASS_POOL if c not in set(chain)] or CLASS_POOL))
        new_premises = [_replace_class_token(p, old_cls, new_cls) for p in premises]
        new_steps = [_replace_class_token(s, old_cls, new_cls) for s in steps]
        new_final = str(example.final_answer)
        return _render_control(
            example=example,
            variant=variant,
            premises=new_premises,
            step_texts=new_steps,
            final_answer=new_final,
            expected_failure_mode=f"lexical_consistent_swap_node_{target_idx}",
        )

    raise ValueError(f"Unsupported variant: {variant}")


def _build_trace_records(
    examples: Sequence[LogicalExample],
    *,
    model_key: str,
    model_family: str,
    num_layers: int,
    hidden_dim: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        for i, text in enumerate(ex.step_texts):
            cls = str(ex.step_classes[min(i, len(ex.step_classes) - 1)]) if ex.step_classes else "unknown"
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
                    "cot_source": str(ex.cot_source),
                    "structured_state": {
                        "step_idx": int(i),
                        "step_type": "logical_step",
                        "step_text": str(text),
                        "entity": str(ex.entity),
                        "conclusion_class": cls,
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
    forward_batch_size: int = 1,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every_controls: int = 0,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    if adapter.tokenizer is None:
        raise RuntimeError("Adapter tokenizer is not loaded")
    pad_token_id = int(
        adapter.tokenizer.pad_token_id
        if adapter.tokenizer.pad_token_id is not None
        else adapter.tokenizer.eos_token_id
    )
    batch_size = max(1, int(forward_batch_size))
    ckpt_every = max(0, int(checkpoint_every_controls))

    rows: List[Dict[str, Any]] = []
    next_control_idx = 0
    if bool(resume) and checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            rows = list(state.get("rows", []))
            next_control_idx = int(state.get("next_control_idx", 0))

    for batch_start in range(int(next_control_idx), len(controls), batch_size):
        batch_end = min(len(controls), batch_start + batch_size)
        ctrl_batch = list(controls[batch_start:batch_end])

        tokenized: List[Tuple[List[int], List[Tuple[int, int]], Dict[str, Any]]] = []
        for ctrl in ctrl_batch:
            token_ids, offsets, tok_meta = adapter.tokenize_with_offsets(str(ctrl.get("cot_text", "")))
            tokenized.append((token_ids, offsets, tok_meta))

        max_len = max(len(toks[0]) for toks in tokenized)
        input_ids = torch.full(
            (len(ctrl_batch), int(max_len)),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=adapter.device,
        )
        attention_mask = torch.zeros((len(ctrl_batch), int(max_len)), dtype=torch.long, device=adapter.device)
        for i, (token_ids, _, _) in enumerate(tokenized):
            if not token_ids:
                continue
            n = len(token_ids)
            input_ids[i, :n] = torch.tensor(token_ids, dtype=torch.long, device=adapter.device)
            attention_mask[i, :n] = 1

        hs, _ = _forward_hidden_states_only(adapter, input_ids, attention_mask=attention_mask)
        if len(hs) != int(num_layers):
            raise RuntimeError(
                f"hidden-state depth mismatch for model_key={model_key!r}: got={len(hs)} expected={num_layers}"
            )

        for bi, ctrl in enumerate(ctrl_batch):
            token_ids, offsets, tok_meta = tokenized[bi]
            step_spans = list(ctrl.get("cot_step_spans", []))
            line_roles = list(ctrl.get("cot_line_roles", []))
            step_line_indices = [i for i, r in enumerate(line_roles) if str(r) == "step"]
            step_states = list(ctrl.get("text_step_states", []))
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
                raw_hidden = torch.stack(
                    [hs[L][bi, int(tok_idx), :].detach().cpu() for L in range(int(num_layers))]
                ).half()
                step_state = (
                    step_states[local_idx]
                    if local_idx < len(step_states) and isinstance(step_states[local_idx], dict)
                    else {}
                )
                structured_state = {
                    "step_idx": int(local_idx),
                    "step_type": str(step_state.get("step_type", "logical_step")),
                    "step_text": str(step_state.get("step_text", "")),
                    "entity": str(step_state.get("entity", "")),
                    "conclusion_class": str(step_state.get("conclusion_class", "unknown")),
                    "operator": "unknown",
                }
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
                    "result_token_id": int(token_ids[int(tok_idx)]) if 0 <= int(tok_idx) < len(token_ids) else -1,
                    "structured_state": structured_state,
                    "cot_source": str(ctrl.get("cot_source", "unknown")),
                    "generation_metadata": clone_jsonable(ctrl.get("generation_metadata", {})),
                }
                rows.append(rec)
                if int(max_records) > 0 and len(rows) >= int(max_records):
                    rows = rows[: int(max_records)]
                    if checkpoint_path is not None:
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        save_pt(
                            checkpoint_path,
                            {
                                "next_control_idx": int(batch_end),
                                "rows": rows,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                    return rows

        if ckpt_every > 0 and checkpoint_path is not None and (int(batch_end) % ckpt_every == 0):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_pt(
                checkpoint_path,
                {
                    "next_control_idx": int(batch_end),
                    "rows": rows,
                    "timestamp": datetime.now().isoformat(),
                },
            )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", default="qwen2.5-7b")
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260309)
    p.add_argument("--chain-len-min", type=int, default=3)
    p.add_argument("--chain-len-max", type=int, default=5)
    p.add_argument(
        "--variants",
        default="faithful,wrong_intermediate,order_flip,skipped_step,wrong_premise,irrelevant_insertion,lexical_consistent_swap",
    )
    p.add_argument("--cot-source", choices=COT_SOURCE_CHOICES, default="synthetic_template")
    p.add_argument("--gen-max-new-tokens", type=int, default=200)
    p.add_argument("--gen-temperature", type=float, default=0.2)
    p.add_argument("--gen-top-p", type=float, default=0.95)
    p.add_argument("--gen-do-sample", action="store_true")
    p.add_argument("--gen-retries", type=int, default=2)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--forward-batch-size", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
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
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

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
    gen_progress_path = out_dir / "meta" / "progress_examples.pt"
    rec_progress_path = out_dir / "meta" / "progress_records.pt"

    adapter = create_adapter(model_key=spec.model_key, device=str(args.device)).load(device=str(args.device))

    gen_cfg = GenerationConfig(
        max_new_tokens=int(args.gen_max_new_tokens),
        temperature=float(args.gen_temperature),
        top_p=float(args.gen_top_p),
        do_sample=bool(args.gen_do_sample),
        retries=int(args.gen_retries),
    )

    all_bases: List[LogicalExample] = [
        _make_example(
            i,
            rng=rng,
            min_hops=int(args.chain_len_min),
            max_hops=int(args.chain_len_max),
        )
        for i in range(int(args.sample_size))
    ]
    shard_index = int(args.shard_index)
    num_shards = int(args.num_shards)
    base_examples = [ex for ex in all_bases if (int(ex.example_idx) % num_shards) == shard_index]

    examples: List[LogicalExample] = []
    generation_stats = {
        "requested": int(args.sample_size),
        "requested_for_shard": int(len(base_examples)),
        "generated_ok": 0,
        "generated_failed": 0,
        "avg_attempt": None,
        "failure_reasons": {},
    }
    attempts: List[int] = []
    start_local_idx = 0
    if bool(args.resume) and gen_progress_path.exists():
        gen_state = torch.load(gen_progress_path, map_location="cpu")
        if isinstance(gen_state, dict):
            examples = [_logical_example_from_dict(d) for d in list(gen_state.get("examples", []))]
            generation_stats = clone_jsonable(gen_state.get("generation_stats", generation_stats))
            attempts = [int(x) for x in list(gen_state.get("attempts", []))]
            start_local_idx = int(gen_state.get("next_local_idx", 0))

    checkpoint_every = max(0, int(args.checkpoint_every))
    gen_batch_size = max(1, int(args.gen_batch_size))
    if str(args.cot_source) == "model_generated":
        for local_start in range(int(start_local_idx), len(base_examples), gen_batch_size):
            local_end = min(len(base_examples), local_start + gen_batch_size)
            chunk = base_examples[local_start:local_end]
            generated = _generate_model_generated_chains(
                adapter=adapter,
                examples=chunk,
                gen_cfg=gen_cfg,
                batch_size=gen_batch_size,
            )
            for base, g in zip(chunk, generated):
                gres = g.get("result", {}) if isinstance(g, dict) else {}
                parse_success = bool(gres.get("parse_success"))
                attempts.append(int(gres.get("attempt", 0)))
                if not parse_success:
                    generation_stats["generated_failed"] = int(generation_stats["generated_failed"]) + 1
                    reason = str(gres.get("parse_reason", "unknown"))
                    fr = generation_stats["failure_reasons"]
                    fr[reason] = int(fr.get(reason, 0) + 1)
                    continue
                generation_stats["generated_ok"] = int(generation_stats["generated_ok"]) + 1
                gen_meta = {
                    "cot_source": "model_generated",
                    "prompt_hash": str(g.get("prompt_hash", "")),
                    "generation_config": clone_jsonable(g.get("config", {})),
                    "parse_success": True,
                    "parse_reason": gres.get("parse_reason"),
                    "retry_count": int(gres.get("attempt", 0)),
                    "generated_text_hash": str(gres.get("generated_text_hash", "")),
                }
                examples.append(
                    LogicalExample(
                        trace_id=str(base.trace_id),
                        example_idx=int(base.example_idx),
                        entity=str(base.entity),
                        classes=list(base.classes),
                        premises=list(base.premises),
                        step_texts=list(gres.get("step_texts", [])),
                        step_classes=list(gres.get("step_classes", [])),
                        final_answer=str(gres.get("final_answer", base.final_answer)),
                        final_class=str(gres.get("final_class", base.final_class)),
                        cot_source="model_generated",
                        generation_metadata=gen_meta,
                    )
                )
            if checkpoint_every > 0 and (local_end % checkpoint_every == 0):
                gen_progress_path.parent.mkdir(parents=True, exist_ok=True)
                save_pt(
                    gen_progress_path,
                    {
                        "next_local_idx": int(local_end),
                        "examples": [_logical_example_to_dict(ex) for ex in examples],
                        "generation_stats": generation_stats,
                        "attempts": attempts,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
    else:
        examples = list(base_examples)

    if str(args.cot_source) == "model_generated":
        if not examples:
            raise RuntimeError("No model-generated examples were parseable; adjust generation settings")
        if attempts:
            generation_stats["avg_attempt"] = float(sum(attempts) / len(attempts))
    if gen_progress_path.exists():
        try:
            gen_progress_path.unlink()
        except OSError:
            pass

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
            vrng = random.Random(_stable_seed(int(args.seed), str(ex.trace_id), str(v)))
            controls.append(_variant_control(ex, v, vrng))

    save_json(
        controls_out,
        {
            "schema_version": "phase7_cot_control_v1",
            "source_dataset": ("prontoqa_model_generated" if str(args.cot_source) == "model_generated" else "prontoqa_synthetic"),
            "model_key": str(spec.model_key),
            "num_controls": int(len(controls)),
            "num_examples": int(len(examples)),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "variants": variants,
            "cot_source": str(args.cot_source),
            "generation_config": clone_jsonable(gen_cfg.__dict__),
            "generation_stats": clone_jsonable(generation_stats),
            "controls": controls,
        },
    )

    records = _build_control_records(
        controls,
        adapter=adapter,
        model_key=spec.model_key,
        num_layers=spec.num_layers,
        hidden_dim=spec.hidden_dim,
        max_records=int(args.max_records),
        forward_batch_size=int(args.forward_batch_size),
        checkpoint_path=rec_progress_path,
        checkpoint_every_controls=int(args.checkpoint_every),
        resume=bool(args.resume),
    )
    if rec_progress_path.exists():
        try:
            rec_progress_path.unlink()
        except OSError:
            pass

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
            "sample_size_requested": int(args.sample_size),
            "sample_size_shard": int(len(base_examples)),
            "sample_size_effective": int(len(examples)),
            "seed": int(args.seed),
            "chain_len_min": int(args.chain_len_min),
            "chain_len_max": int(args.chain_len_max),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "variants": variants,
            "max_records": int(args.max_records),
            "cot_source": str(args.cot_source),
            "gen_batch_size": int(args.gen_batch_size),
            "forward_batch_size": int(args.forward_batch_size),
            "checkpoint_every": int(args.checkpoint_every),
            "resume": bool(args.resume),
            "generation_config": clone_jsonable(gen_cfg.__dict__),
            "generation_stats": clone_jsonable(generation_stats),
        },
        "timestamp": datetime.now().isoformat(),
    }
    save_json(records_out, payload)

    manifest = {
        "schema_version": "phase7_prontoqa_prepare_manifest_v2",
        "status": "ok",
        "model_key": str(spec.model_key),
        "cot_source": str(args.cot_source),
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
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "variants": variants,
        "gen_batch_size": int(args.gen_batch_size),
        "forward_batch_size": int(args.forward_batch_size),
        "checkpoint_every": int(args.checkpoint_every),
        "resume": bool(args.resume),
        "generation_config": clone_jsonable(gen_cfg.__dict__),
        "generation_stats": clone_jsonable(generation_stats),
        "timestamp": datetime.now().isoformat(),
    }
    save_json(manifest_out, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover
    from .common import clone_jsonable, load_json, load_pt, save_json, save_pt, set_seed, sha256_file
    from .model_registry import create_adapter, resolve_model_spec
except ImportError:  # pragma: no cover
    from common import clone_jsonable, load_json, load_pt, save_json, save_pt, set_seed, sha256_file
    from model_registry import create_adapter, resolve_model_spec


STEP_RE = re.compile(r"^STEP\s+(?P<idx>\d+)\s*:\s*(?P<body>.+)$")
FINAL_RE = re.compile(r"FINAL_ANSWER\s*[:=]\s*(?P<value>-?\d+(?:\.\d+)?)", flags=re.IGNORECASE)
FINAL_BOOL_RE = re.compile(r"FINAL_ANSWER\s*[:=]\s*(?P<value>YES|NO|TRUE|FALSE)\b", flags=re.IGNORECASE)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class PairSpec:
    pair_id: str
    pair_type: str  # inverse | equivalent | lexical_control
    relation_type: str  # inverse_addend | inverse_boolean | equal
    lexical_control: bool
    a: int
    b: int
    c: int
    prompt_a: str
    prompt_b: str
    expected_a: float
    expected_b: float
    domain: str = "arithmetic"
    answer_mode: str = "numeric"
    logic_meta: Optional[Dict[str, Any]] = None


LOGICAL_INFERENCE_TYPES = (
    "unknown",
    "fact_assertion",
    "class_subsumption",
    "negation",
    "other",
)
LOGICAL_TRUTH_VALUES = ("unknown", "true", "false", "uncertain")


def _line_spans(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    pos = 0
    for line in str(text).splitlines(keepends=True):
        start = pos
        end = pos + len(line.rstrip("\n"))
        out.append((int(start), int(max(start, end))))
        pos += len(line)
    if not out and text:
        out.append((0, len(text)))
    return out


def _extract_step_spans(text: str) -> List[Dict[str, Any]]:
    spans = _line_spans(text)
    lines = str(text).splitlines()
    out: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        m = STEP_RE.match(str(ln).strip())
        if not m:
            continue
        st, en = spans[i] if i < len(spans) else (0, 0)
        out.append(
            {
                "step_idx": int(m.group("idx")),
                "line_index": int(i),
                "char_start": int(st),
                "char_end": int(en),
                "text": str(ln),
            }
        )
    if not out and text.strip():
        # Fallback single pseudo-step keeps pipeline usable on format drift.
        out.append(
            {
                "step_idx": 0,
                "line_index": 0,
                "char_start": 0,
                "char_end": len(text),
                "text": str(text).strip(),
            }
        )
    out.sort(key=lambda x: (int(x["step_idx"]), int(x["line_index"])))
    return out


def _find_token_for_char(offsets: Sequence[Tuple[int, int]], char_pos: int) -> Optional[int]:
    best = None
    cp = int(char_pos)
    for i, (s, e) in enumerate(offsets):
        si = int(s)
        ei = int(e)
        if si <= cp < ei:
            return int(i)
        if si <= cp:
            best = int(i)
    return best


def _parse_final_answer(text: str) -> Tuple[Optional[float], Dict[str, Any]]:
    raw = str(text or "")
    bm = FINAL_BOOL_RE.search(raw)
    if bm:
        tok = str(bm.group("value")).strip().lower()
        if tok in {"yes", "true"}:
            return 1.0, {"method": "final_answer_bool_regex", "defined": True}
        if tok in {"no", "false"}:
            return 0.0, {"method": "final_answer_bool_regex", "defined": True}
    m = FINAL_RE.search(raw)
    if m:
        try:
            v = float(m.group("value"))
            return v, {"method": "final_answer_regex", "defined": True}
        except Exception:
            pass
    nums = NUM_RE.findall(raw)
    if nums:
        try:
            return float(nums[-1]), {"method": "last_number_fallback", "defined": True}
        except Exception:
            pass
    tail = raw.strip().split()
    if tail:
        tok = str(tail[-1]).strip().lower().strip(".!,;:")
        if tok in {"yes", "true"}:
            return 1.0, {"method": "tail_bool_fallback", "defined": True}
        if tok in {"no", "false"}:
            return 0.0, {"method": "tail_bool_fallback", "defined": True}
    return None, {"method": "none", "defined": False}


def _is_correct_answer(
    *,
    answer: Optional[float],
    expected: float,
    answer_mode: str,
    tol: float,
) -> bool:
    if answer is None or not isinstance(answer, (int, float)):
        return False
    mode = str(answer_mode).strip().lower()
    if mode == "boolean":
        return int(float(answer) >= 0.5) == int(float(expected) >= 0.5)
    return bool(abs(float(answer) - float(expected)) <= float(tol))


def _pair_relation_contradiction(
    spec: PairSpec,
    ans_a: Optional[float],
    ans_b: Optional[float],
    *,
    tol: float,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    if ans_a is None or ans_b is None:
        return None, {"defined": False, "reason": "missing_answer"}
    if str(spec.relation_type) == "equal":
        ok = math.isfinite(ans_a) and math.isfinite(ans_b) and abs(float(ans_a) - float(ans_b)) <= float(tol)
        return (not ok), {"defined": True, "relation": "equal", "ok": bool(ok)}
    if str(spec.relation_type) == "inverse_addend":
        # For pair: member_a predicts c=a+b, member_b predicts a.
        lhs = float(ans_b) + float(spec.b)
        ok = abs(lhs - float(ans_a)) <= float(tol)
        return (not ok), {"defined": True, "relation": "inverse_addend", "ok": bool(ok), "lhs": float(lhs)}
    if str(spec.relation_type) == "inverse_boolean":
        a = int(float(ans_a) >= 0.5)
        b = int(float(ans_b) >= 0.5)
        ok = (a != b)
        return (not ok), {"defined": True, "relation": "inverse_boolean", "ok": bool(ok)}
    return None, {"defined": False, "reason": f"unknown_relation:{spec.relation_type}"}


def _format_prompt(question: str) -> str:
    return "\n".join(
        [
            "Solve the arithmetic problem with concise steps.",
            "Output format (strict):",
            "STEP 0: ...",
            "STEP 1: ...",
            "FINAL_ANSWER: <number>",
            "",
            f"Question: {question}",
            "Output:",
        ]
    )


def _format_logic_prompt(*, facts: Sequence[str], question: str, domain: str) -> str:
    return "\n".join(
        [
            f"Reason over the {domain} facts with concise steps.",
            "Output format (strict):",
            "STEP 0: ...",
            "STEP 1: ...",
            "FINAL_ANSWER: YES or NO",
            "",
            "Facts:",
            *[f"- {str(x)}" for x in facts],
            "",
            f"Question: {question}",
            "Output:",
        ]
    )


def _build_decoder_vocab_manifest(domain: str, pair_specs: Sequence[PairSpec]) -> Dict[str, Any]:
    d = str(domain).strip().lower()
    out: Dict[str, Any] = {
        "schema_version": "phase7_optionc_decoder_vocab_v1",
        "decoder_domain": d,
        "unknown_id": 0,
    }
    if d == "arithmetic":
        return out
    classes: set[str] = set()
    entities: set[str] = set()
    for ps in pair_specs:
        meta = dict(ps.logic_meta or {})
        ent = str(meta.get("entity", "")).strip()
        if ent:
            entities.add(ent)
        for c in list(meta.get("chain", [])):
            cc = str(c).strip().lower()
            if cc:
                classes.add(cc)
    class_vocab = ["__unknown__"] + sorted(classes)
    entity_vocab = ["__unknown__"] + sorted(entities)
    inf_vocab = list(LOGICAL_INFERENCE_TYPES)
    truth_vocab = list(LOGICAL_TRUTH_VALUES)
    out.update(
        {
            "class_vocab": class_vocab,
            "entity_vocab": entity_vocab,
            "inference_type_vocab": inf_vocab,
            "truth_value_vocab": truth_vocab,
            "chain_depth_bins": 5,
            "class_to_id": {k: int(i) for i, k in enumerate(class_vocab)},
            "entity_to_id": {k: int(i) for i, k in enumerate(entity_vocab)},
            "inference_type_to_id": {k: int(i) for i, k in enumerate(inf_vocab)},
            "truth_value_to_id": {k: int(i) for i, k in enumerate(truth_vocab)},
        }
    )
    return out


def _infer_logical_structured_state(
    *,
    pair_spec: PairSpec,
    step_text: str,
    step_idx: int,
    member_correct: bool,
    vocab_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    unknown_id = int(vocab_manifest.get("unknown_id", 0))
    class_to_id = {str(k): int(v) for k, v in dict(vocab_manifest.get("class_to_id", {})).items()}
    entity_to_id = {str(k): int(v) for k, v in dict(vocab_manifest.get("entity_to_id", {})).items()}
    inf_to_id = {str(k): int(v) for k, v in dict(vocab_manifest.get("inference_type_to_id", {})).items()}
    truth_to_id = {str(k): int(v) for k, v in dict(vocab_manifest.get("truth_value_to_id", {})).items()}
    domain = str(pair_spec.domain).strip().lower()
    meta = dict(pair_spec.logic_meta or {})
    chain = [str(x).strip().lower() for x in list(meta.get("chain", []))]
    entity = str(meta.get("entity", "")).strip()
    line = str(step_text or "")
    line_l = line.lower()

    inference_type = "unknown"
    truth_value = "uncertain"
    premise_cls = "__unknown__"
    conclusion_cls = "__unknown__"
    label_source = "fallback"
    parse_confidence = 0.20
    unknown_reason = ""

    class_vocab = [str(x) for x in list(vocab_manifest.get("class_vocab", [])) if str(x) != "__unknown__"]
    mentions = [c for c in class_vocab if re.search(rf"\b{re.escape(c)}\b", line_l)]

    if re.search(r"\bnot\b", line_l):
        inference_type = "negation"
        truth_value = "false"
        label_source = "text_negation"
        parse_confidence = 0.85
    elif "all " in line_l and " are " in line_l:
        inference_type = "class_subsumption"
        truth_value = "true" if bool(member_correct) else "uncertain"
        label_source = "text_subsumption"
        parse_confidence = 0.85
    elif entity and re.search(rf"\b{re.escape(entity.lower())}\b", line_l):
        inference_type = "fact_assertion" if int(step_idx) == 0 else "other"
        truth_value = "true" if bool(member_correct) else "uncertain"
        label_source = "text_entity_assertion"
        parse_confidence = 0.75

    if len(mentions) >= 2:
        premise_cls = str(mentions[0])
        conclusion_cls = str(mentions[-1])
        parse_confidence = max(parse_confidence, 0.90)
    elif len(mentions) == 1:
        conclusion_cls = str(mentions[0])
        if int(step_idx) > 0 and chain:
            premise_cls = str(chain[min(len(chain) - 1, max(0, int(step_idx) - 1))])
        elif chain:
            premise_cls = str(chain[0])
        parse_confidence = max(parse_confidence, 0.70)
    elif chain:
        # Deterministic fallback keeps labels available for decoder training.
        ci = min(len(chain) - 1, max(0, int(step_idx)))
        pi = min(len(chain) - 1, max(0, int(step_idx) - 1))
        conclusion_cls = str(chain[ci])
        premise_cls = str(chain[pi])
        label_source = "template_chain_fallback"
        parse_confidence = max(parse_confidence, 0.55)
        unknown_reason = "no_class_mention_in_step_text"
    else:
        unknown_reason = "missing_chain_metadata"

    if truth_value == "uncertain":
        if bool(member_correct):
            truth_value = "true"
        elif int(step_idx) >= 2:
            truth_value = "false"

    if inference_type == "unknown":
        inference_type = "other"

    return {
        "decoder_domain": domain,
        "inference_type": str(inference_type),
        "inference_type_id": int(inf_to_id.get(str(inference_type), unknown_id)),
        "chain_depth_id": int(min(4, max(0, int(step_idx)))),
        "truth_value": str(truth_value),
        "truth_value_id": int(truth_to_id.get(str(truth_value), unknown_id)),
        "conclusion_class": str(conclusion_cls),
        "conclusion_class_id": int(class_to_id.get(str(conclusion_cls), unknown_id)),
        "premise_class": str(premise_cls),
        "premise_class_id": int(class_to_id.get(str(premise_cls), unknown_id)),
        "target_entity": str(entity),
        "target_entity_id": int(entity_to_id.get(str(entity), unknown_id)),
        "logical_label_defined": bool(parse_confidence >= 0.5),
        "logical_label_source": str(label_source),
        "logical_parse_confidence": float(parse_confidence),
        "logical_unknown_reason": str(unknown_reason),
    }


def _build_pair_specs_arithmetic(*, n_pairs: int, seed: int, lexical_fraction: float) -> List[PairSpec]:
    rng = random.Random(int(seed))
    specs: List[PairSpec] = []
    for i in range(int(n_pairs)):
        a = int(rng.randint(11, 999))
        b = int(rng.randint(11, 999))
        c = int(a + b)
        lexical = bool(rng.random() < float(lexical_fraction))
        if lexical:
            q1 = f"What is {a} + {b}?"
            q2 = f"Compute the sum of {a} and {b}."
            specs.append(
                PairSpec(
                    pair_id=f"pair_{i:06d}",
                    pair_type="lexical_control",
                    relation_type="equal",
                    lexical_control=True,
                    a=a,
                    b=b,
                    c=c,
                    prompt_a=_format_prompt(q1),
                    prompt_b=_format_prompt(q2),
                    expected_a=float(c),
                    expected_b=float(c),
                    domain="arithmetic",
                    answer_mode="numeric",
                )
            )
            continue

        if rng.random() < 0.5:
            # Equivalent wording pair.
            q1 = f"What is {a} + {b}?"
            q2 = f"Add {b} to {a}. What is the result?"
            specs.append(
                PairSpec(
                    pair_id=f"pair_{i:06d}",
                    pair_type="equivalent",
                    relation_type="equal",
                    lexical_control=False,
                    a=a,
                    b=b,
                    c=c,
                    prompt_a=_format_prompt(q1),
                    prompt_b=_format_prompt(q2),
                    expected_a=float(c),
                    expected_b=float(c),
                    domain="arithmetic",
                    answer_mode="numeric",
                )
            )
        else:
            # Inverse framing pair.
            q1 = f"What is {a} + {b}?"
            q2 = f"What number plus {b} equals {c}?"
            specs.append(
                PairSpec(
                    pair_id=f"pair_{i:06d}",
                    pair_type="inverse",
                    relation_type="inverse_addend",
                    lexical_control=False,
                    a=a,
                    b=b,
                    c=c,
                    prompt_a=_format_prompt(q1),
                    prompt_b=_format_prompt(q2),
                    expected_a=float(c),
                    expected_b=float(a),
                    domain="arithmetic",
                    answer_mode="numeric",
                )
            )
    return specs


def _build_pair_specs_prontoqa(*, n_pairs: int, seed: int, lexical_fraction: float) -> List[PairSpec]:
    rng = random.Random(int(seed))
    entities = [
        "Ava", "Ben", "Cara", "Dylan", "Eli", "Faye", "Gus", "Hana", "Ivan", "Jade",
    ]
    classes = [
        "mammal", "vertebrate", "animal", "organism", "living_thing",
        "bird", "reptile", "amphibian", "arthropod", "creature",
        "agent", "entity", "being", "eukaryote",
    ]
    specs: List[PairSpec] = []
    for i in range(int(n_pairs)):
        ent = f"{rng.choice(entities)}_{i:04d}"
        chain = rng.sample(classes, 4)
        facts = [
            f"{ent} is {chain[0]}.",
            f"All {chain[0]} are {chain[1]}.",
            f"All {chain[1]} are {chain[2]}.",
            f"All {chain[2]} are {chain[3]}.",
        ]
        lexical = bool(rng.random() < float(lexical_fraction))
        if lexical:
            q1 = f"Is {ent} a {chain[3]}?"
            q2 = f"Given the facts, does {ent} belong to {chain[3]}?"
            specs.append(
                PairSpec(
                    pair_id=f"pair_{i:06d}",
                    pair_type="lexical_control",
                    relation_type="equal",
                    lexical_control=True,
                    a=0,
                    b=0,
                    c=0,
                    prompt_a=_format_logic_prompt(facts=facts, question=q1, domain="prontoqa"),
                    prompt_b=_format_logic_prompt(facts=facts, question=q2, domain="prontoqa"),
                    expected_a=1.0,
                    expected_b=1.0,
                    domain="prontoqa",
                    answer_mode="boolean",
                    logic_meta={"entity": ent, "chain": list(chain), "facts": list(facts)},
                )
            )
            continue

        if rng.random() < 0.5:
            q1 = f"Is {ent} a {chain[3]}?"
            q2 = f"From these rules, can we conclude {ent} is {chain[3]}?"
            specs.append(
                PairSpec(
                    pair_id=f"pair_{i:06d}",
                    pair_type="equivalent",
                    relation_type="equal",
                    lexical_control=False,
                    a=0,
                    b=0,
                    c=0,
                    prompt_a=_format_logic_prompt(facts=facts, question=q1, domain="prontoqa"),
                    prompt_b=_format_logic_prompt(facts=facts, question=q2, domain="prontoqa"),
                    expected_a=1.0,
                    expected_b=1.0,
                    domain="prontoqa",
                    answer_mode="boolean",
                    logic_meta={"entity": ent, "chain": list(chain), "facts": list(facts)},
                )
            )
        else:
            q1 = f"Is {ent} a {chain[3]}?"
            q2 = f"Is it false that {ent} is {chain[3]}?"
            specs.append(
                PairSpec(
                    pair_id=f"pair_{i:06d}",
                    pair_type="inverse",
                    relation_type="inverse_boolean",
                    lexical_control=False,
                    a=0,
                    b=0,
                    c=0,
                    prompt_a=_format_logic_prompt(facts=facts, question=q1, domain="prontoqa"),
                    prompt_b=_format_logic_prompt(facts=facts, question=q2, domain="prontoqa"),
                    expected_a=1.0,
                    expected_b=0.0,
                    domain="prontoqa",
                    answer_mode="boolean",
                    logic_meta={"entity": ent, "chain": list(chain), "facts": list(facts)},
                )
            )
    return specs


def _build_pair_specs_entailmentbank(*, n_pairs: int, seed: int, lexical_fraction: float) -> List[PairSpec]:
    rng = random.Random(int(seed))
    symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    specs: List[PairSpec] = []
    for i in range(int(n_pairs)):
        x = f"x{i:04d}"
        a, b, c = rng.sample(symbols, 3)
        facts = [
            f"If something is {a}, then it is {b}.",
            f"If something is {b}, then it is {c}.",
            f"{x} is {a}.",
        ]
        lexical = bool(rng.random() < float(lexical_fraction))
        if lexical:
            q1 = f"Does it follow that {x} is {c}?"
            q2 = f"Can we entail {x} is {c} from the premises?"
            specs.append(
                PairSpec(
                    pair_id=f"pair_{i:06d}",
                    pair_type="lexical_control",
                    relation_type="equal",
                    lexical_control=True,
                    a=0,
                    b=0,
                    c=0,
                    prompt_a=_format_logic_prompt(facts=facts, question=q1, domain="entailmentbank"),
                    prompt_b=_format_logic_prompt(facts=facts, question=q2, domain="entailmentbank"),
                    expected_a=1.0,
                    expected_b=1.0,
                    domain="entailmentbank",
                    answer_mode="boolean",
                    logic_meta={"entity": x, "chain": [a.lower(), b.lower(), c.lower()], "facts": list(facts)},
                )
            )
            continue

        if rng.random() < 0.5:
            q1 = f"Does it follow that {x} is {c}?"
            q2 = f"Are the premises sufficient to conclude {x} is {c}?"
            specs.append(
                PairSpec(
                    pair_id=f"pair_{i:06d}",
                    pair_type="equivalent",
                    relation_type="equal",
                    lexical_control=False,
                    a=0,
                    b=0,
                    c=0,
                    prompt_a=_format_logic_prompt(facts=facts, question=q1, domain="entailmentbank"),
                    prompt_b=_format_logic_prompt(facts=facts, question=q2, domain="entailmentbank"),
                    expected_a=1.0,
                    expected_b=1.0,
                    domain="entailmentbank",
                    answer_mode="boolean",
                    logic_meta={"entity": x, "chain": [a.lower(), b.lower(), c.lower()], "facts": list(facts)},
                )
            )
        else:
            q1 = f"Does it follow that {x} is {c}?"
            q2 = f"Does it follow that {x} is not {c}?"
            specs.append(
                PairSpec(
                    pair_id=f"pair_{i:06d}",
                    pair_type="inverse",
                    relation_type="inverse_boolean",
                    lexical_control=False,
                    a=0,
                    b=0,
                    c=0,
                    prompt_a=_format_logic_prompt(facts=facts, question=q1, domain="entailmentbank"),
                    prompt_b=_format_logic_prompt(facts=facts, question=q2, domain="entailmentbank"),
                    expected_a=1.0,
                    expected_b=0.0,
                    domain="entailmentbank",
                    answer_mode="boolean",
                    logic_meta={"entity": x, "chain": [a.lower(), b.lower(), c.lower()], "facts": list(facts)},
                )
            )
    return specs


def _build_pair_specs(*, domain: str, n_pairs: int, seed: int, lexical_fraction: float) -> List[PairSpec]:
    d = str(domain).strip().lower()
    if d == "arithmetic":
        return _build_pair_specs_arithmetic(n_pairs=n_pairs, seed=seed, lexical_fraction=lexical_fraction)
    if d == "prontoqa":
        return _build_pair_specs_prontoqa(n_pairs=n_pairs, seed=seed, lexical_fraction=lexical_fraction)
    if d == "entailmentbank":
        return _build_pair_specs_entailmentbank(n_pairs=n_pairs, seed=seed, lexical_fraction=lexical_fraction)
    raise ValueError(f"Unsupported domain: {domain}")


def _gen_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_new_tokens": int(args.gen_max_new_tokens),
        "do_sample": bool(args.gen_do_sample),
        "temperature": float(args.gen_temperature),
        "top_p": float(args.gen_top_p),
        "retries": int(args.gen_retries),
        "batch_size": int(args.gen_batch_size),
    }


def _run_generation_for_members(
    *,
    adapter,
    prompts: Sequence[str],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    retries = int(cfg["retries"])
    pending = list(range(len(prompts)))
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
    active_batch_size = max(1, int(cfg["batch_size"]))
    oom_count = 0
    for attempt in range(retries + 1):
        if not pending:
            break
        next_pending: List[int] = []
        cursor = 0
        while cursor < len(pending):
            chunk_size = min(active_batch_size, len(pending) - cursor)
            chunk_ids = pending[cursor : cursor + chunk_size]
            chunk_prompts = [prompts[i] for i in chunk_ids]
            try:
                outs = adapter.generate_with_step_hidden_states(
                    chunk_prompts,
                    max_new_tokens=int(cfg["max_new_tokens"]),
                    do_sample=bool(cfg["do_sample"]),
                    temperature=float(cfg["temperature"]),
                    top_p=float(cfg["top_p"]),
                )
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "cuda out of memory" in msg:
                    oom_count += 1
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if chunk_size > 1:
                        active_batch_size = max(1, chunk_size // 2)
                        continue
                    if attempt < retries:
                        next_pending.extend(int(x) for x in chunk_ids)
                        cursor += chunk_size
                        continue
                raise
            cursor += chunk_size
            for local_idx, out in enumerate(outs):
                gi = int(chunk_ids[local_idx])
                gen_text = str(out.get("generated_text", ""))
                step_spans = _extract_step_spans(gen_text)
                parsed_final, parse_diag = _parse_final_answer(gen_text)
                success = parsed_final is not None and len(step_spans) > 0
                out_pack = {
                    **out,
                    "attempt": int(attempt),
                    "step_spans": step_spans,
                    "parsed_final_answer": parsed_final,
                    "final_parse": parse_diag,
                    "parse_success": bool(success),
                }
                results[gi] = out_pack
                if not success and attempt < retries:
                    next_pending.append(gi)
        pending = next_pending
    # Fill impossible holes defensively.
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "prompt": str(prompts[i]),
                "generated_text": "",
                "generated_token_ids": [],
                "hidden_by_generated_token": [],
                "prompt_token_count": 0,
                "generated_token_count": 0,
                "captured_step_count": 0,
                "attempt": int(retries),
                "step_spans": [],
                "parsed_final_answer": None,
                "final_parse": {"method": "none", "defined": False},
                "parse_success": False,
            }
    diag = {
        "initial_batch_size": int(cfg["batch_size"]),
        "effective_batch_size": int(active_batch_size),
        "oom_retry_count": int(oom_count),
    }
    return [dict(x) for x in results if isinstance(x, dict)], diag


def _tokenize_generated_with_offsets(adapter, text: str) -> Tuple[List[int], List[Tuple[int, int]], Dict[str, Any]]:
    tok = adapter.tokenizer
    if tok is None:
        raise RuntimeError("Tokenizer not loaded")
    diag = {"defined": True, "method": "offset_mapping"}
    try:
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        token_ids = [int(x) for x in enc.input_ids[0].tolist()]
        offs = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
        return token_ids, offs, diag
    except Exception:
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        token_ids = [int(x) for x in enc.input_ids[0].tolist()]
        offs = [(0, 0) for _ in token_ids]
        return token_ids, offs, {"defined": False, "method": "fallback_no_offsets"}


def _member_label_from_correctness(correct_a: bool, correct_b: bool) -> Tuple[List[int], bool]:
    # labels: 0 faithful, 1 unfaithful
    if correct_a and correct_b:
        return [0, 0], False
    if correct_a and (not correct_b):
        return [0, 1], False
    if (not correct_a) and correct_b:
        return [1, 0], False
    return [1, 1], True  # both wrong => ambiguous pair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", default="qwen2.5-7b")
    p.add_argument("--domain", choices=["arithmetic", "prontoqa", "entailmentbank"], default="arithmetic")
    p.add_argument(
        "--decoder-domain-labels",
        choices=["auto", "arithmetic", "prontoqa", "entailmentbank"],
        default="auto",
        help="Structured-state label schema. 'auto' uses --domain.",
    )
    p.add_argument("--sample-pairs", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260309)
    p.add_argument("--lexical-fraction", type=float, default=0.20)
    p.add_argument("--answer-tol", type=float, default=1e-6)
    p.add_argument("--gen-max-new-tokens", type=int, default=180)
    p.add_argument("--gen-do-sample", action="store_true")
    p.add_argument("--gen-temperature", type=float, default=0.2)
    p.add_argument("--gen-top-p", type=float, default=0.95)
    p.add_argument("--gen-retries", type=int, default=1)
    p.add_argument("--gen-batch-size", type=int, default=12)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--merge-shards", nargs="*", default=[])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True, help="Output JSON manifest path.")
    return p.parse_args()


def _resolve_rows_pt_path(payload_path: Path, rows_path_raw: Any) -> Path:
    rp = Path(str(rows_path_raw))
    if rp.is_absolute():
        return rp
    cand = (payload_path.parent / rp).resolve()
    if cand.exists():
        return cand
    return rp.resolve()


def _merge_decoder_vocab_manifest(base: Optional[Dict[str, Any]], nxt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if base is None:
        return clone_jsonable(dict(nxt or {})) if nxt is not None else None
    if nxt is None:
        return base
    bd = str(base.get("decoder_domain", "")).strip().lower()
    nd = str(nxt.get("decoder_domain", "")).strip().lower()
    if bd != nd:
        raise RuntimeError(f"decoder vocab domain mismatch across shards: {bd} vs {nd}")
    if bd == "arithmetic":
        return base
    merged = dict(base)
    for key in ("class_vocab", "entity_vocab"):
        vals = set(str(x) for x in list(base.get(key, [])))
        vals.update(str(x) for x in list(nxt.get(key, [])))
        unknown = "__unknown__"
        vocab = [unknown] + sorted(v for v in vals if v and v != unknown)
        merged[key] = vocab
        map_key = "class_to_id" if key == "class_vocab" else "entity_to_id"
        merged[map_key] = {v: int(i) for i, v in enumerate(vocab)}
    return merged


def _merge_shard_payloads(shard_paths: Sequence[str], output: str) -> None:
    if not shard_paths:
        raise RuntimeError("merge requested with no shard paths")
    shard_payloads: List[Tuple[Path, Dict[str, Any], List[Dict[str, Any]]]] = []
    model_key: Optional[str] = None
    num_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    source_scope: Optional[str] = None
    merged_decoder_vocab_manifest: Optional[Dict[str, Any]] = None
    for sp_raw in shard_paths:
        sp = Path(str(sp_raw))
        payload = load_json(sp)
        if str(payload.get("status")) != "ok":
            raise RuntimeError(f"shard payload not ok: {sp}")
        rp = _resolve_rows_pt_path(sp, payload.get("rows_path"))
        if not rp.exists():
            raise RuntimeError(f"missing shard rows file: {rp}")
        expected_rows_sha = str(payload.get("rows_sha256") or "")
        actual_rows_sha = sha256_file(rp)
        if expected_rows_sha and expected_rows_sha != actual_rows_sha:
            raise RuntimeError(f"rows hash mismatch for shard {sp}: expected {expected_rows_sha}, got {actual_rows_sha}")
        rows = list(load_pt(rp))
        merged_decoder_vocab_manifest = _merge_decoder_vocab_manifest(
            merged_decoder_vocab_manifest,
            dict(payload.get("decoder_vocab_manifest") or {}) or None,
        )
        sk = str(payload.get("model_key"))
        if model_key is None:
            model_key = sk
            num_layers = int(payload.get("num_layers", 0))
            hidden_dim = int(payload.get("hidden_dim", 0))
            source_scope = str((payload.get("source") or {}).get("scope", "arithmetic"))
        else:
            if sk != model_key or int(payload.get("num_layers", -1)) != num_layers or int(payload.get("hidden_dim", -1)) != hidden_dim:
                raise RuntimeError(f"incompatible shard metadata in {sp}")
            this_scope = str((payload.get("source") or {}).get("scope", "arithmetic"))
            if this_scope != source_scope:
                raise RuntimeError(f"incompatible shard scope in {sp}: {this_scope} vs {source_scope}")
        shard_payloads.append((sp, payload, rows))

    pairs: List[Dict[str, Any]] = []
    members: List[Dict[str, Any]] = []
    rows_all: List[Dict[str, Any]] = []
    seen_pair: set[str] = set()
    seen_member: set[str] = set()
    shard_meta: List[Dict[str, Any]] = []
    for sp, payload, rows in shard_payloads:
        shard_meta.append(
            {
                "path": str(sp),
                "sha256": sha256_file(sp),
                "rows_path": str(payload.get("rows_path")),
                "rows_sha256": str(payload.get("rows_sha256")),
                "pairs_count": int(payload.get("pairs_count", 0)),
                "members_count": int(payload.get("members_count", 0)),
                "rows_count": int(payload.get("rows_count", 0)),
            }
        )
        for p in list(payload.get("pairs", [])):
            pid = str(p.get("pair_id", ""))
            if not pid or pid in seen_pair:
                continue
            seen_pair.add(pid)
            pairs.append(dict(p))
        for m in list(payload.get("members", [])):
            mid = str(m.get("member_id", ""))
            if not mid or mid in seen_member:
                continue
            seen_member.add(mid)
            members.append(dict(m))
        rows_all.extend(dict(r) for r in rows if isinstance(r, dict))

    # Recompute row windows after merge.
    rows_by_member: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, row in enumerate(rows_all):
        mid = str(row.get("member_id", ""))
        rows_by_member.setdefault(mid, []).append((idx, row))
    for m in members:
        mid = str(m.get("member_id", ""))
        seq = sorted(rows_by_member.get(mid, []), key=lambda t: (int(t[1].get("step_idx", 0)), int(t[1].get("line_index", 0))))
        if not seq:
            m["row_start"] = int(m.get("row_start", 0))
            m["row_end"] = int(m.get("row_start", 0))
            m["step_count"] = 0
            continue
        m["row_start"] = int(seq[0][0])
        m["row_end"] = int(seq[-1][0] + 1)
        m["step_count"] = int(len(seq))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out_path.with_suffix(out_path.suffix + ".rows.pt")
    save_pt(rows_path, rows_all)

    lexical_count = int(sum(1 for p in pairs if bool(p.get("lexical_control"))))
    contradiction_defined = [p for p in pairs if bool(p.get("behavioral_contradiction_defined"))]
    contradiction_rate = (
        float(sum(1 for p in contradiction_defined if bool(p.get("behavioral_contradiction_score", 0.0) >= 0.5)) / max(1, len(contradiction_defined)))
        if contradiction_defined
        else None
    )
    payload = {
        "schema_version": "phase7_optionc_paired_dataset_v1",
        "status": "ok",
        "model_key": str(model_key),
        "num_layers": int(num_layers or 0),
        "hidden_dim": int(hidden_dim or 0),
        "source": {
            "generator": "contradictory_pair_prepare.py#merge",
            "merged_from_shards": shard_meta,
            "merged_shard_count": int(len(shard_meta)),
            "scope": str(source_scope or "arithmetic"),
        },
        "rows_format": "pt",
        "rows_path": str(rows_path),
        "rows_inline": False,
        "rows_count": int(len(rows_all)),
        "rows_sha256": sha256_file(rows_path),
        "pairs_count": int(len(pairs)),
        "members_count": int(len(members)),
        "lexical_pairs_count": int(lexical_count),
        "behavioral_contradiction_defined_pairs": int(len(contradiction_defined)),
        "behavioral_contradiction_rate": contradiction_rate,
        "decoder_vocab_manifest": (merged_decoder_vocab_manifest or {"decoder_domain": str(source_scope or "arithmetic"), "unknown_id": 0}),
        "class_counts_member": {
            "faithful": int(sum(1 for m in members if str(m.get("gold_label")) == "faithful")),
            "unfaithful": int(sum(1 for m in members if str(m.get("gold_label")) == "unfaithful")),
            "pair_ambiguous": int(sum(1 for m in members if bool(m.get("pair_ambiguous")))),
        },
        "pairs": pairs,
        "members": members,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(out_path, payload)
    print(f"Merged {len(shard_meta)} shard payload(s) -> {out_path}")


def main() -> None:
    args = parse_args()
    if args.merge_shards:
        _merge_shard_payloads(args.merge_shards, args.output)
        return

    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >=1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError(f"--shard-index must be in [0, {int(args.num_shards)-1}]")

    set_seed(int(args.seed))
    spec = resolve_model_spec(str(args.model_key))
    adapter = create_adapter(model_key=spec.model_key, device=str(args.device)).load(device=str(args.device))

    pair_specs_all = _build_pair_specs(
        domain=str(args.domain),
        n_pairs=int(args.sample_pairs),
        seed=int(args.seed),
        lexical_fraction=float(args.lexical_fraction),
    )
    decoder_domain_labels = str(args.domain) if str(args.decoder_domain_labels) == "auto" else str(args.decoder_domain_labels)
    decoder_vocab_manifest = _build_decoder_vocab_manifest(str(decoder_domain_labels), pair_specs_all)
    if int(args.num_shards) > 1:
        pair_specs = [ps for i, ps in enumerate(pair_specs_all) if (int(i) % int(args.num_shards)) == int(args.shard_index)]
    else:
        pair_specs = pair_specs_all
    prompts: List[str] = []
    member_refs: List[Tuple[int, str]] = []  # (pair_idx, member a|b)
    for pi, ps in enumerate(pair_specs):
        prompts.append(str(ps.prompt_a))
        member_refs.append((int(pi), "a"))
        prompts.append(str(ps.prompt_b))
        member_refs.append((int(pi), "b"))

    gen_cfg = _gen_cfg_from_args(args)
    generated, gen_run_diag = _run_generation_for_members(adapter=adapter, prompts=prompts, cfg=gen_cfg)
    if len(generated) != len(prompts):
        raise RuntimeError("generation result count mismatch")

    rows: List[Dict[str, Any]] = []
    members: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []

    for pi, ps in enumerate(pair_specs):
        idx_a = int(2 * pi)
        idx_b = int(2 * pi + 1)
        ga = generated[idx_a]
        gb = generated[idx_b]
        pa = ga.get("parsed_final_answer")
        pb = gb.get("parsed_final_answer")
        corr_a = _is_correct_answer(
            answer=pa,
            expected=float(ps.expected_a),
            answer_mode=str(ps.answer_mode),
            tol=float(args.answer_tol),
        )
        corr_b = _is_correct_answer(
            answer=pb,
            expected=float(ps.expected_b),
            answer_mode=str(ps.answer_mode),
            tol=float(args.answer_tol),
        )
        labels, pair_ambiguous = _member_label_from_correctness(corr_a, corr_b)
        contradiction_flag, contradiction_diag = _pair_relation_contradiction(ps, pa, pb, tol=float(args.answer_tol))

        pair_id = str(ps.pair_id)
        pair_obj = {
            "pair_id": pair_id,
            "pair_type": str(ps.pair_type),
            "relation_type": str(ps.relation_type),
            "domain": str(ps.domain),
            "answer_mode": str(ps.answer_mode),
            "lexical_control": bool(ps.lexical_control),
            "a": int(ps.a),
            "b": int(ps.b),
            "c": int(ps.c),
            "member_ids": [f"{pair_id}_a", f"{pair_id}_b"],
            "member_correct": [bool(corr_a), bool(corr_b)],
            "member_labels": [int(labels[0]), int(labels[1])],
            "pair_ambiguous": bool(pair_ambiguous),
            "pair_any_unfaithful": bool(any(int(x) == 1 for x in labels)),
            "behavioral_contradiction_score": (float(bool(contradiction_flag)) if contradiction_flag is not None else None),
            "behavioral_contradiction_defined": bool(contradiction_flag is not None),
            "behavioral_contradiction_diag": contradiction_diag,
            "logic_meta": clone_jsonable(ps.logic_meta) if ps.logic_meta is not None else None,
        }
        pairs.append(pair_obj)

        for side, g, expected, corr, lbl in (
            ("a", ga, ps.expected_a, corr_a, labels[0]),
            ("b", gb, ps.expected_b, corr_b, labels[1]),
        ):
            member_id = f"{pair_id}_{side}"
            gen_text = str(g.get("generated_text", ""))
            gen_ids = [int(x) for x in list(g.get("generated_token_ids", []))]
            hidden_steps = list(g.get("hidden_by_generated_token", []))
            tok_ids, offsets, tok_diag = _tokenize_generated_with_offsets(adapter, gen_text)
            n_align = int(min(len(gen_ids), len(hidden_steps), len(tok_ids), len(offsets)))
            step_spans = list(g.get("step_spans", []))
            row_count_start = len(rows)

            for sp in step_spans:
                if n_align <= 0:
                    continue
                step_idx = int(sp.get("step_idx", 0))
                line_index = int(sp.get("line_index", step_idx))
                char_end = int(sp.get("char_end", 0))
                anchor_char = max(0, int(char_end - 1))
                tok_idx = _find_token_for_char(offsets[:n_align], anchor_char)
                if tok_idx is None or tok_idx < 0 or tok_idx >= n_align:
                    continue
                hidden = hidden_steps[int(tok_idx)]
                if not torch.is_tensor(hidden) or hidden.ndim != 2:
                    continue
                step_text = str(sp.get("text", ""))
                structured_state = {
                    "step_idx": int(step_idx),
                    "step_type": ("operate" if str(ps.domain) == "arithmetic" else "reason"),
                    "operator": "unknown",
                    "magnitude_bucket": "unknown",
                    "sign": "unknown",
                    "subresult_value": None,
                    "lhs_value": None,
                    "rhs_value": None,
                }
                if str(ps.domain).strip().lower() in {"prontoqa", "entailmentbank"}:
                    structured_state.update(
                        _infer_logical_structured_state(
                            pair_spec=ps,
                            step_text=step_text,
                            step_idx=int(step_idx),
                            member_correct=bool(corr),
                            vocab_manifest=decoder_vocab_manifest,
                        )
                    )
                rows.append(
                    {
                        "schema_version": "phase7_optionc_member_step_v1",
                        "trace_id": str(member_id),
                        "pair_id": str(pair_id),
                        "member_id": str(member_id),
                        "member_side": str(side),
                        "example_idx": int(pi),
                        "step_idx": int(step_idx),
                        "line_index": int(line_index),
                        "model_key": str(spec.model_key),
                        "model_family": str(spec.model_family),
                        "num_layers": int(spec.num_layers),
                        "hidden_dim": int(spec.hidden_dim),
                        "control_variant": ("lexical_consistent_swap" if bool(ps.lexical_control) else "contradictory_pair"),
                        "gold_label": ("unfaithful" if int(lbl) == 1 else "faithful"),
                        "anchor_abs": int(anchor_char),
                        "hidden_token_pos_0b": int(tok_idx),
                        "result_token_id": int(gen_ids[int(tok_idx)] if tok_idx < len(gen_ids) else -1),
                        "raw_hidden": hidden.half().cpu(),
                        "structured_state": structured_state,
                        "cot_source": "model_generated",
                        "domain": str(ps.domain),
                    }
                )

            row_count_end = len(rows)
            members.append(
                {
                    "member_id": str(member_id),
                    "pair_id": str(pair_id),
                    "member_side": str(side),
                    "pair_type": str(ps.pair_type),
                    "relation_type": str(ps.relation_type),
                    "domain": str(ps.domain),
                    "answer_mode": str(ps.answer_mode),
                    "lexical_control": bool(ps.lexical_control),
                    "prompt": str(g.get("prompt", "")),
                    "prompt_hash": hashlib.sha256(str(g.get("prompt", "")).encode("utf-8")).hexdigest(),
                    "generated_text": gen_text,
                    "generated_text_hash": hashlib.sha256(gen_text.encode("utf-8")).hexdigest(),
                    "generated_token_count": int(len(gen_ids)),
                    "captured_step_count": int(len(hidden_steps)),
                    "tokenized_generated_count": int(len(tok_ids)),
                    "token_alignment_diag": {
                        "offset_diag": tok_diag,
                        "align_count": int(n_align),
                        "generated_ids_match_retokenized_prefix": bool(
                            len(gen_ids) > 0 and len(tok_ids) > 0 and gen_ids[:n_align] == tok_ids[:n_align]
                        ),
                    },
                    "attempt": int(g.get("attempt", 0)),
                    "final_parse": clone_jsonable(g.get("final_parse", {})),
                    "parsed_final_answer": (float(g["parsed_final_answer"]) if isinstance(g.get("parsed_final_answer"), (int, float)) else None),
                    "expected_answer": float(expected),
                    "is_correct": bool(corr),
                    "gold_label": ("unfaithful" if int(lbl) == 1 else "faithful"),
                    "label_binary": int(lbl),
                    "label_defined": True,
                    "pair_ambiguous": bool(pair_ambiguous),
                    "row_start": int(row_count_start),
                    "row_end": int(row_count_end),
                    "step_count": int(max(0, row_count_end - row_count_start)),
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out_path.with_suffix(out_path.suffix + ".rows.pt")
    save_pt(rows_path, rows)

    lexical_count = int(sum(1 for p in pairs if bool(p.get("lexical_control"))))
    contradiction_defined = [p for p in pairs if bool(p.get("behavioral_contradiction_defined"))]
    contradiction_rate = (
        float(sum(1 for p in contradiction_defined if bool(p.get("behavioral_contradiction_score", 0.0) >= 0.5)) / max(1, len(contradiction_defined)))
        if contradiction_defined
        else None
    )
    payload = {
        "schema_version": "phase7_optionc_paired_dataset_v1",
        "status": "ok",
        "model_key": str(spec.model_key),
        "model_family": str(spec.model_family),
        "num_layers": int(spec.num_layers),
        "hidden_dim": int(spec.hidden_dim),
        "source": {
            "generator": "contradictory_pair_prepare.py",
            "seed": int(args.seed),
            "sample_pairs": int(args.sample_pairs),
            "lexical_fraction": float(args.lexical_fraction),
            "answer_tol": float(args.answer_tol),
            "decoder_domain_labels": str(decoder_domain_labels),
            "generation_config": gen_cfg,
            "generation_runtime_diag": gen_run_diag,
            "device": str(args.device),
            "scope": str(args.domain),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "global_pairs_count": int(len(pair_specs_all)),
            "local_pairs_count": int(len(pair_specs)),
            "worker_pid": int(os.getpid()),
        },
        "rows_format": "pt",
        "rows_path": str(rows_path),
        "rows_inline": False,
        "rows_count": int(len(rows)),
        "rows_sha256": sha256_file(rows_path),
        "pairs_count": int(len(pairs)),
        "members_count": int(len(members)),
        "lexical_pairs_count": int(lexical_count),
        "behavioral_contradiction_defined_pairs": int(len(contradiction_defined)),
        "behavioral_contradiction_rate": contradiction_rate,
        "decoder_vocab_manifest": decoder_vocab_manifest,
        "class_counts_member": {
            "faithful": int(sum(1 for m in members if str(m.get("gold_label")) == "faithful")),
            "unfaithful": int(sum(1 for m in members if str(m.get("gold_label")) == "unfaithful")),
            "pair_ambiguous": int(sum(1 for m in members if bool(m.get("pair_ambiguous")))),
        },
        "pairs": pairs,
        "members": members,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(out_path, payload)
    print(f"Saved Option C paired dataset -> {out_path}")


if __name__ == "__main__":
    main()

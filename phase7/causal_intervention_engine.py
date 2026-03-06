#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover
    from .common import (
        CAUSAL_PATCH_SPEC_SCHEMA,
        MAG_BUCKETS,
        OPERATORS,
        SIGNS,
        STEP_TYPES,
        load_json,
        load_rows_payload,
        save_pt,
        load_pt,
        save_json,
        sha256_file,
        set_seed,
        write_rows_sidecar,
    )
    from .model_adapters import BaseCausalLMAdapter
    from .control_token_anchor import collect_control_step_token_positions
    from .model_registry import create_adapter, resolve_model_spec
    from .state_decoder_core import load_model_from_checkpoint
except ImportError:  # pragma: no cover
    from common import (
        CAUSAL_PATCH_SPEC_SCHEMA,
        MAG_BUCKETS,
        OPERATORS,
        SIGNS,
        STEP_TYPES,
        load_json,
        load_rows_payload,
        save_pt,
        load_pt,
        save_json,
        sha256_file,
        set_seed,
        write_rows_sidecar,
    )
    from model_adapters import BaseCausalLMAdapter
    from control_token_anchor import collect_control_step_token_positions
    from model_registry import create_adapter, resolve_model_spec
    from state_decoder_core import load_model_from_checkpoint

# Phase4 helper module (SAE loaders + normalization stats for GPT-2 assets)
import phase4.causal_patch_test as p4  # type: ignore  # noqa: E402
from phase6.pipeline_utils import build_input_tensor_from_record  # type: ignore  # noqa: E402


def _record_result_token_id(rec: dict) -> int:
    return int(rec.get("result_token_id", rec["structured_state"]["result_token_id"]))


def _record_result_pos(rec: dict) -> int:
    if "result_tok_idx" not in rec:
        raise KeyError("missing result_tok_idx")
    pos_1based = int(rec["result_tok_idx"])
    if pos_1based <= 0:
        raise ValueError(f"invalid result_tok_idx={pos_1based}; expected 1-based positive token index")
    token_ids = rec.get("token_ids")
    if isinstance(token_ids, (list, tuple)) and token_ids:
        if pos_1based > len(token_ids):
            raise ValueError(
                f"result_tok_idx={pos_1based} exceeds token_ids length={len(token_ids)}"
            )
    return pos_1based - 1


def _record_eq_pos(rec: dict) -> int:
    if "eq_tok_idx" not in rec:
        raise KeyError("missing eq_tok_idx")
    pos_1based = int(rec["eq_tok_idx"])
    if pos_1based <= 0:
        raise ValueError(f"invalid eq_tok_idx={pos_1based}; expected 1-based positive token index")
    token_ids = rec.get("token_ids")
    if isinstance(token_ids, (list, tuple)) and token_ids:
        if pos_1based > len(token_ids):
            raise ValueError(
                f"eq_tok_idx={pos_1based} exceeds token_ids length={len(token_ids)}"
            )
    return pos_1based - 1


def _source_input_ids(rec: dict, device: str) -> torch.Tensor:
    return torch.tensor(rec["token_ids"], dtype=torch.long, device=device).unsqueeze(0)


def _tensor_shape_2d(x: object) -> Optional[tuple[int, int]]:
    if isinstance(x, torch.Tensor) and x.ndim == 2:
        return int(x.shape[0]), int(x.shape[1])
    return None


def _validate_records_model_compatibility(
    records: Sequence[dict],
    *,
    model_key: str,
    model_family: str,
    num_layers: int,
    hidden_dim: int,
    tokenizer_id: str,
) -> Dict[str, Any]:
    checks = {
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
            errors.append(f"record[{idx}] missing/invalid raw_hidden tensor")
            continue
        if raw_shape != (int(num_layers), int(hidden_dim)):
            checks["raw_hidden_shape_mismatch_records"] += 1
            errors.append(
                f"record[{idx}] raw_hidden shape={raw_shape} incompatible with model_key={model_key!r} "
                f"expected=({int(num_layers)}, {int(hidden_dim)})"
            )
        if "model_key" in r and str(r.get("model_key")) != str(model_key):
            checks["record_model_key_mismatch"] += 1
            errors.append(f"record[{idx}] model_key={r.get('model_key')!r} != {model_key!r}")
        if "model_family" in r and str(r.get("model_family")) != str(model_family):
            checks["record_model_family_mismatch"] += 1
            errors.append(f"record[{idx}] model_family={r.get('model_family')!r} != {model_family!r}")
        if "num_layers" in r:
            try:
                rec_layers = int(r.get("num_layers"))
            except Exception:
                rec_layers = None
            if rec_layers != int(num_layers):
                checks["record_num_layers_mismatch"] += 1
                errors.append(f"record[{idx}] num_layers={rec_layers!r} != {int(num_layers)}")
        if "hidden_dim" in r:
            try:
                rec_hidden = int(r.get("hidden_dim"))
            except Exception:
                rec_hidden = None
            if rec_hidden != int(hidden_dim):
                checks["record_hidden_dim_mismatch"] += 1
                errors.append(f"record[{idx}] hidden_dim={rec_hidden!r} != {int(hidden_dim)}")
        if "tokenizer_id" in r and str(r.get("tokenizer_id")) != str(tokenizer_id):
            checks["record_tokenizer_id_mismatch"] += 1
            errors.append(f"record[{idx}] tokenizer_id={r.get('tokenizer_id')!r} != {tokenizer_id!r}")
    checks["mismatch_errors_detected"] = int(len(errors))
    if errors:
        head = "\n".join(errors[:20])
        tail = "" if len(errors) <= 20 else f"\n... and {len(errors)-20} more mismatches"
        raise RuntimeError(
            "Causal intervention strict model/data compatibility check failed.\n"
            f"{head}{tail}"
        )
    return checks


def _validate_result_token_positions(records: Sequence[dict]) -> List[str]:
    errors: List[str] = []
    for idx, r in enumerate(records):
        try:
            _ = _record_result_pos(r)
        except Exception as exc:
            errors.append(f"record[{idx}] {exc}")
        try:
            _ = _record_eq_pos(r)
        except Exception as exc:
            errors.append(f"record[{idx}] {exc}")
    return errors


def _index_trace_step_records(records: Sequence[dict]) -> Dict[Tuple[str, int], dict]:
    idx: Dict[Tuple[str, int], dict] = {}
    for r in records:
        key = (str(r.get("trace_id", "")), int(r.get("step_idx", -1)))
        if key not in idx:
            idx[key] = r
    return idx


def _order_controls_for_sampling(
    controls: Sequence[dict],
    *,
    policy: str,
    rng: random.Random,
) -> List[dict]:
    ordered = list(controls)
    if policy == "random":
        rng.shuffle(ordered)
        return ordered

    if policy != "stratified_trace_variant":
        raise ValueError(f"unsupported control_sampling_policy={policy!r}")

    by_trace: Dict[str, List[dict]] = {}
    for c in ordered:
        by_trace.setdefault(str(c.get("trace_id", "")), []).append(c)

    trace_ids = list(by_trace.keys())
    rng.shuffle(trace_ids)
    head: List[dict] = []
    tail: List[dict] = []
    for tid in trace_ids:
        group = list(by_trace.get(tid, []))
        rng.shuffle(group)
        faithful = [c for c in group if str(c.get("gold_label")) == "faithful"]
        unfaithful = [c for c in group if str(c.get("gold_label")) == "unfaithful"]
        other = [c for c in group if str(c.get("gold_label")) not in {"faithful", "unfaithful"}]

        if faithful:
            head.append(faithful.pop())
        if unfaithful:
            head.append(unfaithful.pop())

        remainder = faithful + unfaithful + other
        rng.shuffle(remainder)
        tail.extend(remainder)

    return head + tail


def _build_control_conditioned_records(
    controls_payload: dict,
    trace_records: Sequence[dict],
    *,
    adapter: BaseCausalLMAdapter,
    parse_mode: str,
    token_anchor: str,
    anchor_priority: str,
    max_records: int,
    seed: int,
    control_sampling_policy: str,
    model_key: str,
    model_family: str,
    num_layers: int,
    hidden_dim: int,
    tokenizer_id: str,
) -> Dict[str, Any]:
    controls = list(controls_payload.get("controls", []) or [])
    rng = random.Random(int(seed))
    controls = _order_controls_for_sampling(
        controls,
        policy=str(control_sampling_policy),
        rng=rng,
    )

    trace_step_idx = _index_trace_step_records(trace_records)
    built: List[dict] = []
    stats = {
        "controls_total": int(len(controls)),
        "controls_used": 0,
        "controls_missing_text": 0,
        "controls_no_step_positions": 0,
        "controls_no_matching_trace_steps": 0,
        "records_built": 0,
        "records_skipped_missing_trace_step": 0,
        "anchor_coverage_eq_like_rows": 0,
        "anchor_coverage_line_end_rows": 0,
        "anchor_coverage_fallback_rows": 0,
        "anchor_coverage_total_rows": 0,
        "position_convention_version": "phase7_pos_contract_v1",
        "position_contract_validated": True,
        "offset_alignment_degraded_rows": 0,
        "control_sampling_policy": str(control_sampling_policy),
    }

    for ctrl in controls:
        cot_text = str(ctrl.get("cot_text", "")).strip()
        if not cot_text:
            stats["controls_missing_text"] += 1
            continue
        trace_id = str(ctrl.get("trace_id"))
        variant = str(ctrl.get("variant", ctrl.get("control_variant", "unknown")))
        example_idx = int(ctrl.get("example_idx", -1))

        pos_payload = collect_control_step_token_positions(
            ctrl,
            adapter,
            parse_mode=parse_mode,
            token_anchor=token_anchor,
            anchor_priority=anchor_priority,
        )
        step_rows = list(pos_payload.get("rows", []))
        cov = dict(pos_payload.get("anchor_coverage", {}) or {})
        stats["anchor_coverage_eq_like_rows"] += int(cov.get("eq_like_rows", 0))
        stats["anchor_coverage_line_end_rows"] += int(cov.get("line_end_rows", 0))
        stats["anchor_coverage_fallback_rows"] += int(cov.get("fallback_rows", 0))
        stats["anchor_coverage_total_rows"] += int(cov.get("total_rows", 0))
        for sp in step_rows:
            if bool(sp.get("offset_alignment_degraded", False)):
                stats["offset_alignment_degraded_rows"] += 1
        if not step_rows:
            stats["controls_no_step_positions"] += 1
            continue

        logits, hidden_states = adapter.forward(cot_text)
        if not hidden_states:
            raise RuntimeError("adapter.forward returned empty hidden_states in control-conditioned mode")
        if len(hidden_states) != int(num_layers):
            raise RuntimeError(
                "Control-conditioned hidden-state depth mismatch: "
                f"adapter_layers={len(hidden_states)} expected={num_layers}"
            )
        seq_len = int(logits.shape[1])
        input_ids = adapter.tokenize(cot_text)[0].detach().cpu().tolist()

        matched_any = False
        for sp in step_rows:
            step_idx = int(sp.get("step_idx", -1))
            base = trace_step_idx.get((trace_id, step_idx))
            if base is None:
                stats["records_skipped_missing_trace_step"] += 1
                continue
            matched_any = True
            tok_pos_0b = int(sp.get("hidden_token_pos_0b", sp.get("token_pos", 0)))
            eq_pos_0b = int(sp.get("eq_token_pos_0b", sp.get("eq_token_pos", tok_pos_0b)))
            result_pos_0b = int(sp.get("result_token_pos_0b", sp.get("result_token_pos", min(tok_pos_0b + 1, seq_len - 1))))
            eq_idx_1b = int(sp.get("eq_tok_idx_1b", eq_pos_0b + 1))
            result_idx_1b = int(sp.get("result_tok_idx_1b", result_pos_0b + 1))
            if eq_idx_1b != eq_pos_0b + 1:
                raise RuntimeError(
                    f"position contract violation: eq_tok_idx_1b={eq_idx_1b} != eq_token_pos_0b+1={eq_pos_0b + 1}"
                )
            if result_idx_1b != result_pos_0b + 1:
                raise RuntimeError(
                    "position contract violation: "
                    f"result_tok_idx_1b={result_idx_1b} != result_token_pos_0b+1={result_pos_0b + 1}"
                )
            if token_anchor == "eq_like" and tok_pos_0b != eq_pos_0b:
                raise RuntimeError(
                    "position contract violation for eq_like: "
                    f"hidden_token_pos_0b={tok_pos_0b} != eq_token_pos_0b={eq_pos_0b}"
                )
            tok_pos = max(0, min(int(tok_pos_0b), seq_len - 1))
            eq_pos = max(0, min(int(eq_pos_0b), seq_len - 1))
            result_pos = max(0, min(int(result_pos_0b), seq_len - 1))
            raw_hidden = torch.stack(
                [hidden_states[layer_i][0, tok_pos, :].detach().float().cpu() for layer_i in range(len(hidden_states))],
                dim=0,
            )
            rec = {
                "trace_id": trace_id,
                "control_variant": variant,
                "gold_label": ctrl.get("gold_label"),
                "example_idx": int(base.get("example_idx", example_idx)),
                "step_idx": int(step_idx),
                "structured_state": base.get("structured_state"),
                "token_ids": input_ids,
                "eq_tok_idx": int(eq_pos + 1),  # backward-compatible alias (1-based)
                "result_tok_idx": int(result_pos + 1),  # backward-compatible alias (1-based)
                "hidden_token_pos_0b": int(tok_pos),
                "eq_token_pos_0b": int(eq_pos),
                "result_token_pos_0b": int(result_pos),
                "eq_tok_idx_1b": int(eq_pos + 1),
                "result_tok_idx_1b": int(result_pos + 1),
                "position_convention_version": "phase7_pos_contract_v1",
                "result_token_id": int((base.get("structured_state") or {}).get("result_token_id", base.get("result_token_id", -1))),
                "raw_hidden": raw_hidden,
                "model_key": model_key,
                "model_family": model_family,
                "num_layers": int(num_layers),
                "hidden_dim": int(hidden_dim),
                "tokenizer_id": tokenizer_id,
                "token_anchor_mode": str(sp.get("token_anchor_mode", token_anchor)),
                "token_anchor_reason": str(sp.get("token_anchor_reason", "unknown")),
                "selected_anchor_rule": str(sp.get("selected_anchor_rule", "unknown")),
                "anchor_candidate_matches": list(sp.get("anchor_candidate_matches", [])),
                "anchor_char_index": sp.get("anchor_char_index"),
                "special_tokens_policy": str(sp.get("special_tokens_policy", "unknown")),
                "num_special_tokens_prefix": int(sp.get("num_special_tokens_prefix", 0)),
                "offset_alignment_degraded": bool(sp.get("offset_alignment_degraded", False)),
                "line_index": sp.get("line_index"),
                "schema_version": "phase7_control_conditioned_record_v1",
            }
            built.append(rec)
            if len(built) >= int(max_records):
                break
        if matched_any:
            stats["controls_used"] += 1
        else:
            stats["controls_no_matching_trace_steps"] += 1
        if len(built) >= int(max_records):
            break

    stats["records_built"] = int(len(built))
    stats["controls_used_fraction"] = (
        float(stats["controls_used"] / max(1, stats["controls_total"]))
        if stats["controls_total"] > 0
        else 0.0
    )
    stats["unique_traces_used"] = int(len({str(r.get("trace_id", "")) for r in built}))
    stats["unique_variants_used"] = int(len({str(r.get("control_variant", "")) for r in built}))
    stats["faithful_rows"] = int(sum(1 for r in built if str(r.get("gold_label")) == "faithful"))
    stats["unfaithful_rows"] = int(sum(1 for r in built if str(r.get("gold_label")) == "unfaithful"))
    if not built:
        raise RuntimeError("Control-conditioned mode produced zero records; cannot run causal checks.")
    return {"records": built, "stats": stats}


def _summarize_control_records_prefix(
    records: Sequence[dict],
    stats_base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = dict(stats_base or {})
    rows = list(records)
    controls_total = int(out.get("controls_total", 0) or 0)
    used_keys = {
        (
            str(r.get("trace_id", "")),
            str(r.get("control_variant", "")),
            int(r.get("example_idx", -1)),
        )
        for r in rows
    }
    out["records_built"] = int(len(rows))
    out["controls_used"] = int(len(used_keys))
    out["controls_used_fraction"] = float(len(used_keys) / max(1, controls_total)) if controls_total > 0 else 0.0
    out["unique_traces_used"] = int(len({str(r.get("trace_id", "")) for r in rows}))
    out["unique_variants_used"] = int(len({str(r.get("control_variant", "")) for r in rows}))
    out["faithful_rows"] = int(sum(1 for r in rows if str(r.get("gold_label")) == "faithful"))
    out["unfaithful_rows"] = int(sum(1 for r in rows if str(r.get("gold_label")) == "unfaithful"))
    return out


def _next_max_records_stage(
    current_max_records: int,
    *,
    coverage_fraction: Optional[float],
    controls_used_fraction: Optional[float],
    target_controls_used_fraction: float,
    max_records_cap: int,
) -> Optional[int]:
    """Deterministic escalation helper for run orchestration."""
    cur = int(max(1, current_max_records))
    cap = int(max(1, max_records_cap))
    target = float(target_controls_used_fraction)
    cov_ok = isinstance(coverage_fraction, (int, float)) and float(coverage_fraction) >= target
    ctr_ok = isinstance(controls_used_fraction, (int, float)) and float(controls_used_fraction) >= target
    if cov_ok and ctr_ok:
        return None
    stages = sorted(set([1200, 2200, 6000, cap]))
    for s in stages:
        if s > cur and s <= cap:
            return int(s)
    return None


def _record_identity_key(rec: dict) -> Tuple[str, int, int]:
    return (
        str(rec.get("trace_id", "")),
        int(rec.get("step_idx", -1)),
        int(rec.get("example_idx", -1)),
    )


def _causal_row_identity_key(row: Dict[str, Any], *, variable: str) -> Tuple[str, str, int, str]:
    return (
        str(row.get("source_trace_id", "")),
        str(row.get("source_control_variant", "")),
        int(row.get("source_step_idx", -1)),
        str(variable),
    )


def _save_control_records_artifact(
    output_path: str,
    *,
    records: Sequence[dict],
    stats: Dict[str, Any],
    model_spec: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out_path = Path(output_path)
    rows_path = out_path.with_suffix(out_path.suffix + ".rows.pt")
    save_pt(rows_path, list(records))
    rows_payload = {
        "rows": [],
        "rows_inline": False,
        "rows_path": str(rows_path),
        "rows_format": "pt",
        "rows_count": int(len(records)),
        "rows_sha256": sha256_file(rows_path),
    }
    payload = {
        "schema_version": "phase7_control_conditioned_records_v1",
        "status": "ok",
        "model_metadata": {
            "model_key": str(model_spec.model_key),
            "model_family": str(model_spec.model_family),
            "num_layers": int(model_spec.num_layers),
            "hidden_dim": int(model_spec.hidden_dim),
            "tokenizer_id": str(model_spec.tokenizer_id),
        },
        "controls_source": str(args.controls),
        "controls_source_sha256": (sha256_file(args.controls) if args.controls else None),
        "trace_dataset": str(args.trace_dataset),
        "trace_dataset_sha256": sha256_file(args.trace_dataset),
        "parse_mode": str(args.parse_mode),
        "token_anchor": str(args.token_anchor),
        "anchor_priority": str(args.anchor_priority),
        "control_sampling_policy": str(args.control_sampling_policy),
        "seed": int(args.seed),
        "max_records_cap": int(args.max_records_cap),
        "rows_schema": "phase7_control_conditioned_record_v1",
        "stats": dict(stats),
        **rows_payload,
    }
    save_json(output_path, payload)
    return payload


def _load_control_records_artifact(path: str) -> Tuple[List[dict], Dict[str, Any]]:
    payload = load_json(path)
    rows: List[dict]
    if str(payload.get("rows_format", "")).lower() == "pt":
        rows_path = payload.get("rows_path")
        if not rows_path:
            raise RuntimeError(f"control-records artifact missing rows_path: {path}")
        rp = Path(str(rows_path))
        if not rp.is_absolute():
            # Prefer artifact-relative resolution for short sidecar names, but
            # fall back to workspace-relative paths (e.g. "phase7_results/...").
            rel_candidate = (Path(path).parent / rp).resolve()
            if rel_candidate.exists():
                rp = rel_candidate
            else:
                rp = rp.resolve()
        rows = list(load_pt(rp))
    else:
        rows = load_rows_payload(payload, base_path=path)
    return rows, payload


def _feature_indices_for(specs_payload: dict, variable: str, layer: int) -> List[int]:
    if "specs" in specs_payload:
        for s in specs_payload["specs"]:
            if str(s.get("variable")) == str(variable) and int(s.get("layer")) == int(layer):
                return [int(x) for x in s.get("feature_indices", [])]
    if str(layer) in specs_payload:
        vals = specs_payload[str(layer)]
        return [int(x) for x in vals]
    return []


def _select_control_features(k: int, exclude: Sequence[int], pool_size: int, seed: int = 17) -> List[int]:
    if pool_size <= 0:
        return []
    rng = random.Random(seed)
    exclude_set = set(int(x) for x in exclude)
    pool = [i for i in range(pool_size) if i not in exclude_set]
    rng.shuffle(pool)
    return pool[: min(k, len(pool))]


def _build_subspace_decoder_cols(sae, feat_idx: Sequence[int], device: str) -> torch.Tensor:
    idx = torch.tensor(list(feat_idx), dtype=torch.long, device=device)
    D = sae.decoder.weight[:, idx].float()  # (hidden, k)
    return F.normalize(D, p=2, dim=0)


def _logprob_at_token(logits: torch.Tensor, pos: int, token_id: int) -> float:
    pos = int(pos)
    if pos < 0 or pos >= int(logits.shape[1]):
        raise ValueError(f"token position out of bounds: pos={pos}, seq_len={int(logits.shape[1])}")
    lp = F.log_softmax(logits[0, pos, :], dim=-1)
    return float(lp[int(token_id)].item())


def necessity_ablation(
    source_rec: dict,
    layer: int,
    feature_indices: Sequence[int],
    ctx: "CausalPatchContext",
) -> Dict[str, Any]:
    if not feature_indices:
        return {"supported": False, "reason": "empty_feature_indices"}
    if layer not in ctx.saes:
        return {"supported": False, "reason": f"missing_sae_for_layer_{layer}"}

    src_ids = _source_input_ids(source_rec, ctx.device)
    eq_pos = _record_eq_pos(source_rec)
    pred_pos = _record_result_pos(source_rec)
    correct_tok = _record_result_token_id(source_rec)

    logits_base, _ = ctx.adapter.forward(src_ids)
    lp_base = _logprob_at_token(logits_base, pred_pos, correct_tok)

    h_src = source_rec["raw_hidden"][layer].to(ctx.device).float()
    D_norm = _build_subspace_decoder_cols(ctx.saes[layer], feature_indices, ctx.device)
    coords = D_norm.T @ h_src
    comp = D_norm @ coords
    patch_vec = h_src - comp
    off_ratio = float(comp.norm().item() / max(1e-6, h_src.norm().item()))

    logits_ablate = ctx.adapter.patch_forward(layer=int(layer), token_pos=eq_pos, patch_vector=patch_vec, input_ids=src_ids)
    lp_ablate = _logprob_at_token(logits_ablate, pred_pos, correct_tok)
    return {
        "supported": True,
        "baseline_logprob": lp_base,
        "patched_logprob": lp_ablate,
        "delta_logprob": lp_ablate - lp_base,
        "off_manifold_ratio": off_ratio,
    }


def sufficiency_patch(
    source_rec: dict,
    donor_rec: dict,
    layer: int,
    feature_indices: Sequence[int],
    ctx: "CausalPatchContext",
    control_feature_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if not feature_indices:
        return {"supported": False, "reason": "empty_feature_indices"}
    if layer not in ctx.saes:
        return {"supported": False, "reason": f"missing_sae_for_layer_{layer}"}

    src_ids = _source_input_ids(source_rec, ctx.device)
    eq_pos = _record_eq_pos(source_rec)
    pred_pos = _record_result_pos(source_rec)
    donor_tok = _record_result_token_id(donor_rec)

    logits_base, _ = ctx.adapter.forward(src_ids)
    lp_base = _logprob_at_token(logits_base, pred_pos, donor_tok)

    h_src = source_rec["raw_hidden"][layer].to(ctx.device).float()
    h_don = donor_rec["raw_hidden"][layer].to(ctx.device).float()
    delta_h = h_don - h_src

    D_norm = _build_subspace_decoder_cols(ctx.saes[layer], feature_indices, ctx.device)
    projected = D_norm @ (D_norm.T @ delta_h)
    patch_vec = h_src + projected
    off_ratio = float(projected.norm().item() / max(1e-6, h_src.norm().item()))

    logits_patch = ctx.adapter.patch_forward(layer=int(layer), token_pos=eq_pos, patch_vector=patch_vec, input_ids=src_ids)
    lp_patch = _logprob_at_token(logits_patch, pred_pos, donor_tok)

    control_gain = None
    if control_feature_indices:
        Dc = _build_subspace_decoder_cols(ctx.saes[layer], control_feature_indices, ctx.device)
        proj_c = Dc @ (Dc.T @ delta_h)
        patch_vec_c = h_src + proj_c
        logits_c = ctx.adapter.patch_forward(layer=int(layer), token_pos=eq_pos, patch_vector=patch_vec_c, input_ids=src_ids)
        lp_c = _logprob_at_token(logits_c, pred_pos, donor_tok)
        control_gain = lp_c - lp_base

    return {
        "supported": True,
        "baseline_logprob_for_donor_token": lp_base,
        "patched_logprob_for_donor_token": lp_patch,
        "delta_logprob": lp_patch - lp_base,
        "specificity_control_delta_logprob": control_gain,
        "off_manifold_ratio": off_ratio,
    }


def select_matched_donor(records: Sequence[dict], source: dict, variable: str, seed: int = 17) -> Optional[dict]:
    rng = random.Random(seed + int(source.get("example_idx", 0)) + int(source.get("step_idx", 0)))
    src_state = source["structured_state"]
    src_key = _record_identity_key(source)
    candidates = []
    for r in records:
        if _record_identity_key(r) == src_key:
            continue
        s = r["structured_state"]
        if s.get("step_type") != src_state.get("step_type"):
            continue
        if s.get("operator") != src_state.get("operator"):
            continue
        if s.get("magnitude_bucket") != src_state.get("magnitude_bucket"):
            continue
        if int(r.get("example_idx", -1)) == int(source.get("example_idx", -2)):
            continue
        if variable == "subresult_value":
            try:
                if math.isclose(
                    float(s.get("subresult_value")),
                    float(src_state.get("subresult_value")),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    continue
            except Exception:
                continue
        candidates.append(r)
    if not candidates:
        for r in records:
            if _record_identity_key(r) == src_key:
                continue
            s = r["structured_state"]
            if s.get("step_type") != src_state.get("step_type") or s.get("operator") != src_state.get("operator"):
                continue
            if int(r.get("example_idx", -1)) == int(source.get("example_idx", -2)):
                continue
            candidates.append(r)
    if not candidates:
        return None
    return rng.choice(candidates)


@dataclass
class CausalPatchContext:
    device: str
    model_key: str
    model_family: str
    tokenizer_id: str
    num_layers: int
    hidden_dim: int
    adapter: Optional[BaseCausalLMAdapter]
    saes: Dict[int, Any]
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    latent_dim: int
    resolved_saes_dir: Optional[str] = None
    resolved_activations_dir: Optional[str] = None
    unsupported_reason: Optional[str] = None

    @property
    def supports_subspace_patching(self) -> bool:
        return bool(self.saes)

    @classmethod
    def load(
        cls,
        device: str,
        model_key: str = "gpt2-medium",
        adapter_config: Optional[str] = None,
        saes_dir: Optional[str] = None,
        activations_dir: Optional[str] = None,
        load_model: bool = True,
    ) -> "CausalPatchContext":
        spec = resolve_model_spec(model_key, adapter_config)
        adapter = None
        if load_model:
            adapter = create_adapter(model_key=spec.model_key, device=device, adapter_config=adapter_config).load(device=device)

        resolved_saes_dir = saes_dir if saes_dir is not None else spec.sae_dir
        resolved_activations_dir = activations_dir
        if resolved_activations_dir is None and spec.model_key == "gpt2-medium":
            resolved_activations_dir = "phase2_results/activations"

        saes: Dict[int, Any] = {}
        norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        unsupported_reason: Optional[str] = None
        if resolved_saes_dir:
            if spec.model_key == "gpt2-medium":
                saes = p4.load_saes(Path(resolved_saes_dir), device)
            else:
                unsupported_reason = f"sae_loader_unimplemented_for_model:{spec.model_key}"
        else:
            unsupported_reason = unsupported_reason or f"missing_model_sae_dir:{spec.model_key}"
        if resolved_activations_dir:
            if spec.model_key == "gpt2-medium":
                norm_stats = p4.load_norm_stats(Path(resolved_activations_dir), device)

        latent_dim = int(next(iter(saes.values())).decoder.weight.shape[1]) if saes else 0
        return cls(
            device=device,
            model_key=spec.model_key,
            model_family=spec.model_family,
            tokenizer_id=spec.tokenizer_id,
            num_layers=int(spec.num_layers),
            hidden_dim=int(spec.hidden_dim),
            adapter=adapter,
            saes=saes,
            norm_stats=norm_stats,
            latent_dim=latent_dim,
            resolved_saes_dir=resolved_saes_dir,
            resolved_activations_dir=resolved_activations_dir,
            unsupported_reason=unsupported_reason,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_family": self.model_family,
            "num_layers": int(self.num_layers),
            "hidden_dim": int(self.hidden_dim),
            "tokenizer_id": self.tokenizer_id,
            "latent_dim": int(self.latent_dim),
            "supports_subspace_patching": bool(self.supports_subspace_patching),
            "resolved_saes_dir": self.resolved_saes_dir,
            "resolved_activations_dir": self.resolved_activations_dir,
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass
class MediationContext:
    enabled: bool
    variable: str
    device: str
    model: Optional[torch.nn.Module] = None
    cfg: Optional[Any] = None
    numeric_stats: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None
    reason: Optional[str] = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[str],
        *,
        variable: str,
        device: str,
        expected_model_key: str,
        force_enable: bool,
    ) -> "MediationContext":
        if not force_enable:
            return cls(enabled=False, variable=variable, device=device, reason="disabled_by_flag")
        if checkpoint_path is None:
            return cls(enabled=False, variable=variable, device=device, reason="missing_state_decoder_checkpoint")
        ckpt, cfg, numeric_stats, model = load_model_from_checkpoint(checkpoint_path, device)
        cfg_model_key = str(getattr(cfg, "model_key", "gpt2-medium"))
        if cfg_model_key != str(expected_model_key):
            raise RuntimeError(
                "Mediation checkpoint model mismatch: "
                f"checkpoint model_key={cfg_model_key!r} vs run model_key={expected_model_key!r}"
            )
        return cls(
            enabled=True,
            variable=variable,
            device=device,
            model=model,
            cfg=cfg,
            numeric_stats=numeric_stats,
            checkpoint_path=str(checkpoint_path),
            reason=None,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "variable": self.variable,
            "checkpoint_path": self.checkpoint_path,
            "reason": self.reason,
        }


def _rec_with_raw_layer_patch(source_rec: dict, layer: int, patch_vec: torch.Tensor) -> dict:
    row = dict(source_rec)
    raw = source_rec["raw_hidden"].clone()
    raw[int(layer)] = patch_vec.detach().cpu().float()
    row["raw_hidden"] = raw
    # Patched hidden states invalidate any cached SAE-derived features.
    row.pop("sae_features", None)
    return row


def _normalize_hidden_for_mediation(
    h: torch.Tensor,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if stats is None:
        return h
    mean, std = stats
    return (h - mean) / std


def _derive_sae_features_from_raw_hidden(
    row: dict,
    *,
    ctx: CausalPatchContext,
    cfg: Any,
) -> torch.Tensor:
    raw_hidden = row.get("raw_hidden")
    if not isinstance(raw_hidden, torch.Tensor) or raw_hidden.ndim != 2:
        raise RuntimeError("mediation decode requires raw_hidden tensor with shape (layers, hidden_dim)")
    if not ctx.supports_subspace_patching:
        raise RuntimeError(
            "mediation decode requires SAE-derived features, but subspace patching assets are unavailable: "
            f"{ctx.unsupported_reason or 'saes_unavailable_for_model'}"
        )

    sae_rows: List[torch.Tensor] = []
    for layer_i in range(int(raw_hidden.shape[0])):
        sae = ctx.saes.get(int(layer_i))
        if sae is None:
            raise RuntimeError(f"missing_sae_for_layer_{layer_i} while deriving mediation features")
        h = raw_hidden[layer_i].to(ctx.device).float()
        h_norm = _normalize_hidden_for_mediation(h, ctx.norm_stats.get(int(layer_i)))
        feat = sae.encode(h_norm.unsqueeze(0)).squeeze(0).detach().float().cpu()
        sae_rows.append(feat)
    sae_features = torch.stack(sae_rows, dim=0)
    row["sae_features"] = sae_features
    return sae_features


def _build_decoder_input_for_mediation(
    row: dict,
    *,
    mctx: MediationContext,
    ctx: CausalPatchContext,
) -> Tuple[torch.Tensor, str]:
    if mctx.cfg is None:
        raise RuntimeError("mediation decode missing checkpoint config")
    cfg = mctx.cfg
    input_variant = str(getattr(cfg, "input_variant", "raw"))
    if input_variant == "raw":
        x = build_input_tensor_from_record(row, cfg)
        return x, "raw"

    if input_variant not in {"sae", "hybrid", "hybrid_indexed"}:
        raise RuntimeError(f"unsupported mediation input_variant={input_variant!r}")

    if "sae_features" not in row:
        _derive_sae_features_from_raw_hidden(row, ctx=ctx, cfg=cfg)
        source = "sae_derived_from_raw_hidden"
    else:
        source = "row_sae_features"

    x = build_input_tensor_from_record(row, cfg)
    return x, source


def _decode_decoder_variable(row: dict, mctx: MediationContext, ctx: CausalPatchContext) -> Tuple[Any, str]:
    if not mctx.enabled or mctx.model is None or mctx.cfg is None:
        return None, "disabled"
    x_cpu, feature_source = _build_decoder_input_for_mediation(row, mctx=mctx, ctx=ctx)
    x = x_cpu.unsqueeze(0).to(mctx.device)
    with torch.no_grad():
        out = mctx.model(x)
    var = str(mctx.variable)
    if var == "result_token_id":
        return int(out["result_token_logits"].argmax(dim=-1)[0].item()), feature_source
    if var == "operator":
        idx = int(out["operator_logits"].argmax(dim=-1)[0].item())
        return (OPERATORS[idx] if 0 <= idx < len(OPERATORS) else "unknown"), feature_source
    if var == "magnitude_bucket":
        idx = int(out["magnitude_logits"].argmax(dim=-1)[0].item())
        return (MAG_BUCKETS[idx] if 0 <= idx < len(MAG_BUCKETS) else "[1000+)"), feature_source
    if var == "sign":
        idx = int(out["sign_logits"].argmax(dim=-1)[0].item())
        return (SIGNS[idx] if 0 <= idx < len(SIGNS) else "zero"), feature_source
    if var == "step_type":
        idx = int(out["step_type_logits"].argmax(dim=-1)[0].item())
        return (STEP_TYPES[idx] if 0 <= idx < len(STEP_TYPES) else "operate"), feature_source
    if mctx.numeric_stats is None:
        return None, feature_source
    if var == "subresult_value":
        z = float(out["subresult_pred"][0].item())
        st = mctx.numeric_stats["subresult_value"]
        return float(z * st.std + st.mean), feature_source
    if var == "lhs_value":
        z = float(out["lhs_pred"][0].item())
        st = mctx.numeric_stats["lhs_value"]
        return float(z * st.std + st.mean), feature_source
    if var == "rhs_value":
        z = float(out["rhs_pred"][0].item())
        st = mctx.numeric_stats["rhs_value"]
        return float(z * st.std + st.mean), feature_source
    return None, feature_source


def _is_numeric_variable(variable: str) -> bool:
    return variable in {"subresult_value", "lhs_value", "rhs_value"}


def _latent_shift_score(pre_value: Any, post_value: Any, variable: str) -> Optional[float]:
    if pre_value is None or post_value is None:
        return None
    if _is_numeric_variable(variable):
        try:
            return float(abs(float(post_value) - float(pre_value)))
        except Exception:
            return None
    return 0.0 if str(post_value) == str(pre_value) else 1.0


def _latent_direction_match(pre_value: Any, post_value: Any, target_value: Any, variable: str) -> Optional[bool]:
    if pre_value is None or post_value is None or target_value is None:
        return None
    if _is_numeric_variable(variable):
        try:
            pre_d = abs(float(pre_value) - float(target_value))
            post_d = abs(float(post_value) - float(target_value))
        except Exception:
            return None
        return bool(post_d + 1e-8 < pre_d)
    return bool(str(post_value) == str(target_value))


def _target_value_for_variable(rec: dict, variable: str) -> Any:
    st = rec.get("structured_state", {})
    if variable == "result_token_id":
        return int(st.get("result_token_id", rec.get("result_token_id", -1)))
    return st.get(variable)


def _compute_need_suff_patch_vectors(
    source_rec: dict,
    donor_rec: Optional[dict],
    layer: int,
    feature_indices: Sequence[int],
    ctx: CausalPatchContext,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not feature_indices or layer not in ctx.saes:
        return None, None
    h_src = source_rec["raw_hidden"][layer].to(ctx.device).float()
    D_norm = _build_subspace_decoder_cols(ctx.saes[layer], feature_indices, ctx.device)
    coords = D_norm.T @ h_src
    comp = D_norm @ coords
    need_patch_vec = h_src - comp
    suff_patch_vec = None
    if donor_rec is not None:
        h_don = donor_rec["raw_hidden"][layer].to(ctx.device).float()
        delta_h = h_don - h_src
        projected = D_norm @ (D_norm.T @ delta_h)
        suff_patch_vec = h_src + projected
    return need_patch_vec, suff_patch_vec


def _compute_layer_mediation(
    *,
    source_rec: dict,
    donor_rec: Optional[dict],
    layer: int,
    feature_indices: Sequence[int],
    ctx: CausalPatchContext,
    mctx: Optional[MediationContext],
) -> Dict[str, Any]:
    if mctx is None or not mctx.enabled:
        return {
            "supported": False,
            "reason": (mctx.reason if mctx is not None else "no_mediation_context"),
            "latent_shift_score": None,
            "direction_match": None,
            "pass": None,
        }
    if int(layer) >= source_rec["raw_hidden"].shape[0]:
        return {
            "supported": False,
            "reason": f"layer_out_of_range_for_record:{layer}",
            "latent_shift_score": None,
            "direction_match": None,
            "pass": None,
        }
    need_patch, suff_patch = _compute_need_suff_patch_vectors(
        source_rec=source_rec,
        donor_rec=donor_rec,
        layer=layer,
        feature_indices=feature_indices,
        ctx=ctx,
    )
    if need_patch is None:
        return {
            "supported": False,
            "reason": "missing_patch_vectors_for_mediation",
            "latent_shift_score": None,
            "direction_match": None,
            "pass": None,
        }

    pre_val, pre_src = _decode_decoder_variable(source_rec, mctx, ctx)
    need_row = _rec_with_raw_layer_patch(source_rec, layer, need_patch)
    post_need_val, need_src = _decode_decoder_variable(need_row, mctx, ctx)
    need_shift = _latent_shift_score(pre_val, post_need_val, mctx.variable)
    need_pass = None if need_shift is None else bool(float(need_shift) > 1e-6)

    post_suff_val = None
    suff_src = None
    suff_shift = None
    direction_match = None
    suff_pass = None
    if suff_patch is not None and donor_rec is not None:
        suff_row = _rec_with_raw_layer_patch(source_rec, layer, suff_patch)
        post_suff_val, suff_src = _decode_decoder_variable(suff_row, mctx, ctx)
        suff_shift = _latent_shift_score(pre_val, post_suff_val, mctx.variable)
        target_val = _target_value_for_variable(donor_rec, mctx.variable)
        direction_match = _latent_direction_match(pre_val, post_suff_val, target_val, mctx.variable)
        suff_pass = direction_match

    if need_pass is None and suff_pass is None:
        mediation_pass = None
    elif need_pass is None:
        mediation_pass = bool(suff_pass)
    elif suff_pass is None:
        mediation_pass = bool(need_pass)
    else:
        mediation_pass = bool(need_pass and suff_pass)

    combined_shift = None
    shifts = [x for x in [need_shift, suff_shift] if isinstance(x, (int, float))]
    if shifts:
        combined_shift = float(max(shifts))
    return {
        "supported": True,
        "variable": mctx.variable,
        "pre_value": pre_val,
        "post_necessity_value": post_need_val,
        "post_sufficiency_value": post_suff_val,
        "latent_shift_score": combined_shift,
        "latent_shift_score_necessity": need_shift,
        "latent_shift_score_sufficiency": suff_shift,
        "direction_match": direction_match,
        "necessity_mediation_pass": need_pass,
        "sufficiency_mediation_pass": suff_pass,
        "pass": mediation_pass,
        "mediation_decode_feature_source": (
            pre_src if pre_src == need_src and (suff_src is None or suff_src == pre_src) else "mixed"
        ),
        "mediation_decode_feature_sources": {
            "pre": pre_src,
            "post_necessity": need_src,
            "post_sufficiency": suff_src,
        },
    }


def _iter_record_buffers(records: Sequence[dict], buffer_size: int) -> Iterator[List[dict]]:
    bs = int(buffer_size)
    if bs <= 0:
        raise ValueError(f"record buffer size must be positive, got {buffer_size}")
    for start in range(0, len(records), bs):
        yield list(records[start : start + bs])


def run_causal_checks_on_record(
    source_rec: dict,
    all_records: Sequence[dict],
    specs_payload: dict,
    variable: str,
    layers: Sequence[int],
    ctx: CausalPatchContext,
    mediation_ctx: Optional[MediationContext] = None,
    off_manifold_max_ratio: float = 0.75,
    seed: int = 17,
) -> Dict[str, Any]:
    donor = select_matched_donor(all_records, source_rec, variable=variable, seed=seed)
    out = {
        "variable": variable,
        "source_trace_id": source_rec.get("trace_id"),
        "source_control_variant": source_rec.get("control_variant"),
        "source_step_idx": int(source_rec.get("step_idx", -1)),
        "layers": {},
    }
    if ctx.supports_subspace_patching and ctx.adapter is None:
        raise RuntimeError("CausalPatchContext has SAEs but no loaded adapter/model")
    for layer in layers:
        feats = _feature_indices_for(specs_payload, variable, int(layer))
        ctrl_feats = _select_control_features(len(feats), feats, pool_size=ctx.latent_dim, seed=seed + int(layer)) if feats else []

        if not ctx.supports_subspace_patching:
            reason = ctx.unsupported_reason or "saes_unavailable_for_model"
            need = {"supported": False, "reason": reason}
            suff = {"supported": False, "reason": reason}
        else:
            need = necessity_ablation(source_rec, int(layer), feats, ctx) if feats else {"supported": False, "reason": "empty_feature_indices"}
            suff = (
                sufficiency_patch(source_rec, donor, int(layer), feats, ctx, control_feature_indices=ctrl_feats)
                if donor is not None and feats
                else {"supported": False, "reason": "no_donor" if donor is None else "empty_feature_indices"}
            )

        layer_row = {
            "feature_count": len(feats),
            "donor_example_idx": int(donor.get("example_idx", -1)) if donor else None,
            "necessity": need,
            "sufficiency": suff,
            "specificity": {
                "supported": bool(suff.get("supported", False) and suff.get("specificity_control_delta_logprob") is not None),
                "target_delta_logprob": suff.get("delta_logprob"),
                "control_delta_logprob": suff.get("specificity_control_delta_logprob"),
                "delta_margin": (
                    float(suff["delta_logprob"] - suff["specificity_control_delta_logprob"])
                    if suff.get("supported") and suff.get("specificity_control_delta_logprob") is not None
                    else None
                ),
            },
        }
        layer_row["mediation"] = _compute_layer_mediation(
            source_rec=source_rec,
            donor_rec=donor,
            layer=int(layer),
            feature_indices=feats,
            ctx=ctx,
            mctx=mediation_ctx,
        )
        off_vals = [v for v in [need.get("off_manifold_ratio"), suff.get("off_manifold_ratio")] if isinstance(v, (int, float))]
        layer_row["off_manifold_intervention"] = any(float(v) > off_manifold_max_ratio for v in off_vals)
        out["layers"][str(layer)] = layer_row
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dataset", default="phase7_results/dataset/gsm8k_step_traces_test.pt")
    p.add_argument(
        "--controls",
        default=None,
        help=(
            "Optional controls JSON. When provided, run control-conditioned causal checks keyed by "
            "(trace_id, control_variant, step_idx) using control-text token anchors."
        ),
    )
    p.add_argument("--parse-mode", choices=["template_only", "hybrid"], default="hybrid")
    p.add_argument(
        "--token-anchor",
        choices=["eq_like", "line_end"],
        default="eq_like",
        help="Token anchor strategy for control-conditioned causal checks.",
    )
    p.add_argument(
        "--anchor-priority",
        choices=["template_first", "equation_first", "leftmost_eq"],
        default="template_first",
        help="Anchor rule priority when multiple equation-like candidates exist in a control step line.",
    )
    p.add_argument(
        "--control-sampling-policy",
        choices=["random", "stratified_trace_variant"],
        default="random",
        help=(
            "Control sampling policy in control-conditioned mode. "
            "stratified_trace_variant prioritizes trace-level faithful/unfaithful diversity before filling quota."
        ),
    )
    p.add_argument("--subspace-specs", default="phase7_results/interventions/variable_subspaces.json")
    p.add_argument("--model-key", default="gpt2-medium")
    p.add_argument("--adapter-config", default=None, help="Optional JSON overrides for model registry entry")
    p.add_argument("--variable", default="subresult_value")
    p.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Optional multi-variable mode. When set, keeps model warm and emits one output per variable.",
    )
    p.add_argument(
        "--output-template",
        default=None,
        help=(
            "Required in multi-variable mode. Use '{variable}' placeholder, e.g. "
            "'phase7_results/interventions/causal_checks_<tag>_{variable}.json'."
        ),
    )
    p.add_argument(
        "--state-decoder-checkpoint",
        default=None,
        help="Optional state decoder checkpoint used for latent mediation readout.",
    )
    p.add_argument(
        "--mediation-variable",
        choices=["subresult_value", "lhs_value", "rhs_value", "operator", "magnitude_bucket", "sign", "result_token_id", "step_type"],
        default="subresult_value",
        help="Target variable for mediation readout checks.",
    )
    p.add_argument(
        "--enable-latent-mediation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable latent mediation checks. Defaults to enabled when --state-decoder-checkpoint is set.",
    )
    p.add_argument("--layers", type=int, nargs="*", default=[22])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--saes-dir", default=None, help="Optional SAE directory override (per-model default comes from registry)")
    p.add_argument("--activations-dir", default=None, help="Optional activation stats dir override")
    p.add_argument("--max-records", type=int, default=20)
    p.add_argument(
        "--target-controls-used-fraction",
        type=float,
        default=0.35,
        help=(
            "Coverage target used by orchestration telemetry (fraction of controls used in control-conditioned mode). "
            "Engine does not auto-escalate; runner uses this target."
        ),
    )
    p.add_argument(
        "--max-records-cap",
        type=int,
        default=12000,
        help="Upper cap for orchestration max-records escalation telemetry.",
    )
    p.add_argument(
        "--min-controls-used",
        type=int,
        default=0,
        help="Optional guard: fail if control-conditioned build uses fewer than this many controls.",
    )
    p.add_argument(
        "--record-buffer-size",
        type=int,
        default=256,
        help="Chunk size for causal-check execution loop. Metadata-only buffering; scoring semantics unchanged.",
    )
    p.add_argument(
        "--control-records-input",
        default=None,
        help="Optional prebuilt control-conditioned records artifact (rows sidecar compatible).",
    )
    p.add_argument(
        "--control-records-output",
        default=None,
        help="Optional path to persist built control-conditioned records for incremental reuse.",
    )
    p.add_argument(
        "--resume-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In non-dry runs, skip already-computed row identities when output artifact exists.",
    )
    p.add_argument(
        "--rows-format",
        choices=["json", "jsonl.gz"],
        default="jsonl.gz",
        help="Storage format for large row payloads in outputs.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dry-run", action="store_true", help="Validate setup and emit status JSON without running interventions")
    p.add_argument("--output", default="phase7_results/interventions/causal_checks_smoke.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    variables: List[str]
    if args.variables:
        variables = [str(v) for v in args.variables]
    else:
        variables = [str(args.variable)]
    if not variables:
        raise ValueError("No variables provided")
    if len(variables) > 1:
        if not args.output_template:
            raise ValueError("--output-template is required when --variables is used")
        if "{variable}" not in str(args.output_template):
            raise ValueError("--output-template must include '{variable}' placeholder")

    if int(args.max_records) > int(args.max_records_cap):
        raise ValueError(
            f"--max-records ({args.max_records}) cannot exceed --max-records-cap ({args.max_records_cap})"
        )
    control_conditioned_mode = bool(args.controls or args.control_records_input)
    mediation_enabled = (
        bool(args.enable_latent_mediation)
        if args.enable_latent_mediation is not None
        else bool(args.state_decoder_checkpoint)
    )
    spec = resolve_model_spec(args.model_key, args.adapter_config)
    records_all = load_pt(args.trace_dataset)
    records_all = [r for r in records_all if r.get("gsm8k_split") == "test"] or records_all

    # Strict model slice behavior:
    # - if model_key is present in dataset rows, only matching rows are selected;
    # - if no model_key is present (legacy datasets), rows are retained and shape validation enforces compatibility.
    has_record_model_key = any("model_key" in r for r in records_all)
    if has_record_model_key:
        base_records = [r for r in records_all if str(r.get("model_key")) == spec.model_key]
    else:
        base_records = list(records_all)

    if not base_records:
        payload = {
            "schema_version": "phase7_causal_checks_v2",
            "status": "error_no_records_for_model_key",
            "dry_run": bool(args.dry_run),
            "model_spec": spec.to_dict(),
            "model_metadata": {
                "model_key": spec.model_key,
                "model_family": spec.model_family,
                "num_layers": int(spec.num_layers),
                "hidden_dim": int(spec.hidden_dim),
                "tokenizer_id": spec.tokenizer_id,
                "latent_dim": 0,
                "supports_subspace_patching": False,
                "resolved_saes_dir": args.saes_dir if args.saes_dir is not None else spec.sae_dir,
                "resolved_activations_dir": args.activations_dir,
                "unsupported_reason": f"no_records_for_model_key:{spec.model_key}",
            },
            "records_considered": 0,
            "causal_mode": ("control_conditioned" if control_conditioned_mode else "source_trace"),
            "layers_requested": list(args.layers),
            "variable": variables[0],
            "variables": list(variables),
            "mediation": {
                "enabled": mediation_enabled,
                "variable": args.mediation_variable,
                "checkpoint": args.state_decoder_checkpoint,
            },
            "subspace_specs_schema": CAUSAL_PATCH_SPEC_SCHEMA,
            "rows": [],
        }
        save_json(args.output, payload)
        raise SystemExit(
            f"No trace records matched model_key={spec.model_key!r} in {args.trace_dataset}; "
            f"wrote structured error payload to {args.output}"
        )

    try:
        compatibility_checks = _validate_records_model_compatibility(
            base_records,
            model_key=spec.model_key,
            model_family=spec.model_family,
            num_layers=int(spec.num_layers),
            hidden_dim=int(spec.hidden_dim),
            tokenizer_id=spec.tokenizer_id,
        )
    except RuntimeError as exc:
        payload = {
            "schema_version": "phase7_causal_checks_v2",
            "status": "error_model_data_mismatch",
            "dry_run": bool(args.dry_run),
            "model_spec": spec.to_dict(),
            "model_metadata": {
                "model_key": spec.model_key,
                "model_family": spec.model_family,
                "num_layers": int(spec.num_layers),
                "hidden_dim": int(spec.hidden_dim),
                "tokenizer_id": spec.tokenizer_id,
                "latent_dim": 0,
                "supports_subspace_patching": False,
                "resolved_saes_dir": args.saes_dir if args.saes_dir is not None else spec.sae_dir,
                "resolved_activations_dir": args.activations_dir,
                "unsupported_reason": str(exc),
            },
            "records_considered": len(base_records),
            "causal_mode": ("control_conditioned" if control_conditioned_mode else "source_trace"),
            "layers_requested": list(args.layers),
            "variable": variables[0],
            "variables": list(variables),
            "mediation": {
                "enabled": mediation_enabled,
                "variable": args.mediation_variable,
                "checkpoint": args.state_decoder_checkpoint,
            },
            "subspace_specs_schema": CAUSAL_PATCH_SPEC_SCHEMA,
            "rows": [],
            "compatibility_checks": {"num_records_checked": int(len(base_records)), "validation_failed": True},
        }
        save_json(args.output, payload)
        raise

    control_conditioned_stats: Dict[str, Any] = {}
    records_pool: List[dict] = []
    records: List[dict]
    if control_conditioned_mode and not bool(args.dry_run):
        stats_base: Dict[str, Any] = {}
        if args.control_records_input:
            records_pool, records_payload = _load_control_records_artifact(str(args.control_records_input))
            stats_base = dict(records_payload.get("stats", {}) or {})
        else:
            if not args.controls:
                raise ValueError("--controls is required when --control-records-input is not provided")
            controls_payload = load_json(str(args.controls))
            adapter = create_adapter(model_key=spec.model_key, device=args.device, adapter_config=args.adapter_config).load(
                device=args.device
            )
            if adapter.tokenizer is None:
                raise RuntimeError("Adapter tokenizer is required for control-conditioned causal checks")
            built = _build_control_conditioned_records(
                controls_payload,
                base_records,
                adapter=adapter,
                parse_mode=str(args.parse_mode),
                token_anchor=str(args.token_anchor),
                anchor_priority=str(args.anchor_priority),
                max_records=int(args.max_records_cap),
                seed=int(args.seed),
                control_sampling_policy=str(args.control_sampling_policy),
                model_key=str(spec.model_key),
                model_family=str(spec.model_family),
                num_layers=int(spec.num_layers),
                hidden_dim=int(spec.hidden_dim),
                tokenizer_id=str(spec.tokenizer_id),
            )
            records_pool = list(built["records"])
            stats_base = dict(built.get("stats", {}) or {})
            if args.control_records_output:
                _save_control_records_artifact(
                    str(args.control_records_output),
                    records=records_pool,
                    stats=stats_base,
                    model_spec=spec,
                    args=args,
                )
            # Free temporary adapter before causal patch loop initializes its own adapter.
            del adapter
            gc.collect()
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

        records = list(records_pool[: int(args.max_records)])
        control_conditioned_stats = _summarize_control_records_prefix(records, stats_base)
        control_conditioned_stats["target_controls_used_fraction"] = float(args.target_controls_used_fraction)
        control_conditioned_stats["max_records_cap"] = int(args.max_records_cap)
        min_controls_used = max(0, int(args.min_controls_used))
        if min_controls_used > 0 and int(control_conditioned_stats.get("controls_used", 0) or 0) < min_controls_used:
            raise RuntimeError(
                "Control-conditioned sampling guard failed: "
                f"controls_used={int(control_conditioned_stats.get('controls_used', 0) or 0)} < min_controls_used={min_controls_used}"
            )
    else:
        records = list(base_records[: args.max_records])
        records_pool = list(records)

    if not records:
        payload = {
            "schema_version": "phase7_causal_checks_v2",
            "status": "error_no_records_after_max_records",
            "dry_run": bool(args.dry_run),
            "model_spec": spec.to_dict(),
            "model_metadata": {
                "model_key": spec.model_key,
                "model_family": spec.model_family,
                "num_layers": int(spec.num_layers),
                "hidden_dim": int(spec.hidden_dim),
                "tokenizer_id": spec.tokenizer_id,
                "latent_dim": 0,
                "supports_subspace_patching": False,
                "resolved_saes_dir": args.saes_dir if args.saes_dir is not None else spec.sae_dir,
                "resolved_activations_dir": args.activations_dir,
                "unsupported_reason": "empty_after_max_records",
            },
            "records_considered": 0,
            "causal_mode": ("control_conditioned" if control_conditioned_mode else "source_trace"),
            "layers_requested": list(args.layers),
            "variable": variables[0],
            "variables": list(variables),
            "mediation": {
                "enabled": mediation_enabled,
                "variable": args.mediation_variable,
                "checkpoint": args.state_decoder_checkpoint,
            },
            "subspace_specs_schema": CAUSAL_PATCH_SPEC_SCHEMA,
            "rows": [],
            "compatibility_checks": compatibility_checks,
            "control_conditioned_stats": control_conditioned_stats,
        }
        save_json(args.output, payload)
        raise SystemExit(
            f"No records remain after applying --max-records={args.max_records}; wrote {args.output}"
        )
    position_errors = _validate_result_token_positions(records)
    if position_errors:
        payload = {
            "schema_version": "phase7_causal_checks_v2",
            "status": "error_invalid_result_token_positions",
            "dry_run": bool(args.dry_run),
            "model_spec": spec.to_dict(),
            "model_metadata": {
                "model_key": spec.model_key,
                "model_family": spec.model_family,
                "num_layers": int(spec.num_layers),
                "hidden_dim": int(spec.hidden_dim),
                "tokenizer_id": spec.tokenizer_id,
                "latent_dim": 0,
                "supports_subspace_patching": False,
                "resolved_saes_dir": args.saes_dir if args.saes_dir is not None else spec.sae_dir,
                "resolved_activations_dir": args.activations_dir,
                "unsupported_reason": "invalid_result_token_positions",
            },
            "records_considered": len(records),
            "causal_mode": ("control_conditioned" if control_conditioned_mode else "source_trace"),
            "layers_requested": list(args.layers),
            "variable": variables[0],
            "variables": list(variables),
            "mediation": {
                "enabled": mediation_enabled,
                "variable": args.mediation_variable,
                "checkpoint": args.state_decoder_checkpoint,
            },
            "subspace_specs_schema": CAUSAL_PATCH_SPEC_SCHEMA,
            "rows": [],
            "compatibility_checks": compatibility_checks,
            "control_conditioned_stats": control_conditioned_stats,
            "invalid_result_positions": {
                "num_errors": len(position_errors),
                "examples": position_errors[:20],
            },
        }
        save_json(args.output, payload)
        raise RuntimeError(
            "Causal intervention strict record check failed: invalid/missing result token positions.\n"
            + "\n".join(position_errors[:20])
            + ("" if len(position_errors) <= 20 else f"\n... and {len(position_errors) - 20} more")
        )
    specs = load_json(args.subspace_specs) if Path(args.subspace_specs).exists() else {"schema_version": CAUSAL_PATCH_SPEC_SCHEMA}

    if args.dry_run:
        controls_count = None
        if control_conditioned_mode:
            if args.control_records_input and Path(args.control_records_input).exists():
                _, cp = _load_control_records_artifact(str(args.control_records_input))
                controls_count = int((cp.get("stats") or {}).get("controls_total", 0))
            elif args.controls:
                controls_payload = load_json(str(args.controls))
                controls_count = int(len(list(controls_payload.get("controls", []) or [])))
        resolved_saes_dir = args.saes_dir if args.saes_dir is not None else spec.sae_dir
        resolved_activations_dir = args.activations_dir
        if resolved_activations_dir is None and spec.model_key == "gpt2-medium":
            resolved_activations_dir = "phase2_results/activations"

        unsupported_reason: Optional[str] = None
        supports_subspace = False
        if resolved_saes_dir:
            if spec.model_key != "gpt2-medium":
                unsupported_reason = f"sae_loader_unimplemented_for_model:{spec.model_key}"
            elif not Path(resolved_saes_dir).exists():
                unsupported_reason = f"missing_sae_dir_path:{resolved_saes_dir}"
            else:
                supports_subspace = True
        else:
            unsupported_reason = f"missing_model_sae_dir:{spec.model_key}"

        status = "ready" if supports_subspace else "unsupported_model_causal_subspace"
        payload = {
            "schema_version": "phase7_causal_checks_v2",
            "status": status,
            "dry_run": True,
            "model_spec": spec.to_dict(),
            "model_metadata": {
                "model_key": spec.model_key,
                "model_family": spec.model_family,
                "num_layers": int(spec.num_layers),
                "hidden_dim": int(spec.hidden_dim),
                "tokenizer_id": spec.tokenizer_id,
                "latent_dim": 0,
                "supports_subspace_patching": supports_subspace,
                "resolved_saes_dir": resolved_saes_dir,
                "resolved_activations_dir": resolved_activations_dir,
                "unsupported_reason": unsupported_reason,
            },
            "records_considered": len(records),
            "causal_mode": ("control_conditioned" if control_conditioned_mode else "source_trace"),
            "controls_source": args.controls,
            "controls_count": controls_count,
            "parse_mode": str(args.parse_mode),
            "token_anchor": str(args.token_anchor),
            "anchor_priority": str(args.anchor_priority),
            "control_sampling_policy": str(args.control_sampling_policy),
            "layers_requested": list(args.layers),
            "variable": variables[0],
            "variables": list(variables),
            "mediation": {
                "enabled": mediation_enabled,
                "variable": args.mediation_variable,
                "checkpoint": args.state_decoder_checkpoint,
                "status": (
                    "ready" if (mediation_enabled and args.state_decoder_checkpoint) else
                    "disabled" if not mediation_enabled else "missing_state_decoder_checkpoint"
                ),
            },
            "subspace_specs_schema": specs.get("schema_version", CAUSAL_PATCH_SPEC_SCHEMA),
            "rows": [],
            "compatibility_checks": compatibility_checks,
            "control_conditioned_stats": control_conditioned_stats,
        }
        save_json(args.output, payload)
        print(f"[dry-run] status={status} records={len(records)} unsupported_reason={unsupported_reason}")
        print(f"Saved -> {args.output}")
        return

    print(f"Loading model/adapter for {args.model_key} on {args.device}...")
    ctx = CausalPatchContext.load(
        args.device,
        model_key=args.model_key,
        adapter_config=args.adapter_config,
        saes_dir=args.saes_dir,
        activations_dir=args.activations_dir,
    )
    donor_pool = list(records_pool if control_conditioned_mode else records)
    buffer_size = max(1, int(args.record_buffer_size))
    peak_records_in_memory = int(len(records))
    controls_source_sha256 = sha256_file(args.controls) if args.controls and Path(args.controls).exists() else None
    subspace_specs_sha256 = sha256_file(args.subspace_specs) if Path(args.subspace_specs).exists() else None
    checkpoint_sha256 = (
        sha256_file(args.state_decoder_checkpoint)
        if args.state_decoder_checkpoint and Path(args.state_decoder_checkpoint).exists()
        else None
    )

    for variable in variables:
        output_path = (
            str(args.output_template).format(variable=variable)
            if len(variables) > 1
            else str(args.output)
        )
        med_var = str(args.mediation_variable)
        if med_var == "subresult_value" and variable in {"operator", "magnitude_bucket", "sign", "result_token_id"}:
            med_var = str(variable)
        mctx = MediationContext.from_checkpoint(
            args.state_decoder_checkpoint,
            variable=med_var,
            device=args.device,
            expected_model_key=spec.model_key,
            force_enable=mediation_enabled,
        )

        existing_rows: List[dict] = []
        existing_keys: set[Tuple[str, str, int, str]] = set()
        if bool(args.resume_output) and Path(output_path).exists():
            try:
                prior_payload = load_json(output_path)
                existing_rows = load_rows_payload(prior_payload, base_path=output_path)
                for er in existing_rows:
                    existing_keys.add(_causal_row_identity_key(er, variable=variable))
            except Exception:
                existing_rows = []
                existing_keys = set()

        new_rows: List[dict] = []
        buffer_flush_count = 0
        for buffer in _iter_record_buffers(records, buffer_size):
            buffer_flush_count += 1
            for rec in buffer:
                identity = (
                    str(rec.get("trace_id", "")),
                    str(rec.get("control_variant", "")),
                    int(rec.get("step_idx", -1)),
                    str(variable),
                )
                if identity in existing_keys:
                    continue
                new_rows.append(
                    run_causal_checks_on_record(
                        rec,
                        donor_pool,
                        specs,
                        variable,
                        args.layers,
                        ctx,
                        mediation_ctx=mctx,
                        seed=args.seed,
                    )
                )

        rows = list(existing_rows) + new_rows
        rows_payload = write_rows_sidecar(
            output_path,
            rows,
            rows_format=str(args.rows_format),
            rows_inline=False,
        )
        payload = {
            "schema_version": "phase7_causal_checks_v2",
            "status": "ok" if ctx.supports_subspace_patching else "unsupported_model_causal_subspace",
            "dry_run": False,
            "causal_mode": ("control_conditioned" if control_conditioned_mode else "source_trace"),
            "controls_source": args.controls,
            "controls_source_sha256": controls_source_sha256,
            "trace_dataset": str(args.trace_dataset),
            "trace_dataset_sha256": sha256_file(args.trace_dataset),
            "control_records_input": args.control_records_input,
            "control_records_output": args.control_records_output,
            "subspace_specs_path": str(args.subspace_specs),
            "subspace_specs_sha256": subspace_specs_sha256,
            "state_decoder_checkpoint_sha256": checkpoint_sha256,
            "parse_mode": str(args.parse_mode),
            "token_anchor": str(args.token_anchor),
            "anchor_priority": str(args.anchor_priority),
            "control_sampling_policy": str(args.control_sampling_policy),
            "model_metadata": ctx.metadata(),
            "mediation": mctx.metadata(),
            "mediation_variable": med_var,
            "enable_latent_mediation": mediation_enabled,
            "subspace_specs_schema": specs.get("schema_version", CAUSAL_PATCH_SPEC_SCHEMA),
            "compatibility_checks": compatibility_checks,
            "control_conditioned_stats": control_conditioned_stats,
            "position_convention_version": str(
                (control_conditioned_stats or {}).get("position_convention_version", "phase7_pos_contract_v1")
            ),
            "position_contract_validated": bool(
                (control_conditioned_stats or {}).get("position_contract_validated", True)
            ),
            "variable": str(variable),
            "variables": [str(variable)],
            "execution_telemetry": {
                "record_buffer_size": int(buffer_size),
                "buffer_flush_count": int(buffer_flush_count),
                "peak_records_in_memory": int(peak_records_in_memory),
                "target_controls_used_fraction": float(args.target_controls_used_fraction),
                "max_records_requested": int(args.max_records),
                "max_records_cap": int(args.max_records_cap),
                "controls_used_target_met": bool(
                    isinstance((control_conditioned_stats or {}).get("controls_used_fraction"), (int, float))
                    and float((control_conditioned_stats or {}).get("controls_used_fraction"))
                    >= float(args.target_controls_used_fraction)
                ),
                "resume_output": bool(args.resume_output),
                "resumed_rows": int(len(existing_rows)),
                "new_rows_added": int(len(new_rows)),
            },
            **rows_payload,
        }
        save_json(output_path, payload)
        print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()

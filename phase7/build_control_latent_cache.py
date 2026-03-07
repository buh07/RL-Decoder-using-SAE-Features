#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover
    from .common import load_json, save_json, sha256_file, write_rows_sidecar
    from .control_token_anchor import collect_control_step_token_positions
    from .model_registry import create_adapter, resolve_model_spec
    from .sae_assets import load_norm_stats, load_saes
    from .state_decoder_core import (
        MAG_TO_ID,
        OP_TO_ID,
        SIGN_TO_ID,
        STEP_TO_ID,
        load_model_from_checkpoint,
    )
except ImportError:  # pragma: no cover
    from common import load_json, save_json, sha256_file, write_rows_sidecar
    from control_token_anchor import collect_control_step_token_positions
    from model_registry import create_adapter, resolve_model_spec
    from sae_assets import load_norm_stats, load_saes
    from state_decoder_core import (
        MAG_TO_ID,
        OP_TO_ID,
        SIGN_TO_ID,
        STEP_TO_ID,
        load_model_from_checkpoint,
    )

from phase6.pipeline_utils import build_input_tensor_from_record  # type: ignore  # noqa: E402


def _decode_records_from_features(
    decoder,
    records: List[dict],
    cfg,
    numeric_stats,
    *,
    device: str,
    batch_size: int,
) -> List[dict]:
    inv_op = {v: k for k, v in OP_TO_ID.items()}
    inv_step = {v: k for k, v in STEP_TO_ID.items()}
    inv_mag = {v: k for k, v in MAG_TO_ID.items()}
    inv_sign = {v: k for k, v in SIGN_TO_ID.items()}
    layer_idx = list(getattr(cfg, "layers", ()))
    preds: List[dict] = []

    decoder.eval()
    with torch.no_grad():
        for start in range(0, len(records), max(1, int(batch_size))):
            chunk = records[start : start + max(1, int(batch_size))]
            x = torch.stack([build_input_tensor_from_record(r, cfg).float() for r in chunk], dim=0).to(device)
            out = decoder(x)
            log_probs = F.log_softmax(out["result_token_logits"], dim=-1)
            op_probs = F.softmax(out["operator_logits"], dim=-1).cpu()
            step_probs = F.softmax(out["step_type_logits"], dim=-1).cpu()
            mag_probs = F.softmax(out["magnitude_logits"], dim=-1).cpu()
            sign_probs = F.softmax(out["sign_logits"], dim=-1).cpu()
            top1_tok = out["result_token_logits"].argmax(dim=-1).cpu().tolist()
            top1_lp = log_probs.max(dim=-1).values.cpu().tolist()
            ops = out["operator_logits"].argmax(dim=-1).cpu().tolist()
            steps = out["step_type_logits"].argmax(dim=-1).cpu().tolist()
            mags = out["magnitude_logits"].argmax(dim=-1).cpu().tolist()
            signs = out["sign_logits"].argmax(dim=-1).cpu().tolist()
            lhs_z = out["lhs_pred"].cpu().tolist()
            rhs_z = out["rhs_pred"].cpu().tolist()
            sub_z = out["subresult_pred"].cpu().tolist()

            for i in range(len(chunk)):
                lhs = lhs_z[i] * numeric_stats["lhs_value"].std + numeric_stats["lhs_value"].mean
                rhs = rhs_z[i] * numeric_stats["rhs_value"].std + numeric_stats["rhs_value"].mean
                sub = sub_z[i] * numeric_stats["subresult_value"].std + numeric_stats["subresult_value"].mean
                r = chunk[i]
                preds.append(
                    {
                        "step_idx": int(r["step_idx"]),
                        "trace_id": str(r["trace_id"]),
                        "example_idx": int(r.get("example_idx", -1)),
                        "latent_pred_state": {
                            "step_type": inv_step[int(steps[i])],
                            "operator": inv_op[int(ops[i])],
                            "magnitude_bucket": inv_mag[int(mags[i])],
                            "sign": inv_sign[int(signs[i])],
                            "lhs_value": float(lhs),
                            "rhs_value": float(rhs),
                            "subresult_value": float(sub),
                            "result_token_id": int(top1_tok[i]),
                        },
                        "latent_pred_confidence": {
                            "result_token_logprob_top1": float(top1_lp[i]),
                            "operator_prob": float(op_probs[i, ops[i]].item()),
                            "step_type_prob": float(step_probs[i, steps[i]].item()),
                            "sign_top1_prob": float(sign_probs[i, signs[i]].item()),
                            "magnitude_top1_prob": float(mag_probs[i, mags[i]].item()),
                            "operator_probs": {
                                str(inv_op[j]): float(op_probs[i, j].item()) for j in range(op_probs.shape[1])
                            },
                            "sign_probs": {
                                str(inv_sign[j]): float(sign_probs[i, j].item()) for j in range(sign_probs.shape[1])
                            },
                            "magnitude_probs": {
                                str(inv_mag[j]): float(mag_probs[i, j].item()) for j in range(mag_probs.shape[1])
                            },
                        },
                    }
                )
    return preds


def _normalize_hidden(
    h: torch.Tensor,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if stats is None:
        return h
    mean, std = stats
    return (h - mean) / std


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controls", required=True, help="Controls JSON with `controls` list.")
    p.add_argument("--state-decoder-checkpoint", required=True)
    p.add_argument("--model-key", default="gpt2-medium")
    p.add_argument("--adapter-config", default=None, help="Optional JSON overrides for model registry entry")
    p.add_argument(
        "--saes-dir",
        default=None,
        help="Optional SAE directory override for non-raw checkpoint variants.",
    )
    p.add_argument(
        "--activations-dir",
        default="phase2_results/activations",
        help="Activation stats directory used to normalize hidden states before SAE encoding.",
    )
    p.add_argument("--parse-mode", choices=["template_only", "hybrid"], default="hybrid")
    p.add_argument(
        "--token-anchor",
        choices=["eq_like", "line_end"],
        default="eq_like",
        help=(
            "Token anchor for per-step hidden extraction. eq_like anchors to equation-style '=' "
            "positions when available; line_end keeps legacy behavior."
        ),
    )
    p.add_argument(
        "--anchor-priority",
        choices=["template_first", "equation_first", "leftmost_eq"],
        default="template_first",
        help="Anchor rule priority when multiple equation-like candidates appear on the same line.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-controls", type=int, default=None)
    p.add_argument(
        "--rows-format",
        choices=["json", "jsonl.gz"],
        default="jsonl.gz",
        help="Storage format for row-heavy payload section.",
    )
    p.add_argument(
        "--rows-inline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inline rows in output JSON instead of sidecar rows artifact.",
    )
    p.add_argument(
        "--allow-model-key-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow using a decoder checkpoint with model_key different from --model-key. "
            "Unsafe by default."
        ),
    )
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    controls_payload = load_json(args.controls)
    controls = list(controls_payload.get("controls", []))
    if args.max_controls is not None:
        controls = controls[: args.max_controls]

    ckpt, cfg, numeric_stats, decoder = load_model_from_checkpoint(args.state_decoder_checkpoint, args.device)
    input_variant = str(getattr(cfg, "input_variant", "raw"))
    if input_variant not in {"raw", "sae", "hybrid", "hybrid_indexed"}:
        raise ValueError(
            "variant-conditioned latent cache supports raw/sae/hybrid/hybrid_indexed checkpoints; "
            f"got input_variant={input_variant!r} from checkpoint {args.state_decoder_checkpoint}"
        )

    spec = resolve_model_spec(args.model_key, args.adapter_config)
    ckpt_model_key = str(getattr(cfg, "model_key", ""))
    if (not bool(args.allow_model_key_mismatch)) and ckpt_model_key and ckpt_model_key != str(spec.model_key):
        raise ValueError(
            f"Checkpoint model_key={ckpt_model_key!r} does not match requested --model-key={spec.model_key!r}. "
            "Use --allow-model-key-mismatch only if this is intentional."
        )

    adapter = create_adapter(model_key=spec.model_key, device=args.device, adapter_config=args.adapter_config).load(
        device=args.device
    )
    if adapter.tokenizer is None:
        raise RuntimeError("Adapter tokenizer is not loaded")

    needs_sae_features = input_variant in {"sae", "hybrid", "hybrid_indexed"}
    saes = {}
    norm_stats = {}
    resolved_saes_dir: Optional[str] = None
    resolved_activations_dir: Optional[str] = None
    if needs_sae_features:
        resolved_saes_dir = args.saes_dir if args.saes_dir is not None else spec.sae_dir
        if not resolved_saes_dir:
            raise ValueError(
                f"input_variant={input_variant!r} requires SAE assets but no SAE directory is configured. "
                "Provide --saes-dir or set model spec sae_dir."
            )
        resolved_activations_dir = str(args.activations_dir)
        saes = load_saes(
            saes_dir=Path(resolved_saes_dir),
            model_key=str(spec.model_key),
            num_layers=int(spec.num_layers),
            device=str(args.device),
        )
        norm_stats = load_norm_stats(
            activations_dir=Path(resolved_activations_dir),
            model_key=str(spec.model_key),
            num_layers=int(spec.num_layers),
            device=str(args.device),
        )

    out_rows: List[dict] = []
    skipped_controls = 0
    controls_with_rows = 0
    position_contract_validated = True
    anchor_cov_totals = {
        "eq_like_rows": 0,
        "line_end_rows": 0,
        "fallback_rows": 0,
        "total_rows": 0,
    }
    tokenization_summary = {
        "offset_alignment_degraded_rows": 0,
        "special_tokens_policy_counts": {},
    }

    for i, ctrl in enumerate(controls):
        trace_id = str(ctrl.get("trace_id"))
        variant = str(ctrl.get("variant", ctrl.get("control_variant", "unknown")))
        example_idx = int(ctrl.get("example_idx", -1))
        cot_text = str(ctrl.get("cot_text", ""))
        if not cot_text.strip():
            skipped_controls += 1
            continue

        pos_payload = collect_control_step_token_positions(
            ctrl,
            adapter,
            parse_mode=args.parse_mode,
            token_anchor=args.token_anchor,
            anchor_priority=args.anchor_priority,
        )
        step_pos_rows = list(pos_payload.get("rows", []))
        cov = dict(pos_payload.get("anchor_coverage", {}) or {})
        for k in anchor_cov_totals:
            anchor_cov_totals[k] += int(cov.get(k, 0))
        tok_meta = dict(pos_payload.get("tokenization_metadata", {}) or {})
        policy = str(tok_meta.get("special_tokens_policy", "unknown"))
        tokenization_summary["special_tokens_policy_counts"][policy] = int(
            tokenization_summary["special_tokens_policy_counts"].get(policy, 0) + 1
        )
        if bool(tok_meta.get("offset_alignment_degraded", False)):
            tokenization_summary["offset_alignment_degraded_rows"] += int(len(step_pos_rows))
        if not step_pos_rows:
            skipped_controls += 1
            continue

        logits, hidden_states = adapter.forward(cot_text)
        if not hidden_states:
            raise RuntimeError("Adapter forward returned no hidden states")
        seq_len = int(logits.shape[1])
        num_layers = len(hidden_states)
        cfg_layers = [int(x) for x in getattr(cfg, "layers", ())]
        if cfg_layers:
            if min(cfg_layers) < 0 or max(cfg_layers) >= num_layers:
                raise ValueError(
                    f"Checkpoint layers {tuple(cfg_layers)} are incompatible with adapter hidden-state depth "
                    f"{num_layers} for model_key={spec.model_key!r}"
                )

        records: List[dict] = []
        for sp in step_pos_rows:
            hidden_pos = int(sp.get("hidden_token_pos_0b", sp["token_pos"]))
            eq_pos = int(sp.get("eq_token_pos_0b", sp["eq_token_pos"]))
            result_pos = int(sp.get("result_token_pos_0b", sp["result_token_pos"]))
            eq_idx_1b = int(sp.get("eq_tok_idx_1b", eq_pos + 1))
            result_idx_1b = int(sp.get("result_tok_idx_1b", result_pos + 1))
            if eq_idx_1b != eq_pos + 1:
                raise RuntimeError(
                    f"position contract violation: eq_tok_idx_1b={eq_idx_1b} != eq_token_pos_0b+1={eq_pos + 1}"
                )
            if result_idx_1b != result_pos + 1:
                raise RuntimeError(
                    "position contract violation: "
                    f"result_tok_idx_1b={result_idx_1b} != result_token_pos_0b+1={result_pos + 1}"
                )
            if args.token_anchor == "eq_like" and hidden_pos != eq_pos:
                raise RuntimeError(
                    "position contract violation for eq_like: "
                    f"hidden_token_pos_0b={hidden_pos} != eq_token_pos_0b={eq_pos}"
                )
            pos = max(0, min(int(hidden_pos), seq_len - 1))
            eq_pos = max(0, min(int(eq_pos), seq_len - 1))
            result_pos = max(0, min(int(result_pos), seq_len - 1))
            raw_hidden = torch.stack(
                [hidden_states[layer_i][0, pos, :].detach().float().cpu() for layer_i in range(num_layers)],
                dim=0,
            )
            record = {
                "trace_id": trace_id,
                "example_idx": example_idx,
                "step_idx": int(sp["step_idx"]),
                "raw_hidden": raw_hidden,
                "schema_version": "phase7_control_latent_cache_record_v1",
            }
            if needs_sae_features:
                sae_rows: List[torch.Tensor] = []
                for layer_i in range(num_layers):
                    sae = saes.get(layer_i)
                    if sae is None:
                        raise RuntimeError(f"Missing SAE for layer={layer_i} while building latent cache")
                    h = hidden_states[layer_i][0, pos, :].detach().float().to(args.device)
                    h_norm = _normalize_hidden(h, norm_stats.get(layer_i))
                    sae_dtype = next(sae.parameters()).dtype
                    feat = sae.encode(h_norm.to(dtype=sae_dtype).unsqueeze(0)).squeeze(0).detach().float().cpu()
                    sae_rows.append(feat)
                record["sae_features"] = torch.stack(sae_rows, dim=0)
            records.append(record)

        preds = _decode_records_from_features(
            decoder,
            records,
            cfg,
            numeric_stats,
            device=args.device,
            batch_size=max(1, min(int(args.batch_size), len(records))),
        )
        for p in preds:
            step_idx = int(p["step_idx"])
            line_meta = next((x for x in step_pos_rows if int(x["step_idx"]) == step_idx), None)
            if line_meta is not None and not bool(line_meta.get("anchor_span_contains_anchor_char", False)):
                raise RuntimeError(
                    f"anchor span contract violated for trace_id={trace_id}, variant={variant}, step_idx={step_idx}"
                )
            out_rows.append(
                {
                    "trace_id": trace_id,
                    "control_variant": variant,
                    "step_idx": step_idx,
                    "example_idx": int(p.get("example_idx", example_idx)),
                    "latent_pred_state": p.get("latent_pred_state"),
                    "latent_pred_confidence": p.get("latent_pred_confidence"),
                    "line_index": (int(line_meta["line_index"]) if line_meta is not None else None),
                    "token_pos": (int(line_meta["token_pos"]) if line_meta is not None else None),
                    "eq_token_pos": (int(line_meta["eq_token_pos"]) if line_meta is not None else None),
                    "result_token_pos": (int(line_meta["result_token_pos"]) if line_meta is not None else None),
                    "hidden_token_pos_0b": (
                        int(line_meta["hidden_token_pos_0b"]) if line_meta is not None else None
                    ),
                    "eq_token_pos_0b": (
                        int(line_meta["eq_token_pos_0b"]) if line_meta is not None else None
                    ),
                    "result_token_pos_0b": (
                        int(line_meta["result_token_pos_0b"]) if line_meta is not None else None
                    ),
                    "eq_tok_idx_1b": (int(line_meta["eq_tok_idx_1b"]) if line_meta is not None else None),
                    "result_tok_idx_1b": (
                        int(line_meta["result_tok_idx_1b"]) if line_meta is not None else None
                    ),
                    "position_convention_version": (
                        str(line_meta["position_convention_version"]) if line_meta is not None else "phase7_pos_contract_v1"
                    ),
                    "token_anchor_mode": (
                        str(line_meta["token_anchor_mode"]) if line_meta is not None else str(args.token_anchor)
                    ),
                    "token_anchor_reason": (
                        str(line_meta["token_anchor_reason"]) if line_meta is not None else "unknown"
                    ),
                    "selected_anchor_rule": (
                        str(line_meta.get("selected_anchor_rule", "unknown")) if line_meta is not None else "unknown"
                    ),
                    "anchor_candidate_matches": (
                        list(line_meta.get("anchor_candidate_matches", [])) if line_meta is not None else []
                    ),
                    "anchor_char_index": (int(line_meta["anchor_char_index"]) if line_meta is not None else None),
                    "anchor_token_span_start": (
                        int(line_meta["anchor_token_span_start"]) if line_meta is not None else None
                    ),
                    "anchor_token_span_end": (
                        int(line_meta["anchor_token_span_end"]) if line_meta is not None else None
                    ),
                    "anchor_span_contains_anchor_char": (
                        bool(line_meta["anchor_span_contains_anchor_char"]) if line_meta is not None else None
                    ),
                    "special_tokens_policy": (
                        str(line_meta.get("special_tokens_policy", tok_meta.get("special_tokens_policy", "unknown")))
                        if line_meta is not None
                        else str(tok_meta.get("special_tokens_policy", "unknown"))
                    ),
                    "num_special_tokens_prefix": (
                        int(line_meta.get("num_special_tokens_prefix", tok_meta.get("num_special_tokens_prefix", 0)))
                        if line_meta is not None
                        else int(tok_meta.get("num_special_tokens_prefix", 0))
                    ),
                    "offset_alignment_degraded": (
                        bool(line_meta.get("offset_alignment_degraded", tok_meta.get("offset_alignment_degraded", False)))
                        if line_meta is not None
                        else bool(tok_meta.get("offset_alignment_degraded", False))
                    ),
                }
            )
        controls_with_rows += 1
        if (i + 1) % 25 == 0:
            print(f"[build_control_latent_cache] processed {i + 1}/{len(controls)} controls")

    rows_payload = write_rows_sidecar(
        args.output,
        out_rows,
        rows_format=str(args.rows_format),
        rows_inline=bool(args.rows_inline),
    )
    payload = {
        "schema_version": "phase7_control_latent_cache_v1",
        "controls_source": str(args.controls),
        "controls_source_sha256": sha256_file(args.controls),
        "state_decoder_checkpoint": str(args.state_decoder_checkpoint),
        "state_decoder_checkpoint_sha256": sha256_file(args.state_decoder_checkpoint),
        "latent_source": "variant_conditioned",
        "index_key": ["trace_id", "control_variant", "step_idx"],
        "parse_mode_used": str(args.parse_mode),
        "token_anchor_mode": str(args.token_anchor),
        "anchor_priority": str(args.anchor_priority),
        "position_convention_version": "phase7_pos_contract_v1",
        "position_contract_validated": bool(position_contract_validated),
        "model_metadata": {
            "model_key": str(spec.model_key),
            "model_family": str(spec.model_family),
            "num_layers": int(spec.num_layers),
            "hidden_dim": int(spec.hidden_dim),
            "tokenizer_id": str(spec.tokenizer_id),
        },
        "feature_source_metadata": {
            "input_variant": input_variant,
            "uses_sae_features": bool(needs_sae_features),
            "resolved_saes_dir": resolved_saes_dir,
            "resolved_activations_dir": resolved_activations_dir,
        },
        "decoder_metadata": {
            "config_name": str(getattr(cfg, "name", "unknown")),
            "input_variant": str(getattr(cfg, "input_variant", "unknown")),
            "layers": list(getattr(cfg, "layers", ())),
        },
        "num_controls": int(len(controls)),
        "num_controls_with_rows": int(controls_with_rows),
        "num_controls_skipped": int(skipped_controls),
        "num_rows": int(len(out_rows)),
        "anchor_coverage_summary": {
            **anchor_cov_totals,
            "eq_like_fraction": float(anchor_cov_totals["eq_like_rows"] / max(1, anchor_cov_totals["total_rows"])),
        },
        "tokenization_summary": {
            "offset_alignment_degraded_rows": int(tokenization_summary["offset_alignment_degraded_rows"]),
            "special_tokens_policy_counts": {
                str(k): int(v)
                for k, v in sorted(tokenization_summary["special_tokens_policy_counts"].items())
            },
        },
        **rows_payload,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, payload)
    print(f"Saved control latent cache -> {out_path} (rows={len(out_rows)})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        load_pt,
        save_json,
        set_seed,
    )
    from .model_adapters import BaseCausalLMAdapter
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
        load_pt,
        save_json,
        set_seed,
    )
    from model_adapters import BaseCausalLMAdapter
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


def _record_identity_key(rec: dict) -> Tuple[str, int, int]:
    return (
        str(rec.get("trace_id", "")),
        int(rec.get("step_idx", -1)),
        int(rec.get("example_idx", -1)),
    )


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
    return row


def _decode_decoder_variable(row: dict, mctx: MediationContext) -> Any:
    if not mctx.enabled or mctx.model is None or mctx.cfg is None:
        return None
    x = build_input_tensor_from_record(row, mctx.cfg).unsqueeze(0).to(mctx.device)
    with torch.no_grad():
        out = mctx.model(x)
    var = str(mctx.variable)
    if var == "result_token_id":
        return int(out["result_token_logits"].argmax(dim=-1)[0].item())
    if var == "operator":
        idx = int(out["operator_logits"].argmax(dim=-1)[0].item())
        return OPERATORS[idx] if 0 <= idx < len(OPERATORS) else "unknown"
    if var == "magnitude_bucket":
        idx = int(out["magnitude_logits"].argmax(dim=-1)[0].item())
        return MAG_BUCKETS[idx] if 0 <= idx < len(MAG_BUCKETS) else "[1000+)"
    if var == "sign":
        idx = int(out["sign_logits"].argmax(dim=-1)[0].item())
        return SIGNS[idx] if 0 <= idx < len(SIGNS) else "zero"
    if var == "step_type":
        idx = int(out["step_type_logits"].argmax(dim=-1)[0].item())
        return STEP_TYPES[idx] if 0 <= idx < len(STEP_TYPES) else "operate"
    if mctx.numeric_stats is None:
        return None
    if var == "subresult_value":
        z = float(out["subresult_pred"][0].item())
        st = mctx.numeric_stats["subresult_value"]
        return float(z * st.std + st.mean)
    if var == "lhs_value":
        z = float(out["lhs_pred"][0].item())
        st = mctx.numeric_stats["lhs_value"]
        return float(z * st.std + st.mean)
    if var == "rhs_value":
        z = float(out["rhs_pred"][0].item())
        st = mctx.numeric_stats["rhs_value"]
        return float(z * st.std + st.mean)
    return None


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

    pre_val = _decode_decoder_variable(source_rec, mctx)
    need_row = _rec_with_raw_layer_patch(source_rec, layer, need_patch)
    post_need_val = _decode_decoder_variable(need_row, mctx)
    need_shift = _latent_shift_score(pre_val, post_need_val, mctx.variable)
    need_pass = None if need_shift is None else bool(float(need_shift) > 1e-6)

    post_suff_val = None
    suff_shift = None
    direction_match = None
    suff_pass = None
    if suff_patch is not None and donor_rec is not None:
        suff_row = _rec_with_raw_layer_patch(source_rec, layer, suff_patch)
        post_suff_val = _decode_decoder_variable(suff_row, mctx)
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
    }


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
    p.add_argument("--subspace-specs", default="phase7_results/interventions/variable_subspaces.json")
    p.add_argument("--model-key", default="gpt2-medium")
    p.add_argument("--adapter-config", default=None, help="Optional JSON overrides for model registry entry")
    p.add_argument("--variable", default="subresult_value")
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
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dry-run", action="store_true", help="Validate setup and emit status JSON without running interventions")
    p.add_argument("--output", default="phase7_results/interventions/causal_checks_smoke.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
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
        records = [r for r in records_all if str(r.get("model_key")) == spec.model_key]
    else:
        records = list(records_all)

    if not records:
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
            "layers_requested": list(args.layers),
            "variable": args.variable,
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
            records,
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
            "records_considered": len(records),
            "layers_requested": list(args.layers),
            "variable": args.variable,
            "mediation": {
                "enabled": mediation_enabled,
                "variable": args.mediation_variable,
                "checkpoint": args.state_decoder_checkpoint,
            },
            "subspace_specs_schema": CAUSAL_PATCH_SPEC_SCHEMA,
            "rows": [],
            "compatibility_checks": {"num_records_checked": int(len(records)), "validation_failed": True},
        }
        save_json(args.output, payload)
        raise

    records = records[: args.max_records]
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
            "layers_requested": list(args.layers),
            "variable": args.variable,
            "mediation": {
                "enabled": mediation_enabled,
                "variable": args.mediation_variable,
                "checkpoint": args.state_decoder_checkpoint,
            },
            "subspace_specs_schema": CAUSAL_PATCH_SPEC_SCHEMA,
            "rows": [],
            "compatibility_checks": compatibility_checks,
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
            "layers_requested": list(args.layers),
            "variable": args.variable,
            "mediation": {
                "enabled": mediation_enabled,
                "variable": args.mediation_variable,
                "checkpoint": args.state_decoder_checkpoint,
            },
            "subspace_specs_schema": CAUSAL_PATCH_SPEC_SCHEMA,
            "rows": [],
            "compatibility_checks": compatibility_checks,
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
            "layers_requested": list(args.layers),
            "variable": args.variable,
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
    mctx = MediationContext.from_checkpoint(
        args.state_decoder_checkpoint,
        variable=args.mediation_variable,
        device=args.device,
        expected_model_key=spec.model_key,
        force_enable=mediation_enabled,
    )
    rows = []
    for rec in records:
        rows.append(
            run_causal_checks_on_record(
                rec,
                records,
                specs,
                args.variable,
                args.layers,
                ctx,
                mediation_ctx=mctx,
                seed=args.seed,
            )
        )
        print(f"checked trace={rec.get('trace_id')} step={rec.get('step_idx')} example={rec.get('example_idx')}")

    payload = {
        "schema_version": "phase7_causal_checks_v2",
        "status": "ok" if ctx.supports_subspace_patching else "unsupported_model_causal_subspace",
        "dry_run": False,
        "model_metadata": ctx.metadata(),
        "mediation": mctx.metadata(),
        "rows": rows,
        "mediation_variable": args.mediation_variable,
        "enable_latent_mediation": mediation_enabled,
        "subspace_specs_schema": specs.get("schema_version", CAUSAL_PATCH_SPEC_SCHEMA),
        "compatibility_checks": compatibility_checks,
    }
    save_json(args.output, payload)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()

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
    from .common import CAUSAL_PATCH_SPEC_SCHEMA, load_json, load_pt, save_json, set_seed
    from .model_adapters import BaseCausalLMAdapter
    from .model_registry import create_adapter, resolve_model_spec
except Exception:  # pragma: no cover
    from common import CAUSAL_PATCH_SPEC_SCHEMA, load_json, load_pt, save_json, set_seed
    from model_adapters import BaseCausalLMAdapter
    from model_registry import create_adapter, resolve_model_spec

# Phase4 helper module (SAE loaders + normalization stats for GPT-2 assets)
import phase4.causal_patch_test as p4  # type: ignore  # noqa: E402


def _record_result_token_id(rec: dict) -> int:
    return int(rec.get("result_token_id", rec["structured_state"]["result_token_id"]))


def _record_result_pos(rec: dict) -> int:
    return int(rec.get("result_tok_idx", 0)) - 1


def _record_eq_pos(rec: dict) -> int:
    return int(rec.get("eq_tok_idx", 0))


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
    pos = max(0, min(int(pos), logits.shape[1] - 1))
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
    candidates = []
    for r in records:
        if r is source:
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
            if r is source:
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


def run_causal_checks_on_record(
    source_rec: dict,
    all_records: Sequence[dict],
    specs_payload: dict,
    variable: str,
    layers: Sequence[int],
    ctx: CausalPatchContext,
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
            "schema_version": "phase7_causal_checks_v1",
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
            "schema_version": "phase7_causal_checks_v1",
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
            "subspace_specs_schema": CAUSAL_PATCH_SPEC_SCHEMA,
            "rows": [],
            "compatibility_checks": {"num_records_checked": int(len(records)), "validation_failed": True},
        }
        save_json(args.output, payload)
        raise

    records = records[: args.max_records]
    if not records:
        payload = {
            "schema_version": "phase7_causal_checks_v1",
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
            "subspace_specs_schema": CAUSAL_PATCH_SPEC_SCHEMA,
            "rows": [],
            "compatibility_checks": compatibility_checks,
        }
        save_json(args.output, payload)
        raise SystemExit(
            f"No records remain after applying --max-records={args.max_records}; wrote {args.output}"
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
            "schema_version": "phase7_causal_checks_v1",
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
    rows = []
    for rec in records:
        rows.append(run_causal_checks_on_record(rec, records, specs, args.variable, args.layers, ctx, seed=args.seed))
        print(f"checked trace={rec.get('trace_id')} step={rec.get('step_idx')} example={rec.get('example_idx')}")

    payload = {
        "schema_version": "phase7_causal_checks_v1",
        "status": "ok" if ctx.supports_subspace_patching else "unsupported_model_causal_subspace",
        "dry_run": False,
        "model_metadata": ctx.metadata(),
        "rows": rows,
        "subspace_specs_schema": specs.get("schema_version", CAUSAL_PATCH_SPEC_SCHEMA),
        "compatibility_checks": compatibility_checks,
    }
    save_json(args.output, payload)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()

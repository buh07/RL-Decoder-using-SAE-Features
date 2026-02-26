#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from common import CAUSAL_PATCH_SPEC_SCHEMA, load_json, load_pt, save_json, set_seed

# Phase4 helper module (subspace patching utilities and SAE loaders)
import phase4.causal_patch_test as p4  # type: ignore  # noqa: E402


LATENT_DIM = 12288
HIDDEN_DIM = 1024


def _record_result_token_id(rec: dict) -> int:
    return int(rec.get("result_token_id", rec["structured_state"]["result_token_id"]))


def _record_result_pos(rec: dict) -> int:
    return int(rec.get("result_tok_idx", 0)) - 1


def _record_eq_pos(rec: dict) -> int:
    return int(rec.get("eq_tok_idx", 0))


def _source_input_ids(rec: dict, device: str) -> torch.Tensor:
    return torch.tensor(rec["token_ids"], dtype=torch.long, device=device).unsqueeze(0)


def _feature_indices_for(specs_payload: dict, variable: str, layer: int) -> List[int]:
    if "specs" in specs_payload:
        for s in specs_payload["specs"]:
            if str(s.get("variable")) == str(variable) and int(s.get("layer")) == int(layer):
                return [int(x) for x in s.get("feature_indices", [])]
    # fallback: treat payload as layer->features map
    if str(layer) in specs_payload:
        vals = specs_payload[str(layer)]
        return [int(x) for x in vals]
    return []


def _select_control_features(k: int, exclude: Sequence[int], seed: int = 17) -> List[int]:
    rng = random.Random(seed)
    exclude_set = set(int(x) for x in exclude)
    pool = [i for i in range(LATENT_DIM) if i not in exclude_set]
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


def gpt2_necessity_ablation(
    source_rec: dict,
    layer: int,
    feature_indices: Sequence[int],
    model,
    saes,
    device: str,
) -> Dict[str, Any]:
    if not feature_indices:
        return {"supported": False, "reason": "empty_feature_indices"}
    src_ids = _source_input_ids(source_rec, device)
    eq_pos = _record_eq_pos(source_rec)
    pred_pos = _record_result_pos(source_rec)
    correct_tok = _record_result_token_id(source_rec)

    with torch.no_grad():
        logits_base = model(src_ids).logits
    lp_base = _logprob_at_token(logits_base, pred_pos, correct_tok)

    h_src = source_rec["raw_hidden"][layer].to(device).float()
    D_norm = _build_subspace_decoder_cols(saes[layer], feature_indices, device)
    coords = D_norm.T @ h_src
    comp = D_norm @ coords
    patch_vec = h_src - comp
    off_ratio = float(comp.norm().item() / max(1e-6, h_src.norm().item()))

    logits_ablate = p4.run_patched_forward(model, src_ids, layer, eq_pos, patch_vec)
    lp_ablate = _logprob_at_token(logits_ablate, pred_pos, correct_tok)
    return {
        "supported": True,
        "baseline_logprob": lp_base,
        "patched_logprob": lp_ablate,
        "delta_logprob": lp_ablate - lp_base,
        "off_manifold_ratio": off_ratio,
    }


def gpt2_sufficiency_patch(
    source_rec: dict,
    donor_rec: dict,
    layer: int,
    feature_indices: Sequence[int],
    model,
    saes,
    device: str,
    control_feature_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if not feature_indices:
        return {"supported": False, "reason": "empty_feature_indices"}
    src_ids = _source_input_ids(source_rec, device)
    eq_pos = _record_eq_pos(source_rec)
    pred_pos = _record_result_pos(source_rec)
    donor_tok = _record_result_token_id(donor_rec)

    with torch.no_grad():
        logits_base = model(src_ids).logits
    lp_base = _logprob_at_token(logits_base, pred_pos, donor_tok)

    h_src = source_rec["raw_hidden"][layer].to(device).float()
    h_don = donor_rec["raw_hidden"][layer].to(device).float()
    delta_h = h_don - h_src

    D_norm = _build_subspace_decoder_cols(saes[layer], feature_indices, device)
    projected = D_norm @ (D_norm.T @ delta_h)
    patch_vec = h_src + projected
    off_ratio = float(projected.norm().item() / max(1e-6, h_src.norm().item()))

    logits_patch = p4.run_patched_forward(model, src_ids, layer, eq_pos, patch_vec)
    lp_patch = _logprob_at_token(logits_patch, pred_pos, donor_tok)

    control_gain = None
    if control_feature_indices:
        Dc = _build_subspace_decoder_cols(saes[layer], control_feature_indices, device)
        proj_c = Dc @ (Dc.T @ delta_h)
        patch_vec_c = h_src + proj_c
        logits_c = p4.run_patched_forward(model, src_ids, layer, eq_pos, patch_vec_c)
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
                if math.isclose(float(s.get("subresult_value")), float(src_state.get("subresult_value")), rel_tol=1e-6, abs_tol=1e-6):
                    continue
            except Exception:
                continue
        candidates.append(r)
    if not candidates:
        # Relax magnitude bucket match
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
class Gpt2PatchContext:
    device: str
    model: Any
    saes: Dict[int, Any]
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    @classmethod
    def load(
        cls,
        device: str,
        saes_dir: str = "phase2_results/saes_gpt2_12x_topk/saes",
        activations_dir: str = "phase2_results/activations",
    ) -> "Gpt2PatchContext":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(p4.MODEL_ID, torch_dtype=torch.float32).to(device).eval()
        saes = p4.load_saes(Path(saes_dir), device)
        norm_stats = p4.load_norm_stats(Path(activations_dir), device)
        return cls(device=device, model=model, saes=saes, norm_stats=norm_stats)


def run_causal_checks_on_record(
    source_rec: dict,
    all_records: Sequence[dict],
    specs_payload: dict,
    variable: str,
    layers: Sequence[int],
    ctx: Gpt2PatchContext,
    off_manifold_max_ratio: float = 0.75,
    seed: int = 17,
) -> Dict[str, Any]:
    donor = select_matched_donor(all_records, source_rec, variable=variable, seed=seed)
    out = {"variable": variable, "source_trace_id": source_rec.get("trace_id"), "source_step_idx": int(source_rec.get("step_idx", -1)), "layers": {}}
    for layer in layers:
        feats = _feature_indices_for(specs_payload, variable, int(layer))
        ctrl_feats = _select_control_features(len(feats), feats, seed=seed + int(layer)) if feats else []
        need = gpt2_necessity_ablation(source_rec, int(layer), feats, ctx.model, ctx.saes, ctx.device) if feats else {"supported": False, "reason": "empty_feature_indices"}
        suff = (
            gpt2_sufficiency_patch(source_rec, donor, int(layer), feats, ctx.model, ctx.saes, ctx.device, control_feature_indices=ctrl_feats)
            if donor is not None and feats
            else {"supported": False, "reason": "no_donor" if donor is None else "empty_feature_indices"}
        )
        # Specificity is encoded via control delta in sufficiency patch.
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
    p.add_argument("--variable", default="subresult_value")
    p.add_argument("--layers", type=int, nargs="*", default=[22])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--saes-dir", default="phase2_results/saes_gpt2_12x_topk/saes")
    p.add_argument("--activations-dir", default="phase2_results/activations")
    p.add_argument("--max-records", type=int, default=20)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", default="phase7_results/interventions/causal_checks_smoke.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    records = load_pt(args.trace_dataset)
    records = [r for r in records if r.get("gsm8k_split") == "test"] or records
    records = records[: args.max_records]
    specs = load_json(args.subspace_specs)
    print(f"Loading GPT-2/SAEs on {args.device}...")
    ctx = Gpt2PatchContext.load(args.device, saes_dir=args.saes_dir, activations_dir=args.activations_dir)
    rows = []
    for rec in records:
        rows.append(run_causal_checks_on_record(rec, records, specs, args.variable, args.layers, ctx, seed=args.seed))
        print(f"checked trace={rec.get('trace_id')} step={rec.get('step_idx')} example={rec.get('example_idx')}")
    save_json(args.output, {"schema_version": "phase7_causal_checks_v1", "rows": rows})
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()

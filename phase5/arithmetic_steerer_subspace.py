#!/usr/bin/env python3
"""
Phase 5b — Arithmetic Steerer (Probe-Subspace Variant)

Identical to arithmetic_steerer.py except the mean-difference steering vector
is restricted to the probe feature subspace: only the top-K probe features
(from the ridge probe trained in Phase 4) are perturbed; all other SAE
dimensions are left untouched.

Method:
  1. Load the top-50 probe feature indices per layer from Phase 4
     (top_features_per_layer.json, "result" key).
  2. Split the 674-record dataset into high-|C| and low-|C| halves.
  3. For each layer L:
       Δf_sub[L] = mean(result_features[L][probe_idx] | high-C)
                 − mean(result_features[L][probe_idx] | low-C)
       (a vector living in the 50-dim probe subspace of the 12288-dim latent)
  4. For test examples: encode residual at layer L → add α × Δf_sub[L] to
     probe dimensions only → decode → inject → measure log_prob shift.
  5. Sweep α ∈ {0.5, 1, 2, 4} across all 24 layers.

Parallelism:
  Use --layers to restrict evaluation to a subset of layers, enabling
  multiple GPU runs in parallel. Results are merged with merge_subspace.py.

  GPU 0 (layers 0-7):
    CUDA_VISIBLE_DEVICES=0 python3 phase5/arithmetic_steerer_subspace.py \\
        --layers 0 1 2 3 4 5 6 7 --output phase5_results/steering_subspace/shard_0

  GPU 1 (layers 8-15):
    CUDA_VISIBLE_DEVICES=1 python3 phase5/arithmetic_steerer_subspace.py \\
        --layers 8 9 10 11 12 13 14 15 --output phase5_results/steering_subspace/shard_1

  GPU 2 (layers 16-23):
    CUDA_VISIBLE_DEVICES=2 python3 phase5/arithmetic_steerer_subspace.py \\
        --layers 16 17 18 19 20 21 22 23 --output phase5_results/steering_subspace/shard_2

  Then merge:
    python3 phase5/merge_subspace_results.py

See also (Phase 5c — stronger variant):
    phase5/arithmetic_steerer_nn_transfer.py  (to be implemented if needed)
    Uses nearest-neighbour example-pair feature transfer in the probe subspace,
    which replicates the Phase 4r mechanism (+0.107 Δlog_prob) but generalises
    to unseen test examples via training-set lookup.  For each test record T,
    finds the closest high-C training example S in SAE feature space at layer L,
    then replaces T's probe-subspace feature activations with S's values.
    Expected Δlog_prob: +0.05 to +0.10 (vs subspace mean-diff's ~0 to +0.03).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_architecture import SparseAutoencoder
from sae_config import SAEConfig

NUM_LAYERS  = 24
HIDDEN_DIM  = 1024
LATENT_DIM  = 12288
MODEL_ID    = "gpt2-medium"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",          default="phase4_results/topk/collection/gsm8k_arithmetic_dataset.pt")
    p.add_argument("--saes-dir",         default="phase2_results/saes_gpt2_12x_topk/saes")
    p.add_argument("--activations-dir",  default="phase2_results/activations")
    p.add_argument("--probe-features",   default="phase4_results/topk/probe/top_features_per_layer.json",
                   help="Path to top_features_per_layer.json from Phase 4 probe")
    p.add_argument("--probe-position",   default="result",
                   choices=["eq", "pre_eq", "result"],
                   help="Which token position's probe features to use (default: result)")
    p.add_argument("--layers",           type=int, nargs="+", default=list(range(NUM_LAYERS)),
                   help="Which layers to evaluate (default: all 24). Use to split work across GPUs.")
    p.add_argument("--output",           default="phase5_results/steering_subspace")
    p.add_argument("--device",           default="cuda:0")
    p.add_argument("--alpha",            type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--test-fraction",    type=float, default=0.3)
    p.add_argument("--num-test-pairs",   type=int, default=50)
    return p.parse_args()


# ── SAE + norm loading ────────────────────────────────────────────────────────

def load_saes(saes_dir: Path, device: str, layers: List[int]) -> Dict[int, SparseAutoencoder]:
    saes: Dict[int, SparseAutoencoder] = {}
    for layer_idx in layers:
        ckpt_path = saes_dir / f"gpt2-medium_layer{layer_idx}_sae.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cd = ckpt["config"]
        cfg = SAEConfig(
            input_dim=cd["input_dim"],
            expansion_factor=cd["expansion_factor"],
            use_relu=cd.get("use_relu", True),
            use_topk=cd.get("use_topk", False),
            topk_k=cd.get("topk_k", 0),
            use_amp=False,
        )
        sae = SparseAutoencoder(cfg)
        sae.load_state_dict(ckpt["model_state_dict"])
        sae = sae.to(device).eval()
        saes[layer_idx] = sae
    print(f"  Loaded {len(saes)} SAEs.")
    return saes


def load_norm_stats(activations_dir: Path, device: str, layers: List[int]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in layers:
        path = activations_dir / f"gpt2-medium_layer{layer_idx}_activations.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        acts = payload["activations"] if isinstance(payload, dict) else payload
        if acts.dim() == 3:
            acts = acts.reshape(-1, acts.shape[-1])
        acts = acts.float()
        stats[layer_idx] = (
            acts.mean(dim=0).to(device),
            acts.std(dim=0).clamp_min(1e-6).to(device),
        )
    return stats


def normalize(x: torch.Tensor, stats: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if stats is None:
        return x
    mean, std = stats
    return (x - mean) / std


def unnormalize(x: torch.Tensor, stats: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if stats is None:
        return x
    mean, std = stats
    return x * std + mean


# ── Probe feature loading ─────────────────────────────────────────────────────

def load_probe_indices(probe_features_path: Path, position: str, device: str) -> Dict[int, torch.Tensor]:
    """Returns dict: layer -> LongTensor of probe feature indices (on device)."""
    with open(probe_features_path) as f:
        top_features = json.load(f)
    per_layer = top_features[position]  # list of 24 lists, each with 50 indices
    return {L: torch.tensor(per_layer[L], dtype=torch.long, device=device)
            for L in range(len(per_layer))}


# ── Activation capture + patch hooks ─────────────────────────────────────────

def register_capture_hooks(model) -> Tuple[Dict[int, torch.Tensor], List]:
    store: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(L: int):
        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            store[L] = h.detach().cpu()
        return hook_fn

    for i in range(NUM_LAYERS):
        handles.append(model.transformer.h[i].register_forward_hook(make_hook(i)))
    return store, handles


class PatchedActivationHook:
    def __init__(self, tok_pos: int, patch_vec: torch.Tensor) -> None:
        self.tok_pos = tok_pos
        self.patch_vec = patch_vec
        self.fired = False

    def __call__(self, module, inp, output):
        if self.fired:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        hidden[0, self.tok_pos, :] = self.patch_vec
        self.fired = True
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


# ── Dataset split ─────────────────────────────────────────────────────────────

def split_dataset(records: list, test_fraction: float):
    c_vals = np.array([abs(r["C"]) for r in records])
    median_c = np.median(c_vals)
    high_idx = [i for i, r in enumerate(records) if abs(r["C"]) >= median_c]
    low_idx  = [i for i, r in enumerate(records) if abs(r["C"]) <  median_c]
    n_test = max(1, int(len(records) * test_fraction))
    all_idx = list(range(len(records)))
    train_idx = all_idx[:-n_test]
    test_idx  = all_idx[-n_test:]
    high_train = [i for i in high_idx if i in set(train_idx)]
    low_train  = [i for i in low_idx  if i in set(train_idx)]
    return high_train, low_train, test_idx


# ── Steering vector computation (probe subspace) ──────────────────────────────

def compute_steering_vectors(
    records: list,
    high_idx: List[int],
    low_idx: List[int],
    probe_indices: Dict[int, torch.Tensor],
    layers: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Δf_sub[L]: (LATENT_DIM,) tensor with non-zero values ONLY at probe_indices[L].
    """
    steering = {}
    for layer in layers:
        high_feats = torch.stack([records[i]["result_features"][layer] for i in high_idx]).float()
        low_feats  = torch.stack([records[i]["result_features"][layer] for i in low_idx ]).float()
        delta_full = high_feats.mean(0) - low_feats.mean(0)
        # Zero out non-probe dims
        probe_idx = probe_indices[layer].cpu()
        delta_sub = torch.zeros_like(delta_full)
        delta_sub[probe_idx] = delta_full[probe_idx]
        steering[layer] = delta_sub
    return steering


# ── Single-record steering evaluation ────────────────────────────────────────

@torch.no_grad()
def evaluate_steering_on_record(
    record: dict,
    model,
    store_baseline: Dict[int, torch.Tensor],
    log_prob_base: float,
    saes: Dict[int, SparseAutoencoder],
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    steering_vectors: Dict[int, torch.Tensor],
    layer: int,
    alpha: float,
    device: str,
) -> float:
    """Return Δlog_prob = log_p(correct | steered) − log_p(correct | baseline)."""
    token_ids = torch.tensor(record["token_ids"], device=device).unsqueeze(0)
    result_tok_idx = record["result_tok_idxs"][0]
    result_tok_id  = record["token_ids"][result_tok_idx]

    # Use pre-captured baseline activations
    h_raw = store_baseline[layer][0, result_tok_idx - 1, :].to(device)
    h_norm = normalize(h_raw, norm_stats.get(layer))
    h_feat = saes[layer].encode(h_norm.unsqueeze(0)).squeeze(0)

    # Add steering only in probe subspace
    delta_feat = steering_vectors[layer].to(device)
    h_feat_steered = h_feat + alpha * delta_feat

    h_norm_steered = saes[layer].decode(h_feat_steered.unsqueeze(0)).squeeze(0)
    h_patched = unnormalize(h_norm_steered, norm_stats.get(layer))

    patch_hook = PatchedActivationHook(tok_pos=result_tok_idx - 1, patch_vec=h_patched.float())
    handle = model.transformer.h[layer].register_forward_hook(patch_hook)
    logits_steered = model(token_ids).logits
    handle.remove()

    log_prob_steered = F.log_softmax(logits_steered[0, result_tok_idx - 1, :], dim=-1)[result_tok_id].item()
    return log_prob_steered - log_prob_base


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(args.layers)
    device = args.device

    print(f"Layers to evaluate: {layers}")
    print(f"Loading dataset: {args.dataset}")
    records = torch.load(args.dataset, weights_only=False)
    print(f"  {len(records)} records")

    print(f"Loading probe feature indices: {args.probe_features} (position={args.probe_position})")
    probe_indices = load_probe_indices(Path(args.probe_features), args.probe_position, device)
    probe_k = len(probe_indices[0])
    print(f"  {probe_k} probe features per layer")

    print(f"Loading SAEs for layers {layers}...")
    saes = load_saes(Path(args.saes_dir), device, layers)

    print(f"Loading norm stats for layers {layers}...")
    norm_stats = load_norm_stats(Path(args.activations_dir), device, layers)

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = model.to(device).eval()

    high_idx, low_idx, test_idx = split_dataset(records, args.test_fraction)
    print(f"\nDataset split: {len(high_idx)} high-C train, {len(low_idx)} low-C train, {len(test_idx)} test")

    print(f"Computing subspace steering vectors...")
    steering_vectors = compute_steering_vectors(records, high_idx, low_idx, probe_indices, layers)

    test_records = [records[i] for i in test_idx[:args.num_test_pairs]]
    alphas = args.alpha
    print(f"\nEvaluating {len(test_records)} records × {len(layers)} layers × {len(alphas)} α values")

    results: Dict[int, Dict[float, List[float]]] = {
        L: {a: [] for a in alphas} for L in layers
    }

    for rec_num, rec in enumerate(test_records):
        if rec_num % 10 == 0:
            print(f"  Record {rec_num}/{len(test_records)}...")

        # One baseline forward pass per record, shared across all layers
        token_ids = torch.tensor(rec["token_ids"], device=device).unsqueeze(0)
        result_tok_idx = rec["result_tok_idxs"][0]
        result_tok_id  = rec["token_ids"][result_tok_idx]
        store, handles = register_capture_hooks(model)
        with torch.no_grad():
            logits_base = model(token_ids).logits
        for h in handles:
            h.remove()
        log_prob_base = F.log_softmax(logits_base[0, result_tok_idx - 1, :], dim=-1)[result_tok_id].item()

        for layer in layers:
            for alpha in alphas:
                try:
                    delta = evaluate_steering_on_record(
                        rec, model, store, log_prob_base,
                        saes, norm_stats, steering_vectors, layer, alpha, device
                    )
                    results[layer][alpha].append(delta)
                except Exception as e:
                    print(f"    Error L{layer} α={alpha}: {e}")
                    results[layer][alpha].append(float("nan"))

    mean_results: Dict[int, Dict[float, float]] = {}
    for layer in layers:
        mean_results[layer] = {}
        for alpha in alphas:
            vals = [v for v in results[layer][alpha] if not np.isnan(v)]
            mean_results[layer][alpha] = float(np.mean(vals)) if vals else float("nan")

    print(f"\n{'Layer':>5}  " + "  ".join(f"α={a:4.1f}" for a in alphas))
    print("-" * (7 + 10 * len(alphas)))
    for layer in layers:
        row = f"{layer:5d}  "
        row += "  ".join(f"{mean_results[layer][a]:+7.4f}" for a in alphas)
        print(row)

    best_layer, best_alpha, best_val = layers[0], alphas[0], -999.0
    for layer in layers:
        for alpha in alphas:
            v = mean_results[layer][alpha]
            if not np.isnan(v) and v > best_val:
                best_val = v
                best_layer = layer
                best_alpha = alpha
    print(f"\nBest (this shard): layer={best_layer}, α={best_alpha}, mean Δlog_prob={best_val:+.4f}")

    # Save shard results
    shard_name = f"shard_L{layers[0]}-{layers[-1]}.json"
    results_json = {
        "method": "subspace_mean_diff",
        "probe_position": args.probe_position,
        "probe_k": probe_k,
        "layers_evaluated": layers,
        "alphas": alphas,
        "mean_delta_logprob": {str(L): {str(a): mean_results[L][a] for a in alphas} for L in layers},
        "best_layer_shard": best_layer,
        "best_alpha_shard": best_alpha,
        "best_mean_delta_logprob_shard": best_val,
        "num_test_records": len(test_records),
        "high_c_train": len(high_idx),
        "low_c_train": len(low_idx),
    }
    json_path = out_dir / shard_name
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Shard results: {json_path}")
    print("\nDone. Run merge_subspace_results.py after all shards finish.")


if __name__ == "__main__":
    main()

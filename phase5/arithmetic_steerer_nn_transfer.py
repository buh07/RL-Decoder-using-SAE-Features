#!/usr/bin/env python3
"""
Phase 5c — Arithmetic Steerer (Nearest-Neighbour Feature Transfer)

Instead of using a population mean-difference vector (Phase 5/5b), this script
transfers actual feature values from the nearest high-C training example to the
probe subspace of the test example's SAE features.

Method:
  1. Load the pre-computed SAE features for all 674 records.
  2. Split into high-|C| train, low-|C| train, and test sets (same split as 5/5b).
  3. Pre-compute train feature matrices for each layer.
  4. For each test record T at layer L:
       a. Find nearest high-C train record S by L2 distance in the FULL SAE
          feature space at layer L:
              S = argmin_i ||f_T[L] - f_i[L]||_2,  i in high_C_train
       b. Live-encode T's residual stream activation at layer L through the SAE.
       c. Interpolate T's probe-subspace features toward S's:
              h_feat_patched[probe_idx] = (1-β)*h_feat[probe_idx] + β*f_S[probe_idx]
       d. Decode, unnormalize, inject via PatchedActivationHook.
       e. Measure Δlog_prob for the correct result token.
  5. Sweep β ∈ {0.25, 0.5, 0.75, 1.0} across all 24 layers.

Motivation:
  Phase 4r's causal patch (+0.107 Δlog_prob) worked by transferring actual feature
  values from a selected high-C example.  Phase 5b's mean-diff steering gives −2.385
  because population-average directions destroy the model.  Phase 5c generalises
  Phase 4r to unseen test examples by using nearest-neighbour lookup in feature
  space, keeping the "real example transfer" mechanism intact.

Parallelism (3 GPU shards):
  CUDA_VISIBLE_DEVICES=0 python3 phase5/arithmetic_steerer_nn_transfer.py \\
      --layers 0 1 2 3 4 5 6 7 \\
      --output phase5_results/steering_nn_transfer/shard_L0-7.json

  CUDA_VISIBLE_DEVICES=1 python3 phase5/arithmetic_steerer_nn_transfer.py \\
      --layers 8 9 10 11 12 13 14 15 \\
      --output phase5_results/steering_nn_transfer/shard_L8-15.json

  CUDA_VISIBLE_DEVICES=2 python3 phase5/arithmetic_steerer_nn_transfer.py \\
      --layers 16 17 18 19 20 21 22 23 \\
      --output phase5_results/steering_nn_transfer/shard_L16-23.json

  Then merge:
    python3 phase5/merge_nn_transfer_results.py
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
    p.add_argument("--output",           default="phase5_results/steering_nn_transfer/shard.json")
    p.add_argument("--device",           default="cuda:0")
    p.add_argument("--beta",             type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                   help="Interpolation factor(s): 0=keep T's features, 1=fully replace with S's")
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
    """Identical split logic to Phase 5/5b for reproducibility."""
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


# ── Precompute train feature matrices ─────────────────────────────────────────

def build_train_feature_matrices(
    records: list,
    high_idx: List[int],
    layers: List[int],
    device: str,
) -> Dict[int, torch.Tensor]:
    """
    Returns dict: layer -> (N_high_train, LATENT_DIM) float32 tensor on device.
    Pre-computed so NN search can use torch ops on GPU.
    """
    print(f"  Pre-computing train feature matrices ({len(high_idx)} records × {len(layers)} layers)...")
    matrices: Dict[int, torch.Tensor] = {}
    for layer in layers:
        stacked = torch.stack(
            [records[i]["result_features"][layer].float() for i in high_idx]
        )  # (N_high, 12288) float32
        matrices[layer] = stacked.to(device)
    return matrices


# ── Nearest-neighbour lookup ──────────────────────────────────────────────────

@torch.no_grad()
def find_nearest_neighbour(
    query_feats: torch.Tensor,        # (LATENT_DIM,) float32, on device
    train_matrix: torch.Tensor,       # (N_train, LATENT_DIM) float32, on device
) -> int:
    """Return index into train_matrix of the nearest neighbour by L2 distance."""
    diffs = train_matrix - query_feats.unsqueeze(0)   # (N_train, LATENT_DIM)
    dists = (diffs * diffs).sum(dim=1)                # (N_train,) — squared L2
    return int(dists.argmin().item())


# ── Single-record NN-transfer evaluation ──────────────────────────────────────

@torch.no_grad()
def evaluate_nn_transfer_on_record(
    record: dict,
    rec_idx: int,
    model,
    store_baseline: Dict[int, torch.Tensor],
    log_prob_base: float,
    saes: Dict[int, SparseAutoencoder],
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    probe_indices: Dict[int, torch.Tensor],
    train_matrices: Dict[int, torch.Tensor],   # layer -> (N_high, LATENT_DIM)
    layers: List[int],
    betas: List[float],
    device: str,
) -> Dict[int, Dict[float, float]]:
    """
    For each (layer, beta) pair, return Δlog_prob = log_p(correct|patched) - baseline.

    Within each layer we:
      1. Live-encode T's residual (from baseline capture).
      2. Find nearest high-C train record S using pre-computed features (full space).
      3. Replace probe dims: h_feat_patched[probe_idx] = (1-β)*h_feat[probe_idx] + β*f_S[probe_idx]
      4. Decode, unnormalize, inject, measure shift.
    """
    token_ids      = torch.tensor(record["token_ids"], device=device).unsqueeze(0)
    result_tok_idx = record["result_tok_idxs"][0]
    result_tok_id  = record["token_ids"][result_tok_idx]

    layer_results: Dict[int, Dict[float, float]] = {L: {} for L in layers}

    for layer in layers:
        probe_idx = probe_indices[layer]                        # (50,) LongTensor on device
        train_mat = train_matrices[layer]                       # (N_high, 12288) on device

        # --- T's pre-computed features (for NN search) ---
        f_T_precomp = record["result_features"][layer].float().to(device)  # (12288,)

        # --- Find nearest high-C train example (in full feature space) ---
        nn_s = find_nearest_neighbour(f_T_precomp, train_mat)
        f_S = train_mat[nn_s]                                  # (12288,) S's features

        # --- Live-encode T's residual at this layer ---
        h_raw  = store_baseline[layer][0, result_tok_idx - 1, :].to(device)
        h_norm = normalize(h_raw, norm_stats.get(layer))
        h_feat = saes[layer].encode(h_norm.unsqueeze(0)).squeeze(0)  # (12288,)

        for beta in betas:
            # Interpolate probe dims only
            h_feat_patched = h_feat.clone()
            h_feat_patched[probe_idx] = (
                (1.0 - beta) * h_feat[probe_idx] + beta * f_S[probe_idx]
            )

            h_norm_steered = saes[layer].decode(h_feat_patched.unsqueeze(0)).squeeze(0)
            h_patched      = unnormalize(h_norm_steered, norm_stats.get(layer))

            patch_hook = PatchedActivationHook(tok_pos=result_tok_idx - 1, patch_vec=h_patched.float())
            handle = model.transformer.h[layer].register_forward_hook(patch_hook)
            logits_steered = model(token_ids).logits
            handle.remove()

            log_prob_steered = F.log_softmax(
                logits_steered[0, result_tok_idx - 1, :], dim=-1
            )[result_tok_id].item()
            layer_results[layer][beta] = log_prob_steered - log_prob_base

    return layer_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted(args.layers)
    betas  = args.beta
    device = args.device

    print(f"Layers to evaluate: {layers}")
    print(f"Beta sweep: {betas}")
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

    # Pre-compute train feature matrices (GPU) — enables fast NN lookup
    train_matrices = build_train_feature_matrices(records, high_idx, layers, device)

    test_records = [records[i] for i in test_idx[:args.num_test_pairs]]
    print(f"\nEvaluating {len(test_records)} records × {len(layers)} layers × {len(betas)} β values")

    # results[layer][beta] = list of Δlog_prob values
    results: Dict[int, Dict[float, List[float]]] = {
        L: {b: [] for b in betas} for L in layers
    }

    for rec_num, (rec, rec_global_idx) in enumerate(zip(test_records, test_idx[:args.num_test_pairs])):
        if rec_num % 10 == 0:
            print(f"  Record {rec_num}/{len(test_records)}...")

        # One baseline forward pass per record — captures activations at all layers
        token_ids      = torch.tensor(rec["token_ids"], device=device).unsqueeze(0)
        result_tok_idx = rec["result_tok_idxs"][0]
        result_tok_id  = rec["token_ids"][result_tok_idx]

        store, handles = register_capture_hooks(model)
        with torch.no_grad():
            logits_base = model(token_ids).logits
        for h in handles:
            h.remove()
        log_prob_base = F.log_softmax(
            logits_base[0, result_tok_idx - 1, :], dim=-1
        )[result_tok_id].item()

        try:
            layer_results = evaluate_nn_transfer_on_record(
                rec, rec_global_idx, model, store, log_prob_base,
                saes, norm_stats, probe_indices, train_matrices,
                layers, betas, device
            )
            for layer in layers:
                for beta in betas:
                    results[layer][beta].append(layer_results[layer][beta])
        except Exception as e:
            print(f"    Error record {rec_num}: {e}")
            for layer in layers:
                for beta in betas:
                    results[layer][beta].append(float("nan"))

    # Aggregate
    mean_results: Dict[int, Dict[float, float]] = {}
    for layer in layers:
        mean_results[layer] = {}
        for beta in betas:
            vals = [v for v in results[layer][beta] if not np.isnan(v)]
            mean_results[layer][beta] = float(np.mean(vals)) if vals else float("nan")

    # Print table
    print(f"\n{'Layer':>5}  " + "  ".join(f"β={b:.2f}" for b in betas))
    print("-" * (7 + 10 * len(betas)))
    for layer in layers:
        row = f"{layer:5d}  "
        row += "  ".join(f"{mean_results[layer][b]:+7.4f}" for b in betas)
        print(row)

    best_layer, best_beta, best_val = layers[0], betas[0], -999.0
    for layer in layers:
        for beta in betas:
            v = mean_results[layer][beta]
            if not np.isnan(v) and v > best_val:
                best_val  = v
                best_layer = layer
                best_beta  = beta
    print(f"\nBest (this shard): layer={best_layer}, β={best_beta}, mean Δlog_prob={best_val:+.4f}")

    # Save shard results
    results_json = {
        "method": "nn_transfer",
        "probe_position": args.probe_position,
        "probe_k": probe_k,
        "layers_evaluated": layers,
        "betas": betas,
        "mean_delta_logprob": {
            str(L): {str(b): mean_results[L][b] for b in betas}
            for L in layers
        },
        "best_layer_shard": best_layer,
        "best_beta_shard": best_beta,
        "best_mean_delta_logprob_shard": best_val,
        "num_test_records": len(test_records),
        "high_c_train": len(high_idx),
        "low_c_train": len(low_idx),
        "baseline_phase5_full_space_best": -2.854,
        "baseline_phase5b_subspace_best": -2.385,
        "baseline_phase4r_subspace_patch_best": 0.1066,
    }
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Shard results: {out_path}")
    print("\nDone. Run merge_nn_transfer_results.py after all shards finish.")


if __name__ == "__main__":
    main()

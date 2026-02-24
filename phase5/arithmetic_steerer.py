#!/usr/bin/env python3
"""
Phase 5 — Arithmetic Steerer

Computes a mean-difference steering vector in SAE feature space and applies it
at test time to steer arithmetic predictions.

Method:
  1. Split the 674-record dataset into high-|C| and low-|C| halves.
  2. For each layer L:
       Δf[L] = mean(result_features[L] | high-C records)
             − mean(result_features[L] | low-C records)
  3. For test examples: encode residual at layer L → add α × Δf[L] → decode →
     inject as a patched activation → measure log_prob shift.
  4. Sweep α ∈ {0.5, 1, 2, 4} across all 24 layers; find the (layer, α) pair
     that most reliably increases log_prob of the correct result token.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 phase5/arithmetic_steerer.py \
        --dataset       phase4_results/topk/collection/gsm8k_arithmetic_dataset.pt \
        --saes-dir      phase2_results/saes_gpt2_12x_topk/saes \
        --activations-dir phase2_results/activations \
        --output        phase5_results/steering \
        --device        cuda:0
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
    p.add_argument("--output",           default="phase5_results/steering")
    p.add_argument("--device",           default="cuda:0")
    p.add_argument("--alpha",            type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--test-fraction",    type=float, default=0.3,
                   help="Fraction of records held out for testing (default 0.3)")
    p.add_argument("--num-test-pairs",   type=int, default=50,
                   help="Number of test records to evaluate per (layer, alpha) pair")
    p.add_argument("--batch-size",       type=int, default=1)
    return p.parse_args()


# ── SAE + norm loading (same pattern as causal_patch_test.py) ─────────────────

def load_saes(saes_dir: Path, device: str) -> Dict[int, SparseAutoencoder]:
    saes: Dict[int, SparseAutoencoder] = {}
    for layer_idx in range(NUM_LAYERS):
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


def load_norm_stats(activations_dir: Path, device: str) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in range(NUM_LAYERS):
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
    """Split by median |C| into high/low halves, then reserve test_fraction for evaluation."""
    c_vals = np.array([abs(r["C"]) for r in records])
    median_c = np.median(c_vals)
    high_idx = [i for i, r in enumerate(records) if abs(r["C"]) >= median_c]
    low_idx  = [i for i, r in enumerate(records) if abs(r["C"]) <  median_c]
    # Hold out last test_fraction for test
    n_test = max(1, int(len(records) * test_fraction))
    all_idx = list(range(len(records)))
    train_idx = all_idx[:-n_test]
    test_idx  = all_idx[-n_test:]
    high_train = [i for i in high_idx if i in set(train_idx)]
    low_train  = [i for i in low_idx  if i in set(train_idx)]
    return high_train, low_train, test_idx


# ── Steering vector computation ───────────────────────────────────────────────

def compute_steering_vectors(
    records: list,
    high_idx: List[int],
    low_idx: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Δf[L] = mean(result_features[L] | high-C) − mean(result_features[L] | low-C)
    Returns: dict layer -> (LATENT_DIM,) float32 tensor (on CPU)
    """
    steering = {}
    for layer in range(NUM_LAYERS):
        high_feats = torch.stack([records[i]["result_features"][layer] for i in high_idx]).float()
        low_feats  = torch.stack([records[i]["result_features"][layer] for i in low_idx ]).float()
        delta = high_feats.mean(0) - low_feats.mean(0)
        steering[layer] = delta  # (LATENT_DIM,)
    return steering


# ── Single-record steering evaluation ────────────────────────────────────────

@torch.no_grad()
def evaluate_steering_on_record(
    record: dict,
    model,
    tokenizer,
    saes: Dict[int, SparseAutoencoder],
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    steering_vectors: Dict[int, torch.Tensor],
    layer: int,
    alpha: float,
    device: str,
) -> float:
    """Return Δlog_prob = log_p(correct_result | steered) − log_p(correct_result | baseline)."""
    token_ids = torch.tensor(record["token_ids"], device=device).unsqueeze(0)
    result_tok_idx = record["result_tok_idxs"][0]
    result_tok_id  = record["token_ids"][result_tok_idx]

    # 1. Baseline: capture activations
    store, handles = register_capture_hooks(model)
    logits_base = model(token_ids).logits  # (1, seq, vocab)
    for h in handles:
        h.remove()

    log_prob_base = F.log_softmax(logits_base[0, result_tok_idx - 1, :], dim=-1)[result_tok_id].item()

    # 2. Compute patched activation
    h_raw = store[layer][0, result_tok_idx - 1, :].to(device)  # (HIDDEN_DIM,)

    # Encode raw residual into SAE feature space
    h_norm = normalize(h_raw, norm_stats.get(layer))
    h_feat = saes[layer].encode(h_norm.unsqueeze(0)).squeeze(0)  # (LATENT_DIM,)

    # Add scaled steering vector (in feature space)
    delta_feat = steering_vectors[layer].to(device)
    h_feat_steered = h_feat + alpha * delta_feat

    # Decode back to residual-stream space
    h_norm_steered = saes[layer].decode(h_feat_steered.unsqueeze(0)).squeeze(0)
    h_patched = unnormalize(h_norm_steered, norm_stats.get(layer))

    # 3. Re-run model with patched activation injected
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
    device = args.device

    print(f"Loading dataset: {args.dataset}")
    records = torch.load(args.dataset, weights_only=False)
    print(f"  {len(records)} records")

    print(f"Loading SAEs: {args.saes_dir}")
    saes = load_saes(Path(args.saes_dir), device)

    print(f"Loading norm stats: {args.activations_dir}")
    norm_stats = load_norm_stats(Path(args.activations_dir), device)
    print(f"  Norm stats for {len(norm_stats)} layers")

    # Load model + tokenizer
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = model.to(device).eval()

    # Split dataset
    high_idx, low_idx, test_idx = split_dataset(records, args.test_fraction)
    print(f"\nDataset split: {len(high_idx)} high-C (train), {len(low_idx)} low-C (train), {len(test_idx)} test")

    # Compute steering vectors
    print("Computing steering vectors (mean-diff in SAE feature space)...")
    steering_vectors = compute_steering_vectors(records, high_idx, low_idx)

    # Save steering vectors
    sv_path = out_dir / "steering_vectors.pt"
    torch.save({str(L): steering_vectors[L] for L in range(NUM_LAYERS)}, sv_path)
    print(f"  Saved: {sv_path}")

    # Evaluate across layers and alpha values
    test_records = [records[i] for i in test_idx[:args.num_test_pairs]]
    alphas = args.alpha
    print(f"\nEvaluating {len(test_records)} test records × {NUM_LAYERS} layers × {len(alphas)} α values")

    # results[layer][alpha_idx] = list of delta_logprobs
    results: Dict[int, Dict[float, List[float]]] = {
        L: {a: [] for a in alphas} for L in range(NUM_LAYERS)
    }

    for rec_num, rec in enumerate(test_records):
        if rec_num % 10 == 0:
            print(f"  Record {rec_num}/{len(test_records)}...")
        for layer in range(NUM_LAYERS):
            for alpha in alphas:
                try:
                    delta = evaluate_steering_on_record(
                        rec, model, tokenizer, saes, norm_stats,
                        steering_vectors, layer, alpha, device
                    )
                    results[layer][alpha].append(delta)
                except Exception as e:
                    results[layer][alpha].append(float("nan"))

    # Aggregate: mean Δlog_prob per (layer, alpha)
    mean_results: Dict[int, Dict[float, float]] = {}
    for layer in range(NUM_LAYERS):
        mean_results[layer] = {}
        for alpha in alphas:
            vals = [v for v in results[layer][alpha] if not np.isnan(v)]
            mean_results[layer][alpha] = float(np.mean(vals)) if vals else float("nan")

    # Print summary table
    print(f"\n{'Layer':>5}  " + "  ".join(f"α={a:4.1f}" for a in alphas))
    print("-" * (7 + 10 * len(alphas)))
    for layer in range(NUM_LAYERS):
        row = f"{layer:5d}  "
        row += "  ".join(f"{mean_results[layer][a]:+7.4f}" for a in alphas)
        print(row)

    # Find best (layer, alpha)
    best_layer, best_alpha, best_val = 0, alphas[0], -999.0
    for layer in range(NUM_LAYERS):
        for alpha in alphas:
            v = mean_results[layer][alpha]
            if not np.isnan(v) and v > best_val:
                best_val = v
                best_layer = layer
                best_alpha = alpha
    print(f"\nBest: layer={best_layer}, α={best_alpha}, mean Δlog_prob={best_val:+.4f}")

    # Save results JSON
    results_json = {
        "layers": list(range(NUM_LAYERS)),
        "alphas": alphas,
        "mean_delta_logprob": {str(L): {str(a): mean_results[L][a] for a in alphas} for L in range(NUM_LAYERS)},
        "best_layer": best_layer,
        "best_alpha": best_alpha,
        "best_mean_delta_logprob": best_val,
        "num_test_records": len(test_records),
        "high_c_train": len(high_idx),
        "low_c_train": len(low_idx),
    }
    json_path = out_dir / "steering_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results: {json_path}")

    # Plot heatmap: layers × alphas
    mat = np.array([[mean_results[L][a] for a in alphas] for L in range(NUM_LAYERS)])
    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-0.5, vmax=0.5, origin="upper")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(NUM_LAYERS)])
    ax.set_title(f"Steering Δlog_prob\n(+ve = toward high-|C| result)")
    plt.colorbar(im, ax=ax, label="Mean Δlog_prob")
    plt.tight_layout()
    heatmap_path = out_dir / "steering_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Heatmap: {heatmap_path}")

    # Line plot: best alpha across layers
    best_alpha_vals = [mean_results[L][best_alpha] for L in range(NUM_LAYERS)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.plot(range(NUM_LAYERS), best_alpha_vals, marker="o", linewidth=2, label=f"α={best_alpha}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δlog_prob")
    ax.set_title(f"Arithmetic Steering Effect by Layer (α={best_alpha})")
    ax.legend()
    plt.tight_layout()
    lineplot_path = out_dir / "steering_by_layer.png"
    plt.savefig(lineplot_path, dpi=150)
    plt.close()
    print(f"Line plot: {lineplot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

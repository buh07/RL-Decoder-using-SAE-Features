#!/usr/bin/env python3
"""
Phase 7 — Experiment C: Causal Patch Test
==========================================
Tests whether specific SAE features at the arithmetic ``=`` token *causally*
encode the result value by replacing them across example pairs and measuring
the resulting log-probability shift.

Background
----------
Experiments A and B are *correlational*: they show that certain SAE features
predict or co-activate with arithmetic results, but they don't prove that
*those features cause* the model's output.  Causal patching addresses this.

Method
------
For a source example S and a target example T (same operator, different
numbers — e.g. S: "12+5=17", T: "30+4=34"):

  1. Run a full forward pass for S; record all layer activations.
  2. Run a full forward pass for T; record all layer activations.
  3. At each layer L (0..23):
       a. Take the SAE latent at the ``=`` token for S: h_S  (12288,)
       b. Take the SAE latent at the ``=`` token for T: h_T  (12288,)
       c. Build a "patched" latent:
             h_patch = h_S  with  top-K features replaced by h_T's values
          The top-K features chosen are those from Experiment A's top
          probe features at this layer (most predictive of result value).
       d. Decode: x_patch = SAE_decoder(h_patch)
       e. Un-normalise x_patch → residual-stream space.
       f. Patch the residual stream of S at layer L, position eq_tok,
          then continue the model from layer L+1.
       g. Measure log-prob of predicting T's result token vs. S's result.
          Record: Δlog_prob = log_p(C_T | patched) − log_p(C_T | original)

  4. Repeat across many (S, T) pairs and take the mean Δlog_prob per layer.

A positive Δlog_prob at layer L means: replacing the SAE features at that
layer made the model *more likely* to predict T's answer — evidence that
those features encode the causal arithmetic computation.

Pair selection
--------------
We pair examples that share the same operator (+, -, *, /) so that the
"structure" of the computation is the same and only the numbers differ.
We randomly sample up to ``--num-pairs`` pairs per operator.

Output
------
  phase7/results/patching/
    patching_results.json        — mean Δlog_prob per layer per operator
    delta_logprob_by_layer.png   — line plot of causal effect vs. layer
    causal_effect_heatmap.png    — per-pair Δlog_prob heatmap (pairs × layers)

Usage
-----
  cd "/scratch2/f004ndc/RL-Decoder with SAE Features"
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python3 phase7/causal_patch_test.py \\
      --dataset         phase7/results/collection/gsm8k_arithmetic_dataset.pt \\
      --probe-features  phase7/results/probe/top_features_per_layer.json \\
      --saes-dir        phase5_results/multilayer_gpt2_12x/saes \\
      --activations-dir phase4_results/activations_multilayer \\
      --output-dir      phase7/results/patching \\
      --num-pairs       30 \\
      --patch-k         128 \\
      --device          cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
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

NUM_LAYERS = 24
HIDDEN_DIM = 1024
LATENT_DIM = 12288
MODEL_ID   = "gpt2-medium"


# ---------------------------------------------------------------------------
# Shared utilities (same as in arithmetic_data_collector.py)
# ---------------------------------------------------------------------------

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
            use_amp=False,
        )
        sae = SparseAutoencoder(cfg)
        sae.load_state_dict(ckpt["model_state_dict"])
        sae = sae.to(device).eval()
        saes[layer_idx] = sae
    print(f"  Loaded {len(saes)} SAEs.")
    return saes


def load_norm_stats(
    activations_dir: Path, device: str
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
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


def normalize(
    x: torch.Tensor,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if stats is None:
        return x
    mean, std = stats
    return (x - mean) / std


def unnormalize(
    x: torch.Tensor,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Invert normalisation: SAE-space → raw residual-stream space."""
    if stats is None:
        return x
    mean, std = stats
    return x * std + mean


def register_capture_hooks(model) -> Tuple[Dict[int, torch.Tensor], List]:
    """Read-only hooks: capture residual-stream activations at every layer."""
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


def run_forward_capture(
    model, input_ids: torch.Tensor, store: Dict[int, torch.Tensor]
) -> torch.Tensor:
    """Run a forward pass and return logits; activations captured in ``store``."""
    with torch.no_grad():
        out = model(input_ids)
    return out.logits  # (1, seq_len, vocab_size)


# ---------------------------------------------------------------------------
# Inference with a single-layer activation patch
# ---------------------------------------------------------------------------

class PatchedActivationHook:
    """Forward hook that injects a patched activation at one layer.

    On first call the hook replaces the activation at ``tok_pos`` with the
    pre-computed ``patch_vec`` and then disables itself so subsequent layers
    are unaffected.
    """

    def __init__(
        self,
        tok_pos: int,
        patch_vec: torch.Tensor,  # (HIDDEN_DIM,) on the model's device
    ) -> None:
        self.tok_pos   = tok_pos
        self.patch_vec = patch_vec
        self.fired     = False

    def __call__(self, module, inp, output):
        if self.fired:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        # Replace the activation at the target token position
        hidden = hidden.clone()
        hidden[0, self.tok_pos, :] = self.patch_vec
        self.fired = True
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


def run_patched_forward(
    model,
    input_ids: torch.Tensor,
    patch_layer: int,
    tok_pos: int,
    patch_vec: torch.Tensor,
) -> torch.Tensor:
    """Run a forward pass with a patched activation at ``patch_layer``.

    Parameters
    ----------
    patch_layer : int
        Which transformer block to patch (0-indexed).
    tok_pos : int
        Token position to patch within the sequence.
    patch_vec : Tensor, shape (HIDDEN_DIM,)
        The replacement activation value (in raw residual-stream space,
        *after* un-normalising back from SAE space).

    Returns
    -------
    Tensor
        Logits of shape (1, seq_len, vocab_size).
    """
    patcher = PatchedActivationHook(tok_pos, patch_vec.to(input_ids.device))
    handle = model.transformer.h[patch_layer].register_forward_hook(patcher)
    with torch.no_grad():
        out = model(input_ids)
    handle.remove()
    return out.logits


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def detect_operator(expr_str: str) -> Optional[str]:
    """Infer the primary operator from an expression string.

    Returns '+', '-', '*', '/', or None.
    Note: checks in order *,/,+,- to avoid misidentifying a leading minus.
    """
    for op in ["*", "/"]:
        if op in expr_str:
            return op
    # Subtraction: look for '-' not at start of string
    if "-" in expr_str[1:]:
        return "-"
    if "+" in expr_str:
        return "+"
    return None


def select_pairs(
    records: List[dict],
    num_pairs: int,
    seed: int = 42,
) -> List[Tuple[dict, dict]]:
    """Pair records that share the same operator.

    Returns a list of (source, target) record pairs.
    """
    rng = random.Random(seed)
    by_op: Dict[str, List[dict]] = {}
    for r in records:
        op = detect_operator(r["expr_str"])
        if op:
            by_op.setdefault(op, []).append(r)

    pairs: List[Tuple[dict, dict]] = []
    for op, recs in by_op.items():
        if len(recs) < 2:
            continue
        rng.shuffle(recs)
        for i in range(0, len(recs) - 1, 2):
            pairs.append((recs[i], recs[i + 1]))
        if len(pairs) >= num_pairs:
            break

    pairs = pairs[:num_pairs]
    print(f"  Selected {len(pairs)} (source, target) pairs.")
    return pairs


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_delta_logprob(
    source_rec: dict,
    target_rec: dict,
    model,
    tokenizer,
    saes: Dict[int, SparseAutoencoder],
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    probe_features: List[List[int]],
    patch_k: int,
    device: str,
) -> List[float]:
    """Compute per-layer Δlog_prob for one (source, target) pair.

    For each layer L:
      1. Replace top-K SAE features at source's eq_tok with target's values.
      2. Decode + un-normalise → patched residual-stream vector.
      3. Run model from scratch with this patch injected at layer L.
      4. Compute Δlog_prob = log_p_patched(first result token of target)
                            − log_p_original(first result token of target).

    Parameters
    ----------
    probe_features : List[List[int]]
        Top-K feature indices per layer from Experiment A.
    patch_k : int
        How many features to replace (subset of probe_features[L]).

    Returns
    -------
    List[float]
        Δlog_prob per layer (length NUM_LAYERS).
    """
    # Build input_ids from source record
    src_ids  = torch.tensor(source_rec["token_ids"], dtype=torch.long).unsqueeze(0).to(device)
    tgt_ids  = torch.tensor(target_rec["token_ids"], dtype=torch.long).unsqueeze(0).to(device)

    # Get the first token of target's result
    tgt_result_toks = target_rec["result_tok_idxs"]
    if not tgt_result_toks:
        return [0.0] * NUM_LAYERS
    tgt_result_tok_id = target_rec["token_ids"][tgt_result_toks[0]]

    # Source and target SAE latents at eq_tok, all layers  (NUM_LAYERS, LATENT_DIM)
    h_src = source_rec["eq_features"].float()   # cpu
    h_tgt = target_rec["eq_features"].float()   # cpu

    # Token position in the source sequence to patch
    eq_tok_pos = source_rec["eq_tok_idx"]

    # Baseline: log_p for target result token under *original* (un-patched) source
    src_logits = model(src_ids).logits   # (1, seq_len, vocab)
    # Log-prob at the position BEFORE the result token (the model predicts next token)
    pred_pos = eq_tok_pos  # model uses eq_tok to predict next; fine as proxy
    log_probs_orig = F.log_softmax(src_logits[0, pred_pos, :], dim=-1)
    lp_orig = log_probs_orig[tgt_result_tok_id].item()

    delta_lp: List[float] = []

    for L in range(NUM_LAYERS):
        # Features to swap: top-patch_k from probe_features[L] (or all if fewer)
        top_feats = probe_features[L][:patch_k]
        if not top_feats:
            delta_lp.append(0.0)
            continue

        # Build patched latent: start from source, replace selected features
        h_patch = h_src[L].clone()                    # (LATENT_DIM,)
        h_patch[top_feats] = h_tgt[L][top_feats]      # replace with target values

        # Decode through SAE
        h_patch_gpu = h_patch.unsqueeze(0).to(device)  # (1, LATENT_DIM)
        with torch.no_grad():
            x_decoded = saes[L].decode(h_patch_gpu).squeeze(0)  # (HIDDEN_DIM,)

        # Un-normalise back to raw residual-stream space
        x_patch = unnormalize(x_decoded, norm_stats.get(L))  # (HIDDEN_DIM,)

        # Run patched forward pass
        patched_logits = run_patched_forward(
            model, src_ids, L, eq_tok_pos, x_patch
        )  # (1, seq_len, vocab)

        log_probs_patch = F.log_softmax(patched_logits[0, pred_pos, :], dim=-1)
        lp_patch = log_probs_patch[tgt_result_tok_id].item()

        delta_lp.append(lp_patch - lp_orig)

    return delta_lp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_delta_logprob(
    mean_delta: np.ndarray,
    per_pair: np.ndarray,
    output_dir: Path,
) -> None:
    """Two-panel figure: mean Δlog_prob per layer + per-pair heatmap."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9),
                              gridspec_kw={"height_ratios": [1, 2]})

    # Top panel: mean ± std
    std_delta = per_pair.std(axis=0)
    axes[0].plot(range(NUM_LAYERS), mean_delta, color="#E74C3C",
                 marker="o", markersize=4, linewidth=2, label="Mean Δlog_prob")
    axes[0].fill_between(range(NUM_LAYERS),
                          mean_delta - std_delta,
                          mean_delta + std_delta,
                          alpha=0.2, color="#E74C3C")
    axes[0].axhline(0, color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Δlog_prob", fontsize=11)
    axes[0].set_title(
        "Causal Patching: Δlog_prob of Target Result After SAE Feature Swap\n"
        "(positive = patching pushed model toward predicting target result)",
        fontsize=11,
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.5, NUM_LAYERS - 0.5)

    # Bottom panel: per-pair heatmap
    im = axes[1].imshow(
        per_pair, aspect="auto", cmap="RdBu_r",
        vmin=-1.0, vmax=1.0, interpolation="nearest",
    )
    axes[1].set_xlabel("Layer", fontsize=11)
    axes[1].set_ylabel("Example pair", fontsize=11)
    axes[1].set_title("Per-pair Δlog_prob heatmap", fontsize=10)
    axes[1].set_xticks(range(NUM_LAYERS))
    axes[1].set_xticklabels(range(NUM_LAYERS), fontsize=7)
    plt.colorbar(im, ax=axes[1], fraction=0.02, pad=0.02)

    plt.tight_layout()
    out = output_dir / "delta_logprob_by_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment C: Causal patch test for arithmetic SAE features."
    )
    parser.add_argument(
        "--dataset",
        default="phase4_results/collection/gsm8k_arithmetic_dataset.pt",
    )
    parser.add_argument(
        "--probe-features",
        default="phase4_results/probe/top_features_per_layer.json",
        help="Top features per layer from Experiment A (probe results).",
    )
    parser.add_argument(
        "--saes-dir",
        default="phase2_results/saes_gpt2_12x/saes",
    )
    parser.add_argument(
        "--activations-dir",
        default="phase2_results/activations",
    )
    parser.add_argument(
        "--output-dir",
        default="phase4_results/patching",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=30,
        help="Number of (source, target) pairs to test.",
    )
    parser.add_argument(
        "--patch-k",
        type=int,
        default=128,
        help="Number of top-probe features to replace per patch.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load everything
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Loading dataset …")
    records = torch.load(Path(args.dataset), map_location="cpu", weights_only=False)
    print(f"  {len(records)} annotation records.")

    print(f"\n[2/5] Loading probe features …")
    probe_data = json.loads(Path(args.probe_features).read_text())
    probe_features: List[List[int]] = probe_data["eq"]   # top features per layer

    print(f"\n[3/5] Loading model and SAEs …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(args.device).eval()
    saes = load_saes(Path(args.saes_dir), args.device)
    norm_stats = load_norm_stats(Path(args.activations_dir), args.device)

    # -----------------------------------------------------------------------
    # Select pairs
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Selecting example pairs …")
    pairs = select_pairs(records, args.num_pairs, seed=args.seed)
    if not pairs:
        print("ERROR: No valid pairs found.  Run the data collector first.")
        return

    # -----------------------------------------------------------------------
    # Run causal patching
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Running causal patch test "
          f"({len(pairs)} pairs × {NUM_LAYERS} layers × patch_k={args.patch_k}) …")

    all_deltas: List[List[float]] = []
    for pair_idx, (src, tgt) in enumerate(pairs):
        deltas = compute_delta_logprob(
            src, tgt, model, tokenizer, saes, norm_stats,
            probe_features, args.patch_k, args.device,
        )
        all_deltas.append(deltas)
        if (pair_idx + 1) % 5 == 0 or (pair_idx + 1) == len(pairs):
            mean_so_far = np.mean([d for row in all_deltas for d in row])
            print(f"  {pair_idx+1}/{len(pairs)} pairs done  "
                  f"(running mean Δlog_prob={mean_so_far:.4f})")

    per_pair = np.array(all_deltas, dtype=np.float32)  # (n_pairs, NUM_LAYERS)
    mean_delta = per_pair.mean(axis=0)                  # (NUM_LAYERS,)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "num_pairs":              len(pairs),
        "patch_k":                args.patch_k,
        "mean_delta_logprob":     mean_delta.tolist(),
        "std_delta_logprob":      per_pair.std(axis=0).tolist(),
        "best_layer":             int(np.argmax(mean_delta)),
        "best_mean_delta_logprob": float(mean_delta.max()),
        "worst_layer":            int(np.argmin(mean_delta)),
    }
    json_path = output_dir / "patching_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {json_path}")

    np.save(output_dir / "per_pair_delta_logprob.npy", per_pair)

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("\nGenerating plots …")
    plot_delta_logprob(mean_delta, per_pair, output_dir)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Experiment C Summary ===")
    best_L = results["best_layer"]
    best_d = results["best_mean_delta_logprob"]
    print(f"  Best patch layer : {best_L}  (mean Δlog_prob = {best_d:.4f})")
    print(f"  Mean over all layers: {mean_delta.mean():.4f}")
    print()
    if best_d > 0.1:
        print(f"  POSITIVE RESULT: Patching top-{args.patch_k} probe features at "
              f"layer {best_L} increases log-prob of the target result by {best_d:.3f}.")
        print(f"  This is evidence that those SAE features *causally* encode the")
        print(f"  arithmetic result at that layer.")
    elif best_d > 0:
        print(f"  WEAK POSITIVE: Small positive Δlog_prob at layer {best_L}.")
        print(f"  Consider increasing --patch-k or running more pairs.")
    else:
        print(f"  NULL RESULT: No positive Δlog_prob found.")
        print(f"  Possible causes: (a) arithmetic is distributed across many features,")
        print(f"  (b) the patched features do not include the causal ones,")
        print(f"  (c) the model is not doing arithmetic through the SAE-captured features.")


if __name__ == "__main__":
    main()

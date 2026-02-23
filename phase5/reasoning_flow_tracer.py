#!/usr/bin/env python3
"""
Phase 5.4: Reasoning Flow Tracer
=================================
Traces how mathematical reasoning features evolve layer-by-layer through
GPT-2 medium, using the 24 trained multi-layer SAEs (one per transformer
block).  For every token in an input sequence the script:

  1. Captures the post-block residual-stream activation via a forward hook.
  2. Normalises using statistics derived from the training activation files
     (same normalisation applied during SAE training, so features fire
     correctly on out-of-sample text).
  3. Encodes through the layer's SAE → sparse latent vector h ∈ ℝ¹²²⁸⁸.
  4. Records which features are active (h_i > 0) and their magnitudes.

This yields a [tokens × layers] picture of what the model "computes" at
each depth, letting us ask:

  • Do arithmetic-relevant features activate in a narrow layer band, or
    broadly across all depths?
  • Is there a depth at which the model transitions from surface-form
    features to semantic/computation features?
  • Do computation tokens (numbers, operators) recruit more / different
    features than background tokens (articles, punctuation)?

Output files (all in --output-dir):
  feature_flow_ex{N}_{label}.png  — (layers × tokens) active-feature heatmap
  computation_contrast.png        — comp vs background token mean active count
  feature_persistence.png         — layer-pair Jaccard similarity of active sets
  layer_sparsity.png              — mean active features per layer (all tokens)
  reasoning_flow.json             — full per-token, per-layer feature data

Usage
-----
  cd /scratch2/f004ndc/RL-Decoder\\ with\\ SAE\\ Features
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 phase5/reasoning_flow_tracer.py \\
      --saes-dir phase5_results/multilayer_gpt2_12x/saes \\
      --activations-dir phase4_results/activations_multilayer \\
      --output-dir phase5_results/reasoning_flow_analysis \\
      --device cuda:0

To analyse a custom prompt instead of the built-in GSM8K examples:
  ... --prompt "Alice has 7 cookies. She gives 2 to Bob. How many remain?"

Notes
-----
- SAEs were trained with use_relu=True, so active features are h_i > 0.
- Hooks attach to transformer.h[i] (full block output), matching the
  capture script that produced the training activation files.
- Normalization statistics are loaded from the training activation files;
  if a file is missing the script falls back to per-batch normalization.
- Jaccard similarity in feature_persistence.png is approximated using the
  top-k recorded features (not the full active set) for memory efficiency.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; works without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src/ to path so we can import the project's SAE classes
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_architecture import SparseAutoencoder
from sae_config import SAEConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LAYERS = 24       # GPT-2 medium transformer blocks
HIDDEN_DIM = 1024     # GPT-2 medium hidden dimension
MODEL_ID   = "gpt2-medium"

# Semantic labels for layer ranges (from phase6/layer_semantics.json).
# Keyed by (first_layer, last_layer) inclusive.
LAYER_SEMANTICS: Dict[Tuple[int, int], str] = {
    (0,  3):  "Input\nTokenisation",
    (4,  7):  "Basic\nSyntax",
    (8,  11): "Semantic\nComposition",
    (12, 15): "Reasoning\nPrimitives",
    (16, 19): "Higher-Order\nReasoning",
    (20, 23): "Output\nFormulation",
}

# Colours for each semantic region (matched to LAYER_SEMANTICS order)
_REGION_COLOURS = [
    "#AED6F1",  # light blue   — tokenisation
    "#A9DFBF",  # light green  — syntax
    "#FAD7A0",  # light orange — semantic
    "#F1948A",  # light red    — reasoning
    "#D2B4DE",  # light purple — higher-order
    "#FDFEFE",  # near-white   — output
]

# Regex that matches tokens containing arithmetic content:
#   <<48/2=24>>  — GSM8K calculator annotation
#   42, 3.14     — bare numbers
#   +, -, *, /,  =, ×, ÷  — operators
COMPUTATION_RE = re.compile(
    r"(<<[^>]+=\s*-?\d+\.?\d*>>"  # GSM8K annotation
    r"|\d+\.?\d*"                  # integer or decimal
    r"|[+\-*/=×÷%])"              # arithmetic operator
)

# ---------------------------------------------------------------------------
# Built-in GSM8K-style examples
# Five problems that cover distinct arithmetic patterns so the aggregate
# contrast and persistence plots draw from varied reasoning types.
# ---------------------------------------------------------------------------
BUILTIN_EXAMPLES: List[Tuple[str, str]] = [
    (
        "James has 5 apples. He buys 3 more at the store. "
        "How many apples does he have now?\n"
        "James starts with 5 apples.\n"
        "He buys 3 more, so 5 + 3 = 8.\n"
        "James now has 8 apples.",
        "addition",
    ),
    (
        "A baker made 48 cookies and divided them equally into 6 bags. "
        "How many cookies are in each bag?\n"
        "48 / 6 = 8.\n"
        "Each bag contains 8 cookies.",
        "division",
    ),
    (
        "Sarah earns $12 per hour and works 7 hours a day for 5 days. "
        "How much does she earn in total?\n"
        "Earnings per day: 12 * 7 = 84 dollars.\n"
        "Total: 84 * 5 = 420 dollars.\n"
        "Sarah earns $420 in total.",
        "multi_step",
    ),
    (
        "There are 30 students in a class. 18 are girls. "
        "How many are boys?\n"
        "30 - 18 = 12.\n"
        "There are 12 boys.",
        "subtraction",
    ),
    (
        "A car travels at 60 miles per hour. "
        "How far does it travel in 2.5 hours?\n"
        "Distance = speed × time = 60 * 2.5 = 150 miles.\n"
        "The car travels 150 miles.",
        "rate",
    ),
]

# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class TokenRecord:
    """Per-token analysis record accumulated across all 24 layers.

    Attributes
    ----------
    token_str      : Decoded string representation of this token.
    token_idx      : Position in the tokenised sequence (0-based).
    is_computation : True if the token matches COMPUTATION_RE.
    active_counts  : List of length NUM_LAYERS; active_counts[i] is the
                     number of SAE features active (h > 0) at layer i.
    top_features   : List of length NUM_LAYERS; top_features[i] is a list
                     of (feature_index, activation_value) tuples for the
                     top-k features by magnitude at layer i.
    """
    token_str:      str
    token_idx:      int
    is_computation: bool
    active_counts:  List[int]                        = field(default_factory=list)
    top_features:   List[List[Tuple[int, float]]]    = field(default_factory=list)

# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

def load_saes(saes_dir: Path, device: str) -> Dict[int, SparseAutoencoder]:
    """Load all NUM_LAYERS SAEs from checkpoint files.

    Reconstructs SAEConfig from the saved config dict and loads the
    model_state_dict.  The checkpoints were produced by
    phase5_task4_train_multilayer_saes.py with use_relu=True and
    normalize_decoder() applied after each optimiser step.

    Parameters
    ----------
    saes_dir : Path
        Directory containing ``gpt2-medium_layer{N}_sae.pt`` files.
    device : str
        Torch device string (e.g. ``"cuda:0"``).

    Returns
    -------
    Dict[int, SparseAutoencoder]
        Mapping from layer index to a loaded, eval-mode SAE.
    """
    saes: Dict[int, SparseAutoencoder] = {}
    for layer_idx in range(NUM_LAYERS):
        ckpt_path = saes_dir / f"gpt2-medium_layer{layer_idx}_sae.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAE checkpoint missing: {ckpt_path}\n"
                f"Run phase5_task4_train_multilayer_saes.py first."
            )
        # weights_only=False: checkpoints may embed PosixPath objects (PyTorch 2.6+)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cd = ckpt["config"]

        # Reconstruct only the fields needed for inference; let all others default.
        # This avoids issues with Path-typed fields stored as strings in the JSON.
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

    print(f"  Loaded {len(saes)} SAEs  "
          f"(expansion={saes[0].config.expansion_factor}x, "
          f"latent_dim={saes[0].config.latent_dim})")
    return saes

# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------

def load_norm_stats(
    activations_dir: Path,
    device: str,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Load per-layer normalisation (mean, std) from the training activation files.

    During SAE training each layer's activations were normalised to zero mean
    and unit variance across the training corpus before being passed to the
    encoder.  Applying the same transform at inference time ensures that the
    learned features respond correctly to new inputs.

    Parameters
    ----------
    activations_dir : Path
        Directory containing ``gpt2-medium_layer{N}_activations.pt`` files.
    device : str
        Target device for the returned tensors.

    Returns
    -------
    Dict[int, Tuple[Tensor, Tensor]]
        layer_idx → (mean, std), each of shape (HIDDEN_DIM,).
        Layers without a corresponding file are omitted; the caller falls
        back to per-batch normalisation for those layers.
    """
    stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    missing = []
    for layer_idx in range(NUM_LAYERS):
        path = activations_dir / f"gpt2-medium_layer{layer_idx}_activations.pt"
        if not path.exists():
            missing.append(layer_idx)
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
    if missing:
        print(f"  WARNING: no activation file for layers {missing}; "
              "falling back to per-batch normalisation for those layers.")
    return stats


def _normalize(
    x: torch.Tensor,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Normalise a (D,) or (N, D) activation tensor.

    Uses pre-loaded training statistics when available; otherwise falls back
    to computing mean/std over the token dimension of the current batch.
    """
    if stats is not None:
        mean, std = stats
        return (x - mean) / std
    # Per-batch fallback: normalise over token positions
    if x.dim() == 1:
        return x  # single vector — cannot compute meaningful std; pass through
    m = x.mean(dim=0, keepdim=True)
    s = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - m) / s

# ---------------------------------------------------------------------------
# Forward-hook infrastructure
# ---------------------------------------------------------------------------

def register_hooks(
    model,
) -> Tuple[Dict[int, torch.Tensor], List]:
    """Register output hooks on each GPT-2 medium transformer block.

    GPT-2's ``transformer.h[i]`` (a ``GPT2Block``) returns a tuple whose
    first element is the residual-stream hidden state *after* the self-
    attention and MLP sublayers.  This is the same quantity captured during
    SAE training (see phase5_task4_capture_multi_layer.py).

    Parameters
    ----------
    model : GPT2LMHeadModel
        The loaded GPT-2 medium model.

    Returns
    -------
    activation_store : Dict[int, Tensor]
        Mutable dict populated with shape-(1, seq_len, HIDDEN_DIM) tensors
        on every forward pass.  Values are moved to CPU immediately to keep
        GPU memory usage flat.
    handles : List
        Hook handle objects; call ``h.remove()`` on each to deregister.
    """
    activation_store: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, output):
            # GPT2Block returns (hidden_states, present_key_value, ...)
            hidden = output[0] if isinstance(output, tuple) else output
            # Immediately move to CPU so all 24 copies don't fill GPU memory
            activation_store[layer_idx] = hidden.detach().cpu()
        return hook_fn

    for i in range(NUM_LAYERS):
        handle = model.transformer.h[i].register_forward_hook(make_hook(i))
        handles.append(handle)

    return activation_store, handles

# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

def classify_tokens(tokens: List[str]) -> List[bool]:
    """Return True for tokens that contain arithmetic content.

    "Computation tokens" are those matching COMPUTATION_RE: digits, decimal
    numbers, arithmetic operators (+,-,*,/,=,×,÷,%), and GSM8K calculator
    annotations (``<<expr=result>>``).  Everything else is "background".
    """
    return [bool(COMPUTATION_RE.search(tok)) for tok in tokens]

# ---------------------------------------------------------------------------
# Core per-example analysis
# ---------------------------------------------------------------------------

def analyse_example(
    prompt: str,
    model,
    tokenizer,
    saes: Dict[int, SparseAutoencoder],
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: str,
    top_k: int = 10,
) -> List[TokenRecord]:
    """Run one prompt through GPT-2 medium and collect SAE feature data.

    Performs a single forward pass with hooks attached to capture residual-
    stream activations at all 24 layers.  For each (token, layer) pair the
    activation is normalised and encoded through the corresponding SAE.

    Parameters
    ----------
    prompt : str
        Input text (tokenised by the GPT-2 tokenizer).
    model : GPT2LMHeadModel
        Loaded GPT-2 medium model in eval mode.
    tokenizer : GPT2Tokenizer
        Corresponding tokenizer.
    saes : Dict[int, SparseAutoencoder]
        Loaded SAEs, keyed by layer index.
    norm_stats : Dict[int, Tuple[Tensor, Tensor]]
        Per-layer (mean, std) from the training activation files.
    device : str
        Torch device string.
    top_k : int
        Number of highest-magnitude features to record per (token, layer).

    Returns
    -------
    List[TokenRecord]
        One record per token, in sequence order.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    is_computation = classify_tokens(tokens)

    activation_store, handles = register_hooks(model)
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()

    records: List[TokenRecord] = []

    for tok_idx, (tok_str, is_comp) in enumerate(zip(tokens, is_computation)):
        rec = TokenRecord(
            token_str=tok_str,
            token_idx=tok_idx,
            is_computation=is_comp,
        )

        for layer_idx in range(NUM_LAYERS):
            if layer_idx not in activation_store:
                rec.active_counts.append(0)
                rec.top_features.append([])
                continue

            # Extract this token's activation: (1, seq_len, D) → (D,)
            act = activation_store[layer_idx][0, tok_idx, :].to(device)

            # Apply training-set normalisation
            act_norm = _normalize(act.unsqueeze(0), norm_stats.get(layer_idx)).squeeze(0)

            with torch.no_grad():
                # encode() applies the linear encoder + ReLU
                h = saes[layer_idx].encode(act_norm)   # shape: (latent_dim,)

            # Active features: those with h > 0 (guaranteed by ReLU gate)
            active_mask = h > 0
            n_active = int(active_mask.sum().item())
            rec.active_counts.append(n_active)

            # Top-k features by activation magnitude
            k = min(top_k, h.shape[0])
            top_vals, top_idxs = torch.topk(h, k=k)
            rec.top_features.append([
                (int(idx), float(val))
                for idx, val in zip(top_idxs.tolist(), top_vals.tolist())
                if val > 0.0   # only include genuinely active features
            ])

        records.append(rec)

    return records

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _layer_yticks() -> Tuple[List[int], List[str]]:
    """Return (positions, labels) for the layer Y axis (every 4 layers)."""
    positions = list(range(NUM_LAYERS))
    labels = [str(i) if i % 4 == 0 else "" for i in positions]
    return positions, labels


def _add_semantic_bands(ax, orientation: str = "horizontal") -> None:
    """Shade semantic layer regions with translucent colour bands.

    Parameters
    ----------
    ax : matplotlib Axes
    orientation : "horizontal" shades Y-axis bands (for heatmaps where Y=layer),
                  "vertical" shades X-axis bands (for line plots where X=layer).
    """
    for colour, (lo_hi, label) in zip(_REGION_COLOURS, LAYER_SEMANTICS.items()):
        lo, hi = lo_hi
        if orientation == "horizontal":
            ax.axhspan(lo - 0.5, hi + 0.5, alpha=0.10, color=colour, zorder=0)
        else:
            ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.10, color=colour, zorder=0)

# ---------------------------------------------------------------------------
# Plot 1: Feature-flow heatmap (per example)
# ---------------------------------------------------------------------------

def plot_feature_flow(
    records: List[TokenRecord],
    title_suffix: str,
    output_path: Path,
) -> None:
    """Save a (layers × tokens) heatmap of active SAE feature counts.

    Rows = layers 0–23 (top to bottom).  Columns = token positions.
    Cell colour encodes the number of active (h > 0) SAE features.
    Red vertical lines mark computation tokens.
    Horizontal bands show semantic layer regions.

    Parameters
    ----------
    records : List[TokenRecord]
        Output of analyse_example().
    title_suffix : str
        Short label appended to the plot title (e.g. "addition (ex 1)").
    output_path : Path
        Where to write the PNG file.
    """
    seq_len = len(records)
    # data[layer, token] = active feature count
    data = np.array([r.active_counts for r in records]).T  # (NUM_LAYERS, seq_len)

    fig_w = max(10, seq_len * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest",
                   origin="upper")

    # Shade semantic regions on Y axis
    _add_semantic_bands(ax, orientation="horizontal")

    # Red vertical lines for computation tokens
    for i, rec in enumerate(records):
        if rec.is_computation:
            ax.axvline(i, color="#E74C3C", linewidth=0.7, alpha=0.4, zorder=1)

    # Token labels — truncated to 7 chars, newlines escaped
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(
        [r.token_str[:7].replace("\n", "↵") for r in records],
        rotation=80, ha="right", fontsize=6.5,
    )

    tick_pos, tick_lab = _layer_yticks()
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=8)

    ax.set_ylabel("Layer  (0 = earliest)", fontsize=9)
    ax.set_xlabel("Token position  (red lines = computation tokens)", fontsize=9)
    ax.set_title(f"Active SAE Features  [{title_suffix}]", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label="# active features", shrink=0.8)

    # Add semantic region legend on right margin
    legend_patches = [
        matplotlib.patches.Patch(color=c, alpha=0.5, label=lbl.replace("\n", " "))
        for c, lbl in zip(_REGION_COLOURS, LAYER_SEMANTICS.values())
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              bbox_to_anchor=(1.12, 1.0), fontsize=7, title="Layer semantics",
              title_fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")

# ---------------------------------------------------------------------------
# Plot 2: Computation vs background contrast
# ---------------------------------------------------------------------------

def plot_computation_contrast(
    all_records: List[List[TokenRecord]],
    output_path: Path,
) -> None:
    """Line plot: mean active features per layer for computation vs background tokens.

    Averages over all tokens across all provided examples.  The gap between
    the two curves at each layer indicates how strongly the SAE discriminates
    arithmetic content from linguistic background at that depth.

    Parameters
    ----------
    all_records : List[List[TokenRecord]]
        One inner list per example.
    output_path : Path
        Destination PNG file.
    """
    comp_sum = np.zeros(NUM_LAYERS)
    bg_sum   = np.zeros(NUM_LAYERS)
    comp_n = bg_n = 0

    for records in all_records:
        for rec in records:
            arr = np.array(rec.active_counts, dtype=float)
            if rec.is_computation:
                comp_sum += arr
                comp_n   += 1
            else:
                bg_sum += arr
                bg_n   += 1

    comp_mean = comp_sum / max(comp_n, 1)
    bg_mean   = bg_sum   / max(bg_n,   1)

    layers = np.arange(NUM_LAYERS)

    fig, ax = plt.subplots(figsize=(11, 5))

    _add_semantic_bands(ax, orientation="vertical")

    ax.plot(layers, comp_mean, "o-", color="#E74C3C", linewidth=2.0, markersize=4,
            label=f"Computation tokens  (n={comp_n})", zorder=3)
    ax.plot(layers, bg_mean,   "s-", color="#2980B9", linewidth=2.0, markersize=4,
            label=f"Background tokens   (n={bg_n})",   zorder=3)
    ax.fill_between(layers, comp_mean, bg_mean, alpha=0.08, color="#7D3C98", zorder=2)

    # Annotate the layer with the largest gap
    gap = comp_mean - bg_mean
    peak_layer = int(np.argmax(np.abs(gap)))
    ax.annotate(
        f"Largest gap: L{peak_layer}\n(Δ={gap[peak_layer]:+.1f})",
        xy=(peak_layer, (comp_mean[peak_layer] + bg_mean[peak_layer]) / 2),
        xytext=(peak_layer + 1.5, max(comp_mean) * 0.9),
        fontsize=8, arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean active SAE features per token", fontsize=10)
    ax.set_title(
        "Computation vs Background Token Feature Activation by Layer\n"
        "(larger gap → SAE features at this depth are arithmetic-specific)",
        fontsize=11,
    )
    ax.set_xticks(layers[::2])
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path.name}")

# ---------------------------------------------------------------------------
# Plot 3: Feature persistence (layer-pair Jaccard similarity)
# ---------------------------------------------------------------------------

def plot_feature_persistence(
    all_records: List[List[TokenRecord]],
    output_path: Path,
) -> None:
    """Heatmap of pairwise Jaccard similarity of active feature sets.

    For every pair of layers (i, j) and every token, computes:
        Jaccard(i, j) = |active_i ∩ active_j| / |active_i ∪ active_j|

    where active sets are approximated by the top-k recorded features.
    The result is averaged over all tokens and examples.

    A near-diagonal structure → features are similar in adjacent layers
    (gradual evolution).  Off-diagonal zeros → abrupt reorganisation
    (evidence of a phase transition in the residual stream).

    Parameters
    ----------
    all_records : List[List[TokenRecord]]
        One inner list per example.
    output_path : Path
        Destination PNG file.
    """
    jaccard_acc   = np.zeros((NUM_LAYERS, NUM_LAYERS))
    jaccard_count = 0

    for records in all_records:
        for rec in records:
            # Build feature index sets from the recorded top-k features
            feat_sets = [
                {idx for idx, val in layer_top if val > 0.0}
                for layer_top in rec.top_features
            ]
            # Upper-triangle accumulation (symmetric)
            for i in range(NUM_LAYERS):
                for j in range(i, NUM_LAYERS):
                    inter = len(feat_sets[i] & feat_sets[j])
                    union = len(feat_sets[i] | feat_sets[j])
                    jac = inter / union if union > 0 else 1.0
                    jaccard_acc[i, j] += jac
                    if i != j:
                        jaccard_acc[j, i] += jac
            jaccard_count += 1

    jaccard_avg = jaccard_acc / max(jaccard_count, 1)

    tick_labels = [str(i) if i % 4 == 0 else "" for i in range(NUM_LAYERS)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(jaccard_avg, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto",
                   interpolation="nearest")

    ax.set_xticks(range(NUM_LAYERS))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title(
        "Feature Persistence: Pairwise Jaccard Similarity of Active SAE Features\n"
        f"(top-{10} features per token; averaged over all tokens & examples)\n"
        "Green = high overlap (stable features)   Red = low overlap (reorganisation)",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, label="Jaccard similarity", shrink=0.85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path.name}")

# ---------------------------------------------------------------------------
# Plot 4: Layer sparsity bar chart
# ---------------------------------------------------------------------------

def plot_layer_sparsity(
    all_records: List[List[TokenRecord]],
    output_path: Path,
) -> None:
    """Bar chart of mean active SAE features per layer, collapsed over all tokens.

    Layers with consistently high active-feature counts encode more
    information from the residual stream; layers with low counts may be
    contributing only small deltas.

    Parameters
    ----------
    all_records : List[List[TokenRecord]]
        One inner list per example.
    output_path : Path
        Destination PNG file.
    """
    total = np.zeros(NUM_LAYERS)
    n = 0
    for records in all_records:
        for rec in records:
            total += np.array(rec.active_counts, dtype=float)
            n += 1
    mean = total / max(n, 1)

    fig, ax = plt.subplots(figsize=(11, 4))

    _add_semantic_bands(ax, orientation="vertical")

    ax.bar(range(NUM_LAYERS), mean, color="#5DADE2", edgecolor="white",
           linewidth=0.4, zorder=3)
    ax.axhline(mean.mean(), color="#E74C3C", linewidth=1.2, linestyle="--",
               label=f"Grand mean = {mean.mean():.1f}", zorder=4)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean active features / token", fontsize=10)
    ax.set_title("Mean Active SAE Features per Layer  (all tokens & examples)", fontsize=11)
    ax.set_xticks(range(NUM_LAYERS)[::2])
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path.name}")

# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(
    all_records: List[List[TokenRecord]],
    examples: List[Tuple[str, str]],
    output_path: Path,
) -> None:
    """Export full per-token, per-layer analysis results to JSON.

    The JSON schema is:
    [
      {
        "example_label": str,
        "prompt": str,
        "tokens": [
          {
            "token_str":      str,
            "token_idx":      int,
            "is_computation": bool,
            "active_counts":  [int × NUM_LAYERS],
            "top_features":   [[int, float] × top_k] × NUM_LAYERS
          },
          ...
        ]
      },
      ...
    ]

    Parameters
    ----------
    all_records : List[List[TokenRecord]]
    examples : List[Tuple[str, str]]  — (prompt, label) pairs
    output_path : Path
    """
    data = []
    for (prompt, label), records in zip(examples, all_records):
        entry = {
            "example_label": label,
            "prompt": prompt,
            "tokens": [
                {
                    "token_str":      rec.token_str,
                    "token_idx":      rec.token_idx,
                    "is_computation": rec.is_computation,
                    "active_counts":  rec.active_counts,
                    "top_features":   rec.top_features,
                }
                for rec in records
            ],
        }
        data.append(entry)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {output_path.name}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(all_records: List[List[TokenRecord]], examples: List[Tuple[str, str]]) -> None:
    """Print a concise per-example and aggregate summary to stdout."""
    print("\n" + "─" * 65)
    print("  Summary Statistics")
    print("─" * 65)

    all_comp_acts   = defaultdict(list)  # layer → list of active counts (comp tokens)
    all_bg_acts     = defaultdict(list)  # layer → list of active counts (bg tokens)

    for (_, label), records in zip(examples, all_records):
        n_comp = sum(1 for r in records if r.is_computation)
        n_bg   = len(records) - n_comp
        avg    = np.mean([np.mean(r.active_counts) for r in records])
        print(f"  {label:15s}  tokens={len(records):3d}  "
              f"comp={n_comp:2d}  bg={n_bg:3d}  "
              f"mean_active={avg:.1f}/layer")
        for rec in records:
            for li, cnt in enumerate(rec.active_counts):
                if rec.is_computation:
                    all_comp_acts[li].append(cnt)
                else:
                    all_bg_acts[li].append(cnt)

    print("─" * 65)
    print("  Layers with largest computation-vs-background gap:")
    gaps = {
        li: np.mean(all_comp_acts[li]) - np.mean(all_bg_acts[li])
        for li in range(NUM_LAYERS)
        if all_comp_acts[li] and all_bg_acts[li]
    }
    top5 = sorted(gaps, key=lambda li: abs(gaps[li]), reverse=True)[:5]
    for li in top5:
        print(f"    Layer {li:2d}: Δ = {gaps[li]:+.2f}  "
              f"(comp={np.mean(all_comp_acts[li]):.1f}, "
              f"bg={np.mean(all_bg_acts[li]):.1f})")
    print("─" * 65)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--saes-dir",
        type=Path,
        default=Path("phase5_results/multilayer_gpt2_12x/saes"),
        help="Directory with gpt2-medium_layer{N}_sae.pt checkpoints "
             "(default: phase5_results/multilayer_gpt2_12x/saes)",
    )
    parser.add_argument(
        "--activations-dir",
        type=Path,
        default=Path("phase4_results/activations_multilayer"),
        help="Directory with training activation .pt files used for normalisation "
             "(default: phase4_results/activations_multilayer)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase5_results/reasoning_flow_analysis"),
        help="Directory for output PNGs and JSON (created if absent)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu  (default: cuda:0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single custom prompt to analyse instead of built-in GSM8K examples",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k features to record per (token, layer) in the JSON output "
             "(default: 10)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 5.4 — Reasoning Flow Tracer")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load GPT-2 medium
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading {MODEL_ID} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID)
        .to(args.device)
        .eval()
    )
    print(f"      {NUM_LAYERS} layers, hidden_dim={HIDDEN_DIM}")

    # ------------------------------------------------------------------
    # 2. Load SAEs
    # ------------------------------------------------------------------
    print(f"\n[2/4] Loading SAEs from {args.saes_dir} ...")
    saes = load_saes(args.saes_dir, args.device)

    # ------------------------------------------------------------------
    # 3. Load normalisation statistics
    # ------------------------------------------------------------------
    print(f"\n[3/4] Loading normalisation stats from {args.activations_dir} ...")
    norm_stats = load_norm_stats(args.activations_dir, args.device)
    print(f"      Stats loaded for {len(norm_stats)}/{NUM_LAYERS} layers")

    # ------------------------------------------------------------------
    # 4. Run analysis on examples
    # ------------------------------------------------------------------
    examples: List[Tuple[str, str]] = (
        [(args.prompt, "custom")] if args.prompt else BUILTIN_EXAMPLES
    )

    print(f"\n[4/4] Analysing {len(examples)} example(s) ...")
    all_records: List[List[TokenRecord]] = []

    for ex_idx, (prompt, label) in enumerate(examples):
        short = prompt[:65].replace("\n", " ") + ("…" if len(prompt) > 65 else "")
        print(f"\n  [{ex_idx + 1}/{len(examples)}] {label}: {short!r}")

        records = analyse_example(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            saes=saes,
            norm_stats=norm_stats,
            device=args.device,
            top_k=args.top_k,
        )
        all_records.append(records)

        n_comp = sum(1 for r in records if r.is_computation)
        avg_active = np.mean([np.mean(r.active_counts) for r in records])
        print(f"      tokens={len(records)}  comp={n_comp}  "
              f"mean_active_per_layer={avg_active:.1f}")

        plot_feature_flow(
            records=records,
            title_suffix=f"{label}  (example {ex_idx + 1})",
            output_path=args.output_dir / f"feature_flow_ex{ex_idx + 1:02d}_{label}.png",
        )

    # ------------------------------------------------------------------
    # 5. Aggregate plots + export
    # ------------------------------------------------------------------
    print("\n  Generating aggregate plots …")
    plot_computation_contrast(all_records, args.output_dir / "computation_contrast.png")
    plot_feature_persistence( all_records, args.output_dir / "feature_persistence.png")
    plot_layer_sparsity(      all_records, args.output_dir / "layer_sparsity.png")
    export_json(all_records, examples,    args.output_dir / "reasoning_flow.json")

    print_summary(all_records, examples)

    print(f"\n{'=' * 65}")
    print(f"Done.  All outputs written to: {args.output_dir}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

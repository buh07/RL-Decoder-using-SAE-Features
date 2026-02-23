#!/usr/bin/env python3
"""
Phase 7 — Experiment B: Feature Co-activation Analysis
=======================================================
Identifies SAE features that systematically co-activate across token roles
in arithmetic computations.  Where Experiment A asks "which layer is best?",
Experiment B asks "which specific *features* encode each computational role?"

Token roles
-----------
Each token in an annotation is assigned one of four roles:

  operand   — tokens of the expression before ``=``  (e.g. "48", "/", "2")
  result    — tokens of the result C                 (e.g. "24")
  eq_sign   — the single ``=`` token inside the annotation
  background — any other token in the same example

For a given layer and feature index f, the *activation rate* for role r is:

    rate(f, r, L) = mean(h[f] > 0  for all (tok, layer=L) where tok has role r)

A feature is *role-selective* if its rate is substantially higher for one
role than for the background.  We compute three selectivity scores:

    sel_operand(f, L)  = rate(f, operand, L)  - rate(f, background, L)
    sel_result(f, L)   = rate(f, result, L)   - rate(f, background, L)
    sel_eq(f, L)       = rate(f, eq, L)       - rate(f, background, L)

Features are then classified (per layer) into five functional categories based
on which selectivity score dominates and whether any score exceeds a threshold:

  computation_bridge  — high sel_eq  (active at the computation boundary)
  result_encoder      — high sel_result, lower sel_operand
  operand_tracker     — high sel_operand, lower sel_result
  dual_role           — high both sel_operand and sel_result
  background          — no score above threshold

Output
------
  phase7/results/coactivation/
    coactivation_results.json   — per-layer feature counts & top features
    activation_rates.npz        — full rate arrays (layers × LATENT_DIM × 4 roles)
    selectivity_heatmap.png     — top-100 most-selective features across layers
    category_pie.png            — proportion of features in each category per layer
    role_rate_curves.png        — mean activation rate by role vs. layer

Usage
-----
  cd "/scratch2/f004ndc/RL-Decoder with SAE Features"
  .venv/bin/python3 phase7/feature_coactivation.py \\
      --dataset    phase7/results/collection/gsm8k_arithmetic_dataset.pt \\
      --output-dir phase7/results/coactivation \\
      --top-k      256 \\
      --sel-threshold 0.10
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

NUM_LAYERS = 24
LATENT_DIM = 12288

ROLE_OPERAND    = 0
ROLE_RESULT     = 1
ROLE_EQ         = 2
ROLE_BACKGROUND = 3
N_ROLES = 4
ROLE_NAMES = ["operand", "result", "eq_sign", "background"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> List[dict]:
    records = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(records)} annotation records.")
    return records


# ---------------------------------------------------------------------------
# Activation rate computation
# ---------------------------------------------------------------------------

def compute_activation_rates(
    records: List[dict],
    top_k: int,
) -> np.ndarray:
    """Compute per-layer, per-role activation rates for all SAE features.

    Rather than thresholding the full 12288-dimensional latent vector (which
    would be expensive), we use a top-K approach: for each token position we
    set the top-``top_k`` features by activation magnitude as "active" and
    the rest as inactive.  This is the standard sparsification used when the
    L1 penalty is insufficient to fully enforce sparsity.

    Parameters
    ----------
    records : List[dict]
        Dataset records from arithmetic_data_collector.py.
    top_k : int
        Number of features to treat as active per token position.

    Returns
    -------
    rates : ndarray, shape (NUM_LAYERS, N_ROLES, LATENT_DIM), float32
        rates[L, role, f] = fraction of (token, example) pairs of that role
        at layer L where feature f is in the top-k.
    """
    # Accumulators: sum of top-k indicator and count, per role per layer
    counts   = np.zeros((NUM_LAYERS, N_ROLES), dtype=np.int64)
    feat_sum = np.zeros((NUM_LAYERS, N_ROLES, LATENT_DIM), dtype=np.float32)

    for r in records:
        # ------------------------------------------------------------------
        # Roles:
        #   eq_sign   → eq_features  (single token, already extracted)
        #   operand   → pre_eq_features  (mean already, but we re-use it as
        #               a proxy for each operand token since we don't store
        #               per-token features; the mean still carries the same
        #               top-K structure for co-activation purposes)
        #   result    → result_features
        #   background → we synthesise a background estimate by taking the
        #               complement: any position not in eq / pre_eq / result.
        #               We approximate this as a random-sign baseline drawn
        #               from N(0, 0.1) in float16 — but that would require
        #               actual background tokens.  Instead we use the
        #               *minimum* activation across roles as background proxy.
        # ------------------------------------------------------------------
        role_feats = {
            ROLE_EQ:      r["eq_features"].float().numpy(),      # (24, 12288)
            ROLE_OPERAND: r["pre_eq_features"].float().numpy(),  # (24, 12288)
            ROLE_RESULT:  r["result_features"].float().numpy(),  # (24, 12288)
        }

        # Background: pointwise minimum across the three annotation roles
        # This approximates "what the residual stream looks like when it is
        # NOT processing any annotation token".
        bg = np.minimum(
            np.minimum(role_feats[ROLE_EQ], role_feats[ROLE_OPERAND]),
            role_feats[ROLE_RESULT],
        )  # (24, 12288)
        role_feats[ROLE_BACKGROUND] = bg

        for role_id, feats in role_feats.items():
            for layer in range(NUM_LAYERS):
                vec = feats[layer]   # (12288,)
                # Top-k mask
                topk_idx = np.argpartition(vec, -top_k)[-top_k:]
                feat_sum[layer, role_id, topk_idx] += 1
                counts[layer, role_id] += 1

    # Normalise by count to get activation rates
    rates = np.zeros_like(feat_sum)
    for L in range(NUM_LAYERS):
        for role in range(N_ROLES):
            c = counts[L, role]
            if c > 0:
                rates[L, role] = feat_sum[L, role] / c

    print(f"  Computed activation rates over {counts[0, ROLE_EQ]} examples "
          f"per layer.")
    return rates


# ---------------------------------------------------------------------------
# Selectivity scores and feature categorisation
# ---------------------------------------------------------------------------

def compute_selectivity(
    rates: np.ndarray,
    sel_threshold: float,
) -> Tuple[np.ndarray, Dict[str, List[List[int]]]]:
    """Compute selectivity scores and classify features per layer.

    Parameters
    ----------
    rates : ndarray, shape (NUM_LAYERS, N_ROLES, LATENT_DIM)
    sel_threshold : float
        Minimum selectivity score for a feature to be considered role-specific.

    Returns
    -------
    selectivity : ndarray, shape (NUM_LAYERS, 3, LATENT_DIM)
        Dimensions in axis-1: [sel_operand, sel_result, sel_eq].
    categories : Dict[str, List[List[int]]]
        Mapping from category name → list of feature-index lists, one per layer.
    """
    bg = rates[:, ROLE_BACKGROUND, :]  # (24, 12288)

    sel_operand = rates[:, ROLE_OPERAND, :] - bg   # (24, 12288)
    sel_result  = rates[:, ROLE_RESULT,  :] - bg
    sel_eq      = rates[:, ROLE_EQ,      :] - bg

    selectivity = np.stack([sel_operand, sel_result, sel_eq], axis=1)  # (24,3,12288)

    categories: Dict[str, List[List[int]]] = {
        "computation_bridge": [],
        "result_encoder":     [],
        "operand_tracker":    [],
        "dual_role":          [],
        "background":         [],
    }

    for L in range(NUM_LAYERS):
        so = sel_operand[L]   # (12288,)
        sr = sel_result[L]
        se = sel_eq[L]

        op_hi  = so > sel_threshold
        res_hi = sr > sel_threshold
        eq_hi  = se > sel_threshold

        bridge  = eq_hi & ~op_hi & ~res_hi
        enc     = res_hi & ~op_hi
        tracker = op_hi & ~res_hi
        dual    = op_hi & res_hi
        bkg     = ~op_hi & ~res_hi & ~eq_hi

        categories["computation_bridge"].append(np.where(bridge)[0].tolist())
        categories["result_encoder"].append(np.where(enc)[0].tolist())
        categories["operand_tracker"].append(np.where(tracker)[0].tolist())
        categories["dual_role"].append(np.where(dual)[0].tolist())
        categories["background"].append(np.where(bkg)[0].tolist())

    return selectivity, categories


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_selectivity_heatmap(
    selectivity: np.ndarray,
    output_dir: Path,
) -> None:
    """Show the top-100 most selective features across layers for each role."""
    # Aggregate selectivity across layers by taking max over layers
    max_sel_operand = selectivity[:, 0, :].max(axis=0)  # (12288,)
    max_sel_result  = selectivity[:, 1, :].max(axis=0)
    max_sel_eq      = selectivity[:, 2, :].max(axis=0)

    combined = max_sel_operand + max_sel_result + max_sel_eq  # (12288,)
    top100 = np.argsort(combined)[-100:][::-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    role_titles = ["Operand selectivity", "Result selectivity", "= sign selectivity"]

    for ax_i, (ax, role_idx) in enumerate(zip(axes, range(3))):
        mat = selectivity[:, role_idx, :][:, top100].T  # (100, 24)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-0.3, vmax=0.3, interpolation="nearest")
        ax.set_xlabel("Layer", fontsize=10)
        if ax_i == 0:
            ax.set_ylabel("Top-100 SAE features (by combined selectivity)", fontsize=9)
        ax.set_title(role_titles[role_idx], fontsize=10)
        ax.set_xticks(range(NUM_LAYERS))
        ax.set_xticklabels(range(NUM_LAYERS), fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.suptitle(
        "SAE Feature Selectivity by Token Role\n"
        "(selectivity = activation rate for role − background rate)",
        fontsize=11, y=1.02,
    )
    out = output_dir / "selectivity_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_category_pie(
    categories: Dict[str, List[List[int]]],
    output_dir: Path,
) -> None:
    """Bar chart showing the count of each feature category per layer."""
    cat_names = list(categories.keys())
    cat_colors = ["#E74C3C", "#27AE60", "#3498DB", "#F39C12", "#BDC3C7"]

    counts_per_layer = {cat: [] for cat in cat_names}
    for L in range(NUM_LAYERS):
        for cat in cat_names:
            counts_per_layer[cat].append(len(categories[cat][L]))

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(NUM_LAYERS)
    xs = np.arange(NUM_LAYERS)
    for cat, col in zip(cat_names, cat_colors):
        vals = np.array(counts_per_layer[cat], dtype=float)
        ax.bar(xs, vals, bottom=bottom, color=col, alpha=0.85, label=cat)
        bottom += vals

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Number of SAE features", fontsize=11)
    ax.set_title(
        "Feature Functional Category Distribution Across Layers\n"
        "(top-K thresholded activation with selectivity threshold)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    out = output_dir / "category_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_role_rate_curves(
    rates: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot mean activation rate by role across layers."""
    fig, ax = plt.subplots(figsize=(10, 5))

    role_colors = ["#3498DB", "#27AE60", "#E74C3C", "#95A5A6"]
    for role_id in range(N_ROLES):
        mean_rate = rates[:, role_id, :].mean(axis=1)  # (24,)
        ax.plot(range(NUM_LAYERS), mean_rate,
                color=role_colors[role_id], marker="o", markersize=4,
                label=ROLE_NAMES[role_id])

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean activation rate (top-K threshold)", fontsize=11)
    ax.set_title(
        "Mean SAE Feature Activation Rate by Token Role Across Layers",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = output_dir / "role_rate_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment B: Feature co-activation analysis for arithmetic roles."
    )
    parser.add_argument(
        "--dataset",
        default="phase7/results/collection/gsm8k_arithmetic_dataset.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="phase7/results/coactivation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=256,
        help="Number of top features to treat as active per token/layer.",
    )
    parser.add_argument(
        "--sel-threshold",
        type=float,
        default=0.10,
        help="Minimum selectivity difference to classify a feature as role-specific.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print("\n[1/4] Loading dataset …")
    records = load_dataset(Path(args.dataset))

    # -----------------------------------------------------------------------
    # Compute activation rates
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Computing activation rates (top-k={args.top_k}) …")
    rates = compute_activation_rates(records, args.top_k)
    np.savez(
        output_dir / "activation_rates.npz",
        rates=rates,
        role_names=np.array(ROLE_NAMES),
    )
    print(f"  Saved activation_rates.npz")

    # -----------------------------------------------------------------------
    # Selectivity and classification
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Computing selectivity (threshold={args.sel_threshold}) …")
    selectivity, categories = compute_selectivity(rates, args.sel_threshold)

    # Aggregate category sizes across layers for summary
    cat_totals = {
        cat: sum(len(categories[cat][L]) for L in range(NUM_LAYERS))
        for cat in categories
    }

    # Top-5 features per category at the layer with most in that category
    top_feats_per_cat: Dict[str, dict] = {}
    for cat in categories:
        best_layer = int(np.argmax([len(categories[cat][L]) for L in range(NUM_LAYERS)]))
        feats_at_best = categories[cat][best_layer]
        top_feats_per_cat[cat] = {
            "best_layer": best_layer,
            "count_at_best_layer": len(feats_at_best),
            "sample_features": feats_at_best[:20],
        }

    results = {
        "top_k":             args.top_k,
        "sel_threshold":     args.sel_threshold,
        "num_records":       len(records),
        "category_totals_across_all_layers": cat_totals,
        "top_features_per_category": top_feats_per_cat,
        "mean_rate_by_role_by_layer": {
            ROLE_NAMES[r]: rates[:, r, :].mean(axis=1).tolist()
            for r in range(N_ROLES)
        },
    }
    json_path = output_dir / "coactivation_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("\n[4/4] Generating plots …")
    plot_selectivity_heatmap(selectivity, output_dir)
    plot_category_pie(categories, output_dir)
    plot_role_rate_curves(rates, output_dir)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Experiment B Summary ===")
    for cat, info in top_feats_per_cat.items():
        total = cat_totals[cat]
        best_L = info["best_layer"]
        n_best = info["count_at_best_layer"]
        print(f"  {cat:22s}: {total:6d} feature-layers total, "
              f"peak {n_best:4d} @ layer {best_L:2d}  "
              f"sample={info['sample_features'][:5]}")

    print(f"\n  Interpretation:")
    bridge_layer = top_feats_per_cat["computation_bridge"]["best_layer"]
    result_layer = top_feats_per_cat["result_encoder"]["best_layer"]
    print(f"  Computation-bridge features peak at layer {bridge_layer} — likely")
    print(f"  the transition point where the model moves from processing inputs")
    print(f"  to generating the output result.")
    print(f"  Result-encoder features peak at layer {result_layer}.")


if __name__ == "__main__":
    main()

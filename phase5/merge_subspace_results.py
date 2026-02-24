#!/usr/bin/env python3
"""
Merge Phase 5b shard results into a single results file + plots.

Usage (after all 3 GPU shards finish):
    python3 phase5/merge_subspace_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NUM_LAYERS = 24
SHARD_DIR  = Path("phase5_results/steering_subspace")
OUT_DIR    = Path("phase5_results/steering_subspace")


def main():
    # Collect all shard files
    shard_files = sorted(SHARD_DIR.glob("shard_L*.json"))
    if not shard_files:
        print(f"No shard files found in {SHARD_DIR}. Run the per-GPU scripts first.")
        return
    print(f"Found {len(shard_files)} shard files: {[f.name for f in shard_files]}")

    # Merge
    all_layers = []
    mean_delta: dict = {}
    alphas = None
    probe_k = None
    num_test = None
    high_c = None
    low_c = None
    probe_pos = None

    for sf in shard_files:
        with open(sf) as f:
            d = json.load(f)
        if alphas is None:
            alphas = d["alphas"]
            probe_k = d["probe_k"]
            num_test = d["num_test_records"]
            high_c = d["high_c_train"]
            low_c = d["low_c_train"]
            probe_pos = d["probe_position"]
        for L_str, alpha_dict in d["mean_delta_logprob"].items():
            mean_delta[int(L_str)] = {float(a): v for a, v in alpha_dict.items()}
            all_layers.append(int(L_str))

    all_layers = sorted(set(all_layers))
    missing = [L for L in range(NUM_LAYERS) if L not in all_layers]
    if missing:
        print(f"WARNING: Missing layers {missing} — these will be shown as NaN.")

    # Find global best
    best_layer, best_alpha, best_val = all_layers[0], alphas[0], -999.0
    for L in all_layers:
        for a in alphas:
            v = mean_delta.get(L, {}).get(a, float("nan"))
            if not np.isnan(v) and v > best_val:
                best_val = v
                best_layer = L
                best_alpha = a

    print(f"\n{'Layer':>5}  " + "  ".join(f"α={a:4.1f}" for a in alphas))
    print("-" * (7 + 10 * len(alphas)))
    for L in range(NUM_LAYERS):
        row = f"{L:5d}  "
        row += "  ".join(
            f"{mean_delta.get(L, {}).get(a, float('nan')):+7.4f}" for a in alphas
        )
        print(row)

    print(f"\nGlobal best: layer={best_layer}, α={best_alpha}, mean Δlog_prob={best_val:+.4f}")
    print(f"Phase 5 full-space best:      layer=17, α=0.5,  Δlog_prob=-2.854")
    print(f"Phase 4r subspace-patch best: layer=22,          Δlog_prob=+0.107")

    # Save merged JSON
    merged = {
        "method": "subspace_mean_diff",
        "probe_position": probe_pos,
        "probe_k": probe_k,
        "layers": list(range(NUM_LAYERS)),
        "alphas": alphas,
        "mean_delta_logprob": {
            str(L): {str(a): mean_delta.get(L, {}).get(a, float("nan")) for a in alphas}
            for L in range(NUM_LAYERS)
        },
        "best_layer": best_layer,
        "best_alpha": best_alpha,
        "best_mean_delta_logprob": best_val,
        "num_test_records": num_test,
        "high_c_train": high_c,
        "low_c_train": low_c,
        "baseline_phase5_full_space_best": -2.854,
        "baseline_phase4r_subspace_patch_best": 0.1066,
    }
    out_json = OUT_DIR / "steering_results_subspace.json"
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged results: {out_json}")

    # Heatmap
    mat = np.array([
        [mean_delta.get(L, {}).get(a, float("nan")) for a in alphas]
        for L in range(NUM_LAYERS)
    ])
    fig, ax = plt.subplots(figsize=(7, 10))
    vmax = max(0.2, np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax, origin="upper")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(NUM_LAYERS)])
    ax.set_title(f"Subspace Steering Δlog_prob\n(probe subspace, {probe_k} features, position={probe_pos})")
    plt.colorbar(im, ax=ax, label="Mean Δlog_prob")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "steering_heatmap_subspace.png", dpi=150)
    plt.close()

    # Line plot: all alphas
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(-2.854, color="red", linewidth=0.8, linestyle=":",
               label="Phase 5 full-space best (L17 α=0.5, −2.854)")
    ax.axhline(0.1066, color="green", linewidth=0.8, linestyle=":",
               label="Phase 4r subspace-patch best (L22, +0.107)")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ai, alpha in enumerate(alphas):
        vals = [mean_delta.get(L, {}).get(alpha, float("nan")) for L in range(NUM_LAYERS)]
        ax.plot(range(NUM_LAYERS), vals, marker="o", linewidth=1.5,
                color=colors[ai % len(colors)], label=f"α={alpha}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δlog_prob")
    ax.set_title("Phase 5b Subspace Steering vs Baselines")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "steering_by_layer_subspace.png", dpi=150)
    plt.close()
    print(f"Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

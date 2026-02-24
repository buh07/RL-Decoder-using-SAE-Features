#!/usr/bin/env python3
"""
Merge Phase 5c (NN transfer) shard results into a single results file + plots.

Usage (after all 3 GPU shards finish):
    python3 phase5/merge_nn_transfer_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NUM_LAYERS = 24
SHARD_DIR  = Path("phase5_results/steering_nn_transfer")
OUT_DIR    = Path("phase5_results/steering_nn_transfer")


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
    betas = None
    probe_k = None
    num_test = None
    high_c = None
    low_c = None
    probe_pos = None

    for sf in shard_files:
        with open(sf) as f:
            d = json.load(f)
        if betas is None:
            betas     = d["betas"]
            probe_k   = d["probe_k"]
            num_test  = d["num_test_records"]
            high_c    = d["high_c_train"]
            low_c     = d["low_c_train"]
            probe_pos = d["probe_position"]
        for L_str, beta_dict in d["mean_delta_logprob"].items():
            mean_delta[int(L_str)] = {float(b): v for b, v in beta_dict.items()}
            all_layers.append(int(L_str))

    all_layers = sorted(set(all_layers))
    missing = [L for L in range(NUM_LAYERS) if L not in all_layers]
    if missing:
        print(f"WARNING: Missing layers {missing} — these will be shown as NaN.")

    # Find global best
    best_layer, best_beta, best_val = all_layers[0], betas[0], -999.0
    for L in all_layers:
        for b in betas:
            v = mean_delta.get(L, {}).get(b, float("nan"))
            if not np.isnan(v) and v > best_val:
                best_val  = v
                best_layer = L
                best_beta  = b

    print(f"\n{'Layer':>5}  " + "  ".join(f"β={b:.2f}" for b in betas))
    print("-" * (7 + 10 * len(betas)))
    for L in range(NUM_LAYERS):
        row = f"{L:5d}  "
        row += "  ".join(
            f"{mean_delta.get(L, {}).get(b, float('nan')):+7.4f}" for b in betas
        )
        print(row)

    print(f"\nGlobal best: layer={best_layer}, β={best_beta}, mean Δlog_prob={best_val:+.4f}")
    print(f"Phase 5b subspace mean-diff best: layer=22, α=4.0, Δlog_prob=-2.385")
    print(f"Phase 5  full-space best:         layer=17, α=0.5, Δlog_prob=-2.854")
    print(f"Phase 4r subspace-patch best:     layer=22,         Δlog_prob=+0.107")

    # Save merged JSON
    merged = {
        "method": "nn_transfer",
        "probe_position": probe_pos,
        "probe_k": probe_k,
        "layers": list(range(NUM_LAYERS)),
        "betas": betas,
        "mean_delta_logprob": {
            str(L): {str(b): mean_delta.get(L, {}).get(b, float("nan")) for b in betas}
            for L in range(NUM_LAYERS)
        },
        "best_layer": best_layer,
        "best_beta": best_beta,
        "best_mean_delta_logprob": best_val,
        "num_test_records": num_test,
        "high_c_train": high_c,
        "low_c_train": low_c,
        "baseline_phase5_full_space_best": -2.854,
        "baseline_phase5b_subspace_best": -2.385,
        "baseline_phase4r_subspace_patch_best": 0.1066,
    }
    out_json = OUT_DIR / "steering_results_nn_transfer.json"
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged results: {out_json}")

    # Heatmap: layers × betas
    mat = np.array([
        [mean_delta.get(L, {}).get(b, float("nan")) for b in betas]
        for L in range(NUM_LAYERS)
    ])
    fig, ax = plt.subplots(figsize=(7, 10))
    vmax = max(0.2, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax, origin="upper")
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"β={b}" for b in betas])
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(NUM_LAYERS)])
    ax.set_title(
        f"NN-Transfer Steering Δlog_prob\n"
        f"(probe subspace, {probe_k} features, position={probe_pos})"
    )
    plt.colorbar(im, ax=ax, label="Mean Δlog_prob")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "steering_heatmap_nn_transfer.png", dpi=150)
    plt.close()

    # Line plot: one line per beta value
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(-2.854, color="red", linewidth=0.8, linestyle=":",
               label="Phase 5 full-space best (L17 α=0.5, −2.854)")
    ax.axhline(-2.385, color="orange", linewidth=0.8, linestyle=":",
               label="Phase 5b subspace best (L22 α=4.0, −2.385)")
    ax.axhline(0.1066, color="green", linewidth=0.8, linestyle=":",
               label="Phase 4r subspace-patch best (L22, +0.107)")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for bi, beta in enumerate(betas):
        vals = [mean_delta.get(L, {}).get(beta, float("nan")) for L in range(NUM_LAYERS)]
        ax.plot(range(NUM_LAYERS), vals, marker="o", linewidth=1.5,
                color=colors[bi % len(colors)], label=f"β={beta}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δlog_prob")
    ax.set_title("Phase 5c NN-Transfer Steering vs Baselines")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "steering_by_layer_nn_transfer.png", dpi=150)
    plt.close()
    print(f"Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

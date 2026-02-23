#!/usr/bin/env python3
"""
Phase 5 Task 4: Visualization & Heatmap Generation

Generates layer universality heatmaps and analysis plots from transfer matrix.

Usage:
    python3 phase5_task4_visualization.py \
        --transfer-matrix phase5_results/multilayer_transfer/transfer_matrix.json \
        --output-dir phase5_results/multilayer_transfer \
        --format png pdf
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_transfer_matrix(file_path: Path) -> Dict:
    """Load transfer matrix JSON."""
    with open(file_path, "r") as f:
        return json.load(f)


def build_layer_heatmap_data(
    transfer_matrix: Dict,
    layer_indices: List[int],
    models: List[str],
) -> np.ndarray:
    """
    Build heatmap data showing transfer quality between layers.
    Rows: source layers, Columns: target layers
    """
    n_layers = len(layer_indices)
    heatmap_data = np.zeros((n_layers, n_layers))

    for src_idx, src_layer in enumerate(layer_indices):
        for tgt_idx, tgt_layer in enumerate(layer_indices):
            if src_idx == tgt_idx:
                # Same layer, same model (diagonal = 1.0)
                heatmap_data[src_idx, tgt_idx] = 1.0
            else:
                # Average transfer across models for this layer pair
                transfers = []
                for transfer_key, metrics in transfer_matrix["transfer_matrix"].items():
                    if (
                        f"layer{src_layer}__to__" in transfer_key
                        and f"layer{tgt_layer}" in transfer_key
                    ):
                        transfers.append(metrics["transfer_recon_ratio"])

                if transfers:
                    heatmap_data[src_idx, tgt_idx] = np.mean(transfers)
                else:
                    heatmap_data[src_idx, tgt_idx] = 0.0

    return heatmap_data


def plot_layer_transfer_heatmap(
    transfer_matrix: Dict,
    output_dir: Path,
    formats: List[str],
):
    """Generate layer transfer heatmap."""
    logger.info("Generating layer transfer heatmap...")

    layer_indices = transfer_matrix["config"]["layers"]
    heatmap_data = build_layer_heatmap_data(transfer_matrix, layer_indices, [])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(layer_indices)))
    ax.set_yticks(range(len(layer_indices)))
    ax.set_xticklabels([f"Layer {l}" for l in layer_indices])
    ax.set_yticklabels([f"Layer {l}" for l in layer_indices])

    ax.set_xlabel("Target Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Source Layer", fontsize=12, fontweight="bold")
    ax.set_title(
        "Multi-Layer Feature Transfer Quality\n(Green=Universal, Red=Specialized)",
        fontsize=14,
        fontweight="bold",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Transfer Ratio")

    # Add text annotations
    for i in range(len(layer_indices)):
        for j in range(len(layer_indices)):
            val = heatmap_data[i, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    plt.tight_layout()

    # Save in requested formats
    for fmt in formats:
        output_file = output_dir / f"layer_transfer_heatmap.{fmt}"
        fig.savefig(output_file, dpi=150 if fmt == "png" else 300, bbox_inches="tight")
        logger.info(f"Saved {output_file}")

    plt.close(fig)


def plot_layer_universality_curve(
    transfer_matrix: Dict,
    output_dir: Path,
    formats: List[str],
):
    """Generate curve showing layer universality scores."""
    logger.info("Generating layer universality curve...")

    layer_scores = transfer_matrix["layer_universality_scores"]
    layer_names = sorted(layer_scores.keys(), key=lambda x: int(x.split("_")[1]))
    layer_indices = [int(name.split("_")[1]) for name in layer_names]
    scores = [layer_scores[name] for name in layer_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curve
    ax.plot(layer_indices, scores, "o-", linewidth=2, markersize=8, color="steelblue")

    # Fill area under curve
    ax.fill_between(layer_indices, scores, alpha=0.3, color="steelblue")

    # Add reference lines
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="High universality (0.9)")
    ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.5, label="Medium universality (0.7)")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Low universality (0.5)")

    ax.set_xlabel("Layer Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Transfer Ratio", fontsize=12, fontweight="bold")
    ax.set_title(
        "Feature Universality vs. Network Depth\n(Higher = More Universal Across Models)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    # Save in requested formats
    for fmt in formats:
        output_file = output_dir / f"layer_universality_curve.{fmt}"
        fig.savefig(output_file, dpi=150 if fmt == "png" else 300, bbox_inches="tight")
        logger.info(f"Saved {output_file}")

    plt.close(fig)


def plot_per_model_transfer(
    transfer_matrix: Dict,
    output_dir: Path,
    formats: List[str],
):
    """Generate per-model transfer analysis plots."""
    logger.info("Generating per-model transfer analysis...")

    # Group by model pair
    model_transfers = {}

    for transfer_key, metrics in transfer_matrix["transfer_matrix"].items():
        # Parse key: model_A_layerX__to__model_B_layerY
        parts = transfer_key.split("__to__")
        src = parts[0].rsplit("_layer", 1)[0]
        tgt = parts[1].rsplit("_layer", 1)[0]
        pair = f"{src} → {tgt}"

        if pair not in model_transfers:
            model_transfers[pair] = []

        model_transfers[pair].append(metrics["transfer_recon_ratio"])

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    pairs = list(model_transfers.keys())
    means = [np.mean(model_transfers[p]) for p in pairs]
    stds = [np.std(model_transfers[p]) for p in pairs]

    colors = ["green" if m > 0.8 else "orange" if m > 0.6 else "red" for m in means]

    ax.bar(range(len(pairs)), means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.set_ylabel("Average Transfer Ratio", fontsize=12, fontweight="bold")
    ax.set_title(
        "Cross-Model Feature Transfer Quality\n(per model pair, averaged across layers)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save in requested formats
    for fmt in formats:
        output_file = output_dir / f"per_model_transfer_analysis.{fmt}"
        fig.savefig(output_file, dpi=150 if fmt == "png" else 300, bbox_inches="tight")
        logger.info(f"Saved {output_file}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize Phase 5.4 results")
    parser.add_argument(
        "--transfer-matrix",
        type=Path,
        default=Path("phase5_results/multilayer_transfer/transfer_matrix.json"),
        help="Transfer matrix JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase5_results/multilayer_transfer"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "jpg"],
        help="Output formats",
    )

    args = parser.parse_args()

    transfer_matrix_path = Path(args.transfer_matrix)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not transfer_matrix_path.exists():
        logger.error(f"Transfer matrix file not found: {transfer_matrix_path}")
        return

    logger.info("Loading transfer matrix...")
    transfer_matrix = load_transfer_matrix(transfer_matrix_path)

    logger.info(f"Found {len(transfer_matrix['transfer_matrix'])} transfer pairs")
    logger.info(f"Layers: {transfer_matrix['config']['layers']}")

    # Generate plots
    plot_layer_transfer_heatmap(transfer_matrix, output_dir, args.format)
    plot_layer_universality_curve(transfer_matrix, output_dir, args.format)
    plot_per_model_transfer(transfer_matrix, output_dir, args.format)

    logger.info(f"\n{'='*70}")
    logger.info(f"Visualization complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

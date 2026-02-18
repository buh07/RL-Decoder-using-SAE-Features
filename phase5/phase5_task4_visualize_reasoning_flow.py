#!/usr/bin/env python3
"""
Phase 5 Task 4: Visualize Reasoning Flow Analysis

Creates heatmaps showing layer-to-layer transfer patterns for reasoning flow analysis.

Usage:
    python3 phase5_task4_visualize_reasoning_flow.py \
        --results-file phase5_results/multilayer_transfer/reasoning_flow/reasoning_flow_analysis.json \
        --output-dir phase5_results/multilayer_transfer/reasoning_flow/visualizations
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def plot_transfer_heatmap(
    transfer_matrix: np.ndarray,
    layers: list,
    model_name: str,
    output_path: Path,
):
    """Plot transfer matrix as a heatmap."""
    plt.figure(figsize=(14, 12))
    
    # Use log scale for better visualization of small values
    # Add 1e-6 to avoid log(0)
    matrix_log = np.log10(transfer_matrix + 1e-6)
    
    # Create heatmap
    ax = sns.heatmap(
        matrix_log,
        xticklabels=layers,
        yticklabels=layers,
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Transfer Quality (log10)'},
        vmin=-3,  # log10(0.001)
        vmax=0,   # log10(1.0)
    )
    
    plt.title(f'{model_name.upper()}: Layer-to-Layer Transfer Matrix\nDarker = Better Transfer', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Target Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Source Layer', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved heatmap to {output_path}")


def plot_layer_universality(
    layer_universality: list,
    layers: list,
    model_name: str,
    output_path: Path,
):
    """Plot layer universality scores as a bar chart."""
    plt.figure(figsize=(14, 6))
    
    plt.bar(range(len(layers)), layer_universality, color='steelblue', alpha=0.7)
    plt.axhline(y=np.mean(layer_universality), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(layer_universality):.3f}')
    
    plt.title(f'{model_name.upper()}: Layer Universality Scores\n(How well each layer transfers to other layers)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Universality Score', fontsize=12, fontweight='bold')
    plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved universality chart to {output_path}")


def plot_cross_model_comparison(
    results: dict,
    output_path: Path,
):
    """Plot comparison of transfer quality across models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    model_names = list(results.keys())
    
    for idx, model_name in enumerate(model_names):
        data = results[model_name]
        layers = data['layers']
        layer_universality = data['layer_universality']
        
        ax = axes[idx]
        
        # Plot universality scores
        ax.bar(range(len(layers)), layer_universality, color='steelblue', alpha=0.7)
        ax.axhline(y=np.mean(layer_universality), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(layer_universality):.3f}')
        
        ax.set_title(f'{model_name.upper()}\nMost Universal: Layer {data["summary"]["most_universal_layer"]}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Universality Score', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Set consistent y-axis limits for comparison
        ax.set_ylim(0, max(0.8, max(layer_universality) * 1.1))
    
    plt.suptitle('Cross-Model Comparison: Layer Universality Patterns', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved cross-model comparison to {output_path}")


def plot_adjacent_vs_distant_transfer(
    transfer_matrix: np.ndarray,
    layers: list,
    model_name: str,
    output_path: Path,
):
    """Plot transfer quality vs layer distance."""
    n_layers = len(layers)
    
    # Compute average transfer for each distance
    max_distance = n_layers - 1
    distances = range(1, max_distance + 1)
    avg_transfers = []
    
    for dist in distances:
        transfers = []
        for i in range(n_layers - dist):
            transfers.append(transfer_matrix[i, i + dist])
            transfers.append(transfer_matrix[i + dist, i])
        avg_transfers.append(np.mean(transfers))
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(distances, avg_transfers, marker='o', linewidth=2, markersize=8, color='darkblue')
    plt.fill_between(distances, avg_transfers, alpha=0.3, color='lightblue')
    
    plt.title(f'{model_name.upper()}: Transfer Quality vs Layer Distance\n(Nearby layers vs distant layers)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer Distance', fontsize=12, fontweight='bold')
    plt.ylabel('Average Transfer Quality', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotation for nearby layers
    if len(avg_transfers) > 0:
        plt.annotate(f'Adjacent layers: {avg_transfers[0]:.3f}', 
                    xy=(1, avg_transfers[0]), 
                    xytext=(5, avg_transfers[0] * 1.2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved distance plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize reasoning flow analysis")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("phase5_results/multilayer_transfer/reasoning_flow/reasoning_flow_analysis.json"),
        help="Path to reasoning flow analysis JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase5_results/multilayer_transfer/reasoning_flow/visualizations"),
        help="Output directory for visualizations",
    )
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Generating visualizations for {len(results)} models...")
    
    # Generate per-model visualizations
    for model_name, data in results.items():
        logger.info(f"\nProcessing {model_name}...")
        
        layers = data['layers']
        transfer_matrix = np.array(data['transfer_matrix'])
        layer_universality = data['layer_universality']
        
        # Transfer heatmap
        heatmap_path = output_dir / f"{model_name}_transfer_heatmap.png"
        plot_transfer_heatmap(transfer_matrix, layers, model_name, heatmap_path)
        
        # Layer universality bar chart
        universality_path = output_dir / f"{model_name}_layer_universality.png"
        plot_layer_universality(layer_universality, layers, model_name, universality_path)
        
        # Distance decay plot
        distance_path = output_dir / f"{model_name}_distance_decay.png"
        plot_adjacent_vs_distant_transfer(transfer_matrix, layers, model_name, distance_path)
    
    # Cross-model comparison
    logger.info(f"\nGenerating cross-model comparison...")
    comparison_path = output_dir / "cross_model_comparison.png"
    plot_cross_model_comparison(results, comparison_path)
    
    # Generate summary statistics
    summary_path = output_dir / "summary_statistics.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("REASONING FLOW ANALYSIS - SUMMARY STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        for model_name, data in results.items():
            f.write(f"{model_name.upper()}:\n")
            f.write(f"  Number of layers: {data['summary']['num_layers']}\n")
            f.write(f"  Mean transfer quality: {data['summary']['mean_transfer_quality']:.4f}\n")
            f.write(f"  Max cross-layer transfer: {data['summary']['max_cross_layer_transfer']:.4f}\n")
            f.write(f"  Min cross-layer transfer: {data['summary']['min_cross_layer_transfer']:.4f}\n")
            f.write(f"  Most universal layer: {data['summary']['most_universal_layer']} ")
            f.write(f"(score: {data['summary']['most_universal_score']:.4f})\n")
            
            if data['bidirectional_transfer_pairs']:
                f.write(f"  Strong bi-directional pairs: {len(data['bidirectional_transfer_pairs'])}\n")
                for pair in data['bidirectional_transfer_pairs'][:3]:
                    f.write(f"    Layers {pair['layer_pair']}: {pair['avg_transfer']:.3f}\n")
            else:
                f.write(f"  Strong bi-directional pairs: 0\n")
            
            f.write("\n")
    
    logger.info(f"  Saved summary statistics to {summary_path}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Visualization complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 5 Task 4: Reasoning Flow Analysis - Within-Model Layer-to-Layer Transfer

Computes how well features from one layer can reconstruct activations at other layers
within the same model. This reveals how reasoning develops through network depth.

Usage:
    python3 phase5_task4_reasoning_flow_analysis.py \
        --sae-dir phase5_results/multilayer_transfer/saes \
        --activations-dir phase4_results/activations_multilayer \
        --output-dir phase5_results/multilayer_transfer/reasoning_flow \
        --device cuda:5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_architecture import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _load_activations(path: Path) -> torch.Tensor:
    """Load activation tensor from saved file."""
    payload = torch.load(path, weights_only=False)
    if isinstance(payload, dict) and "activations" in payload:
        acts = payload["activations"]
    else:
        acts = payload

    if acts.dim() == 3:
        acts = acts.reshape(-1, acts.shape[-1])

    return acts.float()


def _normalize_activations(acts: torch.Tensor) -> torch.Tensor:
    """Normalize activations to zero mean, unit variance."""
    mean = acts.mean(dim=0, keepdim=True)
    std = acts.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (acts - mean) / std


def _load_sae(checkpoint_path: Path, device: str) -> SparseAutoencoder:
    """Load SAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config_dict = checkpoint["config"]

    from sae_config import SAEConfig

    config = SAEConfig(**config_dict)
    sae = SparseAutoencoder(config).to(device)
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.eval()

    return sae


def compute_reconstruction_loss(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    device: str,
    batch_size: int = 128,
) -> float:
    """Compute average reconstruction loss."""
    sae.eval()
    losses = []

    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch = activations[i : i + batch_size].to(device)
            x_hat, _ = sae(batch)
            loss = ((batch - x_hat) ** 2).mean().item()
            losses.append(loss)

    return float(np.mean(losses)) if losses else 0.0


def compute_feature_activation_rates(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Compute activation rate (sparsity) for each feature."""
    sae.eval()
    all_features = []

    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch = activations[i : i + batch_size].to(device)
            _, h = sae(batch)
            all_features.append((h > 0).float().cpu())

    features = torch.cat(all_features, dim=0)
    activation_rates = features.mean(dim=0).numpy()

    return activation_rates


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning flow analysis for Phase 5.4"
    )
    parser.add_argument(
        "--sae-dir",
        type=Path,
        default=Path("phase5_results/multilayer_transfer/saes"),
        help="Directory containing SAE checkpoints",
    )
    parser.add_argument(
        "--activations-dir",
        type=Path,
        default=Path("phase4_results/activations_multilayer"),
        help="Directory containing activation files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase5_results/multilayer_transfer/reasoning_flow"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:5",
        help="Device to run on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference",
    )

    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all SAE and activation files
    sae_files = sorted(sae_dir.glob("*_sae.pt"))
    activation_files = sorted(activations_dir.glob("*_activations.pt"))

    logger.info(f"Found {len(sae_files)} SAE checkpoints")
    logger.info(f"Found {len(activation_files)} activation files")

    # Organize by model
    models_data = {}

    # Load activations
    logger.info("\nLoading activation files...")
    for act_file in activation_files:
        name_parts = act_file.stem.split("_")
        
        layer_part = None
        for i, part in enumerate(name_parts):
            if part.startswith("layer"):
                layer_part = part
                layer_idx = int(part[5:])
                model_name = "_".join(name_parts[:i])
                break
        
        if layer_part is None:
            continue
        
        if model_name not in models_data:
            models_data[model_name] = {"activations": {}, "saes": {}, "layers": []}
        
        acts = _load_activations(act_file)
        acts = _normalize_activations(acts)
        models_data[model_name]["activations"][layer_idx] = acts
        models_data[model_name]["layers"].append(layer_idx)
        logger.info(f"  {model_name} layer {layer_idx}: shape {acts.shape}")

    # Load SAEs
    logger.info("\nLoading SAE checkpoints...")
    for sae_file in sae_files:
        name_parts = sae_file.stem.split("_")
        
        layer_part = None
        for i, part in enumerate(name_parts):
            if part.startswith("layer"):
                layer_part = part
                layer_idx = int(part[5:])
                model_name = "_".join(name_parts[:i])
                break
        
        if layer_part is None or model_name not in models_data:
            continue
        
        sae = _load_sae(sae_file, args.device)
        models_data[model_name]["saes"][layer_idx] = sae
        logger.info(f"  {model_name} layer {layer_idx}")

    # Sort layers
    for model_name in models_data:
        models_data[model_name]["layers"] = sorted(set(models_data[model_name]["layers"]))

    # Compute within-model layer-to-layer transfer
    logger.info("\n" + "="*70)
    logger.info("Computing within-model reasoning flow...")
    logger.info("="*70)

    results = {}

    for model_name, data in models_data.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*70}")
        
        layers = data["layers"]
        saes = data["saes"]
        activations = data["activations"]
        
        logger.info(f"  Layers: {len(layers)} ({min(layers)} to {max(layers)})")
        
        # Compute transfer matrix for this model
        transfer_matrix = np.zeros((len(layers), len(layers)))
        
        for i, src_layer in enumerate(layers):
            src_sae = saes[src_layer]
            
            # Compute self-reconstruction first
            src_loss = compute_reconstruction_loss(
                src_sae, activations[src_layer], args.device, args.batch_size
            )
            
            logger.info(f"  Layer {src_layer} self-reconstruction loss: {src_loss:.4f}")
            
            for j, tgt_layer in enumerate(layers):
                if src_layer == tgt_layer:
                    # Perfect transfer to self
                    transfer_matrix[i, j] = 1.0
                    continue
                
                # Compute cross-layer reconstruction
                tgt_loss = compute_reconstruction_loss(
                    src_sae, activations[tgt_layer], args.device, args.batch_size
                )
                
                # Transfer ratio: higher is better (closer to 1.0 means good transfer)
                # Ratio = 1 / (tgt_loss / src_loss) = src_loss / tgt_loss
                # But we want lower tgt_loss to mean better transfer, so:
                transfer_ratio = src_loss / (tgt_loss + 1e-8)
                
                transfer_matrix[i, j] = float(transfer_ratio)
                
                logger.info(f"    Layer {src_layer} → Layer {tgt_layer}: "
                           f"loss={tgt_loss:.4f}, transfer_ratio={transfer_ratio:.3f}")
        
        # Compute feature activation patterns
        logger.info(f"\n  Computing feature activation patterns...")
        feature_patterns = {}
        
        for layer_idx in layers:
            sae = saes[layer_idx]
            acts = activations[layer_idx]
            
            activation_rates = compute_feature_activation_rates(
                sae, acts, args.device, args.batch_size
            )
            
            # Identify top active features
            top_indices = np.argsort(activation_rates)[-50:][::-1]
            
            feature_patterns[layer_idx] = {
                "activation_rates": activation_rates.tolist(),
                "top_50_indices": top_indices.tolist(),
                "top_50_rates": activation_rates[top_indices].tolist(),
                "mean_activation_rate": float(activation_rates.mean()),
                "median_activation_rate": float(np.median(activation_rates)),
            }
            
            logger.info(f"    Layer {layer_idx}: mean_rate={activation_rates.mean():.3f}, "
                       f"median_rate={np.median(activation_rates):.3f}")
        
        # Analyze reasoning flow patterns
        logger.info(f"\n  Analyzing reasoning flow patterns...")
        
        # Find highly universal layers (transfer well to many other layers)
        layer_universality = transfer_matrix.mean(axis=1)  # Average transfer to other layers
        
        # Find layer pairs with high bi-directional transfer
        bidirectional_transfer = []
        for i, layer_i in enumerate(layers):
            for j, layer_j in enumerate(layers):
                if i < j:  # Only upper triangle
                    forward_transfer = transfer_matrix[i, j]
                    backward_transfer = transfer_matrix[j, i]
                    avg_transfer = (forward_transfer + backward_transfer) / 2
                    
                    if avg_transfer > 0.7:  # Threshold for "good" transfer
                        bidirectional_transfer.append({
                            "layer_pair": [layer_i, layer_j],
                            "forward_transfer": float(forward_transfer),
                            "backward_transfer": float(backward_transfer),
                            "avg_transfer": float(avg_transfer),
                        })
        
        # Sort by average transfer quality
        bidirectional_transfer.sort(key=lambda x: x["avg_transfer"], reverse=True)
        
        logger.info(f"    Found {len(bidirectional_transfer)} layer pairs with strong bi-directional transfer (>0.7)")
        if bidirectional_transfer:
            for pair_info in bidirectional_transfer[:5]:
                logger.info(f"      Layers {pair_info['layer_pair']}: avg={pair_info['avg_transfer']:.3f}")
        
        # Store results
        results[model_name] = {
            "layers": layers,
            "transfer_matrix": transfer_matrix.tolist(),
            "layer_universality": layer_universality.tolist(),
            "feature_patterns": feature_patterns,
            "bidirectional_transfer_pairs": bidirectional_transfer,
            "summary": {
                "num_layers": len(layers),
                "mean_transfer_quality": float(transfer_matrix.mean()),
                "max_cross_layer_transfer": float(transfer_matrix[np.triu_indices_from(transfer_matrix, k=1)].max()),
                "min_cross_layer_transfer": float(transfer_matrix[np.triu_indices_from(transfer_matrix, k=1)].min()),
                "most_universal_layer": int(layers[np.argmax(layer_universality)]),
                "most_universal_score": float(layer_universality.max()),
            }
        }
        
        logger.info(f"\n  Summary:")
        logger.info(f"    Mean transfer quality: {results[model_name]['summary']['mean_transfer_quality']:.3f}")
        logger.info(f"    Most universal layer: {results[model_name]['summary']['most_universal_layer']} "
                   f"(score: {results[model_name]['summary']['most_universal_score']:.3f})")

    # Save results
    output_file = output_dir / "reasoning_flow_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Reasoning flow analysis complete!")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

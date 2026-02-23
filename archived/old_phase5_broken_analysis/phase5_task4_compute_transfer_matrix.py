#!/usr/bin/env python3
"""
Phase 5 Task 4: Compute Layer-to-Layer Transfer Matrix

Computes pairwise transfer metrics between all SAE pairs across layers and models.
Generates transfer quality matrix showing which layers are universal vs specialized.

Usage:
    python3 phase5_task4_compute_transfer_matrix.py \
        --sae-dir phase5_results/multilayer_transfer/saes \
        --activations-dir phase4_results/activations_multilayer \
        --output-dir phase5_results/multilayer_transfer \
        --top-k 50 \
        --device cuda:5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

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

    # Create config object
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
    batch_size: int = 64,
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


def compute_feature_variance(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute variance of each feature."""
    sae.eval()
    all_features = []

    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch = activations[i : i + batch_size].to(device)
            _, h = sae(batch)
            all_features.append(h.detach().cpu())

    features = torch.cat(all_features, dim=0)
    variance = (features > 0).float().sum(dim=0).numpy()  # Count of activations

    return variance


def compute_decoder_similarity(
    sae1: SparseAutoencoder,
    sae2: SparseAutoencoder,
    top_k: int = 50,
) -> float:
    """Compute cosine similarity between top decoder vectors."""
    dec1 = sae1.decoder.weight.data[:top_k]  # Shape: (top_k, input_dim)
    dec2 = sae2.decoder.weight.data[:top_k]

    # Normalize
    dec1_norm = dec1 / (torch.norm(dec1, dim=1, keepdim=True) + 1e-8)
    dec2_norm = dec2 / (torch.norm(dec2, dim=1, keepdim=True) + 1e-8)

    # Compute pairwise similarities and take max for each pair
    similarities = []
    for i in range(min(top_k, len(dec1_norm))):
        sims = torch.abs(torch.matmul(dec1_norm[i : i + 1], dec2_norm.t())).max()
        similarities.append(sims.item())

    return float(np.mean(similarities)) if similarities else 0.0


def compute_transfer_quality(
    source_sae: SparseAutoencoder,
    source_activations: torch.Tensor,
    target_activations: torch.Tensor,
    device: str,
    top_k: int = 50,
) -> Dict[str, float]:
    """
    Compute transfer quality when applying source SAE to target activations.

    Returns:
        Dict with metrics:
        - transfer_recon_ratio: target_loss / source_loss
        - feature_variance_ratio: target_variance / source_variance
        - top_k_spearman: Spearman correlation of top-k
        - decoder_similarity: Cosine similarity of decoder vectors
    """

    # Compute reconstruction quality
    source_loss = compute_reconstruction_loss(source_sae, source_activations, device)
    target_loss = compute_reconstruction_loss(source_sae, target_activations, device)

    transfer_ratio = target_loss / (source_loss + 1e-8)

    # Feature variance preservation
    source_variance = compute_feature_variance(source_sae, source_activations, device)
    target_variance = compute_feature_variance(source_sae, target_activations, device)

    top_idx = np.argsort(source_variance)[-top_k:]
    variance_ratio = (
        target_variance[top_idx].mean() / (source_variance[top_idx].mean() + 1e-8)
    )

    # Spearman correlation of top-k
    if len(top_idx) > 1:
        try:
            spearman_corr, _ = spearmanr(
                source_variance[top_idx], target_variance[top_idx]
            )
            spearman_corr = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
        except Exception as e:
            logger.warning(f"Spearman correlation failed: {e}")
            spearman_corr = 0.0
    else:
        spearman_corr = 0.0

    return {
        "transfer_recon_ratio": float(transfer_ratio),
        "feature_variance_ratio": float(variance_ratio),
        "top_k_spearman": float(spearman_corr),
        "decoder_similarity": float(compute_decoder_similarity(source_sae, source_sae, top_k)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute transfer matrix for Phase 5.4"
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
        default=Path("phase5_results/multilayer_transfer"),
        help="Output directory for transfer matrix",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top features to use for metrics",
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
        default=64,
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

    if not sae_files or not activation_files:
        logger.error("Missing SAE or activation files")
        return

    # Load activation data
    logger.info("Loading activation files...")
    activations = {}
    for act_file in activation_files:
        # Parse name: model_layer{idx}_activations.pt
        name_parts = act_file.stem.split("_")
        
        # Find the part that starts with "layer"
        layer_part = None
        for i, part in enumerate(name_parts):
            if part.startswith("layer"):
                layer_part = part
                layer_idx = int(part[5:])  # Extract number after "layer"
                model_name = "_".join(name_parts[:i])
                break
        
        if layer_part is None:
            logger.warning(f"Cannot parse {act_file.name}")
            continue
        
        key = f"{model_name}_layer{layer_idx}"

        acts = _load_activations(act_file)
        acts = _normalize_activations(acts)
        activations[key] = acts
        logger.info(f"  Loaded {key}: shape {acts.shape}")

    # Load SAE checkpoints
    logger.info("Loading SAE checkpoints...")
    saes = {}
    for sae_file in sae_files:
        # Parse name: model_layer{idx}_sae.pt
        name_parts = sae_file.stem.split("_")
        
        # Find the part that starts with "layer"
        layer_part = None
        for i, part in enumerate(name_parts):
            if part.startswith("layer"):
                layer_part = part
                layer_idx = int(part[5:])  # Extract number after "layer"
                model_name = "_".join(name_parts[:i])
                break
        
        if layer_part is None:
            logger.warning(f"Cannot parse {sae_file.name}")
            continue
        
        key = f"{model_name}_layer{layer_idx}"

        sae = _load_sae(sae_file, args.device)
        saes[key] = sae
        logger.info(f"  Loaded {key}")

    # Compute transfer matrix
    logger.info("\nComputing transfer metrics...")
    transfer_matrix = {}
    layer_universality_scores = {}

    sae_keys = sorted(saes.keys())
    total_pairs = len(sae_keys) * (len(sae_keys) - 1)  # Exclude diagonal
    pair_idx = 0

    # Extract unique layer indices
    layer_indices = sorted(set(
        int(k.split("layer")[1]) for k in sae_keys
    ))

    for layer_idx in layer_indices:
        layer_transfers = []

        for src_key in sae_keys:
            if f"layer{layer_idx}" not in src_key:
                continue

            src_sae = saes[src_key]
            src_acts = activations[src_key]

            for tgt_key in sae_keys:
                if f"layer{layer_idx}" not in tgt_key:
                    continue
                if src_key == tgt_key:
                    continue

                pair_idx += 1
                pct = 100.0 * pair_idx / total_pairs
                logger.info(f"  [{pct:5.1f}%] {src_key} → {tgt_key}")

                tgt_sae = saes[tgt_key]
                tgt_acts = activations[tgt_key]

                try:
                    metrics = compute_transfer_quality(
                        src_sae,
                        src_acts,
                        tgt_acts,
                        args.device,
                        top_k=args.top_k,
                    )

                    # Classify transfer quality
                    transfer_ratio = metrics["transfer_recon_ratio"]
                    if transfer_ratio > 0.85:
                        evaluation = "HIGH_UNIVERSALITY"
                    elif transfer_ratio > 0.70:
                        evaluation = "MODERATE_UNIVERSALITY"
                    elif transfer_ratio > 0.50:
                        evaluation = "PARTIAL_UNIVERSALITY"
                    else:
                        evaluation = "SPECIALIZED"

                    transfer_matrix[f"{src_key}__to__{tgt_key}"] = {
                        **metrics,
                        "evaluation": evaluation,
                    }

                    layer_transfers.append(transfer_ratio)

                except Exception as e:
                    logger.error(f"  Failed: {e}")

        if layer_transfers:
            layer_universality_scores[f"layer_{layer_idx}"] = float(
                np.mean(layer_transfers)
            )

    # Build output structure
    output = {
        "config": {
            "models": ["gemma-2b", "gpt2-medium", "phi-2", "pythia-1.4b"],
            "layers": layer_indices,
            "top_k": args.top_k,
            "num_transfer_pairs": len(transfer_matrix),
        },
        "transfer_matrix": transfer_matrix,
        "layer_universality_scores": layer_universality_scores,
    }

    # Save output
    output_file = output_dir / "transfer_matrix.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Transfer matrix computation complete!")
    logger.info(f"Computed {len(transfer_matrix)} transfer pairs")
    logger.info(f"Layer universality scores: {layer_universality_scores}")
    logger.info(f"Output saved to {output_file}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 5 Task 4: Multi-Layer SAE Training

Trains SAEs on activations from each layer of each model.
Applies per-layer sparsity tuning (more sparsity for early layers, less for late layers).

Usage:
    python3 phase5_task4_train_multilayer_saes.py \
        --activations-dir phase4_results/activations_multilayer \
        --output-dir phase5_results/multilayer_transfer/saes \
        --epochs 10 \
        --batch-size 64 \
        --device cuda:5 \
        --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_architecture import SparseAutoencoder
from sae_config import SAEConfig

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _load_activations(path: Path) -> torch.Tensor:
    """Load activation tensor from saved file."""
    payload = torch.load(path, weights_only=False)
    if isinstance(payload, dict) and "activations" in payload:
        acts = payload["activations"]
    else:
        acts = payload

    # Flatten if needed
    if acts.dim() == 3:
        acts = acts.reshape(-1, acts.shape[-1])

    return acts.float()


def _normalize_activations(acts: torch.Tensor) -> torch.Tensor:
    """Normalize activations to zero mean, unit variance."""
    mean = acts.mean(dim=0, keepdim=True)
    std = acts.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (acts - mean) / std


def _iter_batches(acts: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    """Iterate over batches of activations."""
    dataset = TensorDataset(acts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in loader:
        yield batch[0]


def _get_sae_config_for_layer(
    model_name: str,
    layer_idx: int,
    input_dim: int,
) -> SAEConfig:
    """
    Get SAE configuration tuned for specific layer.
    Early layers: Higher sparsity (more interpretable, less task-specific)
    Late layers: Lower sparsity (more distributed, more task-specific)
    """
    base_params = {
        "gemma-2b": {"input_dim": 2048, "expansion_factor": 8, "learning_rate": 1e-4},
        "gpt2-medium": {"input_dim": 1024, "expansion_factor": 8, "learning_rate": 1e-4},
        "phi-2": {"input_dim": 2560, "expansion_factor": 8, "learning_rate": 1e-4},
        "pythia-1.4b": {"input_dim": 2048, "expansion_factor": 8, "learning_rate": 1e-4},
    }

    if model_name not in base_params:
        logger.warning(f"Unknown model {model_name}, using default config")
        base_params[model_name] = {"input_dim": input_dim, "expansion_factor": 8, "learning_rate": 1e-4}

    params = base_params[model_name]

    # Adjust L1 penalty by layer depth
    # Early layers: more sparsity (1.5e-4)
    # Mid layers: medium sparsity (1e-4)
    # Late layers: less sparsity (0.5e-4)
    if layer_idx in [4, 6]:
        l1_coeff = 1.5e-4
    elif layer_idx in [16, 20, 24, 30]:
        l1_coeff = 0.5e-4
    else:
        l1_coeff = 1e-4

    config = SAEConfig(
        input_dim=params["input_dim"],
        expansion_factor=params["expansion_factor"],
        learning_rate=params["learning_rate"],
        batch_size=64,
        max_epochs=10,
        l1_penalty_coeff=l1_coeff,
        decorrelation_coeff=0.01,
        use_relu=False,
    )

    return config


def train_sae_on_layer(
    activations: torch.Tensor,
    model_name: str,
    layer_idx: int,
    device: str,
    epochs: int,
    batch_size: int,
    output_path: Optional[Path] = None,
) -> Tuple[SparseAutoencoder, Dict[str, float]]:
    """
    Train SAE on activations from a specific layer.

    Args:
        activations: Activation tensor (N, D)
        model_name: Name of model
        layer_idx: Index of layer
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size for training
        output_path: Optional path to save checkpoint

    Returns:
        Tuple of (trained SAE, summary dict)
    """
    input_dim = activations.shape[-1]
    config = _get_sae_config_for_layer(model_name, layer_idx, input_dim)

    logger.info(f"Creating SAE for {model_name} layer {layer_idx} (input_dim={input_dim})")
    logger.info(f"  Config: expansion={config.expansion_factor}x, L1={config.l1_penalty_coeff:.2e}")

    sae = SparseAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)

    sae.train()
    losses: List[float] = []
    sparsities: List[float] = []

    for epoch in range(epochs):
        epoch_losses = []
        epoch_sparsities = []

        for batch in _iter_batches(activations, batch_size=batch_size):
            batch = batch.to(device)
            optimizer.zero_grad()

            x_hat, h = sae(batch)
            loss = sae.compute_total_loss(batch, x_hat, h)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

            # Compute sparsity
            with torch.no_grad():
                sparsity = float((h > 0).float().mean())
                epoch_sparsities.append(sparsity)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        avg_sparsity = float(np.mean(epoch_sparsities)) if epoch_sparsities else 0.0
        losses.append(avg_loss)
        sparsities.append(avg_sparsity)

        pct = 100.0 * (epoch + 1) / epochs
        logger.info(
            f"  epoch {epoch + 1:2d}/{epochs}: loss={avg_loss:.6f}, sparsity={avg_sparsity:.1%} [{pct:.0f}%]"
        )

    sae.eval()

    summary = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "input_dim": input_dim,
        "expansion_factor": config.expansion_factor,
        "final_loss": float(losses[-1]),
        "min_loss": float(min(losses)),
        "final_sparsity": float(sparsities[-1]),
        "avg_sparsity": float(np.mean(sparsities)),
        "config": asdict(config),
    }

    # Save checkpoint if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": sae.state_dict(),
            "config": asdict(config),
            "summary": summary,
        }
        torch.save(checkpoint, output_path)
        logger.info(f"Saved checkpoint to {output_path}")

        # Save summary (convert Path to string for JSON serialization)
        summary_path = output_path.with_suffix(".summary.json")
        summary_json = summary.copy()
        # Convert any Path objects in config to strings
        if "config" in summary_json and isinstance(summary_json["config"], dict):
            for key, val in summary_json["config"].items():
                if hasattr(val, "__fspath__"):  # Check if it's a Path-like object
                    summary_json["config"][key] = str(val)
        with open(summary_path, "w") as f:
            json.dump(summary_json, f, indent=2)

    return sae, summary


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-layer SAEs for Phase 5.4"
    )
    parser.add_argument(
        "--activations-dir",
        type=Path,
        default=Path("phase4_results/activations_multilayer"),
        help="Directory containing captured activations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase5_results/multilayer_transfer/saes"),
        help="Output directory for SAE checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:5",
        help="Device to train on",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all activation files
    activation_files = sorted(activations_dir.glob("*_activations.pt"))
    if not activation_files:
        logger.error(f"No activation files found in {activations_dir}")
        return

    logger.info(f"Found {len(activation_files)} activation files")
    logger.info(f"Training SAEs on device {args.device}")
    logger.info(f"Output directory: {output_dir}")

    results = []

    for i, activation_file in enumerate(activation_files):
        logger.info(f"\n{'='*70}")
        logger.info(f"[{i+1}/{len(activation_files)}] Processing {activation_file.name}")
        logger.info(f"{'='*70}")

        # Parse filename to get model and layer
        # Filename format: model-name_layerN_activations
        # e.g., gemma-2b_layer0_activations, gpt2-medium_layer12_activations
        name_parts = activation_file.stem.split("_")
        
        # Find the part that contains layer number
        layer_idx = None
        model_name = None
        
        for part in name_parts:
            if part.startswith("layer"):
                try:
                    layer_idx = int(part[5:])  # Extract number after "layer"
                except (ValueError, IndexError):
                    pass
        
        if layer_idx is None:
            logger.warning(f"Cannot parse layer index from {activation_file.name}")
            continue
        
        # Model name is everything before "layer" in the stem
        stem_parts = activation_file.stem.split("_layer")[0]
        model_name = stem_parts

        try:
            # Load activations
            logger.info(f"Loading activations from {activation_file}...")
            activations = _load_activations(activation_file)
            logger.info(f"Loaded {activations.shape[0]:,} activations of dim {activations.shape[1]}")

            # Normalize
            activations = _normalize_activations(activations)

            # Train SAE
            output_path = output_dir / f"{model_name}_layer{layer_idx}_sae.pt"

            sae, summary = train_sae_on_layer(
                activations=activations,
                model_name=model_name,
                layer_idx=layer_idx,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output_path=output_path,
            )

            results.append(summary)
            logger.info(f"✓ Training complete for {model_name} layer {layer_idx}")

        except Exception as e:
            logger.error(f"Failed to train SAE: {e}", exc_info=args.verbose)

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Training complete!")
    logger.info(f"Trained {len(results)} SAEs")
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

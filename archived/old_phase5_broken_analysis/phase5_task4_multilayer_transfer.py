#!/usr/bin/env python3
"""
Phase 5 Task 4: Multi-Layer Feature Transfer Analysis

Extends Phase 5.3 to test feature transfer across MULTIPLE LAYERS per model.
This reveals how reasoning evolves through the network depth.

Key Research Questions:
1. Do features transfer better at certain network depths?
2. Which layers have UNIVERSAL reasoning (transfer well across models)?
3. Where does TASK-SPECIFIC reasoning emerge?
4. Can we identify the "reasoning pipeline" through layer progression?

Outputs:
- multi_layer_transfer_results.json: Detailed metrics for all layer combinations
- layer_analysis_report.md: Visualization of transfer quality by depth
- visualization_layer_heatmap.png: Heatmap of transfer performance across layers
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_architecture import SparseAutoencoder
from sae_config import SAEConfig

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


LAYER_CONFIGS = {
    "gemma-2b": {"total_layers": 24, "test_layers": [4, 8, 12, 16, 20]},
    "gpt2-medium": {"total_layers": 12, "test_layers": [3, 6, 9, 11]},
    "phi-2": {"total_layers": 32, "test_layers": [4, 8, 16, 24, 30]},
    "pythia-1.4b": {"total_layers": 24, "test_layers": [4, 8, 12, 16, 20]},
}


def _load_activations(path: Path) -> torch.Tensor:
    payload = torch.load(path, weights_only=False)
    if isinstance(payload, dict) and "activations" in payload:
        acts = payload["activations"]
    else:
        acts = payload

    if acts.dim() == 3:
        acts = acts.reshape(-1, acts.shape[-1])

    return acts.float()


def _normalize_activations(acts: torch.Tensor) -> torch.Tensor:
    mean = acts.mean(dim=0, keepdim=True)
    std = acts.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (acts - mean) / std


def _iter_batches(acts: torch.Tensor, batch_size: int):
    dataset = TensorDataset(acts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in loader:
        yield batch[0]


def train_sae(
    acts: torch.Tensor,
    input_dim: int,
    expansion_factor: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Tuple[SparseAutoencoder, Dict[str, float]]:
    config = SAEConfig(
        input_dim=input_dim,
        expansion_factor=expansion_factor,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=epochs,
        l1_penalty_coeff=1e-4,
        decorrelation_coeff=0.01,
        use_relu=False,
    )

    sae = SparseAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    sae.train()
    losses: List[float] = []

    for epoch in range(epochs):
        epoch_losses = []
        for batch in _iter_batches(acts, batch_size=batch_size):
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, h = sae(batch)
            loss = sae.compute_total_loss(batch, x_hat, h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        losses.append(avg_loss)

    summary = {"final_loss": losses[-1] if losses else 0.0, "epochs": epochs}
    sae.eval()
    return sae, summary


def compute_recon_loss(
    sae: SparseAutoencoder, acts: torch.Tensor, device: str, batch_size: int
) -> float:
    sae.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in _iter_batches(acts, batch_size=batch_size):
            batch = batch.to(device)
            x_hat, _ = sae(batch)
            loss = torch.nn.functional.mse_loss(batch, x_hat)
            total_loss += float(loss.detach().cpu()) * batch.shape[0]
            count += batch.shape[0]
    return total_loss / count if count > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5.4: Multi-layer feature transfer analysis")
    parser.add_argument("--activations-dir", type=Path, default=Path("phase4_results/activations"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase5_results/multi_layer_transfer"))
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)  # Faster training per layer
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== MULTI-LAYER TRANSFER ANALYSIS ===")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Expansion factor: {args.expansion_factor}x")
    logger.info(f"Epochs per layer: {args.epochs}")

    # Build index mapping model identifiers to activation files
    act_files = {}
    for f in args.activations_dir.glob("*.pt"):
        fname = f.stem
        # Extract model and layer info from filename pattern
        if "gemma" in fname:
            model_base = "gemma-2b"
        elif "gpt2" in fname:
            model_base = "gpt2-medium"
        elif "phi" in fname:
            model_base = "phi-2"
        elif "pythia" in fname:
            model_base = "pythia-1.4b"
        else:
            continue

        # Parse layer number
        parts = fname.split("_")
        layer_num = None
        for part in parts:
            if part.startswith("layer"):
                layer_num = int(part.replace("layer", ""))
                break

        if layer_num is not None:
            key = f"{model_base}_layer{layer_num}"
            act_files[key] = f

    logger.info(f"\nFound {len(act_files)} activation files for multi-layer analysis")

    # NOTE: In a full implementation, you would:
    # 1. Download additional layer activations (currently only have single layers)
    # 2. Train SAEs for multiple layers per model
    # 3. Compute transfer metrics across layer pairs
    # 4. Generate layer-wise heatmaps and reports

    results = {
        "config": {
            "expansion_factor": args.expansion_factor,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "layer_configs": LAYER_CONFIGS,
        },
        "note": "Full multi-layer analysis would require activations from multiple layers per model. "
                "Currently only have single layers per model-task pair from Phase 4.",
        "available_files": act_files,
        "recommended_extension": {
            "step_1": "Capture activations for multiple layers (e.g., layers 4, 8, 12, 16, 20) for each model",
            "step_2": "Train SAEs on each layer independently",
            "step_3": "Compute transfer metrics: within-model (layer-to-layer) and cross-model (same layer)",
            "step_4": "Visualize as 2D heatmap: X=source_layer, Y=target_layer, color=transfer_quality",
            "step_5": "Analyze which layers are 'universal' and which are 'task-specific'",
        },
        "research_questions": {
            "q1": "Do earlier layers have better cross-model transfer (universal reasoning)?",
            "q2": "At what depth does task-specific reasoning diverge?",
            "q3": "Are there bottleneck layers crucial for all reasoning?",
            "q4": "Can we predict reasoning quality from feature universality?",
        },
    }

    results_path = output_dir / "multi_layer_analysis_plan.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAnalysis plan saved to {results_path}")
    logger.info("\n=== NEXT STEPS ===")
    logger.info("To enable full multi-layer transfer analysis:")
    logger.info("1. Run Phase 3 / Phase 4 with --capture-all-layers flag")
    logger.info("2. This would generate activations for all intermediate layers")
    logger.info("3. Then re-run Phase 5.4 to get complete layer-by-layer transfer matrix")
    logger.info("\nExpected output: layer_transfer_heatmap showing reasoning evolution!")


if __name__ == "__main__":
    main()

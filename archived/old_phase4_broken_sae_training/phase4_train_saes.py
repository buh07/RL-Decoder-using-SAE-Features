#!/usr/bin/env python3
"""Phase 4: Train SAEs on captured activations."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sae_config import SAEConfig
from sae_architecture import SparseAutoencoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


def train_sae(
    activations: torch.Tensor,
    model_name: str,
    benchmark_name: str,
    gpu_id: int,
    num_epochs: int = 20,
) -> Dict:
    """Train SAE on activations."""
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training SAE for {model_name}_{benchmark_name} on {device}")
    
    input_dim = activations.shape[1]
    config = SAEConfig(
        input_dim=input_dim,
        expansion_factor=8,
        l1_penalty_coeff=1e-4,
        decorrelation_coeff=0.01,
        use_relu=False,
    )
    
    sae = SparseAutoencoder(config).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=1e-4)
    
    # Normalize activations
    activations = (activations - activations.mean(dim=0)) / (activations.std(dim=0) + 1e-6)
    
    dataset = TensorDataset(activations.to(device))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0]
            
            latents = sae.encode(x)
            recon = sae.decode(latents)
            
            recon_loss = nn.functional.mse_loss(recon, x)
            sparsity_loss = config.l1_penalty_coeff * torch.mean(torch.abs(latents))
            
            loss = recon_loss + sparsity_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}: loss={total_loss/len(dataloader):.6f}")
    
    return {
        "model": model_name,
        "benchmark": benchmark_name,
        "input_dim": input_dim,
        "latent_dim": config.latent_dim,
        "expansion": 8,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Train SAEs")
    parser.add_argument("--activation-dir", type=Path, default=Path("phase4_results/activations"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase4_results/saes"))
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--num-epochs", type=int, default=20)
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 4: SAE TRAINING")
    logger.info("=" * 60)
    
    results = {}
    
    # Find all activation files
    activation_files = sorted(args.activation_dir.glob("*_activations.pt"))
    
    if not activation_files:
        logger.warning(f"No activation files found in {args.activation_dir}")
        return
    
    for i, act_file in enumerate(activation_files):
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
        
        logger.info(f"\nLoading {act_file.name}...")
        checkpoint = torch.load(act_file)
        activations = checkpoint["activations"]
        model_name = checkpoint["model"]
        benchmark_name = checkpoint["benchmark"]
        
        logger.info(f"Shape: {activations.shape}")
        
        result = train_sae(
            activations,
            model_name,
            benchmark_name,
            gpu_id,
            num_epochs=args.num_epochs,
        )
        
        results[f"{model_name}_{benchmark_name}"] = result
    
    # Save summary
    summary_file = args.output_dir / "training_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTraining complete. Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

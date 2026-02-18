#!/usr/bin/env python3
"""
Phase 1 SAE Training on Ground-Truth Environments
Trains SAEs and validates reconstruction fidelity + causal perturbations.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_environments import BFSEnvironment, StackMachineEnvironment, LogicPuzzleEnvironment
from sae_config import SAEConfig
from sae_architecture import SparseAutoencoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


class Phase1SAETrainer:
    """Train and validate SAE on ground-truth environment data."""
    
    def __init__(self, gpu_id: int, output_dir: Path):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id
        self.results = {}
        
        logger.info(f"Initialized Phase1SAETrainer on {self.device}")
    
    def train_sae_on_environment(
        self,
        env_data: torch.Tensor,
        env_name: str,
        expansion_factor: int = 8,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
    ) -> Tuple[SparseAutoencoder, Dict]:
        """Train SAE on environment data."""
        
        logger.info(f"Training SAE on {env_name} (expansion {expansion_factor}x)")
        
        input_dim = env_data.shape[1]
        
        # Create SAE config
        config = SAEConfig(
            input_dim=input_dim,
            expansion_factor=expansion_factor,
            l1_penalty_coeff=1e-4,  # Reduced from 1e-3 to allow more features to activate
            reconstruction_coeff=1.0,
            decorrelation_coeff=0.01,  # Reduced to focus on reconstruction first
            use_relu=False,  # Disable ReLU to allow negative activations
        )
        
        # Initialize SAE
        sae = SparseAutoencoder(config).to(self.device)
        optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
        
        # Create dataloader
        dataset = TensorDataset(env_data.to(self.device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        epoch_losses = []
        best_recon_loss = float('inf')
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_recon = 0.0
            num_batches = 0
            
            for batch in dataloader:
                x = batch[0]
                
                # Forward pass
                latents = sae.encode(x)
                recon = sae.decode(latents)
                
                # Compute loss components
                recon_loss = F.mse_loss(recon, x)
                sparsity_loss = sae.config.l1_penalty_coeff * torch.mean(torch.abs(latents))
                
                # Decorrelation loss
                W = sae.decoder.weight  # [input_dim, latent_dim]
                gram = torch.matmul(W.T, W)  # [latent_dim, latent_dim]
                decorr_loss = sae.config.decorrelation_coeff * (
                    torch.sum(gram ** 2) - torch.trace(gram) ** 2
                ) / (sae.config.latent_dim ** 2)
                
                loss = recon_loss + sparsity_loss + decorr_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_recon = total_recon / num_batches
            epoch_losses.append(avg_loss)
            
            if avg_recon < best_recon_loss:
                best_recon_loss = avg_recon
            
            if (epoch + 1) % max(1, num_epochs // 5) == 0:
                logger.info(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f}, recon={avg_recon:.6f}")
        
        logger.info(f"Training complete. Final reconstruction loss: {best_recon_loss:.6f}")
        
        return sae, {
            "expansion": expansion_factor,
            "epochs": num_epochs,
            "final_loss": epoch_losses[-1],
            "best_recon_loss": best_recon_loss,
        }
    
    def validate_reconstruction(
        self,
        sae: SparseAutoencoder,
        test_data: torch.Tensor,
        env_name: str,
    ) -> Dict:
        """Validate reconstruction fidelity."""
        
        logger.info(f"Validating reconstruction on {env_name}")
        
        sae.eval()
        test_data = test_data.to(self.device)
        
        with torch.no_grad():
            latents = sae.encode(test_data)
            recon = sae.decode(latents)
            
            mse = F.mse_loss(recon, test_data).item()
            mae = F.l1_loss(recon, test_data).item()
            
            # Reconstruction fidelity (R² score)
            ss_res = torch.sum((test_data - recon) ** 2)
            ss_tot = torch.sum((test_data - test_data.mean()) ** 2)
            r2_score = (1 - ss_res / ss_tot).item()
            
            # Sparsity metrics
            sparsity = (latents == 0).float().mean().item()
            mean_active = (latents != 0).float().sum(dim=1).mean().item()
            
        logger.info(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2_score:.4f}")
        logger.info(f"  Sparsity: {sparsity:.2%}, Mean active features: {mean_active:.1f}")
        
        results = {
            "mse": mse,
            "mae": mae,
            "r2_score": r2_score,
            "sparsity": sparsity,
            "mean_active_features": mean_active,
        }
        
        if r2_score < 0.95:
            logger.warning(f"⚠️  R² score {r2_score:.4f} < 0.95 threshold!")
        else:
            logger.info(f"✅ R² score {r2_score:.4f} > 0.95 threshold")
        
        return results
    
    def causal_perturbation_tests(
        self,
        sae: SparseAutoencoder,
        test_data: torch.Tensor,
        env_name: str,
    ) -> Dict:
        """Test causal perturbations: perturb latents and check reconstruction changes."""
        
        logger.info(f"Running causal perturbation tests on {env_name}")
        
        sae.eval()
        test_data = test_data.to(self.device)
        num_samples = min(500, len(test_data))
        test_batch = test_data[:num_samples]
        
        with torch.no_grad():
            # Baseline reconstruction
            baseline_latents = sae.encode(test_batch)
            baseline_recon = sae.decode(baseline_latents)
            baseline_error = F.mse_loss(baseline_recon, test_batch).item()
            
            perturbation_results = {
                "add_epsilon": [],
                "scale_2x": [],
                "random_replacement": [],
            }
            
            # Test 1: Add small epsilon to each latent dimension
            for dim in range(0, baseline_latents.shape[1], max(1, baseline_latents.shape[1] // 20)):
                perturbed = baseline_latents.clone()
                perturbed[:, dim] += 0.1  # small epsilon
                
                perturbed_recon = sae.decode(perturbed)
                perturbed_error = F.mse_loss(perturbed_recon, test_batch).item()
                error_change = perturbed_error - baseline_error
                
                perturbation_results["add_epsilon"].append(error_change)
            
            # Test 2: Scale latent dimension by 2x
            for dim in range(0, baseline_latents.shape[1], max(1, baseline_latents.shape[1] // 20)):
                perturbed = baseline_latents.clone()
                perturbed[:, dim] *= 2.0
                
                perturbed_recon = sae.decode(perturbed)
                perturbed_error = F.mse_loss(perturbed_recon, test_batch).item()
                error_change = perturbed_error - baseline_error
                
                perturbation_results["scale_2x"].append(error_change)
            
            # Test 3: Random replacement of latent
            for dim in range(0, baseline_latents.shape[1], max(1, baseline_latents.shape[1] // 20)):
                perturbed = baseline_latents.clone()
                perturbed[:, dim] = torch.randn_like(perturbed[:, dim])
                
                perturbed_recon = sae.decode(perturbed)
                perturbed_error = F.mse_loss(perturbed_recon, test_batch).item()
                error_change = perturbed_error - baseline_error
                
                perturbation_results["random_replacement"].append(error_change)
        
        # Compute statistics
        causal_stats = {}
        for perturbation_type, changes in perturbation_results.items():
            changes = torch.tensor(changes)
            causal_stats[perturbation_type] = {
                "mean_error_increase": changes.mean().item(),
                "std_error_increase": changes.std().item(),
                "max_error_increase": changes.max().item(),
                "percent_positive": (changes > 0).float().mean().item(),
            }
            
            logger.info(f"  {perturbation_type}:")
            logger.info(f"    Mean error increase: {causal_stats[perturbation_type]['mean_error_increase']:.6f}")
            logger.info(f"    % dims increase error: {causal_stats[perturbation_type]['percent_positive']:.1%}")
        
        return causal_stats
    
    def run_full_validation(self, env_name: str, env):
        """Run full validation pipeline on one environment."""
        
        logger.info("=" * 80)
        logger.info(f"Phase 1 Validation: {env_name}")
        logger.info("=" * 80)
        
        # Generate dataset
        data, metadata = env.generate_dataset()
        data_tensor = torch.from_numpy(data).float()
        
        # Normalize data to zero mean and unit variance
        data_mean = data_tensor.mean(dim=0, keepdim=True)
        data_std = data_tensor.std(dim=0, keepdim=True) + 1e-6
        data_tensor = (data_tensor - data_mean) / data_std
        
        # Split train/test
        split_idx = int(0.8 * len(data_tensor))
        train_data = data_tensor[:split_idx]
        test_data = data_tensor[split_idx:]
        
        logger.info(f"Dataset: {len(train_data)} train, {len(test_data)} test samples")
        logger.info(f"Data normalized: mean={train_data.mean():.6f}, std={train_data.std():.6f}")
        
        env_results = {
            "environment": env_name,
            "state_dim": env.state_dim,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "expansions": {},
        }
        
        # Train SAEs with different expansions
        for expansion in [4, 8, 12]:
            logger.info(f"\n--- Training {expansion}x SAE ---")
            
            sae, train_info = self.train_sae_on_environment(
                train_data,
                env_name,
                expansion_factor=expansion,
                num_epochs=15,
                batch_size=32,
                learning_rate=1e-4,
            )
            
            # Validate reconstruction
            recon_results = self.validate_reconstruction(sae, test_data, env_name)
            
            # Run causal tests
            causal_results = self.causal_perturbation_tests(sae, test_data, env_name)
            
            env_results["expansions"][f"{expansion}x"] = {
                "training": train_info,
                "reconstruction": recon_results,
                "causal": causal_results,
                "pass": recon_results["r2_score"] > 0.95,
            }
            
            # Save checkpoint
            checkpoint_path = self.output_dir / f"{env_name}_{expansion}x_sae.pt"
            torch.save({
                "sae_state": sae.state_dict(),
                "config": sae.config.__dict__,
                "results": env_results["expansions"][f"{expansion}x"],
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return env_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 SAE Training and Validation")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--output-dir", type=Path, default=Path("phase1_results"), help="Output directory")
    parser.add_argument("--env", type=str, choices=["bfs", "stack", "logic", "all"], default="all",
                       help="Which environment to run")
    
    args = parser.parse_args()
    
    trainer = Phase1SAETrainer(args.gpu_id, args.output_dir)
    
    logger.info("=" * 80)
    logger.info("PHASE 1: GROUND-TRUTH SAE VALIDATION")
    logger.info("=" * 80)
    logger.info(f"GPU: {args.gpu_id}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")
    
    all_results = {
        "gpu_id": args.gpu_id,
        "environments": {}
    }
    
    # Run selected environments
    environments = {
        "bfs": BFSEnvironment(num_sequences=50, max_steps=50),
        "stack": StackMachineEnvironment(num_sequences=50, max_steps=50),
        "logic": LogicPuzzleEnvironment(num_sequences=50, max_steps=50),
    }
    
    envs_to_run = ["bfs", "stack", "logic"] if args.env == "all" else [args.env]
    
    for env_name in envs_to_run:
        env = environments[env_name]
        result = trainer.run_full_validation(env_name, env)
        all_results["environments"][env_name] = result
    
    # Save results
    results_file = args.output_dir / "phase1_results.json"
    with open(results_file, "w") as f:
        # Convert for JSON serialization
        json_results = json.loads(json.dumps(all_results, default=str))
        json.dump(json_results, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1 RESULTS SUMMARY")
    logger.info("=" * 80)
    
    for env_name, env_result in all_results["environments"].items():
        logger.info(f"\n{env_name.upper()}:")
        for expansion, expansion_result in env_result["expansions"].items():
            status = "✅ PASS" if expansion_result["pass"] else "❌ FAIL"
            r2 = expansion_result["reconstruction"]["r2_score"]
            logger.info(f"  {expansion}: {status} (R² = {r2:.4f})")
    
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SAE Training Loop with streaming activation shards.

Trains sparse autoencoders on captured activations from transformers.
Supports mixed precision, WANDB logging, checkpointing, and evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).parent))

from sae_config import SAEConfig, gpt2_sae_config, gpt2_medium_sae_config, pythia_1_4b_sae_config, default_config_for_model
from sae_architecture import SparseAutoencoder

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ActivationShardDataset(IterableDataset):
    """
    Iterable dataset that streams activation shards from disk.
    
    Each shard is a .pt file containing:
        {"input_ids": Tensor[batch, seq_len, hidden_dim],
         "seq_lens": Tensor[batch]}
    """
    
    def __init__(
        self,
        shard_dir: Path,
        batch_size: int = 32,
        max_shards: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.max_shards = max_shards
        self.shuffle = shuffle
        
        # Find all .pt files
        self.shards = sorted(self.shard_dir.glob("shard_*.pt"))
        
        if max_shards:
            self.shards = self.shards[:max_shards]
        
        if not self.shards:
            raise FileNotFoundError(f"No shard_*.pt files found in {shard_dir}")
        
        print(f"[Dataset] Found {len(self.shards)} shards in {shard_dir}")
    
    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        """
        Iterate over shards, yielding batches of activations.
        Each iteration yields a tensor of shape (batch_size, seq_len, hidden_dim) or smaller.
        """
        shards = self.shards.copy()
        if self.shuffle:
            import random
            random.shuffle(shards)
        
        for shard_path in shards:
            payload = torch.load(shard_path, map_location="cpu")
            acts = payload["activations"]  # (num_sequences, seq_len, hidden_dim)
            
            # Iterate over this shard in batches
            for i in range(0, acts.shape[0], self.batch_size):
                batch = acts[i : i + self.batch_size]
                
                # Flatten (batch, seq_len, hidden_dim) -> (batch*seq_len, hidden_dim)
                batch_flat = batch.reshape(-1, batch.shape[-1])
                
                yield batch_flat


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    loss: float
    recon_loss: float
    l1_loss: float
    decorr_loss: float
    probe_loss: float = 0.0
    temporal_loss: float = 0.0
    activation_fraction: float = 0.0
    lr: float = 0.0
    time_per_step: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "loss": self.loss,
            "recon_loss": self.recon_loss,
            "l1_loss": self.l1_loss,
            "decorr_loss": self.decorr_loss,
            "probe_loss": self.probe_loss,
            "temporal_loss": self.temporal_loss,
            "activation_fraction": self.activation_fraction,
            "lr": self.lr,
            "time_per_step_ms": self.time_per_step * 1000,
        }


class SAETrainer:
    """Trainer for sparse autoencoders."""
    
    def __init__(
        self,
        config: SAEConfig,
        sae: SparseAutoencoder,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.sae = sae.to(device)
        self.device = device
        self.dtype = dtype
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.sae.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler (warmup)
        def lr_lambda(step: int):
            if step < config.warmup_steps:
                return float(step) / float(max(1, config.warmup_steps))
            return 1.0
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision
        self.use_amp = config.use_amp
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        
        # W&B
        self.use_wandb = config.wandb_project and HAS_WANDB
        if self.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.to_dict(),
                name=f"sae_{config.input_dim}d_{config.expansion_factor}x",
            )
    
    def train_step(
        self,
        batch: torch.Tensor,
    ) -> TrainingMetrics:
        """
        Single training step.
        
        Args:
            batch: Tensor of shape (batch_size, hidden_dim)
        
        Returns:
            TrainingMetrics with loss and stats
        """
        start_time = time.perf_counter()
        
        batch = batch.to(self.device, dtype=self.dtype)
        
        self.optimizer.zero_grad()
        
        # Forward pass with AMP
        if self.use_amp:
            with autocast(dtype=self.dtype):
                x_hat, h = self.sae(batch)
                loss_dict = self.sae.compute_loss_components(batch, x_hat, h)
                loss = loss_dict["total_loss_tensor"]
            
            # Backward with scaling
            self.scaler.scale(loss).backward()
            
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.config.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            x_hat, h = self.sae(batch)
            loss_dict = self.sae.compute_loss_components(batch, x_hat, h)
            loss = loss_dict["total_loss_tensor"]
            
            loss.backward()
            
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step()
        
        # Normalize decoder weights periodically
        if self.step % self.config.decoder_norm_every == 0:
            self.sae.normalize_decoder()
        
        # Compute stats
        with torch.no_grad():
            activation_stats = self.sae.get_activation_stats(h)
        
        time_per_step = time.perf_counter() - start_time
        
        metrics = TrainingMetrics(
            step=self.step,
            loss=loss_dict["total_loss"],
            recon_loss=loss_dict["recon_loss"],
            l1_loss=loss_dict["l1_loss"],
            decorr_loss=loss_dict["decorr_loss"],
            probe_loss=loss_dict["probe_loss"],
            temporal_loss=loss_dict["temporal_loss"],
            activation_fraction=activation_stats["activation_fraction"],
            lr=self.optimizer.param_groups[0]["lr"],
            time_per_step=time_per_step,
        )
        
        self.step += 1
        
        return metrics
    
    def log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log metrics to console and W&B."""
        if self.step % self.config.log_every == 0:
            log_str = (
                f"[{self.epoch:02d}] step={self.step:06d} | "
                f"loss={metrics.loss:.4f} "
                f"(recon={metrics.recon_loss:.4f}, "
                f"l1={metrics.l1_loss:.4f}, "
                f"decorr={metrics.decorr_loss:.4f}) | "
                f"sparsity={metrics.activation_fraction:.2%} | "
                f"lr={metrics.lr:.2e} | "
                f"{metrics.time_per_step*1000:.1f}ms"
            )
            print(log_str)
            
            if self.use_wandb:
                wandb.log(metrics.to_dict(), step=self.step)
    
    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state": self.sae.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
        }
        
        torch.save(state, path)
        print(f"[checkpoint] saved to {path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        state = torch.load(path, map_location=self.device)
        
        self.step = state["step"]
        self.epoch = state["epoch"]
        self.sae.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        
        print(f"[checkpoint] loaded from {path} (step {self.step}, epoch {self.epoch})")
    
    def train(
        self,
        dataloader: DataLoader,
        max_steps: Optional[int] = None,
    ) -> dict:
        """
        Full training loop.
        
        Args:
            dataloader: DataLoader yielding batches
            max_steps: Stop training after this many steps (None = full epochs)
        
        Returns:
            Dictionary with training summary
        """
        print(f"[training] Starting SAE training")
        print(f"  Config: {self.config.input_dim}D → {self.config.latent_dim}D")
        print(f"  Device: {self.device}, AMP: {self.use_amp}")
        print(f"  Max epochs: {self.config.max_epochs}, Max steps: {max_steps}")
        print()
        
        start_time = time.perf_counter()
        total_steps = 0
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            epoch_start = time.perf_counter()
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                if max_steps and self.step >= max_steps:
                    print(f"[training] Reached max_steps={max_steps}")
                    break
                
                metrics = self.train_step(batch)
                self.log_metrics(metrics)
                epoch_losses.append(metrics.loss)
                total_steps += 1
                
                # Checkpointing
                if self.step % self.config.checkpoint_every == 0:
                    ckpt_path = (
                        self.config.checkpoint_dir
                        / f"sae_{self.config.input_dim}d_step{self.step:06d}.pt"
                    )
                    self.save_checkpoint(ckpt_path)
            
            epoch_time = time.perf_counter() - epoch_start
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            
            print(
                f"[epoch {epoch}] avg_loss={avg_loss:.4f}, "
                f"time={epoch_time:.1f}s ({len(epoch_losses)} steps)"
            )
        
        total_time = time.perf_counter() - start_time
        
        summary = {
            "total_steps": total_steps,
            "total_time_seconds": total_time,
            "avg_time_per_step_ms": (total_time / max(1, total_steps)) * 1000,
            "final_loss": epoch_losses[-1] if epoch_losses else 0,
            "model_saved": str(self.config.checkpoint_dir),
        }
        
        print()
        print(f"[training] Complete!")
        print(f"  Total steps: {total_steps}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Avg time/step: {summary['avg_time_per_step_ms']:.1f}ms")
        
        if self.use_wandb:
            wandb.summary.update(summary)
        
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder on activation shards")
    
    # Data args
    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
        help="Directory containing activation shard files (shard_*.pt)",
    )
    
    # Model args
    parser.add_argument(
        "--model",
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "pythia-1.4b", "custom"],
        help="Model type (for config presets)",
    )
    
    parser.add_argument(
        "--input-dim",
        type=int,
        help="Input dimension (required if --model custom)",
    )
    
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=8,
        help="Latent expansion factor (default: 8)",
    )
    
    # Training args
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate",
    )
    
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Override max epochs",
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop training after this many steps",
    )
    
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    
    # Device args
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    # Checkpoint args
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/sae"),
        help="Directory to save checkpoints",
    )
    
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume training from checkpoint",
    )
    
    # W&B args
    parser.add_argument(
        "--wandb-project",
        default="rl-decoder-sae",
        help="W&B project name",
    )
    
    parser.add_argument(
        "--wandb-entity",
        help="W&B entity name",
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    
    args = parser.parse_args()
    
    # Create config
    if args.model == "gpt2":
        config = gpt2_sae_config(expansion_factor=args.expansion_factor)
    elif args.model == "gpt2-medium":
        config = gpt2_medium_sae_config(expansion_factor=args.expansion_factor)
    elif args.model == "pythia-1.4b":
        config = pythia_1_4b_sae_config(expansion_factor=args.expansion_factor)
    else:  # custom
        if args.input_dim is None:
            raise ValueError("--input-dim required for custom model")
        config = default_config_for_model(
            "custom",
            input_dim=args.input_dim,
            expansion_factor=args.expansion_factor,
        )
    
    # Override config with CLI args
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.no_amp:
        config.use_amp = False
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.no_wandb or not HAS_WANDB:
        config.wandb_project = None
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
    
    # Create model and trainer
    sae = SparseAutoencoder(config)
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(config.dtype, torch.float16)
    
    trainer = SAETrainer(config, sae, device=args.device, dtype=dtype)
    
    # Load checkpoint if provided
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Create dataloader
    dataset = ActivationShardDataset(
        args.shard_dir,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    
    # Train
    summary = trainer.train(dataloader, max_steps=args.max_steps)
    
    # Save final model
    final_path = config.checkpoint_dir / f"sae_{config.input_dim}d_final.pt"
    trainer.save_checkpoint(final_path)
    
    print(f"\n[done] Training summary:")
    for key, val in summary.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()

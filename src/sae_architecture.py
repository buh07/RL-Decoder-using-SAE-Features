#!/usr/bin/env python3
"""
Sparse Autoencoder (SAE) architecture implementation.
Encoder f(x) -> h, Decoder g(h) -> x̂. L2 reconstruction + L1 sparsity + decorrelation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from sae_config import SAEConfig


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with configurable architecture.
    
    Architecture:
    - Encoder: linear projection from input_dim → latent_dim, optional ReLU
    - Decoder: linear projection from latent_dim → input_dim
    
    Loss:
    L = ||X - X̂||_2^2 + λ||h||_1 + β decorr(W) + γ L_probe + δ L_temporal
    """
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        # Encoder (linear + optional ReLU)
        self.encoder = nn.Linear(config.input_dim, config.latent_dim)
        
        # Decoder (linear)
        self.decoder = nn.Linear(config.latent_dim, config.input_dim)
        
        # Initialize decoder weights (small scale to avoid saturation)
        with torch.no_grad():
            self.decoder.weight.normal_(0, config.decoder_init_scale)
            self.decoder.bias.zero_()
        
        # Encoder can have larger init
        with torch.no_grad():
            self.encoder.weight.normal_(0, 1.0 / (config.input_dim ** 0.5))
            self.encoder.bias.zero_()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latents.
        
        Args:
            x: Input activations, shape (batch, seq_len, input_dim) or (batch, input_dim)
        
        Returns:
            h: Latents, same shape with input_dim -> latent_dim substituted
        """
        h = self.encoder(x)
        
        if self.config.use_relu:
            h = F.relu(h)
        
        return h
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to reconstruction.
        
        Args:
            h: Latents from encode()
        
        Returns:
            x̂: Reconstructed activations, same shape as encoder input
        """
        x_hat = self.decoder(h)
        return x_hat
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode + decode.
        
        Args:
            x: Input activations
        
        Returns:
            (x̂, h): Reconstruction and latents
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
    
    def compute_loss_components(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
        probe_loss: Optional[torch.Tensor] = None,
        temporal_loss: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute loss components (useful for logging individual terms).
        
        Args:
            x: Input activations
            x_hat: Reconstructed activations
            h: Latent activations
            probe_loss: Optional probe-guided loss term
            temporal_loss: Optional temporal smoothness loss
        
        Returns:
            Dictionary with loss components:
                recon_loss, l1_loss, decorr_loss, probe_loss, temporal_loss, total_loss
        """
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        
        # L1 sparsity loss
        l1_loss = h.abs().mean()
        
        # Decorrelation loss: penalize similarity between decoder atoms (columns)
        # W: shape (input_dim, latent_dim)
        # Decorr: β * Σ_{i<j} (w_i^T w_j)^2  where w_i are columns
        W = self.decoder.weight  # (input_dim, latent_dim)
        W_normalized = F.normalize(W, p=2, dim=0)  # Normalize columns
        
        # Gram matrix (correlation)
        gram = W_normalized.T @ W_normalized  # (latent_dim, latent_dim)
        
        # Zero out diagonal (self-correlation)
        gram = gram - torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        
        # Sum of squared off-diagonal elements
        decorr_loss = (gram ** 2).sum() / (gram.shape[0] * (gram.shape[0] - 1))
        
        # Total loss
        total_loss = (
            self.config.reconstruction_coeff * recon_loss
            + self.config.l1_penalty_coeff * l1_loss
            + self.config.decorrelation_coeff * decorr_loss
        )
        
        if probe_loss is not None:
            total_loss = total_loss + self.config.probe_loss_coeff * probe_loss
        
        if temporal_loss is not None:
            total_loss = total_loss + self.config.temporal_smoothness_coeff * temporal_loss
        
        return {
            "recon_loss": recon_loss.item(),
            "l1_loss": l1_loss.item(),
            "decorr_loss": decorr_loss.item(),
            "probe_loss": probe_loss.item() if probe_loss is not None else 0.0,
            "temporal_loss": temporal_loss.item() if temporal_loss is not None else 0.0,
            "total_loss": total_loss.item(),
            "total_loss_tensor": total_loss,  # For backprop
        }
    
    def compute_total_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
        probe_loss: Optional[torch.Tensor] = None,
        temporal_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total loss (shorthand for training loop)."""
        components = self.compute_loss_components(x, x_hat, h, probe_loss, temporal_loss)
        return components["total_loss_tensor"]
    
    def normalize_decoder(self) -> None:
        """Normalize decoder weight columns to unit norm (prevents collapse)."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)
    
    def get_activation_stats(self, h: torch.Tensor) -> dict:
        """Compute statistics on latent activations."""
        active = (h != 0).float().mean()
        mean_act = h.mean()
        max_act = h.max()
        
        return {
            "activation_fraction": active.item(),
            "mean_activation": mean_act.item(),
            "max_activation": max_act.item(),
        }
    
    def get_reconstruction_error(self, x: torch.Tensor, x_hat: torch.Tensor) -> float:
        """Compute relative reconstruction error: ||X - X̂||_F / ||X||_F."""
        data_norm = torch.norm(x, p="fro")
        error_norm = torch.norm(x - x_hat, p="fro")
        return (error_norm / (data_norm + 1e-8)).item()
    
    @staticmethod
    def from_config(config: SAEConfig) -> SparseAutoencoder:
        """Create SAE from config."""
        return SparseAutoencoder(config)


if __name__ == "__main__":
    from sae_config import gpt2_sae_config
    
    # Test SAE
    config = gpt2_sae_config(expansion_factor=8)
    sae = SparseAutoencoder(config)
    
    print(f"SAE Configuration:")
    print(f"  Input: {config.input_dim} → Latent: {config.latent_dim} → Output: {config.input_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in sae.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 512, config.input_dim)  # (batch, seq_len, input_dim)
    x_hat, h = sae(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Latent shape: {h.shape}")
    print(f"  Output shape: {x_hat.shape}")
    
    # Test loss computation
    loss_dict = sae.compute_loss_components(x, x_hat, h)
    print(f"\nLoss components:")
    for key, val in loss_dict.items():
        if key != "total_loss_tensor":
            print(f"  {key}: {val:.6f}")
    
    # Test stats
    stats = sae.get_activation_stats(h)
    print(f"\nActivation statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}")

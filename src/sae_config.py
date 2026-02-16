#!/usr/bin/env python3
"""
SAE Configuration and hyperparameter definitions.
Centralizes all SAE training hyperparameters and architecture choices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
import json
from pathlib import Path


@dataclass
class SAEConfig:
    """Sparse Autoencoder configuration."""
    
    # Architecture
    input_dim: int
    """Dimension of input activations (e.g., 768 for GPT-2)."""
    
    expansion_factor: int = 8
    """Latent dimension as multiple of input_dim. Typical: 4-8x for overcomplete basis."""
    
    use_relu: bool = True
    """Use ReLU on latents (default: True). If False, uses directly."""
    
    decoder_init_scale: float = 0.1
    """Initialization scale for decoder weights."""
    
    # Loss Components
    l1_penalty_coeff: float = 1e-3
    """Coefficient for L1 sparsity loss: λ||h||_1."""
    
    reconstruction_coeff: float = 1.0
    """Coefficient for reconstruction loss: ||X - X̂||_2^2."""
    
    decorrelation_coeff: float = 0.1
    """Coefficient for decorrelation loss: β Σ(w_i^T w_j)."""
    
    probe_loss_coeff: float = 0.05
    """Coefficient for probe-guided loss (if using probes): γ L_probe."""
    
    temporal_smoothness_coeff: float = 0.01
    """Coefficient for temporal loss: δ ||h_t - h_{t+1}||_2^2."""
    
    # Training
    learning_rate: float = 1e-4
    """Adam learning rate."""
    
    batch_size: int = 32
    """Training batch size (sequences per step)."""
    
    max_epochs: int = 10
    """Maximum training epochs."""
    
    warmup_steps: int = 1000
    """Learning rate warmup steps."""
    
    grad_clip: float = 1.0
    """Gradient clipping norm (0 = no clipping)."""
    
    use_amp: bool = True
    """Use automatic mixed precision (recommended for RTX 6000)."""
    
    # Optimization
    decoder_norm_every: int = 100
    """Normalize decoder weights every N steps to prevent collapse."""
    
    log_every: int = 100
    """Log metrics every N steps."""
    
    checkpoint_every: int = 500
    """Save checkpoint every N steps."""
    
    eval_every: int = 500
    """Run evaluation every N steps."""
    
    # Probes (optional)
    use_probes: bool = False
    """Whether to use probe-guided training."""
    
    probe_layer_indices: list[int] = field(default_factory=list)
    """Which transformer layers to attach probes to (e.g., [6] for GPT-2 layer 6)."""
    
    probe_task: str = "hypothesis"
    """Probe task type: 'hypothesis', 'constraint', 'reasoning_step'."""
    
    probe_leakage_threshold: float = 0.05
    """Abort if probe gap > SAE gap exceeds this threshold (flag leakage)."""
    
    # Temporal smoothness (optional)
    use_temporal: bool = False
    """Whether activations come with temporal ordering (e.g., within sequence)."""
    
    # Device & Precision
    device: str = "cuda"
    """Device to train on."""
    
    dtype: str = "float16"
    """Precision: float16, bfloat16, or float32."""
    
    # Logging & Checkpointing
    wandb_project: Optional[str] = "rl-decoder-sae"
    """W&B project name (None to disable)."""
    
    wandb_entity: Optional[str] = None
    """W&B entity/team name."""
    
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/sae"))
    """Directory to save checkpoints."""
    
    save_config_to_wandb: bool = True
    """Upload config to W&B."""
    
    @property
    def latent_dim(self) -> int:
        """Calculated latent dimension."""
        return self.input_dim * self.expansion_factor
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        d = {}
        for key, val in self.__dict__.items():
            if isinstance(val, Path):
                d[key] = str(val)
            elif isinstance(val, list):
                d[key] = val
            else:
                d[key] = val
        return d
    
    def save(self, path: Path) -> None:
        """Save config to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> SAEConfig:
        """Load config from JSON."""
        with path.open("r") as f:
            data = json.load(f)
        
        # Convert path strings back
        if "checkpoint_dir" in data:
            data["checkpoint_dir"] = Path(data["checkpoint_dir"])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """Pretty print config."""
        lines = ["SAEConfig("]
        for key, val in self.__dict__.items():
            if isinstance(val, list) and len(val) > 3:
                val = f"{val[:2]}...{val[-1:]}"
            lines.append(f"  {key}={val}")
        lines.append(")")
        return "\n".join(lines)


# Preset configurations for different models/phases

def gpt2_sae_config(expansion_factor: int = 8, **kwargs) -> SAEConfig:
    """SAE config optimized for GPT-2 small (12L, 768D hidden)."""
    config = SAEConfig(
        input_dim=768,
        expansion_factor=expansion_factor,
        learning_rate=1e-4,
        batch_size=32,
        max_epochs=10,
        l1_penalty_coeff=1e-3,
        decorrelation_coeff=0.1,
        **kwargs
    )
    return config


def gpt2_medium_sae_config(expansion_factor: int = 4, **kwargs) -> SAEConfig:
    """SAE config for GPT-2 medium (24L, 1024D) - reduced expansion for memory."""
    config = SAEConfig(
        input_dim=1024,
        expansion_factor=expansion_factor,
        learning_rate=5e-5,
        batch_size=16,
        max_epochs=10,
        l1_penalty_coeff=1e-3,
        decorrelation_coeff=0.1,
        **kwargs
    )
    return config


def pythia_1_4b_sae_config(expansion_factor: int = 4, **kwargs) -> SAEConfig:
    """SAE config for Pythia 1.4B (24L, 2048D)."""
    config = SAEConfig(
        input_dim=2048,
        expansion_factor=expansion_factor,
        learning_rate=5e-5,
        batch_size=8,
        max_epochs=5,
        l1_penalty_coeff=1e-3,
        decorrelation_coeff=0.1,
        **kwargs
    )
    return config


def default_config_for_model(
    model_name: str,
    input_dim: int,
    expansion_factor: int = 8,
    **kwargs
) -> SAEConfig:
    """Create config for arbitrary model."""
    # Auto-tune batch size and learning rate based on input dimension
    if input_dim <= 768:
        batch_size = 32
        lr = 1e-4
        max_epochs = 10
    elif input_dim <= 1024:
        batch_size = 16
        lr = 5e-5
        max_epochs = 10
    else:  # 2048+
        batch_size = 8
        lr = 5e-5
        max_epochs = 5
    
    config = SAEConfig(
        input_dim=input_dim,
        expansion_factor=expansion_factor,
        learning_rate=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        l1_penalty_coeff=1e-3,
        decorrelation_coeff=0.1,
        **kwargs
    )
    return config


if __name__ == "__main__":
    # Display default configs
    print("SAE Configurations")
    print("=" * 60)
    
    configs = [
        ("GPT-2 (baseline)", gpt2_sae_config()),
        ("GPT-2 Medium", gpt2_medium_sae_config()),
        ("Pythia-1.4B", pythia_1_4b_sae_config()),
    ]
    
    for name, cfg in configs:
        print(f"\n{name}:")
        print(f"  Input: {cfg.input_dim}D → Latent: {cfg.latent_dim}D ({cfg.expansion_factor}x)")
        print(f"  Batch: {cfg.batch_size}, LR: {cfg.learning_rate}, Epochs: {cfg.max_epochs}")
        print(f"  Loss: recon={cfg.reconstruction_coeff}, L1={cfg.l1_penalty_coeff}, "
              f"decorr={cfg.decorrelation_coeff}")

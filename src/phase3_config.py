#!/usr/bin/env python3
"""
Phase 3 Configuration: Controlled Chain-of-Thought Analysis
Reusable across models and SAE expansion levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class Phase3Config:
    """Configuration for Phase 3: Controlled CoT alignment and causal analysis."""

    # Dataset and Model Setup
    dataset: Literal["gsm8k", "custom"] = "gsm8k"
    """Dataset name (gsm8k or custom JSONL with 'reasoning_trace' and 'answer' fields)."""

    model_name: str = "gpt2"
    """Model identifier (gpt2, gpt2-medium, etc.) used for activation capture."""

    layer: int = 6
    """Layer index to analyze."""

    activation_type: Literal["residual", "mlp_hidden"] = "mlp_hidden"
    """Type of activation to use (residual or mlp_hidden post-layer)."""

    # SAE Checkpoint(s)
    sae_checkpoints: list[Path] = field(default_factory=list)
    """List of SAE checkpoint paths to evaluate. Can be multiple expansions."""

    # Input/Output Paths
    train_activation_dir: Path = Path("/tmp/gpt2_gsm8k_acts/gsm8k/train")
    """Directory containing sharded training activations."""

    test_activation_dir: Path = Path("/tmp/gpt2_gsm8k_acts_test/gsm8k/test")
    """Directory containing sharded test activations."""

    tokenizer_path: Optional[Path] = None
    """Path to tokenizer (if None, inferred from model_name via model_registry)."""

    output_dir: Path = Path("phase3_results")
    """Output directory for alignment, probes, evaluations."""

    # Step Extraction and Alignment
    step_extraction_method: Literal["regex", "similarity", "hybrid"] = "regex"
    """Method to extract reasoning steps from chain-of-thought text."""

    step_regex_patterns: dict[str, str] = field(
        default_factory=lambda: {
            "step_header": r"(?:Step|step)\s+\d+[:.]?",
            "equation": r"(\d+[\s+\-*/]*)+\s*=",
            "comparison": r"(>|<|>=|<=|==|!=)",
            "reasoning": r"(?:therefore|so|thus|hence|thus|which means)",
        }
    )
    """Regex patterns for step type detection."""

    similarity_threshold: float = 0.5
    """Threshold for similarity-based step boundaries (0-1)."""

    min_step_length: int = 5
    """Minimum tokens per step to avoid noise."""

    max_step_length: int = 256
    """Maximum tokens per step."""

    # Probe Training
    probe_batch_size: int = 32
    """Batch size for probe training."""

    probe_learning_rate: float = 1e-3
    """Learning rate for probe training."""

    probe_epochs: int = 10
    """Number of epochs for probe training."""

    probe_hidden_dim: Optional[int] = None
    """Hidden dimension for probe (None = linear probe)."""

    probe_dropout: float = 0.1
    """Dropout rate for probe."""

    probe_weight_decay: float = 1e-5
    """L2 regularization for probe."""

    leakage_threshold: float = 0.05
    """Threshold for probe-vs-baseline gap to flag leakage (delta > 5%)."""

    # Causal Ablation
    ablation_method: Literal["zero", "mean", "noise"] = "zero"
    """How to ablate features (zero, mean, or gaussian noise)."""

    causal_batch_size: int = 32
    """Batch size for causal evaluation."""

    num_perturbations_per_feature: int = 1
    """Number of random seeds per feature ablation (for MC sampling)."""

    compute_feature_importance_top_k: int = 100
    """Compute full importance for top-k most active features (rest: approx)."""

    # Evaluation and Validation
    task_accuracy_metric: str = "exact_match"
    """Metric for task performance (exact_match, contains_answer, numeric_close)."""

    numeric_tolerance: float = 1e-3
    """Numeric tolerance for numeric_close metric."""

    compute_statistics: bool = True
    """Compute and log sparsity, cluster coherence, silhouette scores."""

    # Logging and Tracking
    use_wandb: bool = False
    """Whether to log to Weights & Biases."""

    wandb_project: str = "phase3-sae-cot"
    """W&B project name."""

    wandb_entity: Optional[str] = None
    """W&B entity/team."""

    verbose: bool = True
    """Verbose logging."""

    # GPU and Performance
    device: str = "cuda:0"
    """Device to use for compute."""

    num_workers: int = 0
    """Number of workers for data loading."""

    def validate(self):
        """Validate configuration."""
        if not self.sae_checkpoints:
            raise ValueError("Must specify at least one SAE checkpoint")
        for ckpt in self.sae_checkpoints:
            if not ckpt.exists():
                raise FileNotFoundError(f"SAE checkpoint not found: {ckpt}")

        if self.dataset == "gsm8k" and not self.train_activation_dir.exists():
            raise FileNotFoundError(f"Training activations not found: {self.train_activation_dir}")

        if self.output_dir.exists() is False:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "dataset": self.dataset,
            "model": self.model_name,
            "layer": self.layer,
            "sae_checkpoints": [str(p) for p in self.sae_checkpoints],
            "step_extraction_method": self.step_extraction_method,
            "probe_epochs": self.probe_epochs,
            "ablation_method": self.ablation_method,
        }

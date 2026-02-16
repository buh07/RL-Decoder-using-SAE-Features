#!/usr/bin/env python3
"""
Model registry with layer probe defaults for capture hooks.
Defines available models, their architectures, and recommended layer choices for activation capture.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelSpec:
    """Configuration for a model available for activation capture."""
    name: str
    model_id: str
    """HuggingFace model ID or path."""
    
    hf_local_path: Path
    """Local path where model weights are cached (e.g., from LLM Second-Order Effects/models)."""
    
    num_layers: int
    """Total number of transformer layers."""
    
    hidden_dim: int
    """Hidden dimension size."""
    
    mlp_dim: Optional[int]
    """MLP intermediate dimension (if different from hidden_dim)."""
    
    default_probe_layer: int
    """Recommended mid-layer for initial probing experiments."""
    
    notes: str
    """Architecture notes, licensing, and relevant details."""


# Available models with local paths relative to LLM Second-Order Effects/models
MODELS: dict[str, ModelSpec] = {
    "gpt2": ModelSpec(
        name="GPT-2 Small",
        model_id="gpt2",
        hf_local_path=Path("/scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2"),
        num_layers=12,
        hidden_dim=768,
        mlp_dim=3072,
        default_probe_layer=6,
        notes=(
            "OpenAI GPT-2 small. 12 layers, 768D hidden, 3072D MLP. "
            "Baseline for reproducibility; causal LM trained on WebText (2019). "
            "License: MIT. Probing layer 6 (middle) for interpretation."
        ),
    ),
    "gpt2-medium": ModelSpec(
        name="GPT-2 Medium",
        model_id="gpt2-medium",
        hf_local_path=Path("/scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2-medium"),
        num_layers=24,
        hidden_dim=1024,
        mlp_dim=4096,
        default_probe_layer=12,
        notes=(
            "OpenAI GPT-2 medium. 24 layers, 1024D hidden, 4096D MLP. "
            "First scaling test; approx 2x params vs small. "
            "License: MIT. Probing layer 12 (middle) for interpretation."
        ),
    ),
    "pythia-1.4b": ModelSpec(
        name="Pythia 1.4B",
        model_id="EleutherAI/pythia-1.4b",
        hf_local_path=Path("/scratch2/f004ndc/LLM Second-Order Effects/models/models--EleutherAI--pythia-1.4b"),
        num_layers=24,
        hidden_dim=2048,
        mlp_dim=8192,
        default_probe_layer=12,
        notes=(
            "EleutherAI Pythia 1.4B. 24 layers, 2048D hidden, 8192D MLP. "
            "Designed for interpretability research. Intermediate scale. "
            "License: Apache-2.0. Probing layer 12 recommended."
        ),
    ),
    "gemma-2b": ModelSpec(
        name="Gemma 2B",
        model_id="google/gemma-2b",
        hf_local_path=Path("/scratch2/f004ndc/LLM Second-Order Effects/models/models--google--gemma-2b"),
        num_layers=18,
        hidden_dim=2048,
        mlp_dim=8192,
        default_probe_layer=9,
        notes=(
            "Google Gemma 2B. 18 layers, 2048D hidden, 8192D MLP. "
            "Modern open-weight model. Reasonable scale for reasoning tasks. "
            "License: Gemma License. Probing layer 9 (middle) recommended."
        ),
    ),
    "llama-3-8b": ModelSpec(
        name="Llama 3 8B",
        model_id="meta-llama/Meta-Llama-3-8B",
        hf_local_path=Path("/scratch2/f004ndc/LLM Second-Order Effects/models/models--meta-llama--Meta-Llama-3-8B"),
        num_layers=32,
        hidden_dim=4096,
        mlp_dim=14336,
        default_probe_layer=16,
        notes=(
            "Meta Llama 3 8B. 32 layers, 4096D hidden, 14336D MLP. "
            "Frontier-scale reasoning model. High-quality training data. "
            "License: Llama 3 Community. Probing layer 16 (middle) recommended. "
            "Warning: ~30 GB weights; ensure GPU memory."
        ),
    ),
    "phi-2": ModelSpec(
        name="Phi-2",
        model_id="microsoft/phi-2",
        hf_local_path=Path("/scratch2/f004ndc/LLM Second-Order Effects/models/models--microsoft--phi-2"),
        num_layers=32,
        hidden_dim=2560,
        mlp_dim=10240,
        default_probe_layer=16,
        notes=(
            "Microsoft Phi-2. 32 layers, 2560D hidden, 10240D MLP. "
            "Compact but capable reasoning model. Designed for instruction following. "
            "License: MIT. Probing layer 16 (middle) recommended."
        ),
    ),
}


def get_model(name: str) -> ModelSpec:
    """Retrieve a model spec by slug."""
    if name not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name]


def list_models() -> list[str]:
    """Return sorted list of available model slugs."""
    return sorted(MODELS.keys())


if __name__ == "__main__":
    print("Available models for activation capture:")
    print()
    for slug, spec in MODELS.items():
        print(f"  {slug:20} | {spec.name:25} | {spec.num_layers:2} layers | probe={spec.default_probe_layer}")
        print(f"    {spec.hf_local_path}")
        print(f"    {spec.notes}")
        print()

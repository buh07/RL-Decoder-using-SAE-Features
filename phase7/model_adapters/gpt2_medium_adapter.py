#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from .base import BaseCausalLMAdapter


@dataclass
class GPT2MediumAdapter(BaseCausalLMAdapter):
    model_key: str = "gpt2-medium"
    hf_model_id: str = "gpt2-medium"
    tokenizer_id: str = "gpt2-medium"
    model_family: str = "gpt2"
    num_layers: int = 24
    hidden_dim: int = 1024
    default_dtype: str = "float32"
    device: str = "cuda:0"

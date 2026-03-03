#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseCausalLMAdapter


@dataclass
class Qwen25_7BAdapter(BaseCausalLMAdapter):
    model_key: str = "qwen2.5-7b"
    hf_model_id: str = "Qwen/Qwen2.5-7B"
    tokenizer_id: str = "Qwen/Qwen2.5-7B"
    model_family: str = "qwen2.5"
    num_layers: int = 28
    hidden_dim: int = 3584
    default_dtype: str = "bfloat16"
    device: str = "cuda:0"
    load_kwargs: dict = field(default_factory=lambda: {"low_cpu_mem_usage": True})

#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseCausalLMAdapter


@dataclass
class Phi2Adapter(BaseCausalLMAdapter):
    model_key: str = "phi-2"
    hf_model_id: str = "microsoft/phi-2"
    tokenizer_id: str = "microsoft/phi-2"
    model_family: str = "phi"
    num_layers: int = 32
    hidden_dim: int = 2560
    default_dtype: str = "float16"
    device: str = "cuda:0"
    load_kwargs: dict = field(default_factory=lambda: {"low_cpu_mem_usage": True, "trust_remote_code": True})
    # Phi-2 behaves like decoder-only chat-less LM and does not require special-prefix tokens.
    add_special_tokens_policy: bool = False

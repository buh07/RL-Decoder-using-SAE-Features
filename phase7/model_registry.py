#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover
    from .model_adapters import GPT2MediumAdapter, Qwen25_7BAdapter
except ImportError:  # pragma: no cover
    from model_adapters import GPT2MediumAdapter, Qwen25_7BAdapter


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    hf_model_id: str
    tokenizer_id: str
    num_layers: int
    hidden_dim: int
    default_dtype: str
    adapter_class: str
    sae_dir: Optional[str] = None
    model_family: str = "unknown"
    vocab_size: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


_KNOWN_ADAPTER_CLASSES = {"GPT2MediumAdapter", "Qwen25_7BAdapter"}


_MODEL_SPECS: Dict[str, ModelSpec] = {
    "gpt2-medium": ModelSpec(
        model_key="gpt2-medium",
        hf_model_id="gpt2-medium",
        tokenizer_id="gpt2-medium",
        num_layers=24,
        hidden_dim=1024,
        default_dtype="float32",
        adapter_class="GPT2MediumAdapter",
        sae_dir="phase2_results/saes_gpt2_12x_topk/saes",
        model_family="gpt2",
        vocab_size=50257,
    ),
    "qwen2.5-7b": ModelSpec(
        model_key="qwen2.5-7b",
        hf_model_id="Qwen/Qwen2.5-7B",
        tokenizer_id="Qwen/Qwen2.5-7B",
        num_layers=28,
        hidden_dim=3584,
        default_dtype="bfloat16",
        adapter_class="Qwen25_7BAdapter",
        sae_dir="phase2_results/saes_qwen25_7b_12x_topk/saes",
        model_family="qwen2.5",
        vocab_size=152064,
    ),
}


def validate_model_spec(spec: ModelSpec) -> None:
    if not isinstance(spec.model_key, str) or not spec.model_key:
        raise ValueError("ModelSpec.model_key must be a non-empty string")
    if not isinstance(spec.hf_model_id, str) or not spec.hf_model_id:
        raise ValueError(f"ModelSpec.hf_model_id must be a non-empty string for {spec.model_key!r}")
    if not isinstance(spec.tokenizer_id, str) or not spec.tokenizer_id:
        raise ValueError(f"ModelSpec.tokenizer_id must be a non-empty string for {spec.model_key!r}")
    if int(spec.num_layers) <= 0:
        raise ValueError(f"ModelSpec.num_layers must be >0 for {spec.model_key!r}")
    if int(spec.hidden_dim) <= 0:
        raise ValueError(f"ModelSpec.hidden_dim must be >0 for {spec.model_key!r}")
    if spec.adapter_class not in _KNOWN_ADAPTER_CLASSES:
        raise ValueError(
            f"ModelSpec.adapter_class={spec.adapter_class!r} is not supported for {spec.model_key!r}; "
            f"expected one of {sorted(_KNOWN_ADAPTER_CLASSES)}"
        )
    if spec.vocab_size is not None and int(spec.vocab_size) <= 0:
        raise ValueError(f"ModelSpec.vocab_size must be >0 when set for {spec.model_key!r}")


for _k, _spec in _MODEL_SPECS.items():
    validate_model_spec(_spec)


def get_model_spec(model_key: str) -> ModelSpec:
    key = str(model_key)
    if key not in _MODEL_SPECS:
        raise KeyError(f"Unknown model_key={model_key!r}; available={sorted(_MODEL_SPECS)}")
    spec = _MODEL_SPECS[key]
    validate_model_spec(spec)
    return spec


def list_model_specs() -> Dict[str, Dict]:
    return {k: v.to_dict() for k, v in sorted(_MODEL_SPECS.items())}


def _load_adapter_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"--adapter-config must point to a JSON object, got {type(payload).__name__}")
    return payload


def resolve_model_spec(model_key: str, adapter_config: Optional[str] = None) -> ModelSpec:
    base = get_model_spec(model_key).to_dict()
    overrides = _load_adapter_config(adapter_config)
    if not overrides:
        return ModelSpec(**base)
    if "model_key" in overrides and str(overrides["model_key"]) != str(model_key):
        raise ValueError(
            f"--adapter-config model_key={overrides['model_key']!r} does not match --model-key={model_key!r}"
        )
    for k, v in overrides.items():
        if k in base:
            base[k] = v
    spec = ModelSpec(**base)
    validate_model_spec(spec)
    return spec


def create_adapter(model_key: str, device: str = "cuda:0", adapter_config: Optional[str] = None):
    spec = resolve_model_spec(model_key, adapter_config)
    kwargs = {
        "model_key": spec.model_key,
        "hf_model_id": spec.hf_model_id,
        "tokenizer_id": spec.tokenizer_id,
        "model_family": spec.model_family,
        "num_layers": int(spec.num_layers),
        "hidden_dim": int(spec.hidden_dim),
        "default_dtype": spec.default_dtype,
        "device": device,
    }
    if spec.adapter_class == "GPT2MediumAdapter":
        return GPT2MediumAdapter(**kwargs)
    if spec.adapter_class == "Qwen25_7BAdapter":
        return Qwen25_7BAdapter(**kwargs)
    raise KeyError(f"Unknown adapter_class={spec.adapter_class!r} for model_key={model_key!r}")

#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple


VOCAB_SIZE_GPT2 = 50257
SINGLE_LAYER = (22,)
MULTI_LAYERS = (7, 12, 17, 22)
VALID_INPUT_VARIANTS = ("raw", "sae", "hybrid")


@dataclass
class DecoderExperimentConfig:
    name: str
    input_variant: str  # raw | sae | hybrid
    layers: Tuple[int, ...]
    d_model: int = 256
    n_heads: int = 4
    n_decoder_layers: int = 2
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE_GPT2
    aggregator: str = "transformer"  # transformer | mlp
    hybrid_topk_values: int = 50
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    seed: int = 17
    val_fraction: float = 0.2
    early_stop_patience: int = 10

    def input_dim(self) -> int:
        if self.input_variant == "raw":
            return 1024
        if self.input_variant == "sae":
            return 12288
        if self.input_variant == "hybrid":
            return 1024 + self.hybrid_topk_values
        raise ValueError(f"Unsupported input_variant: {self.input_variant}")

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["layers"] = list(self.layers)
        data["input_dim"] = self.input_dim()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "DecoderExperimentConfig":
        d = dict(data)
        d["layers"] = tuple(d["layers"])
        d.pop("input_dim", None)
        return cls(**d)


def default_experiment_configs() -> Dict[str, DecoderExperimentConfig]:
    configs: List[DecoderExperimentConfig] = [
        DecoderExperimentConfig(name="raw_single_l22", input_variant="raw", layers=SINGLE_LAYER),
        DecoderExperimentConfig(name="raw_multi_l7_l12_l17_l22", input_variant="raw", layers=MULTI_LAYERS),
        DecoderExperimentConfig(name="sae_single_l22", input_variant="sae", layers=SINGLE_LAYER),
        DecoderExperimentConfig(name="sae_multi_l7_l12_l17_l22", input_variant="sae", layers=MULTI_LAYERS),
        DecoderExperimentConfig(name="hybrid_single_l22", input_variant="hybrid", layers=SINGLE_LAYER),
        DecoderExperimentConfig(name="hybrid_multi_l7_l12_l17_l22", input_variant="hybrid", layers=MULTI_LAYERS),
    ]
    return {cfg.name: cfg for cfg in configs}


def normalize_layers(layers: Iterable[int], *, layers_total: int = 24) -> Tuple[int, ...]:
    vals = tuple(int(x) for x in layers)
    if not vals:
        raise ValueError("layers must be non-empty")
    if len(set(vals)) != len(vals):
        raise ValueError(f"layers contain duplicates: {vals}")
    if any(x < 0 or x >= layers_total for x in vals):
        raise ValueError(f"layers must be in [0, {layers_total - 1}]")
    return tuple(sorted(vals))


def _default_custom_name(input_variant: str, layers: Tuple[int, ...]) -> str:
    if len(layers) == 1:
        return f"{input_variant}_single_l{layers[0]:02d}"
    return f"{input_variant}_custom_" + "_".join(f"l{x:02d}" for x in layers)


def make_custom_config(
    *,
    input_variant: str,
    layers: Iterable[int],
    name: Optional[str] = None,
    base_name: Optional[str] = None,
) -> DecoderExperimentConfig:
    if input_variant not in VALID_INPUT_VARIANTS:
        raise ValueError(f"Unsupported input_variant={input_variant!r}; expected one of {VALID_INPUT_VARIANTS}")
    layer_tuple = normalize_layers(layers)
    final_name = name or _default_custom_name(input_variant, layer_tuple)
    cfg = None
    if base_name:
        try:
            cfg = get_config(base_name)
        except KeyError:
            cfg = None
    if cfg is None:
        cfg = next((c for c in default_experiment_configs().values() if c.input_variant == input_variant), None)
    if cfg is None:
        raise RuntimeError(f"No base config available for input_variant={input_variant!r}")
    base = asdict(cfg)
    base.update({"name": final_name, "input_variant": input_variant, "layers": layer_tuple})
    return DecoderExperimentConfig(**base)


def get_config(name: str) -> DecoderExperimentConfig:
    configs = default_experiment_configs()
    if name not in configs:
        raise KeyError(f"Unknown config '{name}'. Available: {sorted(configs)}")
    return configs[name]


def list_config_names() -> List[str]:
    return sorted(default_experiment_configs().keys())

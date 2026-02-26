#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseProjection(nn.Module):
    """Projection wrapper for SAE inputs (currently dense matmul; sparse-aware API surface)."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass
class ArithmeticDecoderConfig:
    input_dim: int
    n_layers_input: int
    d_model: int = 256
    n_heads: int = 4
    n_decoder_layers: int = 2
    vocab_size: int = 50257
    dropout: float = 0.1
    aggregator: str = "transformer"
    use_sparse_input: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ArithmeticDecoderConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return dict(
            input_dim=self.input_dim,
            n_layers_input=self.n_layers_input,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_decoder_layers=self.n_decoder_layers,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            aggregator=self.aggregator,
            use_sparse_input=self.use_sparse_input,
        )


class ArithmeticDecoder(nn.Module):
    """
    Reads features from frozen GPT-2 layer states/SAE latents and predicts the result token.

    Input shape: (batch, n_layers_input, input_dim)
    Output shape: (batch, vocab_size)
    """

    def __init__(self, config: ArithmeticDecoderConfig):
        super().__init__()
        self.config = config
        proj_cls = SparseProjection if config.use_sparse_input else nn.Linear
        self.layer_projections = nn.ModuleList(
            [proj_cls(config.input_dim, config.d_model) for _ in range(config.n_layers_input)]
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.layer_pos = nn.Parameter(torch.randn(1, config.n_layers_input + 1, config.d_model) * 0.02)

        if config.aggregator == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.cross_layer_encoder = nn.TransformerEncoder(
                enc_layer,
                num_layers=config.n_decoder_layers,
                enable_nested_tensor=False,
            )
            self.mlp_aggregator = None
        elif config.aggregator == "mlp":
            self.cross_layer_encoder = None
            flat_dim = (config.n_layers_input + 1) * config.d_model
            hidden_dim = max(config.d_model * 2, 256)
            self.mlp_aggregator = nn.Sequential(
                nn.LayerNorm(flat_dim),
                nn.Linear(flat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim, config.d_model),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported aggregator={config.aggregator}")

        self.out_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def _project_layers(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        bsz, n_layers, _ = x.shape
        if n_layers != self.config.n_layers_input:
            raise ValueError(f"Expected {self.config.n_layers_input} input layers, got {n_layers}")
        projected = []
        for i in range(n_layers):
            projected.append(self.layer_projections[i](x[:, i, :]))
        return torch.stack(projected, dim=1)  # (B, L, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        projected = self._project_layers(x)
        bsz = projected.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        seq = torch.cat([cls, projected], dim=1) + self.layer_pos[:, : projected.shape[1] + 1, :]

        if self.cross_layer_encoder is not None:
            encoded = self.cross_layer_encoder(seq)
            pooled = encoded[:, 0, :]
        else:
            pooled = self.mlp_aggregator(seq.reshape(bsz, -1))

        return self.out_norm(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.encode(x))

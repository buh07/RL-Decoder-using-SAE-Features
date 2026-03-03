#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class BaseCausalLMAdapter(ABC):
    model_key: str
    hf_model_id: str
    tokenizer_id: str
    model_family: str
    num_layers: int
    hidden_dim: int
    default_dtype: str = "float32"
    device: str = "cuda:0"
    model: Any = None
    tokenizer: Any = None
    load_kwargs: dict = field(default_factory=dict)

    def _resolved_dtype(self) -> torch.dtype:
        dtype = _DTYPE_MAP.get(str(self.default_dtype).lower(), torch.float32)
        if str(self.device).startswith("cpu") and dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return dtype

    def load(self, device: str | None = None) -> "BaseCausalLMAdapter":
        if device is not None:
            self.device = str(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"dtype": self._resolved_dtype()}
        kwargs.update(self.load_kwargs or {})
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_model_id, **kwargs).to(self.device).eval()
        return self

    def tokenize(self, text_or_ids: Any) -> torch.Tensor:
        if isinstance(text_or_ids, torch.Tensor):
            ids = text_or_ids.long()
        elif isinstance(text_or_ids, str):
            if self.tokenizer is None:
                raise RuntimeError("Adapter tokenizer is not loaded")
            ids = self.tokenizer(text_or_ids, return_tensors="pt").input_ids
        elif isinstance(text_or_ids, Sequence) and text_or_ids and isinstance(text_or_ids[0], int):
            ids = torch.tensor([list(text_or_ids)], dtype=torch.long)
        elif isinstance(text_or_ids, Sequence) and text_or_ids and isinstance(text_or_ids[0], Sequence):
            ids = torch.tensor(text_or_ids, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported tokenize() input type: {type(text_or_ids).__name__}")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.to(self.device)

    def forward(self, input_ids: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.model is None:
            raise RuntimeError("Adapter model is not loaded")
        ids = self.tokenize(input_ids)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True, return_dict=True)
        hidden_states = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
        return out.logits, hidden_states

    def logprob_at(self, input_ids: Any, pred_pos: int, token_id: int) -> float:
        logits, _ = self.forward(input_ids)
        pos = max(0, min(int(pred_pos), logits.shape[1] - 1))
        lp = torch.log_softmax(logits[0, pos, :], dim=-1)
        return float(lp[int(token_id)].item())

    def _transformer_blocks(self):
        if self.model is None:
            raise RuntimeError("Adapter model is not loaded")
        m = self.model
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        raise RuntimeError(f"Unsupported model architecture for patch hooks: model_key={self.model_key}")

    def patch_forward(self, layer: int, token_pos: int, patch_vector: torch.Tensor, input_ids: Any) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Adapter model is not loaded")
        ids = self.tokenize(input_ids)
        blocks = self._transformer_blocks()
        n_blocks = len(blocks)
        if int(layer) < 0 or int(layer) >= n_blocks:
            raise ValueError(f"layer out of range: {layer} not in [0, {n_blocks - 1}]")
        pos = max(0, min(int(token_pos), ids.shape[1] - 1))
        vec = patch_vector.to(self.device)

        def _hook(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if vec.ndim != 1 or vec.numel() != hidden.shape[-1]:
                raise ValueError(
                    f"patch_vector shape mismatch: got {tuple(vec.shape)}, expected ({hidden.shape[-1]},)"
                )
            patched = hidden.clone()
            patched[0, pos, :] = vec.to(dtype=patched.dtype)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        handle = blocks[int(layer)].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                out = self.model(ids)
            return out.logits
        finally:
            handle.remove()

    def metadata(self) -> dict:
        return {
            "model_key": self.model_key,
            "model_family": self.model_family,
            "num_layers": int(self.num_layers),
            "hidden_dim": int(self.hidden_dim),
            "tokenizer_id": self.tokenizer_id,
        }

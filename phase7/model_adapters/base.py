#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    add_special_tokens_policy: bool = True

    def _tokenize_add_special_tokens(self) -> bool:
        # One canonical special-token policy used by tokenize(), tokenize_with_offsets(),
        # and forward() callsites to avoid position drift.
        return bool(self.add_special_tokens_policy)

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
            ids = self.tokenizer(
                text_or_ids,
                return_tensors="pt",
                add_special_tokens=self._tokenize_add_special_tokens(),
            ).input_ids
        elif isinstance(text_or_ids, Sequence) and text_or_ids and isinstance(text_or_ids[0], int):
            ids = torch.tensor([list(text_or_ids)], dtype=torch.long)
        elif isinstance(text_or_ids, Sequence) and text_or_ids and isinstance(text_or_ids[0], Sequence):
            ids = torch.tensor(text_or_ids, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported tokenize() input type: {type(text_or_ids).__name__}")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.to(self.device)

    def tokenize_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]], Dict[str, Any]]:
        if self.tokenizer is None:
            raise RuntimeError("Adapter tokenizer is not loaded")
        add_special_tokens = self._tokenize_add_special_tokens()
        degraded = False
        try:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=add_special_tokens,
            )
            if "offset_mapping" not in enc:
                raise KeyError("offset_mapping missing from tokenizer output")
            offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
            token_ids = [int(x) for x in enc.input_ids[0].tolist()]
        except Exception:
            # Fallback keeps caller running but marks degraded alignment quality.
            degraded = True
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
            )
            token_ids = [int(x) for x in enc.input_ids[0].tolist()]
            offsets = [(0, 0) for _ in token_ids]

        prefix_special = 0
        for s, e in offsets:
            if int(e) <= int(s):
                prefix_special += 1
            else:
                break
        meta = {
            "special_tokens_policy": ("add_special_tokens_true" if add_special_tokens else "add_special_tokens_false"),
            "num_special_tokens_prefix": int(prefix_special),
            "offset_alignment_degraded": bool(degraded),
        }
        return token_ids, offsets, meta

    def forward(self, input_ids: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.model is None:
            raise RuntimeError("Adapter model is not loaded")
        ids = self.tokenize(input_ids)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True, return_dict=True)
        hidden_states = tuple(out.hidden_states[1:]) if out.hidden_states is not None else tuple()
        return out.logits, hidden_states

    @torch.no_grad()
    def generate_with_step_hidden_states(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Adapter model/tokenizer is not loaded")
        if not prompts:
            return []

        original_padding_side = str(getattr(self.tokenizer, "padding_side", "right"))
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                add_special_tokens=self._tokenize_add_special_tokens(),
            )
            input_ids = enc.input_ids.to(self.device)
            attention_mask = getattr(enc, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            prompt_lens = (
                attention_mask.sum(dim=1).detach().cpu().tolist()
                if attention_mask is not None
                else [int(input_ids.shape[1])] * int(input_ids.shape[0])
            )
            pad_token_id = int(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(do_sample),
                "pad_token_id": int(pad_token_id),
                "return_dict_in_generate": True,
                "output_hidden_states": True,
            }
            if bool(do_sample):
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            seq = out.sequences.detach().cpu()
            step_hidden = out.hidden_states

            results: List[Dict[str, Any]] = []
            for bi in range(int(seq.shape[0])):
                prompt_len = int(prompt_lens[bi])
                gen_ids = seq[bi, prompt_len:].tolist()
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                step_tensors: List[torch.Tensor] = []
                if isinstance(step_hidden, (tuple, list)):
                    for hs_step in step_hidden:
                        if not isinstance(hs_step, (tuple, list)):
                            continue
                        per_layer: List[torch.Tensor] = []
                        for li, layer_out in enumerate(hs_step):
                            if li == 0:  # skip embedding layer
                                continue
                            if not torch.is_tensor(layer_out):
                                continue
                            lt = layer_out.detach()
                            if lt.ndim == 3:
                                # [batch, seq, hidden] or [batch, 1, hidden]
                                vec = lt[bi, -1, :]
                            elif lt.ndim == 2:
                                # [batch, hidden]
                                vec = lt[bi, :]
                            else:
                                continue
                            per_layer.append(vec.float().cpu())
                        if per_layer:
                            step_tensors.append(torch.stack(per_layer, dim=0))

                expected_steps = int(len(gen_ids))
                available_steps = int(len(step_tensors))
                if available_steps > expected_steps:
                    step_tensors = step_tensors[:expected_steps]
                if available_steps < expected_steps and available_steps > 0:
                    # Align lengths conservatively by truncating generated ids to captured states.
                    gen_ids = gen_ids[:available_steps]
                    gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                results.append(
                    {
                        "prompt": str(prompts[bi]),
                        "generated_text": str(gen_text),
                        "generated_token_ids": [int(x) for x in gen_ids],
                        "hidden_by_generated_token": step_tensors,
                        "prompt_token_count": int(prompt_len),
                        "generated_token_count": int(len(gen_ids)),
                        "captured_step_count": int(len(step_tensors)),
                    }
                )
            return results
        finally:
            self.tokenizer.padding_side = original_padding_side

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

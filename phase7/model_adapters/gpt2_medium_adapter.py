#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "gpt2-medium"


@dataclass
class GPT2MediumAdapter:
    device: str = "cuda:0"
    model: Any = None
    tokenizer: Any = None

    def load(self) -> "GPT2MediumAdapter":
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(self.device).eval()
        return self

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(input_ids.to(self.device)).logits

    def logprob(self, input_ids: torch.Tensor, pred_pos: int, token_id: int) -> float:
        logits = self.logits(input_ids)
        pred_pos = max(0, min(int(pred_pos), logits.shape[1] - 1))
        lp = torch.log_softmax(logits[0, pred_pos, :], dim=-1)
        return float(lp[int(token_id)].item())

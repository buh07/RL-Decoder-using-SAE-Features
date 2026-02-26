#!/usr/bin/env python3
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: str | Path, max_records: Optional[int] = None) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    records = torch.load(p, weights_only=False)
    if not isinstance(records, list):
        raise TypeError(f"Expected list of records in {p}, got {type(records).__name__}")
    if max_records is not None:
        records = records[:max_records]
    return records


def verify_schema(records: Sequence[dict], expected: str = "phase6_v1") -> None:
    if not records:
        return
    missing = [i for i, r in enumerate(records[:10]) if r.get("schema_version") != expected]
    if missing:
        raise ValueError(
            f"Unexpected schema_version in dataset (expected {expected}); "
            f"examples with mismatch in first 10: {missing}"
        )


def extract_operator(expr_str: str) -> str:
    # Prefer explicit arithmetic operators; avoid unary '-' false positives.
    for ch in ("+", "*", "/"):
        if ch in expr_str:
            return ch
    for i, ch in enumerate(expr_str):
        if ch == "-" and i > 0:
            return "-"
    return "unknown"


def magnitude_bucket(c_val: float) -> str:
    mag = abs(float(c_val))
    if mag < 10:
        return "[0,10)"
    if mag < 100:
        return "[10,100)"
    if mag < 1000:
        return "[100,1000)"
    return "[1000+)"


def topk_values_signed(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: (..., D), returns signed values of top-|x| entries, shape (..., k)
    k = min(k, x.shape[-1])
    idx = x.abs().topk(k, dim=-1).indices
    return x.gather(-1, idx)


def build_input_tensor_from_record(record: dict, cfg) -> torch.Tensor:
    """
    Returns tensor of shape (n_layers_input, input_dim) in float32 on CPU.
    `cfg` may be DecoderExperimentConfig or dict-like with the same fields.
    """
    input_variant = cfg.input_variant if hasattr(cfg, "input_variant") else cfg["input_variant"]
    layers = cfg.layers if hasattr(cfg, "layers") else tuple(cfg["layers"])
    hybrid_topk_values = (
        cfg.hybrid_topk_values if hasattr(cfg, "hybrid_topk_values") else int(cfg.get("hybrid_topk_values", 50))
    )

    raw = record["raw_hidden"].float()   # (24, 1024)
    sae = record["sae_features"].float() # (24, 12288)

    raw_sel = raw[list(layers), :]
    sae_sel = sae[list(layers), :]

    if input_variant == "raw":
        return raw_sel
    if input_variant == "sae":
        return sae_sel
    if input_variant == "hybrid":
        topvals = topk_values_signed(sae_sel, hybrid_topk_values)
        return torch.cat([raw_sel, topvals], dim=-1)

    raise ValueError(f"Unsupported input_variant={input_variant}")


def split_records_by_example(records: Sequence[dict], val_fraction: float, seed: int) -> Tuple[List[dict], List[dict]]:
    by_example: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        by_example[int(r["example_idx"])].append(r)

    ex_ids = sorted(by_example.keys())
    rng = random.Random(seed)
    rng.shuffle(ex_ids)

    n_val = max(1, int(round(len(ex_ids) * val_fraction))) if ex_ids else 0
    n_val = min(n_val, max(0, len(ex_ids) - 1)) if len(ex_ids) > 1 else n_val

    val_ids = set(ex_ids[:n_val])
    train_records: List[dict] = []
    val_records: List[dict] = []
    for ex_id, group in by_example.items():
        (val_records if ex_id in val_ids else train_records).extend(group)
    return train_records, val_records


@dataclass
class RecordItem:
    x: torch.Tensor
    y: int
    baseline_logprob: float
    operator: str
    magnitude_bucket: str
    expr_str: str
    c_value: float
    example_idx: int


class Phase6RecordDataset(Dataset):
    def __init__(self, records: Sequence[dict], cfg):
        self.records = list(records)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> RecordItem:
        r = self.records[idx]
        x = build_input_tensor_from_record(r, self.cfg)
        return RecordItem(
            x=x,
            y=int(r["result_token_id"]),
            baseline_logprob=float(r["baseline_logprob"]),
            operator=extract_operator(str(r.get("expr_str", ""))),
            magnitude_bucket=magnitude_bucket(float(r.get("C", 0.0))),
            expr_str=str(r.get("expr_str", "")),
            c_value=float(r.get("C", 0.0)),
            example_idx=int(r.get("example_idx", -1)),
        )


def collate_record_items(items: Sequence[RecordItem]) -> Dict[str, Any]:
    x = torch.stack([it.x for it in items], dim=0)
    y = torch.tensor([it.y for it in items], dtype=torch.long)
    baseline_logprob = torch.tensor([it.baseline_logprob for it in items], dtype=torch.float32)
    return {
        "x": x,
        "y": y,
        "baseline_logprob": baseline_logprob,
        "operator": [it.operator for it in items],
        "magnitude_bucket": [it.magnitude_bucket for it in items],
        "expr_str": [it.expr_str for it in items],
        "c_value": [it.c_value for it in items],
        "example_idx": [it.example_idx for it in items],
    }


def logits_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    correct_lp = log_probs.gather(1, targets[:, None]).squeeze(1)
    top1 = (logits.argmax(dim=-1) == targets).float()
    top5 = logits.topk(5, dim=-1).indices.eq(targets[:, None]).any(dim=-1).float()
    return {
        "correct_logprob": correct_lp,
        "top1": top1,
        "top5": top5,
    }


def evaluate_batches(model: torch.nn.Module, loader, device: str) -> Dict[str, Any]:
    model.eval()
    all_lp: List[float] = []
    all_baseline_lp: List[float] = []
    all_top1: List[float] = []
    all_top5: List[float] = []
    by_op = defaultdict(lambda: {"n": 0, "top1": 0.0, "top5": 0.0})
    by_mag = defaultdict(lambda: {"n": 0, "top1": 0.0, "top5": 0.0})

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            m = logits_metrics(logits, y)

            lp = m["correct_logprob"].cpu().tolist()
            t1 = m["top1"].cpu().tolist()
            t5 = m["top5"].cpu().tolist()
            bl = batch["baseline_logprob"].tolist()

            all_lp.extend(lp)
            all_top1.extend(t1)
            all_top5.extend(t5)
            all_baseline_lp.extend(bl)

            for op, mag, v1, v5 in zip(batch["operator"], batch["magnitude_bucket"], t1, t5):
                by_op[op]["n"] += 1
                by_op[op]["top1"] += float(v1)
                by_op[op]["top5"] += float(v5)
                by_mag[mag]["n"] += 1
                by_mag[mag]["top1"] += float(v1)
                by_mag[mag]["top5"] += float(v5)

    def _finalize_group(gdict):
        out = {}
        for k, v in sorted(gdict.items()):
            n = max(1, int(v["n"]))
            out[k] = {
                "n": int(v["n"]),
                "top1_accuracy": float(v["top1"] / n),
                "top5_accuracy": float(v["top5"] / n),
            }
        return out

    n = max(1, len(all_lp))
    return {
        "num_records": len(all_lp),
        "top1_accuracy": float(sum(all_top1) / n),
        "top5_accuracy": float(sum(all_top5) / n),
        "mean_logprob_correct": float(sum(all_lp) / n),
        "mean_baseline_logprob": float(sum(all_baseline_lp) / n) if all_baseline_lp else None,
        "delta_logprob_vs_gpt2": float((sum(all_lp) - sum(all_baseline_lp)) / n) if all_baseline_lp else None,
        "per_operator": _finalize_group(by_op),
        "per_magnitude_bucket": _finalize_group(by_mag),
    }

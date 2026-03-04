#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "phase6"))

from decoder_model import ArithmeticDecoder, ArithmeticDecoderConfig  # type: ignore  # noqa: E402
from pipeline_utils import build_input_tensor_from_record  # type: ignore  # noqa: E402

try:  # pragma: no cover
    from .common import MAG_BUCKETS, OPERATORS, SIGNS, STEP_TYPES, magnitude_bucket, set_seed, sign_label
except ImportError:  # pragma: no cover
    from common import MAG_BUCKETS, OPERATORS, SIGNS, STEP_TYPES, magnitude_bucket, set_seed, sign_label

DEFAULT_VOCAB_SIZE = 50257
VALID_INPUT_VARIANTS = ("raw", "sae", "hybrid", "hybrid_indexed")

OP_TO_ID = {v: i for i, v in enumerate(OPERATORS)}
STEP_TO_ID = {v: i for i, v in enumerate(STEP_TYPES)}
MAG_TO_ID = {v: i for i, v in enumerate(MAG_BUCKETS)}
SIGN_TO_ID = {v: i for i, v in enumerate(SIGNS)}


@dataclass
class StateDecoderExperimentConfig:
    name: str
    input_variant: str  # raw|sae|hybrid
    layers: Tuple[int, ...]
    vocab_size: int = DEFAULT_VOCAB_SIZE
    model_key: str = "gpt2-medium"
    model_family: str = "gpt2"
    tokenizer_id: str = "gpt2-medium"
    model_num_layers: int = 24
    model_hidden_dim: int = 1024
    model_sae_dim: int = 12288
    d_model: int = 256
    n_heads: int = 4
    n_decoder_layers: int = 2
    dropout: float = 0.1
    aggregator: str = "transformer"
    hybrid_topk_values: int = 50
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    seed: int = 17
    val_fraction: float = 0.2
    early_stop_patience: int = 10
    loss_w_result_token: float = 1.0
    loss_w_operator: float = 0.5
    loss_w_step_type: float = 0.25
    loss_w_magnitude: float = 0.25
    loss_w_sign: float = 0.25
    loss_w_subresult: float = 0.15
    loss_w_lhs: float = 0.05
    loss_w_rhs: float = 0.05

    def input_dim(self) -> int:
        if self.input_variant == "raw":
            return int(self.model_hidden_dim)
        if self.input_variant == "sae":
            return int(self.model_sae_dim)
        if self.input_variant == "hybrid":
            return int(self.model_hidden_dim) + int(self.hybrid_topk_values)
        if self.input_variant == "hybrid_indexed":
            return int(self.model_hidden_dim) + int(2 * self.hybrid_topk_values)
        raise ValueError(f"Unsupported input_variant={self.input_variant}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["layers"] = list(self.layers)
        d["input_dim"] = self.input_dim()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateDecoderExperimentConfig":
        d = dict(d)
        d["layers"] = tuple(d["layers"])
        d.pop("input_dim", None)
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)


MULTI_LAYERS = (7, 12, 17, 22)


def default_state_decoder_configs() -> Dict[str, StateDecoderExperimentConfig]:
    cfgs = [
        StateDecoderExperimentConfig(name="state_raw_multi_l7_l12_l17_l22", input_variant="raw", layers=MULTI_LAYERS),
        StateDecoderExperimentConfig(name="state_sae_multi_l7_l12_l17_l22", input_variant="sae", layers=MULTI_LAYERS),
        StateDecoderExperimentConfig(name="state_hybrid_multi_l7_l12_l17_l22", input_variant="hybrid", layers=MULTI_LAYERS),
    ]
    return {c.name: c for c in cfgs}


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
    return f"state_{input_variant}_custom_" + "_".join(f"l{x:02d}" for x in layers)


def make_custom_state_decoder_config(
    *,
    input_variant: str,
    layers: Iterable[int],
    name: Optional[str] = None,
    base_name: Optional[str] = None,
) -> StateDecoderExperimentConfig:
    if input_variant not in VALID_INPUT_VARIANTS:
        raise ValueError(f"Unsupported input_variant={input_variant!r}; expected one of {VALID_INPUT_VARIANTS}")
    layer_tuple = normalize_layers(layers)
    final_name = name or _default_custom_name(input_variant, layer_tuple)
    cfg = None
    if base_name:
        cfg = default_state_decoder_configs().get(base_name)
    if cfg is None:
        cfg = next((c for c in default_state_decoder_configs().values() if c.input_variant == input_variant), None)
    if cfg is None:
        raise RuntimeError(f"No base state decoder config for input_variant={input_variant!r}")
    base = asdict(cfg)
    base.update({"name": final_name, "input_variant": input_variant, "layers": layer_tuple})
    return StateDecoderExperimentConfig(**base)


def apply_model_metadata_to_config(
    cfg: StateDecoderExperimentConfig,
    model_meta: Dict[str, Any],
    vocab_size_override: Optional[int] = None,
) -> StateDecoderExperimentConfig:
    d = cfg.to_dict()
    d["model_key"] = str(model_meta.get("model_key", d.get("model_key", "gpt2-medium")))
    d["model_family"] = str(model_meta.get("model_family", d.get("model_family", "unknown")))
    d["tokenizer_id"] = str(model_meta.get("tokenizer_id", d.get("tokenizer_id", "")))
    d["model_num_layers"] = int(model_meta.get("num_layers", d.get("model_num_layers", 24)))
    d["model_hidden_dim"] = int(model_meta.get("hidden_dim", d.get("model_hidden_dim", 1024)))
    d["model_sae_dim"] = int(model_meta.get("sae_dim", d.get("model_sae_dim", 12288)))
    if vocab_size_override is not None:
        d["vocab_size"] = int(vocab_size_override)
    elif model_meta.get("vocab_size") is not None:
        d["vocab_size"] = int(model_meta["vocab_size"])
    return StateDecoderExperimentConfig.from_dict(d)


def split_by_example(records: Sequence[dict], val_fraction: float, seed: int) -> Tuple[List[dict], List[dict]]:
    by_ex: Dict[int, List[dict]] = {}
    for r in records:
        by_ex.setdefault(int(r["example_idx"]), []).append(r)
    ids = sorted(by_ex)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction))) if ids else 0
    n_val = min(n_val, max(0, len(ids) - 1)) if len(ids) > 1 else n_val
    val_ids = set(ids[:n_val])
    tr, va = [], []
    for ex_id, rows in by_ex.items():
        (va if ex_id in val_ids else tr).extend(rows)
    return tr, va


@dataclass
class NumericNormStats:
    mean: float
    std: float

    def to_dict(self) -> Dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "NumericNormStats":
        arr = np.array([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
        if arr.size == 0:
            return cls(0.0, 1.0)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return cls(mean, max(std, 1e-6))

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "NumericNormStats":
        return cls(float(d["mean"]), float(d["std"]))


class Phase7StateDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict],
        cfg: StateDecoderExperimentConfig,
        numeric_stats: Dict[str, NumericNormStats],
        *,
        cache_inputs: str = "off",
        cache_max_gb: float = 2.0,
    ):
        self.records = list(records)
        self.cfg = cfg
        self.numeric_stats = numeric_stats
        self.cache_inputs = str(cache_inputs).lower()
        if self.cache_inputs not in {"off", "auto", "on"}:
            raise ValueError(f"cache_inputs must be one of off|auto|on, got {cache_inputs!r}")
        self.cache_max_gb = float(cache_max_gb)
        self._cached_x: Optional[List[torch.Tensor]] = None
        if self.records and self.cache_inputs != "off":
            first_x = build_input_tensor_from_record(self.records[0], self.cfg)
            est_bytes = int(len(self.records) * first_x.numel() * first_x.element_size())
            allow_bytes = int(max(0.0, self.cache_max_gb) * (1024 ** 3))
            should_cache = self.cache_inputs == "on" or (self.cache_inputs == "auto" and est_bytes <= allow_bytes)
            if should_cache:
                cached = [first_x]
                for r in self.records[1:]:
                    cached.append(build_input_tensor_from_record(r, self.cfg))
                self._cached_x = cached

    def __len__(self) -> int:
        return len(self.records)

    def _enc_numeric(self, key: str, x: Optional[float]) -> Tuple[float, float]:
        if x is None or not math.isfinite(float(x)):
            return 0.0, 0.0
        st = self.numeric_stats[key]
        return (float(x) - st.mean) / st.std, 1.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        s = r["structured_state"]
        x = self._cached_x[idx] if self._cached_x is not None else build_input_tensor_from_record(r, self.cfg)
        op = str(s.get("operator", "unknown"))
        mag = str(s.get("magnitude_bucket", magnitude_bucket(float(r.get("C", 0.0)))))
        sign = str(s.get("sign", sign_label(float(r.get("C", 0.0)))))
        step_type = str(s.get("step_type", "operate"))
        y_sub, m_sub = self._enc_numeric("subresult_value", s.get("subresult_value"))
        y_lhs, m_lhs = self._enc_numeric("lhs_value", s.get("lhs_value"))
        y_rhs, m_rhs = self._enc_numeric("rhs_value", s.get("rhs_value"))
        return {
            "x": x,
            "result_token_id": int(s.get("result_token_id", r["result_token_id"])),
            "operator_id": OP_TO_ID.get(op, OP_TO_ID["unknown"]),
            "step_type_id": STEP_TO_ID.get(step_type, STEP_TO_ID["operate"]),
            "magnitude_id": MAG_TO_ID.get(mag, MAG_TO_ID["[1000+)"]),
            "sign_id": SIGN_TO_ID.get(sign, SIGN_TO_ID["zero"]),
            "subresult_z": y_sub,
            "lhs_z": y_lhs,
            "rhs_z": y_rhs,
            "mask_subresult": m_sub,
            "mask_lhs": m_lhs,
            "mask_rhs": m_rhs,
            "baseline_logprob": float(r.get("baseline_logprob", float("nan"))),
            "operator": op,
            "magnitude_bucket": mag,
            "trace_id": str(r.get("trace_id", "")),
            "step_idx": int(r.get("step_idx", -1)),
            "example_idx": int(r.get("example_idx", -1)),
        }


def collate_state_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([it["x"] for it in items], dim=0)
    batch = {
        "x": x,
        "result_token_id": torch.tensor([it["result_token_id"] for it in items], dtype=torch.long),
        "operator_id": torch.tensor([it["operator_id"] for it in items], dtype=torch.long),
        "step_type_id": torch.tensor([it["step_type_id"] for it in items], dtype=torch.long),
        "magnitude_id": torch.tensor([it["magnitude_id"] for it in items], dtype=torch.long),
        "sign_id": torch.tensor([it["sign_id"] for it in items], dtype=torch.long),
        "subresult_z": torch.tensor([it["subresult_z"] for it in items], dtype=torch.float32),
        "lhs_z": torch.tensor([it["lhs_z"] for it in items], dtype=torch.float32),
        "rhs_z": torch.tensor([it["rhs_z"] for it in items], dtype=torch.float32),
        "mask_subresult": torch.tensor([it["mask_subresult"] for it in items], dtype=torch.float32),
        "mask_lhs": torch.tensor([it["mask_lhs"] for it in items], dtype=torch.float32),
        "mask_rhs": torch.tensor([it["mask_rhs"] for it in items], dtype=torch.float32),
        "baseline_logprob": torch.tensor([it["baseline_logprob"] for it in items], dtype=torch.float32),
        "operator": [it["operator"] for it in items],
        "magnitude_bucket": [it["magnitude_bucket"] for it in items],
        "trace_id": [it["trace_id"] for it in items],
        "step_idx": [it["step_idx"] for it in items],
        "example_idx": [it["example_idx"] for it in items],
    }
    return batch


def dataloader_perf_kwargs(
    *,
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": int(max(0, num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(max(1, prefetch_factor))
    return kwargs


def move_batch_to_device(batch: Dict[str, Any], device: str, *, non_blocking: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


class MultiHeadStateDecoder(nn.Module):
    def __init__(self, cfg: StateDecoderExperimentConfig):
        super().__init__()
        self.exp_cfg = cfg
        self.backbone_cfg = ArithmeticDecoderConfig(
            input_dim=cfg.input_dim(),
            n_layers_input=len(cfg.layers),
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_decoder_layers=cfg.n_decoder_layers,
            vocab_size=cfg.vocab_size,
            dropout=cfg.dropout,
            aggregator=cfg.aggregator,
            use_sparse_input=(cfg.input_variant == "sae"),
            use_output_head=False,
        )
        self.backbone = ArithmeticDecoder(self.backbone_cfg)
        d = cfg.d_model
        self.result_token_head = nn.Linear(d, cfg.vocab_size)
        self.operator_head = nn.Linear(d, len(OPERATORS))
        self.step_type_head = nn.Linear(d, len(STEP_TYPES))
        self.magnitude_head = nn.Linear(d, len(MAG_BUCKETS))
        self.sign_head = nn.Linear(d, len(SIGNS))
        self.subresult_head = nn.Linear(d, 1)
        self.lhs_head = nn.Linear(d, 1)
        self.rhs_head = nn.Linear(d, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        return {
            "z": z,
            "result_token_logits": self.result_token_head(z),
            "operator_logits": self.operator_head(z),
            "step_type_logits": self.step_type_head(z),
            "magnitude_logits": self.magnitude_head(z),
            "sign_logits": self.sign_head(z),
            "subresult_pred": self.subresult_head(z).squeeze(-1),
            "lhs_pred": self.lhs_head(z).squeeze(-1),
            "rhs_pred": self.rhs_head(z).squeeze(-1),
        }


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    return (diff2 * mask).sum() / mask.sum().clamp_min(1.0)


def compute_multitask_loss(model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: StateDecoderExperimentConfig):
    losses: Dict[str, torch.Tensor] = {}
    losses["result_token"] = F.cross_entropy(model_out["result_token_logits"], batch["result_token_id"])
    losses["operator"] = F.cross_entropy(model_out["operator_logits"], batch["operator_id"])
    losses["step_type"] = F.cross_entropy(model_out["step_type_logits"], batch["step_type_id"])
    losses["magnitude"] = F.cross_entropy(model_out["magnitude_logits"], batch["magnitude_id"])
    losses["sign"] = F.cross_entropy(model_out["sign_logits"], batch["sign_id"])
    losses["subresult"] = _masked_mse(model_out["subresult_pred"], batch["subresult_z"], batch["mask_subresult"])
    losses["lhs"] = _masked_mse(model_out["lhs_pred"], batch["lhs_z"], batch["mask_lhs"])
    losses["rhs"] = _masked_mse(model_out["rhs_pred"], batch["rhs_z"], batch["mask_rhs"])
    total = (
        cfg.loss_w_result_token * losses["result_token"]
        + cfg.loss_w_operator * losses["operator"]
        + cfg.loss_w_step_type * losses["step_type"]
        + cfg.loss_w_magnitude * losses["magnitude"]
        + cfg.loss_w_sign * losses["sign"]
        + cfg.loss_w_subresult * losses["subresult"]
        + cfg.loss_w_lhs * losses["lhs"]
        + cfg.loss_w_rhs * losses["rhs"]
    )
    losses["total"] = total
    return losses


def _acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().item())


def _topk_acc(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    return float(logits.topk(k, dim=-1).indices.eq(target[:, None]).any(dim=-1).float().mean().item())


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple:
    """Returns (sum_of_abs_errors, valid_count) to allow correct aggregation."""
    valid = float(mask.sum().item())
    if valid <= 0:
        return 0.0, 0
    return float(((pred - target).abs() * mask).sum().item()), int(valid)


def evaluate_state_model(
    model: MultiHeadStateDecoder,
    loader: DataLoader,
    device: str,
    numeric_stats: Dict[str, NumericNormStats],
    *,
    non_blocking_transfer: bool = False,
) -> Dict[str, Any]:
    model.eval()
    n = 0
    agg = {
        "loss_total": 0.0,
        "loss_result_token": 0.0,
        "loss_operator": 0.0,
        "loss_step_type": 0.0,
        "loss_magnitude": 0.0,
        "loss_sign": 0.0,
        "loss_subresult": 0.0,
        "loss_lhs": 0.0,
        "loss_rhs": 0.0,
        "result_token_top1": 0.0,
        "result_token_top5": 0.0,
        "operator_acc": 0.0,
        "step_type_acc": 0.0,
        "magnitude_acc": 0.0,
        "sign_acc": 0.0,
        "subresult_mae_z_sum": 0.0,
        "subresult_mae_z_n": 0,
        "lhs_mae_z_sum": 0.0,
        "lhs_mae_z_n": 0,
        "rhs_mae_z_sum": 0.0,
        "rhs_mae_z_n": 0,
        "correct_logprob": 0.0,
        "baseline_logprob": 0.0,
    }
    by_op: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for batch in loader:
            dev_batch = move_batch_to_device(batch, device, non_blocking=non_blocking_transfer)
            x = dev_batch["x"]
            out = model(x)
            losses = compute_multitask_loss(out, dev_batch, model.exp_cfg)
            bsz = x.shape[0]
            n += bsz
            for k, v in losses.items():
                agg[f"loss_{k}"] += float(v.item()) * bsz
            agg["result_token_top1"] += _acc(out["result_token_logits"], dev_batch["result_token_id"]) * bsz
            agg["result_token_top5"] += _topk_acc(out["result_token_logits"], dev_batch["result_token_id"], 5) * bsz
            agg["operator_acc"] += _acc(out["operator_logits"], dev_batch["operator_id"]) * bsz
            agg["step_type_acc"] += _acc(out["step_type_logits"], dev_batch["step_type_id"]) * bsz
            agg["magnitude_acc"] += _acc(out["magnitude_logits"], dev_batch["magnitude_id"]) * bsz
            agg["sign_acc"] += _acc(out["sign_logits"], dev_batch["sign_id"]) * bsz
            for _key, _pred_k, _tgt_k, _mask_k in [
                ("subresult", "subresult_pred", "subresult_z", "mask_subresult"),
                ("lhs", "lhs_pred", "lhs_z", "mask_lhs"),
                ("rhs", "rhs_pred", "rhs_z", "mask_rhs"),
            ]:
                _s, _v = _masked_mae(out[_pred_k], dev_batch[_tgt_k], dev_batch[_mask_k])
                agg[f"{_key}_mae_z_sum"] += _s
                agg[f"{_key}_mae_z_n"] += _v

            log_probs = F.log_softmax(out["result_token_logits"], dim=-1)
            corr_lp = log_probs.gather(1, dev_batch["result_token_id"][:, None]).squeeze(1)
            agg["correct_logprob"] += float(corr_lp.sum().item())
            agg["baseline_logprob"] += float(dev_batch["baseline_logprob"].sum().item())

            pred = out["result_token_logits"].argmax(dim=-1).detach().cpu().tolist()
            y = dev_batch["result_token_id"].detach().cpu().tolist()
            for op, pp, yy in zip(batch["operator"], pred, y):
                d = by_op.setdefault(op, {"n": 0.0, "top1": 0.0})
                d["n"] += 1.0
                d["top1"] += 1.0 if pp == yy else 0.0

    if n == 0:
        return {"num_records": 0}
    outm = {"num_records": n}
    for k, v in agg.items():
        if k.endswith("_mae_z_sum") or k.endswith("_mae_z_n"):
            continue  # handled below
        if k in {"correct_logprob", "baseline_logprob"}:
            outm[f"mean_{k}"] = float(v / n)
        else:
            outm[k] = float(v / n)
    # Compute masked MAE metrics from sum/count
    for key in ("subresult", "lhs", "rhs"):
        s = agg[f"{key}_mae_z_sum"]
        cnt = agg[f"{key}_mae_z_n"]
        outm[f"{key}_mae_z"] = float(s / cnt) if cnt > 0 else float("nan")
    outm["delta_logprob_vs_gpt2"] = float((agg["correct_logprob"] - agg["baseline_logprob"]) / n)
    outm["per_operator_result_top1"] = {
        op: {"n": int(d["n"]), "top1_accuracy": float(d["top1"] / max(1.0, d["n"]))} for op, d in sorted(by_op.items())
    }
    # de-normalized MAE for interpretability
    for key, stat_name in [("subresult", "subresult_value"), ("lhs", "lhs_value"), ("rhs", "rhs_value")]:
        z_key = f"{key}_mae_z"
        if z_key in outm and math.isfinite(outm[z_key]):
            outm[f"{key}_mae"] = float(outm[z_key] * numeric_stats[stat_name].std)
    return outm


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    if total_steps <= 0:
        return None

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def numeric_stats_from_records(records: Sequence[dict]) -> Dict[str, NumericNormStats]:
    keys = ["subresult_value", "lhs_value", "rhs_value"]
    return {
        k: NumericNormStats.from_values([r["structured_state"].get(k) for r in records])
        for k in keys
    }


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(path: str | Path, device: str):
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = StateDecoderExperimentConfig.from_dict(ckpt["experiment_config"])
    model = MultiHeadStateDecoder(cfg)
    incompat = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    allowed_unexpected = {"backbone.output_head.weight", "backbone.output_head.bias"}
    unexpected = [k for k in incompat.unexpected_keys if k not in allowed_unexpected]
    missing = list(incompat.missing_keys)
    if unexpected or missing:
        raise RuntimeError(
            "Checkpoint/state-dict incompatibility detected while loading phase7 model. "
            f"unexpected_keys={unexpected} missing_keys={missing}"
        )
    model = model.to(device).eval()
    numeric_stats = {k: NumericNormStats.from_dict(v) for k, v in ckpt["numeric_stats"].items()}
    return ckpt, cfg, numeric_stats, model


def decode_latent_pred_states(
    model: MultiHeadStateDecoder,
    records: Sequence[dict],
    cfg: StateDecoderExperimentConfig,
    numeric_stats: Dict[str, NumericNormStats],
    device: str,
    batch_size: int = 64,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    cache_inputs: str = "off",
    cache_max_gb: float = 2.0,
    non_blocking_transfer: bool = False,
) -> List[dict]:
    ds = Phase7StateDataset(
        records,
        cfg,
        numeric_stats,
        cache_inputs=cache_inputs,
        cache_max_gb=cache_max_gb,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_state_batch,
        **dataloader_perf_kwargs(
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )
    preds: List[dict] = []
    inv_op = {v: k for k, v in OP_TO_ID.items()}
    inv_step = {v: k for k, v in STEP_TO_ID.items()}
    inv_mag = {v: k for k, v in MAG_TO_ID.items()}
    inv_sign = {v: k for k, v in SIGN_TO_ID.items()}
    model.eval()
    with torch.no_grad():
        for batch in dl:
            dev_batch = move_batch_to_device(batch, device, non_blocking=non_blocking_transfer)
            x = dev_batch["x"]
            out = model(x)
            log_probs = F.log_softmax(out["result_token_logits"], dim=-1)
            top1_tok = out["result_token_logits"].argmax(dim=-1).cpu().tolist()
            top1_lp = log_probs.max(dim=-1).values.cpu().tolist()
            ops = out["operator_logits"].argmax(dim=-1).cpu().tolist()
            steps = out["step_type_logits"].argmax(dim=-1).cpu().tolist()
            mags = out["magnitude_logits"].argmax(dim=-1).cpu().tolist()
            signs = out["sign_logits"].argmax(dim=-1).cpu().tolist()
            lhs_z = out["lhs_pred"].cpu().tolist()
            rhs_z = out["rhs_pred"].cpu().tolist()
            sub_z = out["subresult_pred"].cpu().tolist()
            for i in range(len(top1_tok)):
                lhs = lhs_z[i] * numeric_stats["lhs_value"].std + numeric_stats["lhs_value"].mean
                rhs = rhs_z[i] * numeric_stats["rhs_value"].std + numeric_stats["rhs_value"].mean
                sub = sub_z[i] * numeric_stats["subresult_value"].std + numeric_stats["subresult_value"].mean
                preds.append(
                    {
                        "step_idx": int(batch["step_idx"][i]),
                        "trace_id": str(batch["trace_id"][i]),
                        "example_idx": int(batch["example_idx"][i]),
                        "latent_pred_state": {
                            "step_type": inv_step[int(steps[i])],
                            "operator": inv_op[int(ops[i])],
                            "magnitude_bucket": inv_mag[int(mags[i])],
                            "sign": inv_sign[int(signs[i])],
                            "lhs_value": float(lhs),
                            "rhs_value": float(rhs),
                            "subresult_value": float(sub),
                            "result_token_id": int(top1_tok[i]),
                        },
                        "latent_pred_confidence": {
                            "result_token_logprob_top1": float(top1_lp[i]),
                            "operator_prob": float(F.softmax(out["operator_logits"], dim=-1)[i, ops[i]].item()),
                            "step_type_prob": float(F.softmax(out["step_type_logits"], dim=-1)[i, steps[i]].item()),
                        },
                    }
                )
    return preds

#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LOGICAL_DOMAINS = {"prontoqa", "entailmentbank"}
DEFAULT_DOMAIN_HEAD_LOSS_WEIGHTS: Dict[str, float] = {
    "inference": 1.0,
    "chain_depth": 1.0,
    "truth": 2.0,
    "conclusion": 2.0,
    "premise": 1.5,
    "entity": 1.0,
}


@dataclass
class OptionCDomainDecoderConfig:
    decoder_domain: str
    model_key: str
    layers: Tuple[int, ...]
    hidden_dim: int
    d_model: int = 768
    dropout: float = 0.1
    logical_chain_depth_bins: int = 5
    class_vocab: Tuple[str, ...] = ("__unknown__",)
    entity_vocab: Tuple[str, ...] = ("__unknown__",)
    inference_type_vocab: Tuple[str, ...] = ("unknown", "fact_assertion", "class_subsumption", "negation", "other")
    truth_value_vocab: Tuple[str, ...] = ("unknown", "true", "false", "uncertain")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["layers"] = [int(x) for x in self.layers]
        d["class_vocab"] = [str(x) for x in self.class_vocab]
        d["entity_vocab"] = [str(x) for x in self.entity_vocab]
        d["inference_type_vocab"] = [str(x) for x in self.inference_type_vocab]
        d["truth_value_vocab"] = [str(x) for x in self.truth_value_vocab]
        d["input_dim"] = int(self.hidden_dim * len(self.layers))
        return d

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OptionCDomainDecoderConfig":
        d = dict(payload)
        d["layers"] = tuple(int(x) for x in list(d.get("layers", [])))
        d["class_vocab"] = tuple(str(x) for x in list(d.get("class_vocab", ["__unknown__"])))
        d["entity_vocab"] = tuple(str(x) for x in list(d.get("entity_vocab", ["__unknown__"])))
        d["inference_type_vocab"] = tuple(
            str(x) for x in list(d.get("inference_type_vocab", ["unknown", "fact_assertion", "class_subsumption", "negation", "other"]))
        )
        d["truth_value_vocab"] = tuple(str(x) for x in list(d.get("truth_value_vocab", ["unknown", "true", "false", "uncertain"])))
        d.pop("input_dim", None)
        return cls(**d)


class OptionCDomainDecoder(nn.Module):
    def __init__(self, cfg: OptionCDomainDecoderConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = int(cfg.hidden_dim * len(cfg.layers))
        d = int(cfg.d_model)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
        )
        self.inference_head = nn.Linear(d, len(cfg.inference_type_vocab))
        self.chain_depth_head = nn.Linear(d, int(cfg.logical_chain_depth_bins))
        self.truth_head = nn.Linear(d, len(cfg.truth_value_vocab))
        self.conclusion_head = nn.Linear(d, len(cfg.class_vocab))
        self.premise_head = nn.Linear(d, len(cfg.class_vocab))
        self.entity_head = nn.Linear(d, len(cfg.entity_vocab))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {
            "z": z,
            "inference_logits": self.inference_head(z),
            "chain_depth_logits": self.chain_depth_head(z),
            "truth_logits": self.truth_head(z),
            "conclusion_logits": self.conclusion_head(z),
            "premise_logits": self.premise_head(z),
            "entity_logits": self.entity_head(z),
        }


def _macro_f1_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    ignore_id: int = 0,
) -> float:
    if int(mask.sum().item()) <= 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    pred = pred[mask.bool()]
    tgt = target[mask.bool()]
    classes = sorted(set(int(x) for x in tgt.tolist()) - {int(ignore_id)})
    if not classes:
        return 0.0
    f1_vals: List[float] = []
    for c in classes:
        tp = int(((pred == c) & (tgt == c)).sum().item())
        fp = int(((pred == c) & (tgt != c)).sum().item())
        fn = int(((pred != c) & (tgt == c)).sum().item())
        prec = float(tp / max(1, tp + fp))
        rec = float(tp / max(1, tp + fn))
        f1 = 0.0 if (prec + rec) <= 0 else float(2.0 * prec * rec / (prec + rec))
        f1_vals.append(f1)
    return float(sum(f1_vals) / max(1, len(f1_vals)))


class OptionCDomainDataset(Dataset):
    def __init__(self, rows: Sequence[dict], cfg: OptionCDomainDecoderConfig):
        self.rows = list(rows)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        s = dict(row.get("structured_state") or {})
        hid = row["raw_hidden"]
        x = torch.cat([hid[int(l)].float() for l in self.cfg.layers], dim=0)

        def get_id(key: str, max_id: int) -> int:
            v = int(s.get(key, 0))
            if 0 <= v < int(max_id):
                return int(v)
            return 0

        inference_id = get_id("inference_type_id", len(self.cfg.inference_type_vocab))
        chain_depth_id = get_id("chain_depth_id", int(self.cfg.logical_chain_depth_bins))
        truth_id = get_id("truth_value_id", len(self.cfg.truth_value_vocab))
        conc_id = get_id("conclusion_class_id", len(self.cfg.class_vocab))
        prem_id = get_id("premise_class_id", len(self.cfg.class_vocab))
        ent_id = get_id("target_entity_id", len(self.cfg.entity_vocab))

        return {
            "x": x,
            "inference_id": inference_id,
            "chain_depth_id": chain_depth_id,
            "truth_id": truth_id,
            "conclusion_id": conc_id,
            "premise_id": prem_id,
            "entity_id": ent_id,
            "mask_inference": float(inference_id > 0),
            "mask_chain_depth": float(chain_depth_id >= 0),
            "mask_truth": float(truth_id > 0),
            "mask_conclusion": float(conc_id > 0),
            "mask_premise": float(prem_id > 0),
            "mask_entity": float(ent_id > 0),
            "pair_id": str(row.get("pair_id", "")),
            "member_id": str(row.get("member_id", "")),
            "step_idx": int(row.get("step_idx", -1)),
        }


def collate_optionc_domain_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([it["x"] for it in items], dim=0),
        "inference_id": torch.tensor([it["inference_id"] for it in items], dtype=torch.long),
        "chain_depth_id": torch.tensor([it["chain_depth_id"] for it in items], dtype=torch.long),
        "truth_id": torch.tensor([it["truth_id"] for it in items], dtype=torch.long),
        "conclusion_id": torch.tensor([it["conclusion_id"] for it in items], dtype=torch.long),
        "premise_id": torch.tensor([it["premise_id"] for it in items], dtype=torch.long),
        "entity_id": torch.tensor([it["entity_id"] for it in items], dtype=torch.long),
        "mask_inference": torch.tensor([it["mask_inference"] for it in items], dtype=torch.float32),
        "mask_chain_depth": torch.tensor([it["mask_chain_depth"] for it in items], dtype=torch.float32),
        "mask_truth": torch.tensor([it["mask_truth"] for it in items], dtype=torch.float32),
        "mask_conclusion": torch.tensor([it["mask_conclusion"] for it in items], dtype=torch.float32),
        "mask_premise": torch.tensor([it["mask_premise"] for it in items], dtype=torch.float32),
        "mask_entity": torch.tensor([it["mask_entity"] for it in items], dtype=torch.float32),
        "pair_id": [it["pair_id"] for it in items],
        "member_id": [it["member_id"] for it in items],
        "step_idx": [it["step_idx"] for it in items],
    }


def _masked_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask_b = mask.bool()
    if int(mask_b.sum().item()) <= 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask_b], target[mask_b], weight=class_weights)


def _class_weights_from_ids(
    ids: torch.Tensor,
    mask: torch.Tensor,
    n_classes: int,
    *,
    max_ratio: float = 5.0,
) -> Optional[torch.Tensor]:
    mask_b = mask.bool()
    if int(mask_b.sum().item()) <= 0:
        return None
    y = ids[mask_b]
    counts = torch.bincount(y, minlength=int(n_classes)).float()
    counts[0] = 0.0
    valid = counts > 0
    if int(valid.sum().item()) <= 1:
        return None
    mean_c = counts[valid].mean().clamp_min(1.0)
    w = torch.ones_like(counts)
    w[valid] = mean_c / counts[valid]
    pos = w[valid]
    if pos.numel() > 0:
        lo = float(pos.min().item())
        hi = float(pos.max().item())
        if lo > 0 and hi / lo > float(max_ratio):
            cap_hi = lo * float(max_ratio)
            w = torch.clamp(w, min=lo, max=cap_hi)
    w[0] = 0.0
    return w


def compute_domain_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    class_weights: Optional[Dict[str, torch.Tensor]] = None,
    head_loss_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    cw = class_weights or {}
    hlw = dict(DEFAULT_DOMAIN_HEAD_LOSS_WEIGHTS)
    if head_loss_weights:
        for k, v in head_loss_weights.items():
            if k in hlw:
                try:
                    hlw[k] = float(v)
                except Exception:
                    continue
    losses = {
        "inference": _masked_ce(out["inference_logits"], batch["inference_id"], batch["mask_inference"], class_weights=cw.get("inference")),
        "chain_depth": _masked_ce(
            out["chain_depth_logits"],
            batch["chain_depth_id"],
            batch["mask_chain_depth"],
            class_weights=cw.get("chain_depth"),
        ),
        "truth": _masked_ce(out["truth_logits"], batch["truth_id"], batch["mask_truth"], class_weights=cw.get("truth")),
        "conclusion": _masked_ce(
            out["conclusion_logits"],
            batch["conclusion_id"],
            batch["mask_conclusion"],
            class_weights=cw.get("conclusion"),
        ),
        "premise": _masked_ce(out["premise_logits"], batch["premise_id"], batch["mask_premise"], class_weights=cw.get("premise")),
        "entity": _masked_ce(out["entity_logits"], batch["entity_id"], batch["mask_entity"], class_weights=cw.get("entity")),
    }
    # Truth/conclusion are high-value heads for logical transition checks by default.
    total = (
        hlw["inference"] * losses["inference"]
        + hlw["chain_depth"] * losses["chain_depth"]
        + hlw["truth"] * losses["truth"]
        + hlw["conclusion"] * losses["conclusion"]
        + hlw["premise"] * losses["premise"]
        + hlw["entity"] * losses["entity"]
    )
    losses["total"] = total
    return losses


def evaluate_domain_decoder(
    model: OptionCDomainDecoder,
    loader: DataLoader,
    device: str,
    *,
    head_loss_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    model.eval()
    agg: Dict[str, float] = {
        "n": 0.0,
        "loss_total": 0.0,
        "inference_acc": 0.0,
        "chain_depth_acc": 0.0,
        "truth_acc": 0.0,
        "conclusion_acc": 0.0,
        "premise_acc": 0.0,
        "entity_acc": 0.0,
    }
    truth_logits_all: List[torch.Tensor] = []
    truth_ids_all: List[torch.Tensor] = []
    truth_mask_all: List[torch.Tensor] = []
    conc_logits_all: List[torch.Tensor] = []
    conc_ids_all: List[torch.Tensor] = []
    conc_mask_all: List[torch.Tensor] = []
    prem_logits_all: List[torch.Tensor] = []
    prem_ids_all: List[torch.Tensor] = []
    prem_mask_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(dev["x"])
            losses = compute_domain_loss(out, dev, head_loss_weights=head_loss_weights)
            bsz = int(dev["x"].shape[0])
            agg["n"] += float(bsz)
            agg["loss_total"] += float(losses["total"].item()) * bsz
            for key, logit_key, id_key, mask_key in (
                ("inference_acc", "inference_logits", "inference_id", "mask_inference"),
                ("chain_depth_acc", "chain_depth_logits", "chain_depth_id", "mask_chain_depth"),
                ("truth_acc", "truth_logits", "truth_id", "mask_truth"),
                ("conclusion_acc", "conclusion_logits", "conclusion_id", "mask_conclusion"),
                ("premise_acc", "premise_logits", "premise_id", "mask_premise"),
                ("entity_acc", "entity_logits", "entity_id", "mask_entity"),
            ):
                mask = dev[mask_key].bool()
                if int(mask.sum().item()) <= 0:
                    continue
                pred = out[logit_key].argmax(dim=-1)
                acc = float((pred[mask] == dev[id_key][mask]).float().mean().item())
                agg[key] += float(acc) * bsz
            truth_logits_all.append(out["truth_logits"].detach().cpu())
            truth_ids_all.append(dev["truth_id"].detach().cpu())
            truth_mask_all.append(dev["mask_truth"].detach().cpu())
            conc_logits_all.append(out["conclusion_logits"].detach().cpu())
            conc_ids_all.append(dev["conclusion_id"].detach().cpu())
            conc_mask_all.append(dev["mask_conclusion"].detach().cpu())
            prem_logits_all.append(out["premise_logits"].detach().cpu())
            prem_ids_all.append(dev["premise_id"].detach().cpu())
            prem_mask_all.append(dev["mask_premise"].detach().cpu())

    n = max(1.0, float(agg["n"]))
    outm = {
        "loss_total": float(agg["loss_total"] / n),
        "inference_acc": float(agg["inference_acc"] / n),
        "chain_depth_acc": float(agg["chain_depth_acc"] / n),
        "truth_acc": float(agg["truth_acc"] / n),
        "conclusion_acc": float(agg["conclusion_acc"] / n),
        "premise_acc": float(agg["premise_acc"] / n),
        "entity_acc": float(agg["entity_acc"] / n),
        "truth_macro_f1": 0.0,
        "conclusion_macro_f1": 0.0,
        "premise_macro_f1": 0.0,
    }
    if truth_logits_all:
        outm["truth_macro_f1"] = _macro_f1_from_logits(
            torch.cat(truth_logits_all, dim=0),
            torch.cat(truth_ids_all, dim=0),
            torch.cat(truth_mask_all, dim=0),
            ignore_id=0,
        )
        outm["conclusion_macro_f1"] = _macro_f1_from_logits(
            torch.cat(conc_logits_all, dim=0),
            torch.cat(conc_ids_all, dim=0),
            torch.cat(conc_mask_all, dim=0),
            ignore_id=0,
        )
        outm["premise_macro_f1"] = _macro_f1_from_logits(
            torch.cat(prem_logits_all, dim=0),
            torch.cat(prem_ids_all, dim=0),
            torch.cat(prem_mask_all, dim=0),
            ignore_id=0,
        )
    return outm


def save_optionc_domain_decoder_checkpoint(
    path: str | Path,
    *,
    model: OptionCDomainDecoder,
    cfg: OptionCDomainDecoderConfig,
    best_epoch: int,
    best_val: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    train_settings: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "schema_version": "phase7_optionc_domain_decoder_v1",
        "decoder_domain": str(cfg.decoder_domain),
        "config": cfg.to_dict(),
        "model_state_dict": model.state_dict(),
        "best_epoch": int(best_epoch),
        "best_val": dict(best_val),
        "history": [dict(x) for x in history],
        "train_settings": dict(train_settings or {}),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_optionc_domain_decoder_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], OptionCDomainDecoderConfig, OptionCDomainDecoder]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if str(ckpt.get("schema_version")) != "phase7_optionc_domain_decoder_v1":
        raise RuntimeError(f"Unsupported checkpoint schema for optionc domain decoder: {checkpoint_path}")
    cfg = OptionCDomainDecoderConfig.from_dict(dict(ckpt.get("config") or {}))
    model = OptionCDomainDecoder(cfg).to(device)
    model.load_state_dict(dict(ckpt.get("model_state_dict") or {}))
    model.eval()
    return ckpt, cfg, model


def decode_optionc_domain_states(
    model: OptionCDomainDecoder,
    cfg: OptionCDomainDecoderConfig,
    rows: Sequence[dict],
    *,
    device: str,
    batch_size: int = 256,
) -> List[Dict[str, Any]]:
    ds = OptionCDomainDataset(rows, cfg)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate_optionc_domain_batch)
    inf_vocab = list(cfg.inference_type_vocab)
    truth_vocab = list(cfg.truth_value_vocab)
    class_vocab = list(cfg.class_vocab)
    entity_vocab = list(cfg.entity_vocab)
    preds: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(dev["x"])
            inf_probs = F.softmax(out["inference_logits"], dim=-1).detach().cpu()
            chain_probs = F.softmax(out["chain_depth_logits"], dim=-1).detach().cpu()
            truth_probs = F.softmax(out["truth_logits"], dim=-1).detach().cpu()
            conc_probs = F.softmax(out["conclusion_logits"], dim=-1).detach().cpu()
            prem_probs = F.softmax(out["premise_logits"], dim=-1).detach().cpu()
            ent_probs = F.softmax(out["entity_logits"], dim=-1).detach().cpu()

            inf_id = inf_probs.argmax(dim=-1).tolist()
            chain_id = chain_probs.argmax(dim=-1).tolist()
            truth_id = truth_probs.argmax(dim=-1).tolist()
            conc_id = conc_probs.argmax(dim=-1).tolist()
            prem_id = prem_probs.argmax(dim=-1).tolist()
            ent_id = ent_probs.argmax(dim=-1).tolist()
            for i in range(len(inf_id)):
                iid = int(inf_id[i])
                tid = int(truth_id[i])
                cid = int(conc_id[i])
                pid = int(prem_id[i])
                eid = int(ent_id[i])
                did = int(chain_id[i])
                preds.append(
                    {
                        "latent_pred_state": {
                            "decoder_domain": str(cfg.decoder_domain),
                            "inference_type_id": iid,
                            "inference_type": (str(inf_vocab[iid]) if 0 <= iid < len(inf_vocab) else "unknown"),
                            "chain_depth_id": did,
                            "truth_value_id": tid,
                            "truth_value": (str(truth_vocab[tid]) if 0 <= tid < len(truth_vocab) else "unknown"),
                            "conclusion_class_id": cid,
                            "conclusion_class": (str(class_vocab[cid]) if 0 <= cid < len(class_vocab) else "__unknown__"),
                            "premise_class_id": pid,
                            "premise_class": (str(class_vocab[pid]) if 0 <= pid < len(class_vocab) else "__unknown__"),
                            "target_entity_id": eid,
                            "target_entity": (str(entity_vocab[eid]) if 0 <= eid < len(entity_vocab) else "__unknown__"),
                        },
                        "latent_pred_confidence": {
                            "inference_top1_prob": float(inf_probs[i, iid].item()),
                            "chain_depth_top1_prob": float(chain_probs[i, did].item()),
                            "truth_top1_prob": float(truth_probs[i, tid].item()),
                            "conclusion_top1_prob": float(conc_probs[i, cid].item()),
                            "premise_top1_prob": float(prem_probs[i, pid].item()),
                            "target_entity_top1_prob": float(ent_probs[i, eid].item()),
                        },
                    }
                )
    return preds

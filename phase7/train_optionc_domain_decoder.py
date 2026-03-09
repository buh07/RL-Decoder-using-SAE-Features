#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

try:  # pragma: no cover
    from .common import load_json, load_pt, save_json, set_seed
    from .optionc_domain_decoder import (
        DEFAULT_DOMAIN_HEAD_LOSS_WEIGHTS,
        LOGICAL_DOMAINS,
        OptionCDomainDataset,
        OptionCDomainDecoder,
        OptionCDomainDecoderConfig,
        collate_optionc_domain_batch,
        compute_domain_loss,
        evaluate_domain_decoder,
        save_optionc_domain_decoder_checkpoint,
        _class_weights_from_ids,  # type: ignore[attr-defined]
    )
except ImportError:  # pragma: no cover
    from common import load_json, load_pt, save_json, set_seed
    from optionc_domain_decoder import (
        DEFAULT_DOMAIN_HEAD_LOSS_WEIGHTS,
        LOGICAL_DOMAINS,
        OptionCDomainDataset,
        OptionCDomainDecoder,
        OptionCDomainDecoderConfig,
        collate_optionc_domain_batch,
        compute_domain_loss,
        evaluate_domain_decoder,
        save_optionc_domain_decoder_checkpoint,
        _class_weights_from_ids,  # type: ignore[attr-defined]
    )


def _parse_int_csv(value: str) -> Tuple[int, ...]:
    out: List[int] = []
    seen = set()
    for tok in str(value or "").split(","):
        t = tok.strip()
        if not t:
            continue
        v = int(t)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    if not out:
        raise ValueError("--layers must include at least one layer")
    return tuple(sorted(out))


def _parse_head_loss_weights(value: str) -> Dict[str, float]:
    if not str(value or "").strip():
        return {}
    out: Dict[str, float] = {}
    for part in str(value).split(","):
        tok = part.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"Invalid head weight token (expected key:value): {tok}")
        k, v = tok.split(":", 1)
        key = str(k).strip()
        if key not in DEFAULT_DOMAIN_HEAD_LOSS_WEIGHTS:
            raise ValueError(f"Unknown decoder head in weight override: {key}")
        out[key] = float(v.strip())
    return out


def _split_rows_by_pair(rows: Sequence[dict], val_fraction: float, seed: int) -> Tuple[List[dict], List[dict]]:
    pair_ids = sorted({str(r.get("pair_id", "")) for r in rows if str(r.get("pair_id", ""))})
    if len(pair_ids) < 2:
        raise RuntimeError("Need at least two pair IDs for train/val split")
    rng = random.Random(int(seed))
    ids = list(pair_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * float(val_fraction))))
    n_val = min(n_val, max(1, len(ids) - 1))
    val_set = set(ids[:n_val])
    tr = [r for r in rows if str(r.get("pair_id", "")) not in val_set]
    va = [r for r in rows if str(r.get("pair_id", "")) in val_set]
    return tr, va


def _build_class_weights(
    rows: Sequence[dict],
    cfg: OptionCDomainDecoderConfig,
    *,
    device: str,
    max_ratio: float,
) -> Dict[str, torch.Tensor]:
    ds = OptionCDomainDataset(rows, cfg)
    dl = DataLoader(ds, batch_size=1024, shuffle=False, collate_fn=collate_optionc_domain_batch)
    ids: Dict[str, List[torch.Tensor]] = {
        "inference": [],
        "chain_depth": [],
        "truth": [],
        "conclusion": [],
        "premise": [],
        "entity": [],
    }
    masks: Dict[str, List[torch.Tensor]] = {
        "inference": [],
        "chain_depth": [],
        "truth": [],
        "conclusion": [],
        "premise": [],
        "entity": [],
    }
    for b in dl:
        ids["inference"].append(b["inference_id"])
        ids["chain_depth"].append(b["chain_depth_id"])
        ids["truth"].append(b["truth_id"])
        ids["conclusion"].append(b["conclusion_id"])
        ids["premise"].append(b["premise_id"])
        ids["entity"].append(b["entity_id"])

        masks["inference"].append(b["mask_inference"])
        masks["chain_depth"].append(b["mask_chain_depth"])
        masks["truth"].append(b["mask_truth"])
        masks["conclusion"].append(b["mask_conclusion"])
        masks["premise"].append(b["mask_premise"])
        masks["entity"].append(b["mask_entity"])

    out: Dict[str, torch.Tensor] = {}
    for key, n_cls in (
        ("inference", len(cfg.inference_type_vocab)),
        ("chain_depth", int(cfg.logical_chain_depth_bins)),
        ("truth", len(cfg.truth_value_vocab)),
        ("conclusion", len(cfg.class_vocab)),
        ("premise", len(cfg.class_vocab)),
        ("entity", len(cfg.entity_vocab)),
    ):
        if not ids[key]:
            continue
        y = torch.cat(ids[key], dim=0)
        m = torch.cat(masks[key], dim=0)
        w = _class_weights_from_ids(y, m, int(n_cls), max_ratio=float(max_ratio))
        if w is not None:
            out[key] = w.to(device)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-dataset", required=True)
    p.add_argument("--decoder-domain", choices=["prontoqa", "entailmentbank"], required=True)
    p.add_argument("--model-key", default="qwen2.5-7b")
    p.add_argument("--layers", default="8,14,20")
    p.add_argument("--seed", type=int, default=20260309)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--class-weight-max-ratio", type=float, default=5.0)
    p.add_argument(
        "--head-loss-weights",
        default="",
        help="CSV key:value overrides for head weights. Keys: inference,chain_depth,truth,conclusion,premise,entity.",
    )
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-checkpoint", required=True)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    head_loss_weights = dict(DEFAULT_DOMAIN_HEAD_LOSS_WEIGHTS)
    head_loss_weights.update(_parse_head_loss_weights(str(args.head_loss_weights)))
    payload = load_json(args.paired_dataset)
    domain = str(args.decoder_domain).strip().lower()
    if domain not in LOGICAL_DOMAINS:
        raise RuntimeError(f"Unsupported logical decoder domain: {domain}")
    source_domain = str((payload.get("source") or {}).get("scope", "")).strip().lower()
    if source_domain and source_domain != domain:
        raise RuntimeError(f"Dataset domain mismatch: dataset={source_domain}, decoder_domain={domain}")
    vocab_manifest = dict(payload.get("decoder_vocab_manifest") or {})
    if str(vocab_manifest.get("decoder_domain", "")).strip().lower() != domain:
        raise RuntimeError(f"paired dataset missing matching decoder_vocab_manifest for domain={domain}")

    rows_path = payload.get("rows_path")
    if not rows_path:
        raise RuntimeError("paired dataset missing rows_path")
    rp = Path(str(rows_path))
    if not rp.is_absolute():
        cand = (Path(args.paired_dataset).parent / rp).resolve()
        rp = cand if cand.exists() else rp.resolve()
    rows = [r for r in list(load_pt(rp)) if isinstance(r, dict)]
    members = {str(m.get("member_id", "")): dict(m) for m in list(payload.get("members", []))}
    faithful_ids = {
        mid
        for mid, m in members.items()
        if mid
        and bool(m.get("label_defined", False))
        and (not bool(m.get("pair_ambiguous", False)))
        and str(m.get("gold_label", "")) == "faithful"
    }
    rows = [r for r in rows if str(r.get("member_id", "")) in faithful_ids]
    if len(rows) < 200:
        raise RuntimeError(f"Insufficient faithful rows for domain decoder training: {len(rows)}")

    layers = _parse_int_csv(str(args.layers))
    hidden_dim = int(rows[0]["raw_hidden"].shape[-1])
    cfg = OptionCDomainDecoderConfig(
        decoder_domain=domain,
        model_key=str(args.model_key),
        layers=layers,
        hidden_dim=int(hidden_dim),
        d_model=int(args.d_model),
        dropout=float(args.dropout),
        logical_chain_depth_bins=int(vocab_manifest.get("chain_depth_bins", 5)),
        class_vocab=tuple(str(x) for x in list(vocab_manifest.get("class_vocab", ["__unknown__"]))),
        entity_vocab=tuple(str(x) for x in list(vocab_manifest.get("entity_vocab", ["__unknown__"]))),
        inference_type_vocab=tuple(str(x) for x in list(vocab_manifest.get("inference_type_vocab", []))),
        truth_value_vocab=tuple(str(x) for x in list(vocab_manifest.get("truth_value_vocab", []))),
    )

    train_rows, val_rows = _split_rows_by_pair(rows, float(args.val_fraction), int(args.seed))
    train_ds = OptionCDomainDataset(train_rows, cfg)
    val_ds = OptionCDomainDataset(val_rows, cfg)
    train_dl = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_optionc_domain_batch)
    val_dl = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_optionc_domain_batch)

    model = OptionCDomainDecoder(cfg).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    class_weights = _build_class_weights(train_rows, cfg, device=str(args.device), max_ratio=float(args.class_weight_max_ratio))

    best_state = None
    best_epoch = -1
    best_metric = float("-inf")
    no_improve = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        seen = 0
        train_loss = 0.0
        for b in train_dl:
            dev = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(dev["x"])
            losses = compute_domain_loss(
                out,
                dev,
                class_weights=class_weights,
                head_loss_weights=head_loss_weights,
            )
            optim.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            bsz = int(dev["x"].shape[0])
            seen += bsz
            train_loss += float(losses["total"].item()) * bsz

        val = evaluate_domain_decoder(model, val_dl, str(args.device), head_loss_weights=head_loss_weights)
        composite = float(val["truth_macro_f1"]) + 0.5 * (float(val["conclusion_macro_f1"]) + float(val["premise_macro_f1"]))
        row = {
            "epoch": int(epoch),
            "train_loss_total": float(train_loss / max(1, seen)),
            "val": dict(val),
            "val_composite": float(composite),
        }
        history.append(row)
        print(
            f"[optionc_decoder:{domain}] epoch {epoch}/{int(args.epochs)} "
            f"train_total={row['train_loss_total']:.4f} "
            f"val_total={val['loss_total']:.4f} "
            f"truth_acc={val['truth_acc']:.4f} truth_f1={val['truth_macro_f1']:.4f} "
            f"conc_acc={val['conclusion_acc']:.4f} prem_acc={val['premise_acc']:.4f}"
        )
        if composite > best_metric:
            best_metric = float(composite)
            best_epoch = int(epoch)
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(args.early_stop_patience):
                print(f"[optionc_decoder:{domain}] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No best state captured for optionc domain decoder")
    model.load_state_dict(best_state)
    best_val = evaluate_domain_decoder(model, val_dl, str(args.device), head_loss_weights=head_loss_weights)
    save_optionc_domain_decoder_checkpoint(
        args.output_checkpoint,
        model=model,
        cfg=cfg,
        best_epoch=int(best_epoch),
        best_val=best_val,
        history=history,
        train_settings={
            "head_loss_weights": dict(head_loss_weights),
        },
    )
    result = {
        "schema_version": "phase7_optionc_domain_decoder_train_result_v1",
        "status": "ok",
        "decoder_domain": str(domain),
        "paired_dataset": str(args.paired_dataset),
        "rows_path": str(rp),
        "model_key": str(args.model_key),
        "experiment_config": cfg.to_dict(),
        "seed": int(args.seed),
        "num_rows_total": int(len(rows)),
        "num_rows_train": int(len(train_rows)),
        "num_rows_val": int(len(val_rows)),
        "best_epoch": int(best_epoch),
        "best_val": best_val,
        "head_loss_weights": dict(head_loss_weights),
        "checkpoint_path": str(args.output_checkpoint),
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output_json, result)
    print(f"Saved optionc domain decoder checkpoint -> {args.output_checkpoint}")
    print(f"Saved optionc domain decoder results    -> {args.output_json}")


if __name__ == "__main__":
    main()

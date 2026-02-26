#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_pt, save_json, set_seed
from state_decoder_core import (
    Phase7StateDataset,
    StateDecoderExperimentConfig,
    MultiHeadStateDecoder,
    collate_state_batch,
    compute_multitask_loss,
    default_state_decoder_configs,
    evaluate_state_model,
    make_custom_state_decoder_config,
    make_scheduler,
    numeric_stats_from_records,
    save_checkpoint,
    split_by_example,
)

try:
    from experiments.layer_sweep_manifest import get_layer_set, infer_layer_set_id_from_layers, load_manifest
except Exception:  # pragma: no cover
    get_layer_set = None
    infer_layer_set_id_from_layers = None
    load_manifest = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-train", default="phase7_results/dataset/gsm8k_step_traces_train.pt")
    p.add_argument("--config-name", default=None)
    p.add_argument("--all-configs", action="store_true")
    p.add_argument("--manifest", default=None, help="Layer sweep manifest JSON (for --layer-set-id / metadata)")
    p.add_argument("--layer-set-id", default=None, help="Layer-set ID from manifest for a custom single run")
    p.add_argument("--layers", type=int, nargs="+", default=None, help="Custom selected layers (alternative to --layer-set-id)")
    p.add_argument("--input-variant", choices=["raw", "sae", "hybrid"], default=None, help="Required with --layer-set-id/--layers")
    p.add_argument("--custom-config-name", default=None, help="Override auto-generated config name for custom runs")
    p.add_argument("--sweep-run-id", default=None, help="Optional sweep run ID metadata")
    p.add_argument("--parent-baseline", default=None, help="Optional baseline layer_set_id for comparisons")
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--checkpoints-dir", default="phase7_results/checkpoints")
    p.add_argument("--results-dir", default="phase7_results/results")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-records", type=int, default=None)
    return p.parse_args()


def _clone_cfg(cfg: StateDecoderExperimentConfig) -> StateDecoderExperimentConfig:
    return StateDecoderExperimentConfig.from_dict(cfg.to_dict())


def _load_manifest_payload(path: Optional[str]):
    if not path:
        return None
    if load_manifest is None:
        raise RuntimeError("Manifest support unavailable (experiments.layer_sweep_manifest import failed)")
    return load_manifest(path)


def _infer_manifest_row(manifest_payload, cfg: StateDecoderExperimentConfig):
    if manifest_payload is None or infer_layer_set_id_from_layers is None or get_layer_set is None:
        return None
    layer_set_id = infer_layer_set_id_from_layers(manifest_payload, cfg.layers)
    if layer_set_id is None:
        return None
    return get_layer_set(manifest_payload, layer_set_id)


def _build_sweep_metadata(*, args: argparse.Namespace, cfg: StateDecoderExperimentConfig, manifest_row: Optional[Dict]) -> Optional[Dict]:
    if not (args.sweep_run_id or manifest_row or args.layer_set_id or args.layers):
        return None
    row = dict(manifest_row or {})
    return {
        "schema_version": "layer_sweep_result_v1",
        "phase": "phase7",
        "input_variant": cfg.input_variant,
        "layers": list(cfg.layers),
        "num_layers": len(cfg.layers),
        "layer_set_id": row.get("layer_set_id", args.layer_set_id),
        "layer_set_family": row.get("family"),
        "layer_set_sweep_group": row.get("sweep_group"),
        "sweep_run_id": args.sweep_run_id,
        "seed": int(cfg.seed),
        "parent_baseline": args.parent_baseline,
        "manifest_path": str(args.manifest) if args.manifest else None,
    }


def _resolve_selected_configs(args: argparse.Namespace) -> Tuple[List[StateDecoderExperimentConfig], Optional[Dict]]:
    custom_requested = any([args.layer_set_id, args.layers, args.input_variant])
    if args.config_name and custom_requested:
        raise ValueError("Use either preset --config-name or custom layer selection (--manifest/--layer-set-id/--layers), not both")
    if args.all_configs and custom_requested:
        raise ValueError("Use either --all-configs or custom layer selection, not both")

    manifest_payload = _load_manifest_payload(args.manifest)
    if args.layer_set_id:
        if manifest_payload is None or get_layer_set is None:
            raise ValueError("--layer-set-id requires --manifest with a valid layer sweep manifest")
        if not args.input_variant:
            raise ValueError("--layer-set-id requires --input-variant")
        row = get_layer_set(manifest_payload, args.layer_set_id)
        cfg_name = args.custom_config_name or f"state_{args.input_variant}_{row['layer_set_id']}"
        cfg = make_custom_state_decoder_config(input_variant=args.input_variant, layers=row["layers"], name=cfg_name)
        if args.seed is not None:
            cfg.seed = int(args.seed)
        return [cfg], manifest_payload

    if args.layers:
        if not args.input_variant:
            raise ValueError("--layers requires --input-variant")
        cfg = make_custom_state_decoder_config(input_variant=args.input_variant, layers=args.layers, name=args.custom_config_name)
        if args.seed is not None:
            cfg.seed = int(args.seed)
        return [cfg], manifest_payload

    cfgs = default_state_decoder_configs()
    if args.config_name:
        if args.config_name not in cfgs:
            raise KeyError(f"Unknown config {args.config_name}; available={sorted(cfgs)}")
        selected = [_clone_cfg(cfgs[args.config_name])]
    else:
        selected = [_clone_cfg(cfgs[k]) for k in sorted(cfgs)]
    if args.seed is not None:
        for cfg in selected:
            cfg.seed = int(args.seed)
    return selected, manifest_payload


def train_one(
    cfg: StateDecoderExperimentConfig,
    records: List[dict],
    args: argparse.Namespace,
    manifest_payload=None,
) -> Dict:
    set_seed(cfg.seed)
    train_records, val_records = split_by_example(records, cfg.val_fraction, cfg.seed)
    numeric_stats = numeric_stats_from_records(train_records)

    train_ds = Phase7StateDataset(train_records, cfg, numeric_stats)
    val_ds = Phase7StateDataset(val_records, cfg, numeric_stats)
    batch_size = args.batch_size or cfg.batch_size
    epochs = args.epochs or cfg.epochs
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_state_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_state_batch)

    model = MultiHeadStateDecoder(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, epochs * max(1, len(train_loader)), cfg.warmup_steps)

    best_state = None
    best_epoch = -1
    best_key = None
    no_improve = 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        seen = 0
        train_loss = 0.0
        train_loss_result = 0.0
        for batch in train_loader:
            x = batch["x"].to(args.device)
            dev_batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(x)
            losses = compute_multitask_loss(out, dev_batch, cfg)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            bsz = x.shape[0]
            seen += bsz
            train_loss += float(losses["total"].item()) * bsz
            train_loss_result += float(losses["result_token"].item()) * bsz

        val_metrics = evaluate_state_model(model, val_loader, args.device, numeric_stats)
        row = {
            "epoch": epoch,
            "train_loss_total": train_loss / max(1, seen),
            "train_loss_result_token": train_loss_result / max(1, seen),
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[{cfg.name}] epoch {epoch}/{epochs} "
            f"train_total={row['train_loss_total']:.4f} "
            f"val_total={val_metrics['loss_total']:.4f} "
            f"res_top1={val_metrics['result_token_top1']:.4f} "
            f"op_acc={val_metrics['operator_acc']:.4f} "
            f"step_acc={val_metrics['step_type_acc']:.4f}"
        )

        val_key = (
            float(val_metrics["loss_total"]),
            -float(val_metrics["result_token_top1"]),
            -float(val_metrics["operator_acc"]),
        )
        if best_key is None or val_key < best_key:
            best_key = val_key
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"[{cfg.name}] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"No best model captured for {cfg.name}")
    model.load_state_dict(best_state)
    best_val_metrics = evaluate_state_model(model, val_loader, args.device, numeric_stats)
    manifest_row = _infer_manifest_row(manifest_payload, cfg)
    sweep_metadata = _build_sweep_metadata(args=args, cfg=cfg, manifest_row=manifest_row)

    ckpt_path = Path(args.checkpoints_dir) / f"{cfg.name}.pt"
    save_checkpoint(
        ckpt_path,
        {
            "stage": "phase7_state_decoder_supervised",
            "experiment_config": cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "backbone_model_config": model.backbone_cfg.to_dict(),
            "numeric_stats": {k: v.to_dict() for k, v in numeric_stats.items()},
            "best_epoch": best_epoch,
            "history": history,
            "sweep_metadata": sweep_metadata,
        },
    )

    result = {
        "config_name": cfg.name,
        "stage": "phase7_state_decoder_supervised",
        "dataset_train_path": str(args.dataset_train),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_train_examples": len({int(r['example_idx']) for r in train_records}),
        "num_val_examples": len({int(r['example_idx']) for r in val_records}),
        "best_epoch": best_epoch,
        "best_val": best_val_metrics,
        "numeric_stats": {k: v.to_dict() for k, v in numeric_stats.items()},
        "experiment_config": cfg.to_dict(),
        "checkpoint_path": str(ckpt_path),
        "history": history,
        "schema_version": "phase7_state_decoder_train_result_v1",
    }
    if sweep_metadata is not None:
        result["sweep_metadata"] = sweep_metadata
        result.update(sweep_metadata)
    out_path = Path(args.results_dir) / f"state_decoder_supervised_{cfg.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, result)
    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Saved results    -> {out_path}")
    return result


def main() -> None:
    args = parse_args()
    if args.config_name and args.all_configs:
        raise ValueError("Use either --config-name or --all-configs (or neither), not both")
    selected, manifest_payload = _resolve_selected_configs(args)

    records = load_pt(args.dataset_train)
    if args.max_records is not None:
        records = records[: args.max_records]
    records = [r for r in records if r.get("gsm8k_split") == "train"] or records
    print(f"Loaded {len(records)} trace step records from {args.dataset_train}")

    summaries = [train_one(cfg, records, args, manifest_payload=manifest_payload) for cfg in selected]
    if len(summaries) > 1:
        save_json(Path(args.results_dir) / "state_decoder_supervised_comparison_partial.json", summaries)


if __name__ == "__main__":
    main()

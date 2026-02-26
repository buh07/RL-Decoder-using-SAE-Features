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
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from decoder_config import DecoderExperimentConfig, default_experiment_configs, make_custom_config  # noqa: E402
from decoder_model import ArithmeticDecoder, ArithmeticDecoderConfig  # noqa: E402
from pipeline_utils import (  # noqa: E402
    Phase6RecordDataset,
    collate_record_items,
    evaluate_batches,
    load_records,
    set_seed,
    split_records_by_example,
    verify_schema,
)

try:  # noqa: E402
    from experiments.layer_sweep_manifest import (
        get_layer_set,
        infer_layer_set_id_from_layers,
        load_manifest,
    )
except Exception:  # pragma: no cover - optional in non-sweep usage
    get_layer_set = None
    infer_layer_set_id_from_layers = None
    load_manifest = None


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


def evaluate_with_loss(model, loader, device: str) -> Dict:
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_n += int(y.numel())

    metrics = evaluate_batches(model, loader, device)
    metrics["cross_entropy_loss"] = (total_loss / max(1, total_n))
    return metrics


def _clone_cfg(cfg: DecoderExperimentConfig) -> DecoderExperimentConfig:
    return DecoderExperimentConfig.from_dict(cfg.to_dict())


def _load_manifest_payload(path: Optional[str]):
    if not path:
        return None
    if load_manifest is None:
        raise RuntimeError("Manifest support unavailable (experiments.layer_sweep_manifest import failed)")
    return load_manifest(path)


def _infer_manifest_row(manifest_payload, cfg: DecoderExperimentConfig):
    if manifest_payload is None or infer_layer_set_id_from_layers is None or get_layer_set is None:
        return None
    layer_set_id = infer_layer_set_id_from_layers(manifest_payload, cfg.layers)
    if layer_set_id is None:
        return None
    return get_layer_set(manifest_payload, layer_set_id)


def _build_sweep_metadata(
    *,
    args: argparse.Namespace,
    cfg: DecoderExperimentConfig,
    manifest_row: Optional[Dict],
) -> Optional[Dict]:
    if not (args.sweep_run_id or manifest_row or args.layer_set_id or args.layers):
        return None
    row = dict(manifest_row or {})
    return {
        "schema_version": "layer_sweep_result_v1",
        "phase": "phase6",
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


def _resolve_selected_configs(args: argparse.Namespace) -> Tuple[List[DecoderExperimentConfig], Optional[Dict]]:
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
        cfg_name = args.custom_config_name or f"{args.input_variant}_{row['layer_set_id']}"
        cfg = make_custom_config(input_variant=args.input_variant, layers=row["layers"], name=cfg_name)
        if args.seed is not None:
            cfg.seed = int(args.seed)
        return [cfg], manifest_payload

    if args.layers:
        if not args.input_variant:
            raise ValueError("--layers requires --input-variant")
        cfg = make_custom_config(input_variant=args.input_variant, layers=args.layers, name=args.custom_config_name)
        if args.seed is not None:
            cfg.seed = int(args.seed)
        return [cfg], manifest_payload

    cfgs = default_experiment_configs()
    if args.config_name:
        if args.config_name not in cfgs:
            raise KeyError(f"Unknown config '{args.config_name}'. Available: {sorted(cfgs)}")
        selected = [_clone_cfg(cfgs[args.config_name])]
    else:
        # `--all-configs` is redundant with the default but kept as an explicit CLI affordance.
        selected = [_clone_cfg(cfgs[name]) for name in sorted(cfgs)]
    if args.seed is not None:
        for cfg in selected:
            cfg.seed = int(args.seed)
    return selected, manifest_payload


def train_one_config(
    cfg: DecoderExperimentConfig,
    records: List[dict],
    args: argparse.Namespace,
    manifest_payload=None,
) -> Dict:
    set_seed(cfg.seed)
    train_records, val_records = split_records_by_example(records, cfg.val_fraction, cfg.seed)
    if not train_records or not val_records:
        raise RuntimeError(f"Train/val split failed for {cfg.name}: train={len(train_records)} val={len(val_records)}")

    train_ds = Phase6RecordDataset(train_records, cfg)
    val_ds = Phase6RecordDataset(val_records, cfg)

    batch_size = args.batch_size or cfg.batch_size
    epochs = args.epochs or cfg.epochs

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_record_items,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_record_items,
    )

    model_cfg = ArithmeticDecoderConfig(
        input_dim=cfg.input_dim(),
        n_layers_input=len(cfg.layers),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_decoder_layers=cfg.n_decoder_layers,
        vocab_size=cfg.vocab_size,
        dropout=cfg.dropout,
        aggregator=cfg.aggregator,
        use_sparse_input=(cfg.input_variant == "sae"),
    )
    model = ArithmeticDecoder(model_cfg).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = epochs * max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_steps=cfg.warmup_steps)

    best_state = None
    best_val = None
    best_epoch = -1
    no_improve_epochs = 0
    history: List[Dict] = []

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["x"].to(args.device)
            y = batch["y"].to(args.device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            running_loss += float(loss.item()) * int(y.numel())
            seen += int(y.numel())

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate_with_loss(model, val_loader, args.device)
        epoch_row = {
            "epoch": epoch,
            "train_cross_entropy_loss": train_loss,
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        print(
            f"[{cfg.name}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['cross_entropy_loss']:.4f} "
            f"val_top1={val_metrics['top1_accuracy']:.4f} val_top5={val_metrics['top5_accuracy']:.4f}"
        )

        val_key = (val_metrics["cross_entropy_loss"], -val_metrics["top1_accuracy"])
        if best_val is None or val_key < best_val:
            best_val = val_key
            best_epoch = epoch
            no_improve_epochs = 0
            best_state = deepcopy(model.state_dict())
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= cfg.early_stop_patience:
            print(f"[{cfg.name}] early stopping at epoch {epoch} (patience={cfg.early_stop_patience})")
            break

    if best_state is None:
        raise RuntimeError(f"No model state captured for {cfg.name}")

    model.load_state_dict(best_state)
    best_val_metrics = evaluate_with_loss(model, val_loader, args.device)
    manifest_row = _infer_manifest_row(manifest_payload, cfg)
    sweep_metadata = _build_sweep_metadata(args=args, cfg=cfg, manifest_row=manifest_row)

    ckpt_path = Path(args.checkpoints_dir) / f"{cfg.name}_supervised.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "supervised",
            "experiment_config": cfg.to_dict(),
            "model_config": model_cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "history": history,
            "sweep_metadata": sweep_metadata,
        },
        ckpt_path,
    )

    result = {
        "config_name": cfg.name,
        "stage": "supervised",
        "dataset_train_path": str(args.dataset_train),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_train_examples": len({int(r["example_idx"]) for r in train_records}),
        "num_val_examples": len({int(r["example_idx"]) for r in val_records}),
        "best_epoch": best_epoch,
        "best_val": best_val_metrics,
        "checkpoint_path": str(ckpt_path),
        "experiment_config": cfg.to_dict(),
        "model_config": model_cfg.to_dict(),
        "history": history,
        "schema_version": "phase6_supervised_result_v1",
    }
    if sweep_metadata is not None:
        result["sweep_metadata"] = sweep_metadata
        result.update(sweep_metadata)

    results_path = Path(args.results_dir) / f"supervised_{cfg.name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{cfg.name}] saved checkpoint -> {ckpt_path}")
    print(f"[{cfg.name}] saved results    -> {results_path}")

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-train", default="phase6_results/dataset/gsm8k_expanded_train.pt")
    p.add_argument("--config-name", default=None, help="Run only one named config")
    p.add_argument("--all-configs", action="store_true", help="Run all default configs (default if no config-name)")
    p.add_argument("--manifest", default=None, help="Layer sweep manifest JSON (for --layer-set-id / metadata)")
    p.add_argument("--layer-set-id", default=None, help="Layer-set ID from manifest for a custom single run")
    p.add_argument("--layers", type=int, nargs="+", default=None, help="Custom selected layers (alternative to --layer-set-id)")
    p.add_argument("--input-variant", choices=["raw", "sae", "hybrid"], default=None, help="Required with --layer-set-id/--layers")
    p.add_argument("--custom-config-name", default=None, help="Override auto-generated config name for custom runs")
    p.add_argument("--sweep-run-id", default=None, help="Optional run ID propagated into sweep metadata")
    p.add_argument("--parent-baseline", default=None, help="Optional baseline layer_set_id for sweep comparisons")
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--checkpoints-dir", default="phase6_results/checkpoints")
    p.add_argument("--results-dir", default="phase6_results/results")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-records", type=int, default=None, help="Limit records (smoke tests)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.config_name and args.all_configs:
        raise ValueError("Use either --config-name or --all-configs (or neither), not both")
    selected, manifest_payload = _resolve_selected_configs(args)

    records = load_records(args.dataset_train, max_records=args.max_records)
    verify_schema(records)
    train_only_records = [r for r in records if r.get("gsm8k_split") == "train"]
    if train_only_records:
        records = train_only_records
    print(f"Loaded {len(records)} records from {args.dataset_train}")

    all_results = []
    for cfg in selected:
        all_results.append(train_one_config(cfg, records, args, manifest_payload=manifest_payload))

    if len(all_results) > 1:
        summary_path = Path(args.results_dir) / "supervised_comparison_partial.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved multi-config summary -> {summary_path}")


if __name__ == "__main__":
    main()

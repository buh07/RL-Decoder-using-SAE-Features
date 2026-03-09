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

try:  # pragma: no cover
    from .common import load_pt, save_json, set_seed
    from .model_registry import resolve_model_spec
    from .state_decoder_core import (
        OP_TO_ID,
        Phase7StateDataset,
        StateDecoderExperimentConfig,
        MultiHeadStateDecoder,
        apply_model_metadata_to_config,
        collate_state_batch,
        compute_multitask_loss,
        dataloader_perf_kwargs,
        default_state_decoder_configs,
        evaluate_state_model,
        make_custom_state_decoder_config,
        make_scheduler,
        move_batch_to_device,
        numeric_stats_from_records,
        save_checkpoint,
        split_by_example,
    )
except ImportError:  # pragma: no cover
    from common import load_pt, save_json, set_seed
    from model_registry import resolve_model_spec
    from state_decoder_core import (
        OP_TO_ID,
        Phase7StateDataset,
        StateDecoderExperimentConfig,
        MultiHeadStateDecoder,
        apply_model_metadata_to_config,
        collate_state_batch,
        compute_multitask_loss,
        dataloader_perf_kwargs,
        default_state_decoder_configs,
        evaluate_state_model,
        make_custom_state_decoder_config,
        make_scheduler,
        move_batch_to_device,
        numeric_stats_from_records,
        save_checkpoint,
        split_by_example,
    )

try:
    from experiments.layer_sweep_manifest import get_layer_set, infer_layer_set_id_from_layers, load_manifest
except ImportError:  # pragma: no cover
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
    p.add_argument("--input-variant", choices=["raw", "sae", "hybrid", "hybrid_indexed"], default=None, help="Required with --layer-set-id/--layers")
    p.add_argument("--custom-config-name", default=None, help="Override auto-generated config name for custom runs")
    p.add_argument("--sweep-run-id", default=None, help="Optional sweep run ID metadata")
    p.add_argument("--parent-baseline", default=None, help="Optional baseline layer_set_id for comparisons")
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--model-key", default="gpt2-medium")
    p.add_argument("--adapter-config", default=None, help="Optional JSON overrides for model registry entry")
    p.add_argument("--vocab-size", type=int, default=None, help="Optional vocab size override for decoder token head")
    p.add_argument("--checkpoints-dir", default="phase7_results/checkpoints")
    p.add_argument("--results-dir", default="phase7_results/results")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None, help="Override config learning rate")
    p.add_argument("--early-stop-patience", type=int, default=None, help="Override config early-stop patience")
    p.add_argument("--d-model", type=int, default=None, help="Override decoder d_model")
    p.add_argument("--n-heads", type=int, default=None, help="Override decoder attention heads")
    p.add_argument("--n-decoder-layers", type=int, default=None, help="Override decoder layer depth")
    p.add_argument("--raw-anchor-mode", choices=["eq_only", "multi_anchor"], default=None, help="Raw-hidden anchor mode.")
    p.add_argument("--label-smoothing-cls", type=float, default=None, help="Override label smoothing for classification heads")
    p.add_argument(
        "--operator-loss-mode",
        choices=["ce", "weighted_ce", "focal"],
        default=None,
        help="Override operator loss mode (ce|weighted_ce|focal).",
    )
    p.add_argument(
        "--operator-focal-gamma",
        type=float,
        default=None,
        help="Override focal gamma for operator head when --operator-loss-mode=focal.",
    )
    p.add_argument(
        "--operator-class-weight-scope",
        choices=["all_steps", "operate_only", "operate_known_only"],
        default=None,
        help="Rows used to compute operator class weights when weighted/focal loss is enabled.",
    )
    p.add_argument(
        "--operator-class-weight-max-ratio",
        type=float,
        default=None,
        help="Optional cap for max/min positive operator class weight ratio.",
    )
    p.add_argument(
        "--operator-operate-only-supervision",
        action="store_true",
        help="Train operator head only on rows where step_type=='operate'.",
    )
    p.add_argument(
        "--operator-known-only-supervision",
        action="store_true",
        help="Train operator head only on operate rows with known operator labels (exclude unknown).",
    )
    p.add_argument("--loss-w-operator", type=float, default=None, help="Override operator loss weight")
    p.add_argument("--loss-w-step-type", type=float, default=None, help="Override step_type loss weight")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--non-blocking-transfer", action="store_true")
    p.add_argument("--torch-num-threads", type=int, default=None)
    p.add_argument("--cache-inputs", choices=["off", "auto", "on"], default="off")
    p.add_argument("--cache-max-gb", type=float, default=2.0)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument(
        "--allow-missing-split-field",
        action="store_true",
        help="Unsafe legacy mode: allow datasets without gsm8k_split and use all records.",
    )
    p.add_argument(
        "--early-stop-metric",
        choices=["result_top1", "loss_total", "composite", "operator_acc_operate_known_only"],
        default="result_top1",
        help="Metric priority used for best-checkpoint selection/early stopping.",
    )
    p.add_argument(
        "--operator-early-stop-guard",
        action="store_true",
        help=(
            "Do not early-stop while operator_acc is still improving. "
            "Useful when result-token loss plateaus before operator head converges."
        ),
    )
    p.add_argument(
        "--operator-guard-min-delta",
        type=float,
        default=0.002,
        help="Minimum operator_acc improvement to reset early-stop counter when --operator-early-stop-guard is set.",
    )
    p.add_argument(
        "--improvement-profile",
        choices=["none", "d1_tier1", "d2_tier2", "d3_operate_known"],
        default="none",
        help=(
            "Apply pre-registered decoder improvement profile. "
            "d1_tier1: layers 4-7, longer training, lower LR, label smoothing, stronger operator/step losses. "
            "d2_tier2: d1 + larger model capacity. "
            "d3_operate_known: operate-known supervision + capped weighted CE + operator-focused early-stop."
        ),
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable torch deterministic algorithms for stronger reproducibility guarantees.",
    )
    p.add_argument(
        "--allow-mixed-schema",
        action="store_true",
        help="Unsafe compatibility mode: allow mixed phase7 trace schema versions in one training run.",
    )
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
        "model_key": cfg.model_key,
        "model_family": cfg.model_family,
        "num_layers_total": int(cfg.model_num_layers),
        "hidden_dim": int(cfg.model_hidden_dim),
        "tokenizer_id": cfg.tokenizer_id,
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


def _filter_records_by_split(records: List[dict], split: str, allow_missing_split_field: bool) -> List[dict]:
    has_split_field = any("gsm8k_split" in r for r in records)
    if not has_split_field:
        if allow_missing_split_field:
            return list(records)
        raise RuntimeError(
            f"Dataset is missing gsm8k_split field; refusing to continue for split={split!r}. "
            "Use --allow-missing-split-field only for legacy compatibility."
        )
    filtered = [r for r in records if r.get("gsm8k_split") == split]
    if not filtered:
        raise RuntimeError(
            f"No records found for split={split!r}. "
            "Refusing silent fallback to all records to avoid leakage."
        )
    return filtered


def _normalize_schema_version(rec: dict) -> str:
    v = rec.get("schema_version")
    if v is None:
        return "phase7_trace_v1"
    return str(v)


def _validate_schema_versions(records: List[dict], allow_mixed_schema: bool) -> str:
    versions = sorted({_normalize_schema_version(r) for r in records})
    if not versions:
        raise RuntimeError("No records available after split filtering")
    allowed = {"phase7_trace_v1", "phase7_trace_v2"}
    bad = [v for v in versions if v not in allowed]
    if bad:
        raise RuntimeError(f"Unsupported schema versions in dataset: {bad}; supported={sorted(allowed)}")
    if len(versions) > 1 and not allow_mixed_schema:
        raise RuntimeError(
            f"Mixed schema versions detected: {versions}. "
            "Use --allow-mixed-schema only if you intentionally accept mixed ontology semantics."
        )
    return versions[0]


def _resolve_selected_configs(
    args: argparse.Namespace,
    *,
    model_num_layers: int,
) -> Tuple[List[StateDecoderExperimentConfig], Optional[Dict]]:
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
        cfg = make_custom_state_decoder_config(
            input_variant=args.input_variant,
            layers=row["layers"],
            name=cfg_name,
            layers_total=int(model_num_layers),
        )
        if args.seed is not None:
            cfg.seed = int(args.seed)
        return [cfg], manifest_payload

    if args.layers:
        if not args.input_variant:
            raise ValueError("--layers requires --input-variant")
        cfg = make_custom_state_decoder_config(
            input_variant=args.input_variant,
            layers=args.layers,
            name=args.custom_config_name,
            layers_total=int(model_num_layers),
        )
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


def _apply_user_overrides(cfg: StateDecoderExperimentConfig, args: argparse.Namespace) -> StateDecoderExperimentConfig:
    updated = StateDecoderExperimentConfig.from_dict(cfg.to_dict())
    if args.epochs is not None:
        updated.epochs = int(args.epochs)
    if args.batch_size is not None:
        updated.batch_size = int(args.batch_size)
    if args.learning_rate is not None:
        updated.learning_rate = float(args.learning_rate)
    if args.early_stop_patience is not None:
        updated.early_stop_patience = int(args.early_stop_patience)
    if args.d_model is not None:
        updated.d_model = int(args.d_model)
    if args.n_heads is not None:
        updated.n_heads = int(args.n_heads)
    if args.n_decoder_layers is not None:
        updated.n_decoder_layers = int(args.n_decoder_layers)
    if args.raw_anchor_mode is not None:
        updated.raw_anchor_mode = str(args.raw_anchor_mode)
    if args.label_smoothing_cls is not None:
        updated.label_smoothing_cls = float(args.label_smoothing_cls)
    if args.operator_loss_mode is not None:
        updated.operator_loss_mode = str(args.operator_loss_mode)
    if args.operator_focal_gamma is not None:
        updated.operator_focal_gamma = float(args.operator_focal_gamma)
    if args.operator_class_weight_scope is not None:
        updated.operator_class_weight_scope = str(args.operator_class_weight_scope)
    if args.operator_class_weight_max_ratio is not None:
        updated.operator_class_weight_max_ratio = float(args.operator_class_weight_max_ratio)
    if args.operator_operate_only_supervision:
        updated.operator_operate_only_supervision = True
    if args.operator_known_only_supervision:
        updated.operator_known_only_supervision = True
    if args.loss_w_operator is not None:
        updated.loss_w_operator = float(args.loss_w_operator)
    if args.loss_w_step_type is not None:
        updated.loss_w_step_type = float(args.loss_w_step_type)
    return updated


def _apply_improvement_profile(cfg: StateDecoderExperimentConfig, profile: str) -> StateDecoderExperimentConfig:
    updated = StateDecoderExperimentConfig.from_dict(cfg.to_dict())
    if profile == "none":
        return updated
    if profile == "d1_tier1":
        updated.layers = (4, 5, 6, 7)
        updated.name = f"{updated.name}_d1tier1"
        updated.epochs = 150
        updated.early_stop_patience = 20
        updated.learning_rate = 5e-5
        updated.label_smoothing_cls = 0.10
        updated.loss_w_operator = 2.0
        updated.loss_w_step_type = 1.0
        return updated
    if profile == "d2_tier2":
        updated.layers = (0, 1, 2, 3, 4, 5, 6, 7)
        updated.name = f"{updated.name}_d2tier2"
        updated.epochs = 150
        updated.early_stop_patience = 20
        updated.learning_rate = 5e-5
        updated.label_smoothing_cls = 0.10
        updated.loss_w_operator = 2.0
        updated.loss_w_step_type = 1.0
        updated.d_model = 384
        updated.n_heads = 8
        updated.n_decoder_layers = 3
        updated.operator_operate_only_supervision = True
        updated.operator_loss_mode = "weighted_ce"
        updated.operator_class_weight_scope = "operate_only"
        updated.operator_focal_gamma = 2.0
        return updated
    if profile == "d3_operate_known":
        updated.layers = (0, 1, 2, 3, 4, 5, 6, 7)
        updated.name = f"{updated.name}_d3known"
        updated.epochs = 150
        updated.early_stop_patience = 20
        updated.learning_rate = 5e-5
        updated.label_smoothing_cls = 0.10
        updated.loss_w_operator = 2.0
        updated.loss_w_step_type = 1.0
        updated.d_model = 384
        updated.n_heads = 8
        updated.n_decoder_layers = 3
        updated.operator_operate_only_supervision = True
        updated.operator_known_only_supervision = True
        updated.operator_loss_mode = "weighted_ce"
        updated.operator_class_weight_scope = "operate_known_only"
        updated.operator_class_weight_max_ratio = 5.0
        updated.operator_focal_gamma = 2.0
        return updated
    raise ValueError(f"Unsupported improvement profile: {profile!r}")


def _compute_operator_class_weights(
    records: List[dict],
    cfg: StateDecoderExperimentConfig,
    device: str,
) -> Optional[torch.Tensor]:
    mode = str(cfg.operator_loss_mode or "ce")
    if mode not in {"weighted_ce", "focal"}:
        return None
    scope = str(cfg.operator_class_weight_scope or "all_steps")
    counts = [0.0 for _ in range(len(OP_TO_ID))]
    unknown_id = int(OP_TO_ID.get("unknown", len(OP_TO_ID) - 1))
    for r in records:
        s = r.get("structured_state") or {}
        if scope == "operate_only" and str(s.get("step_type", "")) != "operate":
            continue
        if scope == "operate_known_only" and (
            str(s.get("step_type", "")) != "operate" or str(s.get("operator", "unknown")) == "unknown"
        ):
            continue
        op = str(s.get("operator", "unknown"))
        idx = int(OP_TO_ID.get(op, OP_TO_ID["unknown"]))
        counts[idx] += 1.0
    inv = [0.0 if c <= 0.0 else 1.0 / c for c in counts]
    nz = [v for v in inv if v > 0.0]
    if not nz:
        return None
    mean_inv = sum(nz) / float(len(nz))
    weights = [(v / mean_inv) if v > 0.0 else 0.0 for v in inv]
    if scope == "operate_known_only":
        weights[unknown_id] = 0.0
    ratio_cap = cfg.operator_class_weight_max_ratio
    if ratio_cap is not None and math.isfinite(float(ratio_cap)) and float(ratio_cap) > 1.0:
        pos = [w for w in weights if w > 0.0]
        if pos:
            min_w = min(pos)
            max_w = min_w * float(ratio_cap)
            weights = [min(max_w, w) if w > 0.0 else 0.0 for w in weights]
            pos2 = [w for w in weights if w > 0.0]
            if pos2:
                mean2 = sum(pos2) / float(len(pos2))
                weights = [(w / mean2) if w > 0.0 else 0.0 for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one(
    cfg: StateDecoderExperimentConfig,
    records: List[dict],
    args: argparse.Namespace,
    manifest_payload=None,
) -> Dict:
    set_seed(cfg.seed)
    train_records, val_records = split_by_example(records, cfg.val_fraction, cfg.seed)
    numeric_stats = numeric_stats_from_records(train_records)
    train_shuffle_generator = torch.Generator()
    train_shuffle_generator.manual_seed(int(cfg.seed))

    train_ds = Phase7StateDataset(
        train_records,
        cfg,
        numeric_stats,
        cache_inputs=args.cache_inputs,
        cache_max_gb=args.cache_max_gb,
    )
    val_ds = Phase7StateDataset(
        val_records,
        cfg,
        numeric_stats,
        cache_inputs=args.cache_inputs,
        cache_max_gb=args.cache_max_gb,
    )
    batch_size = args.batch_size or cfg.batch_size
    epochs = args.epochs or cfg.epochs
    operator_class_weights = _compute_operator_class_weights(train_records, cfg, args.device)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_state_batch,
        generator=train_shuffle_generator,
        **dataloader_perf_kwargs(
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_state_batch,
        **dataloader_perf_kwargs(
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        ),
    )

    # Keep model-init RNG stable across runtime DataLoader settings (e.g., num_workers).
    set_seed(cfg.seed)
    model = MultiHeadStateDecoder(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, epochs * max(1, len(train_loader)), cfg.warmup_steps)
    operator_guard_enabled = bool(
        args.operator_early_stop_guard or args.improvement_profile in {"d1_tier1", "d2_tier2", "d3_operate_known"}
    )
    effective_early_stop_metric = str(args.early_stop_metric)
    if args.improvement_profile == "d3_operate_known" and effective_early_stop_metric == "result_top1":
        effective_early_stop_metric = "operator_acc_operate_known_only"

    best_state = None
    best_epoch = -1
    best_key = None
    best_operator_acc = float("-inf")
    no_improve = 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        seen = 0
        train_loss = 0.0
        train_loss_result = 0.0
        for batch in train_loader:
            dev_batch = move_batch_to_device(batch, args.device, non_blocking=args.non_blocking_transfer)
            x = dev_batch["x"]
            out = model(x)
            losses = compute_multitask_loss(
                out,
                dev_batch,
                cfg,
                operator_class_weights=operator_class_weights,
            )
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

        val_metrics = evaluate_state_model(
            model,
            val_loader,
            args.device,
            numeric_stats,
            cfg_for_loss=cfg,
            operator_class_weights=operator_class_weights,
            non_blocking_transfer=args.non_blocking_transfer,
        )
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
            f"op_acc_all={val_metrics['operator_acc']:.4f} "
            f"op_acc_operate={float(val_metrics.get('operator_acc_operate_only') or 0.0):.4f} "
            f"op_acc_operate_known={float(val_metrics.get('operator_acc_operate_known_only') or 0.0):.4f} "
            f"step_acc={val_metrics['step_type_acc']:.4f} "
            f"mag_acc={val_metrics['magnitude_acc']:.4f} "
            f"sign_acc={val_metrics['sign_acc']:.4f}"
        )

        operator_metric = val_metrics.get("operator_acc_operate_known_only")
        if operator_metric is None:
            operator_metric = val_metrics.get("operator_acc_operate_only")
        if operator_metric is None:
            operator_metric = val_metrics.get("operator_acc", 0.0)
        operator_metric = float(operator_metric)

        if effective_early_stop_metric == "loss_total":
            val_key = (
                float(val_metrics["loss_total"]),
                -float(val_metrics["result_token_top1"]),
                -operator_metric,
            )
        elif effective_early_stop_metric == "composite":
            comp = (
                float(val_metrics["result_token_top1"])
                + operator_metric
                + float(val_metrics["step_type_acc"])
            ) / 3.0
            val_key = (-comp, float(val_metrics["loss_total"]), -float(val_metrics["result_token_top1"]))
        elif effective_early_stop_metric == "operator_acc_operate_known_only":
            op_known = val_metrics.get("operator_acc_operate_known_only")
            op_known = float(op_known if op_known is not None else operator_metric)
            val_key = (-op_known, float(val_metrics["loss_total"]), -float(val_metrics["result_token_top1"]))
        else:
            val_key = (
                -float(val_metrics["result_token_top1"]),
                float(val_metrics["loss_total"]),
                -operator_metric,
            )
        if best_key is None or val_key < best_key:
            best_key = val_key
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            best_operator_acc = max(best_operator_acc, operator_metric)
        else:
            operator_acc = operator_metric
            guard_delta = float(max(0.0, args.operator_guard_min_delta))
            if operator_guard_enabled and operator_acc >= (best_operator_acc + guard_delta):
                best_operator_acc = operator_acc
                no_improve = 0
                print(
                    f"[{cfg.name}] operator guard kept training at epoch {epoch}: "
                    f"operator metric improved to {operator_acc:.4f} (+{guard_delta:.4f} min)"
                )
                continue
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"[{cfg.name}] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"No best model captured for {cfg.name}")
    model.load_state_dict(best_state)
    best_val_metrics = evaluate_state_model(
        model,
        val_loader,
        args.device,
        numeric_stats,
        cfg_for_loss=cfg,
        operator_class_weights=operator_class_weights,
        non_blocking_transfer=args.non_blocking_transfer,
    )
    manifest_row = _infer_manifest_row(manifest_payload, cfg)
    sweep_metadata = _build_sweep_metadata(args=args, cfg=cfg, manifest_row=manifest_row)

    ckpt_path = Path(args.checkpoints_dir) / f"{cfg.name}.pt"
    save_checkpoint(
        ckpt_path,
        {
            "stage": "phase7_state_decoder_supervised",
            "model_metadata": {
                "model_key": cfg.model_key,
                "model_family": cfg.model_family,
                "num_layers": int(cfg.model_num_layers),
                "hidden_dim": int(cfg.model_hidden_dim),
                "tokenizer_id": cfg.tokenizer_id,
                "vocab_size": int(cfg.vocab_size),
            },
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
        "model_key": cfg.model_key,
        "model_family": cfg.model_family,
        "model_num_layers": int(cfg.model_num_layers),
        "model_hidden_dim": int(cfg.model_hidden_dim),
        "tokenizer_id": cfg.tokenizer_id,
        "vocab_size": int(cfg.vocab_size),
        "dataset_train_path": str(args.dataset_train),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_train_examples": len({int(r['example_idx']) for r in train_records}),
        "num_val_examples": len({int(r['example_idx']) for r in val_records}),
        "best_epoch": best_epoch,
        "best_val": best_val_metrics,
        "numeric_stats": {k: v.to_dict() for k, v in numeric_stats.items()},
        "experiment_config": cfg.to_dict(),
        "early_stop_metric": effective_early_stop_metric,
        "operator_early_stop_guard": bool(operator_guard_enabled),
        "operator_guard_min_delta": float(args.operator_guard_min_delta),
        "operator_loss_mode": str(cfg.operator_loss_mode),
        "operator_focal_gamma": float(cfg.operator_focal_gamma),
        "operator_class_weight_scope": str(cfg.operator_class_weight_scope),
        "operator_class_weight_max_ratio": cfg.operator_class_weight_max_ratio,
        "operator_operate_only_supervision": bool(cfg.operator_operate_only_supervision),
        "operator_known_only_supervision": bool(cfg.operator_known_only_supervision),
        "improvement_profile": str(args.improvement_profile),
        "checkpoint_path": str(ckpt_path),
        "history": history,
        "schema_version": "phase7_state_decoder_train_result_v1",
    }
    if sweep_metadata is not None:
        result["sweep_metadata"] = sweep_metadata
        for k, v in sweep_metadata.items():
            result.setdefault(k, v)
    out_path = Path(args.results_dir) / f"state_decoder_supervised_{cfg.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, result)
    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Saved results    -> {out_path}")
    return result


def main() -> None:
    args = parse_args()
    if args.torch_num_threads is not None:
        if args.torch_num_threads <= 0:
            raise ValueError("--torch-num-threads must be > 0 when set")
        torch.set_num_threads(int(args.torch_num_threads))
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    if args.config_name and args.all_configs:
        raise ValueError("Use either --config-name or --all-configs (or neither), not both")
    spec = resolve_model_spec(args.model_key, args.adapter_config)
    selected, manifest_payload = _resolve_selected_configs(args, model_num_layers=int(spec.num_layers))
    model_meta = spec.to_dict()
    resolved_cfgs: List[StateDecoderExperimentConfig] = []
    for cfg in selected:
        cfg1 = _apply_improvement_profile(cfg, str(args.improvement_profile))
        cfg2 = _apply_user_overrides(cfg1, args)
        cfg3 = apply_model_metadata_to_config(cfg2, model_meta, vocab_size_override=args.vocab_size)
        resolved_cfgs.append(cfg3)
    selected = resolved_cfgs

    records = load_pt(args.dataset_train)
    if args.max_records is not None:
        records = records[: args.max_records]
    records = _filter_records_by_split(records, "train", bool(args.allow_missing_split_field))
    _validate_schema_versions(records, bool(args.allow_mixed_schema))
    print(f"Loaded {len(records)} trace step records from {args.dataset_train}")
    print(f"Detected schema_version(s): {sorted({_normalize_schema_version(r) for r in records})}")

    summaries = [train_one(cfg, records, args, manifest_payload=manifest_payload) for cfg in selected]
    if len(summaries) > 1:
        save_json(Path(args.results_dir) / "state_decoder_supervised_comparison_partial.json", summaries)


if __name__ == "__main__":
    main()

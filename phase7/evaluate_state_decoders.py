#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover
    from .common import load_pt, save_json
    from .model_registry import resolve_model_spec
    from .state_decoder_core import (
        Phase7StateDataset,
        collate_state_batch,
        decode_latent_pred_states,
        evaluate_state_model,
        load_model_from_checkpoint,
        split_by_example,
    )
except Exception:  # pragma: no cover
    from common import load_pt, save_json
    from model_registry import resolve_model_spec
    from state_decoder_core import (
        Phase7StateDataset,
        collate_state_batch,
        decode_latent_pred_states,
        evaluate_state_model,
        load_model_from_checkpoint,
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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-train", default="phase7_results/dataset/gsm8k_step_traces_train.pt")
    p.add_argument("--dataset-test", default="phase7_results/dataset/gsm8k_step_traces_test.pt")
    p.add_argument("--eval-split", choices=["val", "test", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--model-key", default="gpt2-medium")
    p.add_argument("--adapter-config", default=None, help="Optional JSON overrides for model registry entry")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default=None)
    p.add_argument("--manifest", default=None, help="Optional layer sweep manifest for metadata inference")
    p.add_argument("--sweep-run-id", default=None, help="Optional sweep run ID propagated into outputs")
    p.add_argument("--parent-baseline", default=None, help="Optional baseline layer_set_id for comparisons")
    p.add_argument("--max-records-train", type=int, default=None)
    p.add_argument("--max-records-test", type=int, default=None)
    p.add_argument("--emit-latent-preds", action="store_true", help="Store per-step latent predictions")
    p.add_argument("--latent-preds-output", default=None)
    p.add_argument("--grad-topn", type=int, default=0, help="If >0 and SAE input, compute gradient saliency top-N per selected layer")
    p.add_argument("--grad-saliency-output", default=None)
    p.add_argument(
        "--allow-legacy-metadata-mismatch",
        action="store_true",
        help=(
            "Unsafe compatibility mode: allow evaluation even when dataset record metadata/tensor shapes "
            "do not match the checkpoint model metadata."
        ),
    )
    return p.parse_args()


def _tensor_shape_2d(x: object) -> Optional[tuple[int, int]]:
    if isinstance(x, torch.Tensor) and x.ndim == 2:
        return int(x.shape[0]), int(x.shape[1])
    return None


def _validate_records_compatibility(
    records: Sequence[dict],
    *,
    split_name: str,
    expected_model_key: str,
    expected_model_family: str,
    expected_num_layers: int,
    expected_hidden_dim: int,
    expected_tokenizer_id: str,
    allow_legacy_metadata_mismatch: bool,
) -> Dict[str, object]:
    checks = {
        "split": split_name,
        "num_records_checked": int(len(records)),
        "raw_hidden_shape_mismatch_records": 0,
        "record_model_key_mismatch": 0,
        "record_model_family_mismatch": 0,
        "record_num_layers_mismatch": 0,
        "record_hidden_dim_mismatch": 0,
        "record_tokenizer_id_mismatch": 0,
    }
    errors: List[str] = []

    for idx, r in enumerate(records):
        raw_shape = _tensor_shape_2d(r.get("raw_hidden"))
        if raw_shape is None:
            errors.append(f"[{split_name}] record[{idx}] missing/invalid raw_hidden tensor")
            continue
        if raw_shape != (expected_num_layers, expected_hidden_dim):
            checks["raw_hidden_shape_mismatch_records"] += 1
            errors.append(
                f"[{split_name}] record[{idx}] raw_hidden shape={raw_shape} incompatible with "
                f"checkpoint expected=({expected_num_layers}, {expected_hidden_dim})"
            )

        if "model_key" in r and str(r.get("model_key")) != expected_model_key:
            checks["record_model_key_mismatch"] += 1
            errors.append(
                f"[{split_name}] record[{idx}] model_key={r.get('model_key')!r} != {expected_model_key!r}"
            )
        if "model_family" in r and str(r.get("model_family")) != expected_model_family:
            checks["record_model_family_mismatch"] += 1
            errors.append(
                f"[{split_name}] record[{idx}] model_family={r.get('model_family')!r} != {expected_model_family!r}"
            )
        if "num_layers" in r:
            try:
                num_layers = int(r.get("num_layers"))
            except Exception:
                num_layers = None
            if num_layers != expected_num_layers:
                checks["record_num_layers_mismatch"] += 1
                errors.append(
                    f"[{split_name}] record[{idx}] num_layers={num_layers!r} != {expected_num_layers}"
                )
        if "hidden_dim" in r:
            try:
                hidden_dim = int(r.get("hidden_dim"))
            except Exception:
                hidden_dim = None
            if hidden_dim != expected_hidden_dim:
                checks["record_hidden_dim_mismatch"] += 1
                errors.append(
                    f"[{split_name}] record[{idx}] hidden_dim={hidden_dim!r} != {expected_hidden_dim}"
                )
        if "tokenizer_id" in r and str(r.get("tokenizer_id")) != expected_tokenizer_id:
            checks["record_tokenizer_id_mismatch"] += 1
            errors.append(
                f"[{split_name}] record[{idx}] tokenizer_id={r.get('tokenizer_id')!r} != {expected_tokenizer_id!r}"
            )

    checks["mismatch_errors_detected"] = int(len(errors))
    if errors and not allow_legacy_metadata_mismatch:
        head = "\n".join(errors[:20])
        tail = "" if len(errors) <= 20 else f"\n... and {len(errors) - 20} more mismatches"
        raise RuntimeError(
            "Phase7 evaluate strict checkpoint/data compatibility check failed.\n"
            "Use --allow-legacy-metadata-mismatch only if you intentionally accept this unsafe mix.\n"
            f"{head}{tail}"
        )
    return checks


def _load_manifest_payload(path: Optional[str]):
    if not path:
        return None
    if load_manifest is None:
        raise RuntimeError("Manifest support unavailable (experiments.layer_sweep_manifest import failed)")
    return load_manifest(path)


def _infer_manifest_row(manifest_payload, cfg):
    if manifest_payload is None or infer_layer_set_id_from_layers is None or get_layer_set is None:
        return None
    layer_set_id = infer_layer_set_id_from_layers(manifest_payload, cfg.layers)
    if layer_set_id is None:
        return None
    return get_layer_set(manifest_payload, layer_set_id)


def _merge_sweep_metadata(args, ckpt: Dict, cfg, manifest_row: Optional[Dict]) -> Optional[Dict]:
    md = dict(ckpt.get("sweep_metadata") or {})
    row = dict(manifest_row or {})
    if row:
        md.setdefault("layer_set_id", row.get("layer_set_id"))
        md.setdefault("layer_set_family", row.get("family"))
        md.setdefault("layer_set_sweep_group", row.get("sweep_group"))
    if args.sweep_run_id:
        md["sweep_run_id"] = args.sweep_run_id
    if args.parent_baseline:
        md["parent_baseline"] = args.parent_baseline
    if args.manifest:
        md.setdefault("manifest_path", str(args.manifest))
    if not md and not (args.manifest or args.sweep_run_id):
        return None
    md.setdefault("schema_version", "layer_sweep_result_v1")
    md.setdefault("phase", "phase7")
    md.setdefault("input_variant", cfg.input_variant)
    md.setdefault("model_key", getattr(cfg, "model_key", args.model_key))
    md.setdefault("model_family", getattr(cfg, "model_family", "unknown"))
    md.setdefault("num_layers_total", int(getattr(cfg, "model_num_layers", -1)))
    md.setdefault("hidden_dim", int(getattr(cfg, "model_hidden_dim", -1)))
    md.setdefault("tokenizer_id", getattr(cfg, "tokenizer_id", ""))
    md.setdefault("layers", list(cfg.layers))
    md.setdefault("num_layers", len(cfg.layers))
    md.setdefault("seed", int(cfg.seed))
    return md


def _make_loader(records: List[dict], cfg, numeric_stats, batch_size: int, num_workers: int):
    ds = Phase7StateDataset(records, cfg, numeric_stats)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_state_batch)


def _compute_grad_sae_saliency(model, records: List[dict], cfg, numeric_stats, device: str, topn: int) -> Dict:
    # Only meaningful for SAE-only input where input dims map directly to SAE feature indices.
    if cfg.input_variant != "sae":
        return {"supported": False, "reason": f"input_variant={cfg.input_variant} does not map gradient dims to SAE features"}
    ds = Phase7StateDataset(records, cfg, numeric_stats)
    if len(ds) == 0:
        return {"supported": False, "reason": "empty dataset"}
    # Accumulate mean absolute gradient wrt input features per selected layer for result-token CE.
    accum = torch.zeros(len(cfg.layers), cfg.input_dim(), dtype=torch.float64)
    n = 0
    for i in range(min(len(ds), 256)):
        item = ds[i]
        x = item["x"].unsqueeze(0).to(device).requires_grad_(True)
        y = torch.tensor([item["result_token_id"]], dtype=torch.long, device=device)
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out["result_token_logits"], y)
        loss.backward()
        g = x.grad.detach().abs().squeeze(0).cpu().double()
        accum += g
        n += 1
        model.zero_grad(set_to_none=True)
    if n > 0:
        accum /= float(n)
    per_layer = {}
    for li, layer in enumerate(cfg.layers):
        vals, idx = torch.topk(accum[li], k=min(topn, accum.shape[1]))
        per_layer[str(layer)] = [
            {"feature_index": int(i), "mean_abs_grad": float(v)} for i, v in zip(idx.tolist(), vals.tolist())
        ]
    flat_vals = []
    for li, layer in enumerate(cfg.layers):
        vals, idx = torch.topk(accum[li], k=min(topn, accum.shape[1]))
        for i, v in zip(idx.tolist(), vals.tolist()):
            flat_vals.append({"layer": int(layer), "feature_index": int(i), "mean_abs_grad": float(v)})
    flat_vals.sort(key=lambda x: x["mean_abs_grad"], reverse=True)
    return {
        "supported": True,
        "num_examples_used": n,
        "top_features_per_selected_layer": per_layer,
        "top_features_flat": flat_vals[: topn * len(cfg.layers)],
    }


def main() -> None:
    args = parse_args()
    ckpt, cfg, numeric_stats, model = load_model_from_checkpoint(args.checkpoint, args.device)
    spec = resolve_model_spec(getattr(cfg, "model_key", args.model_key), args.adapter_config)
    manifest_payload = _load_manifest_payload(args.manifest)
    manifest_row = _infer_manifest_row(manifest_payload, cfg)
    sweep_metadata = _merge_sweep_metadata(args, ckpt, cfg, manifest_row)

    result = {
        "schema_version": "phase7_state_decoder_eval_result_v1",
        "checkpoint": str(args.checkpoint),
        "stage": ckpt.get("stage"),
        "config_name": cfg.name,
        "model_metadata": {
            "model_key": getattr(cfg, "model_key", spec.model_key),
            "model_family": getattr(cfg, "model_family", spec.model_family),
            "num_layers": int(getattr(cfg, "model_num_layers", spec.num_layers)),
            "hidden_dim": int(getattr(cfg, "model_hidden_dim", spec.hidden_dim)),
            "tokenizer_id": getattr(cfg, "tokenizer_id", spec.tokenizer_id),
            "vocab_size": int(getattr(cfg, "vocab_size", spec.vocab_size or 0)),
        },
        "experiment_config": cfg.to_dict(),
        "evaluations": {},
    }
    result["compatibility_mode"] = {
        "allow_legacy_metadata_mismatch": bool(args.allow_legacy_metadata_mismatch),
    }
    result.update(
        {
            "model_key": result["model_metadata"]["model_key"],
            "model_family": result["model_metadata"]["model_family"],
            "model_num_layers": result["model_metadata"]["num_layers"],
            "model_hidden_dim": result["model_metadata"]["hidden_dim"],
            "tokenizer_id": result["model_metadata"]["tokenizer_id"],
            "vocab_size": result["model_metadata"]["vocab_size"],
        }
    )
    if sweep_metadata is not None:
        result["sweep_metadata"] = sweep_metadata
        result.update(sweep_metadata)

    train_records = None
    if args.eval_split in {"val", "both"} or args.emit_latent_preds or args.grad_topn > 0:
        train_records = load_pt(args.dataset_train)
        if args.max_records_train is not None:
            train_records = train_records[: args.max_records_train]
        train_records = [r for r in train_records if r.get("gsm8k_split") == "train"] or train_records
        train_checks = _validate_records_compatibility(
            train_records,
            split_name="train",
            expected_model_key=str(result["model_metadata"]["model_key"]),
            expected_model_family=str(result["model_metadata"]["model_family"]),
            expected_num_layers=int(result["model_metadata"]["num_layers"]),
            expected_hidden_dim=int(result["model_metadata"]["hidden_dim"]),
            expected_tokenizer_id=str(result["model_metadata"]["tokenizer_id"]),
            allow_legacy_metadata_mismatch=bool(args.allow_legacy_metadata_mismatch),
        )
        result.setdefault("dataset_compatibility_checks", {})["train"] = train_checks

    if args.eval_split in {"val", "both"}:
        _, val_records = split_by_example(train_records, cfg.val_fraction, cfg.seed)  # type: ignore[arg-type]
        val_loader = _make_loader(val_records, cfg, numeric_stats, args.batch_size, args.num_workers)
        m = evaluate_state_model(model, val_loader, args.device, numeric_stats)
        m["num_examples"] = len({int(r["example_idx"]) for r in val_records})
        m["dataset_path"] = str(args.dataset_train)
        result["evaluations"]["val"] = m

    if args.eval_split in {"test", "both"}:
        test_records = load_pt(args.dataset_test)
        if args.max_records_test is not None:
            test_records = test_records[: args.max_records_test]
        test_records = [r for r in test_records if r.get("gsm8k_split") == "test"] or test_records
        test_checks = _validate_records_compatibility(
            test_records,
            split_name="test",
            expected_model_key=str(result["model_metadata"]["model_key"]),
            expected_model_family=str(result["model_metadata"]["model_family"]),
            expected_num_layers=int(result["model_metadata"]["num_layers"]),
            expected_hidden_dim=int(result["model_metadata"]["hidden_dim"]),
            expected_tokenizer_id=str(result["model_metadata"]["tokenizer_id"]),
            allow_legacy_metadata_mismatch=bool(args.allow_legacy_metadata_mismatch),
        )
        result.setdefault("dataset_compatibility_checks", {})["test"] = test_checks
        test_loader = _make_loader(test_records, cfg, numeric_stats, args.batch_size, args.num_workers)
        m = evaluate_state_model(model, test_loader, args.device, numeric_stats)
        m["num_examples"] = len({int(r["example_idx"]) for r in test_records})
        m["dataset_path"] = str(args.dataset_test)
        result["evaluations"]["test"] = m

    out_path = Path(args.output) if args.output else Path("phase7_results/results") / f"state_decoder_eval_{cfg.name}.json"

    if args.emit_latent_preds:
        if train_records is None:
            raise RuntimeError("Internal error: train_records not loaded")
        # Emit on test split by default for auditing.
        test_records = load_pt(args.dataset_test)
        if args.max_records_test is not None:
            test_records = test_records[: args.max_records_test]
        test_records = [r for r in test_records if r.get("gsm8k_split") == "test"] or test_records
        pred_checks = _validate_records_compatibility(
            test_records,
            split_name="test_emit_latent_preds",
            expected_model_key=str(result["model_metadata"]["model_key"]),
            expected_model_family=str(result["model_metadata"]["model_family"]),
            expected_num_layers=int(result["model_metadata"]["num_layers"]),
            expected_hidden_dim=int(result["model_metadata"]["hidden_dim"]),
            expected_tokenizer_id=str(result["model_metadata"]["tokenizer_id"]),
            allow_legacy_metadata_mismatch=bool(args.allow_legacy_metadata_mismatch),
        )
        result.setdefault("dataset_compatibility_checks", {})["test_emit_latent_preds"] = pred_checks
        preds = decode_latent_pred_states(model, test_records, cfg, numeric_stats, args.device, batch_size=args.batch_size)
        pred_out = Path(args.latent_preds_output) if args.latent_preds_output else Path("phase7_results/interp") / f"latent_preds_{cfg.name}.json"
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": "phase7_latent_preds_v1", "config_name": cfg.name, "predictions": preds}
        if sweep_metadata is not None:
            payload["sweep_metadata"] = sweep_metadata
            payload.update({k: v for k, v in sweep_metadata.items() if k not in payload})
        save_json(pred_out, payload)
        print(f"Saved latent predictions -> {pred_out}")

    if args.grad_topn > 0:
        if train_records is None:
            raise RuntimeError("Internal error: train_records not loaded")
        sal = _compute_grad_sae_saliency(model, train_records, cfg, numeric_stats, args.device, args.grad_topn)
        sal_out = Path(args.grad_saliency_output) if args.grad_saliency_output else Path("phase7_results/interp") / f"grad_sae_saliency_{cfg.name}.json"
        sal_out.parent.mkdir(parents=True, exist_ok=True)
        sal_payload = {"schema_version": "phase7_grad_saliency_v1", "config_name": cfg.name, **sal}
        if sweep_metadata is not None:
            sal_payload["sweep_metadata"] = sweep_metadata
            sal_payload.update({k: v for k, v in sweep_metadata.items() if k not in sal_payload})
        save_json(sal_out, sal_payload)
        print(f"Saved grad saliency -> {sal_out}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, result)
    print(f"Saved evaluation -> {out_path}")


if __name__ == "__main__":
    main()

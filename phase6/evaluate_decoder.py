#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from decoder_config import DecoderExperimentConfig  # noqa: E402
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
    from experiments.layer_sweep_manifest import get_layer_set, infer_layer_set_id_from_layers, load_manifest
except Exception:  # pragma: no cover
    get_layer_set = None
    infer_layer_set_id_from_layers = None
    load_manifest = None


def load_model_from_checkpoint(checkpoint_path: str | Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint missing required keys: {checkpoint_path}")
    model_cfg = ArithmeticDecoderConfig.from_dict(ckpt["model_config"])
    model = ArithmeticDecoder(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    exp_cfg = None
    if "experiment_config" in ckpt:
        exp_cfg = DecoderExperimentConfig.from_dict(ckpt["experiment_config"])
    return ckpt, exp_cfg, model_cfg, model


def make_loader(records: List[dict], exp_cfg: DecoderExperimentConfig, batch_size: int, num_workers: int):
    ds = Phase6RecordDataset(records, exp_cfg)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_record_items,
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-train", default="phase6_results/dataset/gsm8k_expanded_train.pt")
    p.add_argument("--dataset-test", default="phase6_results/dataset/gsm8k_expanded_test.pt")
    p.add_argument("--eval-split", choices=["val", "test", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default=None)
    p.add_argument("--manifest", default=None, help="Optional layer sweep manifest for metadata inference")
    p.add_argument("--sweep-run-id", default=None, help="Optional run ID propagated into evaluation metadata")
    p.add_argument("--parent-baseline", default=None, help="Optional baseline layer_set_id for sweep comparisons")
    p.add_argument("--max-records-train", type=int, default=None)
    p.add_argument("--max-records-test", type=int, default=None)
    return p.parse_args()


def _load_manifest_payload(path: Optional[str]):
    if not path:
        return None
    if load_manifest is None:
        raise RuntimeError("Manifest support unavailable (experiments.layer_sweep_manifest import failed)")
    return load_manifest(path)


def _infer_manifest_row(manifest_payload, exp_cfg: DecoderExperimentConfig):
    if manifest_payload is None or infer_layer_set_id_from_layers is None or get_layer_set is None:
        return None
    layer_set_id = infer_layer_set_id_from_layers(manifest_payload, exp_cfg.layers)
    if layer_set_id is None:
        return None
    return get_layer_set(manifest_payload, layer_set_id)


def _merge_sweep_metadata(args, ckpt: Dict, exp_cfg: DecoderExperimentConfig, manifest_row: Optional[Dict]) -> Optional[Dict]:
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
    md.setdefault("phase", "phase6")
    md.setdefault("input_variant", exp_cfg.input_variant)
    md.setdefault("layers", list(exp_cfg.layers))
    md.setdefault("num_layers", len(exp_cfg.layers))
    md.setdefault("seed", int(exp_cfg.seed))
    return md


def main():
    args = parse_args()
    ckpt, exp_cfg, model_cfg, model = load_model_from_checkpoint(args.checkpoint, args.device)
    if exp_cfg is None:
        raise ValueError("Checkpoint does not contain experiment_config; cannot reconstruct feature inputs")

    manifest_payload = _load_manifest_payload(args.manifest)
    manifest_row = _infer_manifest_row(manifest_payload, exp_cfg)
    sweep_metadata = _merge_sweep_metadata(args, ckpt, exp_cfg, manifest_row)
    result = {
        "schema_version": "phase6_eval_result_v1",
        "checkpoint": str(args.checkpoint),
        "stage": ckpt.get("stage"),
        "config_name": exp_cfg.name,
        "experiment_config": exp_cfg.to_dict(),
        "model_config": model_cfg.to_dict(),
        "evaluations": {},
    }
    if sweep_metadata is not None:
        result["sweep_metadata"] = sweep_metadata
        result.update(sweep_metadata)

    if args.eval_split in {"val", "both"}:
        train_records = load_records(args.dataset_train, max_records=args.max_records_train)
        verify_schema(train_records)
        train_records = [r for r in train_records if r.get("gsm8k_split") == "train"] or train_records
        set_seed(exp_cfg.seed)
        _, val_records = split_records_by_example(train_records, exp_cfg.val_fraction, exp_cfg.seed)
        val_loader = make_loader(val_records, exp_cfg, args.batch_size, args.num_workers)
        result["evaluations"]["val"] = evaluate_batches(model, val_loader, args.device)
        result["evaluations"]["val"]["num_examples"] = len({int(r["example_idx"]) for r in val_records})
        result["evaluations"]["val"]["dataset_path"] = str(args.dataset_train)

    if args.eval_split in {"test", "both"}:
        test_records = load_records(args.dataset_test, max_records=args.max_records_test)
        verify_schema(test_records)
        test_records = [r for r in test_records if r.get("gsm8k_split") == "test"] or test_records
        test_loader = make_loader(test_records, exp_cfg, args.batch_size, args.num_workers)
        result["evaluations"]["test"] = evaluate_batches(model, test_loader, args.device)
        result["evaluations"]["test"]["num_examples"] = len({int(r["example_idx"]) for r in test_records})
        result["evaluations"]["test"]["dataset_path"] = str(args.dataset_test)

    out_path = Path(args.output) if args.output else Path("phase6_results/results") / f"eval_{exp_cfg.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved evaluation -> {out_path}")
    if "val" in result["evaluations"]:
        v = result["evaluations"]["val"]
        print(f"VAL  top1={v['top1_accuracy']:.4f} top5={v['top5_accuracy']:.4f} Δlogp={v['delta_logprob_vs_gpt2']:.4f}")
    if "test" in result["evaluations"]:
        v = result["evaluations"]["test"]
        print(f"TEST top1={v['top1_accuracy']:.4f} top5={v['top5_accuracy']:.4f} Δlogp={v['delta_logprob_vs_gpt2']:.4f}")


if __name__ == "__main__":
    main()

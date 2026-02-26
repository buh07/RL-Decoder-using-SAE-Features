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
from pipeline_utils import Phase6RecordDataset, collate_record_items, load_records, verify_schema  # noqa: E402

try:  # noqa: E402
    from experiments.layer_sweep_manifest import get_layer_set, infer_layer_set_id_from_layers, load_manifest
except Exception:  # pragma: no cover
    get_layer_set = None
    infer_layer_set_id_from_layers = None
    load_manifest = None


def load_checkpoint(path: str | Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    exp_cfg = DecoderExperimentConfig.from_dict(ckpt["experiment_config"])
    model_cfg = ArithmeticDecoderConfig.from_dict(ckpt["model_config"])
    model = ArithmeticDecoder(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return ckpt, exp_cfg, model_cfg, model


def flatten_rankings(grad_accum: torch.Tensor, layers: List[int], top_n: int) -> List[Dict]:
    # grad_accum: (n_layers_input, input_dim)
    flat = grad_accum.reshape(-1)
    n = min(top_n, flat.numel())
    vals, idxs = flat.topk(n)
    input_dim = grad_accum.shape[1]
    rows = []
    for rank, (val, flat_idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        local_layer_idx = flat_idx // input_dim
        feat_idx = flat_idx % input_dim
        rows.append(
            {
                "rank": rank,
                "selected_layer_position": int(local_layer_idx),
                "model_layer": int(layers[local_layer_idx]),
                "feature_index": int(feat_idx),
                "mean_abs_grad": float(val),
            }
        )
    return rows


def _extract_feature_ids(items) -> set[int]:
    feats: set[int] = set()
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and "feature_idx" in item:
                feats.add(int(item["feature_idx"]))
            elif isinstance(item, (list, tuple)) and item:
                feats.add(int(item[0]))
            elif isinstance(item, int):
                feats.add(int(item))
    elif isinstance(items, dict):
        for item in items.get("top_features", []):
            if isinstance(item, dict) and "feature_idx" in item:
                feats.add(int(item["feature_idx"]))
            elif isinstance(item, (list, tuple)) and item:
                feats.add(int(item[0]))
            elif isinstance(item, int):
                feats.add(int(item))
    return feats


def load_probe_top_features(probe_path: str | Path, probe_position: str = "result") -> Dict[int, set]:
    data = json.load(open(probe_path))
    layer_map: Dict[int, set] = {}
    # Supports:
    # 1) role-keyed shape: {"eq": [layer0_feats,...], "pre_eq": [...], "result": [...]}
    # 2) layer-keyed shape: {"0": [...], "1": [...], ...}
    if isinstance(data, dict) and probe_position in data and isinstance(data[probe_position], list):
        for layer, items in enumerate(data[probe_position]):
            layer_map[layer] = _extract_feature_ids(items)
        return layer_map

    # Fallback: layer-keyed object
    for k, v in data.items():
        try:
            layer = int(k)
        except Exception:
            continue
        layer_map[layer] = _extract_feature_ids(v)
    return layer_map


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="phase6_results/dataset/gsm8k_expanded_test.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--probe-top-features", default="phase4_results/topk/probe/top_features_per_layer.json")
    p.add_argument("--probe-position", choices=["eq", "pre_eq", "result"], default="result")
    p.add_argument("--manifest", default=None, help="Optional layer sweep manifest for metadata inference")
    p.add_argument("--sweep-run-id", default=None, help="Optional run ID propagated into interpret metadata")
    p.add_argument("--parent-baseline", default=None, help="Optional baseline layer_set_id for sweep comparisons")
    p.add_argument("--output", default=None)
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
    ckpt, exp_cfg, _, model = load_checkpoint(args.checkpoint, args.device)
    manifest_payload = _load_manifest_payload(args.manifest)
    manifest_row = _infer_manifest_row(manifest_payload, exp_cfg)
    sweep_metadata = _merge_sweep_metadata(args, ckpt, exp_cfg, manifest_row)

    records = load_records(args.dataset, max_records=args.max_records)
    verify_schema(records)
    ds = Phase6RecordDataset(records, exp_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_record_items)

    grad_accum = None
    num_records = 0
    model.eval()
    for batch in loader:
        x = batch["x"].to(args.device)
        y = batch["y"].to(args.device)
        x.requires_grad_(True)
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        score = log_probs.gather(1, y[:, None]).sum()
        grads = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
        g = grads.detach().abs().sum(dim=0).cpu()  # (n_layers_input, input_dim)
        grad_accum = g if grad_accum is None else (grad_accum + g)
        num_records += int(x.shape[0])

    if grad_accum is None:
        raise RuntimeError("No gradients accumulated (empty dataset?)")
    grad_accum = grad_accum / max(1, num_records)

    per_layer = {}
    for i, layer in enumerate(exp_cfg.layers):
        vals, idxs = grad_accum[i].topk(min(args.top_n, grad_accum.shape[1]))
        per_layer[str(layer)] = [
            {"feature_index": int(idx), "mean_abs_grad": float(val)}
            for idx, val in zip(idxs.tolist(), vals.tolist())
        ]

    result = {
        "schema_version": "phase6_interpret_result_v1",
        "checkpoint": str(args.checkpoint),
        "config_name": exp_cfg.name,
        "input_variant": exp_cfg.input_variant,
        "layers": list(exp_cfg.layers),
        "num_records": num_records,
        "top_features_flat": flatten_rankings(grad_accum, list(exp_cfg.layers), args.top_n),
        "top_features_per_selected_layer": per_layer,
    }
    if sweep_metadata is not None:
        result["sweep_metadata"] = sweep_metadata
        result.update(sweep_metadata)

    probe_path = Path(args.probe_top_features)
    if exp_cfg.input_variant == "sae" and probe_path.exists():
        probe_map = load_probe_top_features(probe_path, probe_position=args.probe_position)
        overlap = {}
        for layer in exp_cfg.layers:
            grad_top = {row["feature_index"] for row in per_layer[str(layer)]}
            probe_top = probe_map.get(int(layer), set())
            if probe_top:
                overlap[str(layer)] = {
                    "grad_top_n": len(grad_top),
                    "probe_top_n": len(probe_top),
                    "intersection": len(grad_top & probe_top),
                    "overlap_fraction_vs_grad": float(len(grad_top & probe_top) / max(1, len(grad_top))),
                    "overlap_fraction_vs_probe": float(len(grad_top & probe_top) / max(1, len(probe_top))),
                }
        result["probe_overlap"] = overlap
        result["probe_top_features_path"] = str(probe_path)
        result["probe_position"] = args.probe_position

    out_path = Path(args.output) if args.output else Path("phase6_results/results") / f"interpret_{exp_cfg.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved interpretability results -> {out_path}")
    print(f"  config={exp_cfg.name} records={num_records}")


if __name__ == "__main__":
    main()

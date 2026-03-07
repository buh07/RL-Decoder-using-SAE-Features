#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover
    from .common import load_json, save_json, sha256_file
    from .sae_trajectory_coherence_discrimination import (
        FEATURE_SET_CHOICES,
        METRIC_KEYS,
        _build_pair_descriptors,
        _cosine_smoothness,
        _feature_variance_coherence,
        _load_control_records_artifact,
        _magnitude_features_for_layer,
        _magnitude_monotonicity_coherence,
        _select_feature_indices,
    )
except ImportError:  # pragma: no cover
    from common import load_json, save_json, sha256_file
    from sae_trajectory_coherence_discrimination import (
        FEATURE_SET_CHOICES,
        METRIC_KEYS,
        _build_pair_descriptors,
        _cosine_smoothness,
        _feature_variance_coherence,
        _load_control_records_artifact,
        _magnitude_features_for_layer,
        _magnitude_monotonicity_coherence,
        _select_feature_indices,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from sae_architecture import SparseAutoencoder  # type: ignore
from sae_config import SAEConfig  # type: ignore


def _parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    for tok in str(value or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("Expected at least one integer")
    return out


def _load_sae(layer: int, saes_dir: Path, model_key: str, device: str) -> SparseAutoencoder:
    model_key_safe = str(model_key).replace("/", "-")
    ckpt_path = saes_dir / f"{model_key_safe}_layer{int(layer)}_sae.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = dict(ckpt["config"])
    cfg = SAEConfig(
        input_dim=int(c["input_dim"]),
        expansion_factor=int(c["expansion_factor"]),
        use_relu=bool(c.get("use_relu", True)),
        use_topk=bool(c.get("use_topk", False)),
        topk_k=int(c.get("topk_k", 0)),
        use_amp=False,
    )
    sae = SparseAutoencoder(cfg)
    sae.load_state_dict(ckpt["model_state_dict"])
    return sae.to(device).eval()


def _load_norm_stats(layer: int, activations_dir: Path, model_key: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    model_key_safe = str(model_key).replace("/", "-")
    p = activations_dir / f"{model_key_safe}_layer{int(layer)}_activations.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing activation stats: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    acts = payload["activations"] if isinstance(payload, dict) else payload
    if not isinstance(acts, torch.Tensor):
        raise TypeError(f"Unexpected activation payload type for layer={layer}: {type(acts).__name__}")
    if acts.ndim == 3:
        acts = acts.reshape(-1, acts.shape[-1])
    acts = acts.float()
    return acts.mean(dim=0).to(device), acts.std(dim=0).clamp_min(1e-6).to(device)


def _compute_layer_block_vectors(
    rows: Sequence[dict],
    *,
    layer: int,
    feature_indices: Sequence[int],
    saes_dir: Path,
    activations_dir: Path,
    model_key: str,
    device: str,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    sae = _load_sae(layer=layer, saes_dir=saes_dir, model_key=model_key, device=device)
    mean, std = _load_norm_stats(layer=layer, activations_dir=activations_dir, model_key=model_key, device=device)
    fidx = torch.tensor([int(x) for x in feature_indices], dtype=torch.long, device=device)

    # Decoder atoms for selected SAE features define arithmetic subspace for projected block.
    D = sae.decoder.weight.detach().to(device=device, dtype=torch.float32).index_select(dim=1, index=fidx)
    if D.numel() == 0:
        raise RuntimeError(f"No decoder atoms available for layer={layer}")
    Q, _ = torch.linalg.qr(D, mode="reduced")

    raw_chunks: List[torch.Tensor] = []
    sae_chunks: List[torch.Tensor] = []
    proj_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start : start + int(batch_size)]
            x = torch.stack([r["raw_hidden"][int(layer)].float() for r in chunk], dim=0).to(device)
            x_norm = (x - mean) / std
            z = sae.encode(x_norm).index_select(dim=1, index=fidx)
            x_proj = (x @ Q) @ Q.T

            raw_chunks.append(x.detach().cpu())
            sae_chunks.append(z.detach().cpu())
            proj_chunks.append(x_proj.detach().cpu())

    del sae
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    def _cat(chunks: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, 0), dtype=torch.float32)

    return {
        "B1_sae": _cat(sae_chunks),
        "B2_raw": _cat(raw_chunks),
        "B3_proj": _cat(proj_chunks),
    }


def _metrics_for_traj(traj: torch.Tensor, mag_cols: Sequence[int]) -> Dict[str, Optional[float]]:
    return {
        "cosine_smoothness": _cosine_smoothness(traj),
        "feature_variance_coherence": _feature_variance_coherence(traj),
        "magnitude_monotonicity_coherence": _magnitude_monotonicity_coherence(traj, mag_cols),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-records", required=True)
    p.add_argument("--model-key", default="auto")
    p.add_argument("--layers", default="4,7,22")
    p.add_argument("--saes-dir", default="phase2_results/saes_gpt2_12x_topk/saes")
    p.add_argument("--activations-dir", default="phase2_results/activations")
    p.add_argument("--phase4-top-features", default="phase4_results/topk/probe/top_features_per_layer.json")
    p.add_argument(
        "--feature-set",
        choices=list(FEATURE_SET_CHOICES),
        default="eq_pre_result_150",
    )
    p.add_argument(
        "--divergent-source",
        default="phase7_results/results/phase7_sae_feature_discrimination_phase7_sae_20260306_224419_phase7_sae.json",
    )
    p.add_argument(
        "--subspace-specs",
        default="phase7_results/interventions/variable_subspaces_phase7_causal_recovery_r2p4_20260305_133136_phase7_causal_recovery_r2p4.json",
    )
    p.add_argument("--sample-traces", type=int, default=0)
    p.add_argument("--min-common-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260307)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-tag", default="")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    run_tag = str(args.run_tag).strip() or f"mixed_traj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    layers = _parse_int_csv(args.layers)

    rows, payload = _load_control_records_artifact(args.control_records)
    model_metadata = payload.get("model_metadata") if isinstance(payload, dict) else {}
    payload_model_key = str((model_metadata or {}).get("model_key") or "").strip()
    if str(args.model_key).strip().lower() == "auto":
        model_key = payload_model_key or "gpt2-medium"
    else:
        model_key = str(args.model_key).strip()
    if not model_key:
        raise RuntimeError("Could not resolve model_key; pass --model-key explicitly")

    if str(args.feature_set) in {"eq_top50", "result_top50", "eq_pre_result_150"}:
        top_features = load_json(args.phase4_top_features)
        declared_model = str((top_features.get("model_key") if isinstance(top_features, dict) else "") or "").strip()
        if declared_model and declared_model != model_key:
            raise RuntimeError(
                f"phase4 top-features model_key={declared_model!r} does not match requested model_key={model_key!r}"
            )

    rows_used, pair_desc, pair_diag = _build_pair_descriptors(
        rows,
        min_common_steps=int(args.min_common_steps),
        sample_traces=(None if int(args.sample_traces) <= 0 else int(args.sample_traces)),
        seed=int(args.seed),
    )

    if not rows_used or not pair_desc:
        out = {
            "schema_version": "phase7_mixed_trajectory_feature_blocks_v1",
            "status": "blocked_no_pairable_trajectories",
            "run_tag": run_tag,
            "model_key": str(model_key),
            "layers": [int(x) for x in layers],
            "source_control_records": str(args.control_records),
            "source_control_records_sha256": sha256_file(args.control_records),
            "coverage_diagnostics": pair_diag,
            "timestamp": datetime.now().isoformat(),
        }
        save_json(args.output, out)
        print(f"Saved blocked mixed-feature artifact -> {args.output}")
        return

    layer_vectors: Dict[int, Dict[str, torch.Tensor]] = {}
    layer_feature_indices: Dict[int, List[int]] = {}
    layer_mag_cols: Dict[str, Dict[int, List[int]]] = {"B1_sae": {}, "B2_raw": {}, "B3_proj": {}}
    selection_meta: Dict[int, Dict[str, Any]] = {}

    for layer in layers:
        feats, meta = _select_feature_indices(
            feature_set=str(args.feature_set),
            layer=int(layer),
            phase4_top_features_path=args.phase4_top_features,
            divergent_source_path=args.divergent_source,
        )
        layer_feature_indices[int(layer)] = [int(x) for x in feats]
        selection_meta[int(layer)] = dict(meta)

        vecs = _compute_layer_block_vectors(
            rows_used,
            layer=int(layer),
            feature_indices=feats,
            saes_dir=Path(args.saes_dir),
            activations_dir=Path(args.activations_dir),
            model_key=model_key,
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        layer_vectors[int(layer)] = vecs

        mag_feats = set(_magnitude_features_for_layer(args.subspace_specs, layer=int(layer)))
        b1_cols = [i for i, f in enumerate(feats) if int(f) in mag_feats]
        layer_mag_cols["B1_sae"][int(layer)] = b1_cols if b1_cols else list(range(len(feats)))
        # raw/proj blocks do not map by SAE feature index; use all dims for monotonicity fallback.
        layer_mag_cols["B2_raw"][int(layer)] = []
        layer_mag_cols["B3_proj"][int(layer)] = []

    block_feature_names: Dict[str, List[str]] = {"B1_sae": [], "B2_raw": [], "B3_proj": []}
    for block in block_feature_names:
        for layer in layers:
            for metric in METRIC_KEYS:
                block_feature_names[block].append(f"{block}_layer{int(layer)}_{metric}")

    samples: List[Dict[str, Any]] = []
    by_variant_pairs: Dict[str, int] = {}
    for p in pair_desc:
        by_variant_pairs[p.unfaithful_variant] = int(by_variant_pairs.get(p.unfaithful_variant, 0)) + 1
        for label, idxs in (("faithful", p.faithful_row_indices), ("unfaithful", p.unfaithful_row_indices)):
            sample: Dict[str, Any] = {
                "trace_id": str(p.trace_id),
                "variant": str(p.unfaithful_variant),
                "label": str(label),
                "step_count": int(len(idxs)),
            }
            for layer in layers:
                lv = layer_vectors[int(layer)]
                for block in ("B1_sae", "B2_raw", "B3_proj"):
                    traj = lv[block][idxs, :].float()
                    mcols = layer_mag_cols[block][int(layer)]
                    metrics = _metrics_for_traj(traj, mcols)
                    for metric in METRIC_KEYS:
                        sample[f"{block}_layer{int(layer)}_{metric}"] = (
                            float(metrics[metric]) if isinstance(metrics[metric], (int, float)) else None
                        )
            samples.append(sample)

    out = {
        "schema_version": "phase7_mixed_trajectory_feature_blocks_v1",
        "status": "ok",
        "run_tag": run_tag,
        "source_control_records": str(args.control_records),
        "source_control_records_sha256": sha256_file(args.control_records),
        "model_key": str(model_key),
        "source_control_records_stats": payload.get("stats"),
        "layers": [int(x) for x in layers],
        "feature_set": str(args.feature_set),
        "block_feature_names": block_feature_names,
        "coverage_diagnostics": {
            **pair_diag,
            "pair_descriptor_count": int(len(pair_desc)),
            "variant_pair_counts": {k: int(v) for k, v in sorted(by_variant_pairs.items())},
            "samples_count": int(len(samples)),
            "samples_faithful": int(sum(1 for s in samples if s["label"] == "faithful")),
            "samples_unfaithful": int(sum(1 for s in samples if s["label"] == "unfaithful")),
        },
        "layer_feature_selection": {
            str(k): {
                "feature_indices": [int(x) for x in layer_feature_indices[k]],
                "selection_meta": selection_meta[k],
            }
            for k in sorted(layer_feature_indices.keys())
        },
        "analysis_config": {
            "seed": int(args.seed),
            "sample_traces": int(args.sample_traces),
            "min_common_steps": int(args.min_common_steps),
            "batch_size": int(args.batch_size),
            "device": str(args.device),
            "metric_policy": "coherence_faithful_high",
            "blocks": ["B1_sae", "B2_raw", "B3_proj"],
        },
        "samples": samples,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(args.output, out)
    print(f"Saved mixed trajectory feature blocks -> {args.output}")


if __name__ == "__main__":
    main()

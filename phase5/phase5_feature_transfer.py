#!/usr/bin/env python3
"""
Phase 5 Task 3: Feature Transfer Analysis

Trains SAEs on Phase 4 activation files and evaluates whether top features
transfer across models with compatible activation dimensions.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_architecture import SparseAutoencoder
from sae_config import SAEConfig

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _load_activations(path: Path) -> torch.Tensor:
    payload = torch.load(path, weights_only=False)
    if isinstance(payload, dict) and "activations" in payload:
        acts = payload["activations"]
    else:
        acts = payload

    if acts.dim() == 3:
        acts = acts.reshape(-1, acts.shape[-1])

    return acts.float()


def _normalize_activations(acts: torch.Tensor) -> torch.Tensor:
    mean = acts.mean(dim=0, keepdim=True)
    std = acts.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (acts - mean) / std


def _iter_batches(acts: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    dataset = TensorDataset(acts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in loader:
        yield batch[0]


def train_sae(
    acts: torch.Tensor,
    input_dim: int,
    expansion_factor: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Tuple[SparseAutoencoder, Dict[str, float]]:
    config = SAEConfig(
        input_dim=input_dim,
        expansion_factor=expansion_factor,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=epochs,
        l1_penalty_coeff=1e-4,
        decorrelation_coeff=0.01,
        use_relu=False,
    )

    sae = SparseAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    sae.train()
    losses: List[float] = []

    for epoch in range(epochs):
        epoch_losses = []
        for batch in _iter_batches(acts, batch_size=batch_size):
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, h = sae(batch)
            loss = sae.compute_total_loss(batch, x_hat, h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        losses.append(avg_loss)
        logger.info(f"  epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}")

    summary = {
        "final_loss": losses[-1] if losses else 0.0,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }

    return sae, summary


def compute_recon_loss(
    sae: SparseAutoencoder,
    acts: torch.Tensor,
    device: str,
    batch_size: int,
) -> float:
    sae.eval()
    losses = []
    with torch.no_grad():
        for batch in _iter_batches(acts, batch_size=batch_size):
            batch = batch.to(device)
            x_hat, _ = sae(batch)
            loss = torch.nn.functional.mse_loss(x_hat, batch)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def compute_feature_variance(
    sae: SparseAutoencoder,
    acts: torch.Tensor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    sae.eval()
    latents_list = []
    with torch.no_grad():
        for batch in _iter_batches(acts, batch_size=batch_size):
            batch = batch.to(device)
            latents = sae.encode(batch).detach().cpu().numpy()
            latents_list.append(latents)
    if not latents_list:
        return np.zeros((sae.config.latent_dim,), dtype=np.float32)
    latents_all = np.concatenate(latents_list, axis=0)
    return np.var(latents_all, axis=0)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    ar = _rankdata(a)
    br = _rankdata(b)
    ar = (ar - ar.mean()) / (ar.std() + 1e-8)
    br = (br - br.mean()) / (br.std() + 1e-8)
    return float(np.mean(ar * br))


def compute_decoder_similarity(
    source_sae: SparseAutoencoder,
    target_sae: SparseAutoencoder,
    top_indices: List[int],
) -> Dict[str, float]:
    src = source_sae.decoder.weight.detach().cpu()
    tgt = target_sae.decoder.weight.detach().cpu()

    src = torch.nn.functional.normalize(src, p=2, dim=0)
    tgt = torch.nn.functional.normalize(tgt, p=2, dim=0)

    src_top = src[:, top_indices]  # (input_dim, k)
    sim = torch.matmul(src_top.t(), tgt)  # (k, latent_dim)
    max_sim = sim.max(dim=1).values.numpy()

    return {
        "mean_max_cosine": float(np.mean(max_sim)),
        "pct_above_0_7": float(np.mean(max_sim >= 0.7)),
        "pct_above_0_5": float(np.mean(max_sim >= 0.5)),
    }


def build_activation_index(activations_dir: Path) -> Dict[str, Dict[str, Path]]:
    index: Dict[str, Dict[str, Path]] = {}
    for path in sorted(activations_dir.glob("*_layer*_activations.pt")):
        stem = path.name.replace("_activations.pt", "")
        parts = stem.split("_layer")
        if len(parts) != 2:
            continue
        model_benchmark = parts[0]
        if "_" not in model_benchmark:
            continue
        model, benchmark = model_benchmark.split("_", 1)
        key = f"{model}_{benchmark}"
        index[key] = {
            "model": model,
            "benchmark": benchmark,
            "path": path,
        }
    return index


def save_checkpoint(path: Path, sae: SparseAutoencoder, train_summary: Dict[str, float]) -> None:
    payload = {
        "config": asdict(sae.config),
        "model_state": sae.state_dict(),
        "train_summary": train_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str) -> SparseAutoencoder:
    state = torch.load(path, map_location=device, weights_only=False)
    config = SAEConfig(**state["config"])
    sae = SparseAutoencoder(config)
    sae.load_state_dict(state["model_state"])
    sae.to(device)
    sae.eval()
    return sae


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5.3: Feature transfer analysis")
    parser.add_argument("--activations-dir", type=Path, default=Path("phase4_results/activations"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase5_results/transfer_analysis"))
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-train", action="store_true", help="Retrain SAEs even if checkpoints exist")
    parser.add_argument("--backup-existing", action="store_true", help="Backup existing checkpoints before retraining")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "sae_checkpoints"
    
    # Handle backup of existing checkpoints if --backup-existing is set
    if args.backup_existing and ckpt_dir.exists() and list(ckpt_dir.glob("*.pt")):
        from datetime import datetime
        backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = output_dir / f"sae_checkpoints_backup_{backup_suffix}"
        import shutil
        shutil.copytree(ckpt_dir, backup_dir)
        logger.info(f"Backed up existing checkpoints to {backup_dir}")
        
        # Also backup transfer results if they exist
        results_path = output_dir / "transfer_results.json"
        if results_path.exists():
            shutil.copy2(results_path, output_dir / f"transfer_results_backup_{backup_suffix}.json")
            logger.info(f"Backed up transfer results")
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    activation_index = build_activation_index(args.activations_dir)
    if not activation_index:
        raise SystemExit(f"No activation files found in {args.activations_dir}")

    logger.info("Found activation files:")
    for key, info in activation_index.items():
        logger.info(f"  {key}: {info['path'].name}")

    # Train/load SAEs per activation file
    sae_registry: Dict[str, SparseAutoencoder] = {}
    activation_cache: Dict[str, torch.Tensor] = {}
    recon_losses: Dict[str, float] = {}
    feature_variances: Dict[str, np.ndarray] = {}

    for key, info in activation_index.items():
        acts = _load_activations(info["path"])
        acts = _normalize_activations(acts)
        activation_cache[key] = acts

        input_dim = acts.shape[1]
        ckpt_path = ckpt_dir / f"{key}_sae.pt"

        if ckpt_path.exists() and not args.force_train:
            logger.info(f"Loading SAE checkpoint for {key}")
            sae = load_checkpoint(ckpt_path, args.device)
        else:
            logger.info(f"Training SAE for {key} (input_dim={input_dim})")
            sae, train_summary = train_sae(
                acts=acts,
                input_dim=input_dim,
                expansion_factor=args.expansion_factor,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            save_checkpoint(ckpt_path, sae, train_summary)

        sae_registry[key] = sae
        recon_losses[key] = compute_recon_loss(
            sae, acts, device=args.device, batch_size=args.batch_size
        )
        feature_variances[key] = compute_feature_variance(
            sae, acts, device=args.device, batch_size=args.batch_size
        )

    # Compute transfer metrics across pairs
    results = {
        "config": {
            "expansion_factor": args.expansion_factor,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
        },
        "pairs": {},
    }

    keys = sorted(sae_registry.keys())
    for src_key in keys:
        for tgt_key in keys:
            if src_key == tgt_key:
                continue

            src_sae = sae_registry[src_key]
            tgt_sae = sae_registry[tgt_key]
            src_dim = src_sae.config.input_dim
            tgt_dim = tgt_sae.config.input_dim

            pair_key = f"{src_key}__to__{tgt_key}"

            if src_dim != tgt_dim:
                results["pairs"][pair_key] = {
                    "status": "skipped",
                    "reason": f"dimension_mismatch_{src_dim}_vs_{tgt_dim}",
                }
                continue

            src_acts = activation_cache[src_key]
            tgt_acts = activation_cache[tgt_key]

            # Apply source SAE to target activations
            tgt_recon_loss = compute_recon_loss(
                src_sae, tgt_acts, device=args.device, batch_size=args.batch_size
            )

            src_loss = recon_losses[src_key]
            transfer_ratio = tgt_recon_loss / (src_loss + 1e-8)

            src_var = feature_variances[src_key]
            tgt_var = compute_feature_variance(
                src_sae, tgt_acts, device=args.device, batch_size=args.batch_size
            )

            top_k = min(args.top_k, src_var.shape[0])
            top_idx = np.argsort(src_var)[-top_k:][::-1]

            mean_top_var_tgt = float(np.mean(tgt_var[top_idx])) if top_idx.size else 0.0
            mean_all_var_tgt = float(np.mean(tgt_var)) if tgt_var.size else 0.0
            variance_ratio = mean_top_var_tgt / (mean_all_var_tgt + 1e-8)

            spearman = spearman_corr(src_var[top_idx], tgt_var[top_idx]) if top_idx.size else 0.0

            similarity_metrics = compute_decoder_similarity(src_sae, tgt_sae, top_idx.tolist())

            results["pairs"][pair_key] = {
                "status": "ok",
                "source_dim": src_dim,
                "target_dim": tgt_dim,
                "source_recon_loss": src_loss,
                "target_recon_loss_with_source_sae": tgt_recon_loss,
                "transfer_recon_ratio": transfer_ratio,
                "top_k": int(top_k),
                "top_k_variance_ratio": variance_ratio,
                "top_k_spearman": spearman,
                "decoder_similarity": similarity_metrics,
            }

    # Save results
    results_path = output_dir / "transfer_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write summary markdown
    summary_lines = [
        "# Phase 5.3 Feature Transfer Analysis",
        "",
        f"Activation source: {args.activations_dir}",
        f"Expansion factor: {args.expansion_factor}",
        f"Epochs: {args.epochs}",
        f"Top-k features: {args.top_k}",
        "",
        "## Pairwise Transfer Summary",
        "",
        "| Source -> Target | Status | Recon Ratio | Top-k Var Ratio | Spearman | Mean Max Cosine |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for pair_key, data in results["pairs"].items():
        if data["status"] != "ok":
            summary_lines.append(
                f"| {pair_key} | {data['status']} ({data['reason']}) | - | - | - | - |"
            )
            continue

        summary_lines.append(
            "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                pair_key,
                data["status"],
                data["transfer_recon_ratio"],
                data["top_k_variance_ratio"],
                data["top_k_spearman"],
                data["decoder_similarity"]["mean_max_cosine"],
            )
        )

    summary_path = output_dir / "TRANSFER_REPORT.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info(f"Saved transfer results to {results_path}")
    logger.info(f"Saved summary report to {summary_path}")


if __name__ == "__main__":
    main()

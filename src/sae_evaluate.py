#!/usr/bin/env python3
"""
Evaluate SAE checkpoints on held-out activation shards.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from sae_architecture import SparseAutoencoder
from sae_config import SAEConfig
from sae_training import ActivationShardDataset

# Register custom classes for safe loading
torch.serialization.add_safe_globals([SAEConfig])


def _iter_batches(
    shard_dir: Path,
    batch_size: int,
) -> Iterable[torch.Tensor]:
    dataset = ActivationShardDataset(shard_dir, batch_size=batch_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    for batch in dataloader:
        yield batch


def _evaluate_checkpoint(
    checkpoint_path: Path,
    shard_dir: Path,
    device: str,
    max_batches: int | None,
) -> dict:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state["config"]

    sae = SparseAutoencoder(config)
    sae.load_state_dict(state["model_state"])
    sae.to(device)
    sae.eval()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(getattr(config, "dtype", "float16"), torch.float16)

    total_loss = 0.0
    total_recon = 0.0
    total_l1 = 0.0
    total_decorr = 0.0
    total_sparsity = 0.0
    batches = 0

    start = time.perf_counter()
    for batch in _iter_batches(shard_dir, batch_size=config.batch_size):
        if max_batches is not None and batches >= max_batches:
            break

        batch = batch.to(device)
        with torch.no_grad():
            if device.startswith("cuda") and config.use_amp:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    x_hat, h = sae(batch)
                    losses = sae.compute_loss_components(batch, x_hat, h)
            else:
                x_hat, h = sae(batch)
                losses = sae.compute_loss_components(batch, x_hat, h)

        stats = sae.get_activation_stats(h)

        total_loss += float(losses["total_loss"])
        total_recon += float(losses["recon_loss"])
        total_l1 += float(losses["l1_loss"])
        total_decorr += float(losses["decorr_loss"])
        total_sparsity += float(stats["activation_fraction"])
        batches += 1

    elapsed = time.perf_counter() - start
    avg_time_ms = (elapsed / max(1, batches)) * 1000

    return {
        "checkpoint": str(checkpoint_path),
        "input_dim": config.input_dim,
        "latent_dim": config.latent_dim,
        "expansion_factor": int(config.latent_dim / config.input_dim),
        "batches": batches,
        "avg_time_per_step_ms": avg_time_ms,
        "total_loss": total_loss / max(1, batches),
        "recon_loss": total_recon / max(1, batches),
        "l1_loss": total_l1 / max(1, batches),
        "decorr_loss": total_decorr / max(1, batches),
        "activation_fraction": total_sparsity / max(1, batches),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAE checkpoints on activation shards")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="Checkpoint path(s) to evaluate",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
        help="Activation shard directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Limit number of batches for quick eval",
    )
    args = parser.parse_args()

    print("SAE Test Evaluation")
    print("=" * 80)

    for ckpt in args.checkpoint:
        result = _evaluate_checkpoint(
            checkpoint_path=ckpt,
            shard_dir=args.shard_dir,
            device=args.device,
            max_batches=args.max_batches,
        )

        print(f"\nCheckpoint: {result['checkpoint']}")
        print(
            f"  Config: {result['input_dim']}D → {result['latent_dim']}D "
            f"({result['expansion_factor']}x)")
        print(f"  Batches: {result['batches']}")
        print(f"  Avg time/step: {result['avg_time_per_step_ms']:.1f}ms")
        print(
            f"  Loss: total={result['total_loss']:.4f} "
            f"recon={result['recon_loss']:.4f} l1={result['l1_loss']:.4f} "
            f"decorr={result['decorr_loss']:.4f}")
        print(f"  Sparsity: {result['activation_fraction']:.2%}")


if __name__ == "__main__":
    main()

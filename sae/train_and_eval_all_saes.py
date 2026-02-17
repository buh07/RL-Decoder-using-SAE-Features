#!/usr/bin/env python3
"""
Train all SAE expansion levels (4x-20x) on GSM8K and evaluate on test set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sae_config import SAEConfig
from sae_architecture import SparseAutoencoder
from sae_training import ActivationShardDataset


def get_config(expansion: int) -> SAEConfig:
    """Create SAE config for given expansion factor."""
    bs = max(16 // (expansion // 4), 2) if expansion > 8 else 32
    lr = 3e-4 / max(expansion / 4, 1)
    return SAEConfig(
        input_dim=768,
        expansion_factor=expansion,
        batch_size=bs,
        learning_rate=lr,
        max_epochs=1,
        checkpoint_every=500,
        use_amp=True,
        dtype="float16",
    )


def train_sae(expansion: int, gpu_id: int, output_dir: Path) -> tuple[Path, dict]:
    """Train SAE for given expansion level."""
    config = get_config(expansion)
    checkpoint_path = output_dir / f"sae_768d_{expansion}x_final.pt"
    latent_dim = config.input_dim * config.expansion_factor

    print(f"\n[GPU {gpu_id}] Training {expansion}x SAE (768D → {latent_dim}D)...")

    # Load training activations
    train_acts_dir = Path("/tmp/gpt2_gsm8k_acts/gsm8k/train")
    if not train_acts_dir.exists():
        raise FileNotFoundError(f"Training activations not found: {train_acts_dir}")

    dataset = ActivationShardDataset(train_acts_dir, batch_size=config.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # Create and train SAE
    sae = SparseAutoencoder(config)
    sae.to(f"cuda:{gpu_id}")
    sae.train()

    if config.use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler("cuda")
    else:
        scaler = None

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    step = 0
    best_loss = float("inf")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.float16)

    for batch in dataloader:
        batch = batch.to(f"cuda:{gpu_id}")
        optimizer.zero_grad()

        if config.use_amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                x_hat, h = sae(batch)
                loss = sae.compute_total_loss(batch, x_hat, h)
        else:
            x_hat, h = sae(batch)
            loss = sae.compute_total_loss(batch, x_hat, h)

        if config.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()

        if loss < best_loss:
            best_loss = loss

        step += 1
        if step % 100 == 0:
            stats = sae.get_activation_stats(h)
            print(
                f"  Step {step}: loss={loss:.4f}, sparsity={stats['activation_fraction']:.1%}"
            )

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "model_state": sae.state_dict(),
            "final_loss": float(best_loss),
            "step": step,
        },
        checkpoint_path,
    )
    print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    return checkpoint_path, {
        "expansion": expansion,
        "final_loss": float(best_loss),
        "steps": step,
    }


def evaluate_sae(checkpoint_path: Path, gpu_id: int) -> dict:
    """Evaluate SAE on test activations."""
    print(f"\n[GPU {gpu_id}] Evaluating {checkpoint_path.name}...")

    # Load checkpoint
    torch.serialization.add_safe_globals([SAEConfig])
    state = torch.load(checkpoint_path, map_location=f"cuda:{gpu_id}", weights_only=False)
    config = state["config"]

    sae = SparseAutoencoder(config)
    sae.load_state_dict(state["model_state"])
    sae.to(f"cuda:{gpu_id}")
    sae.eval()

    # Load test activations
    test_acts_dir = Path("/tmp/gpt2_gsm8k_acts_test/gsm8k/test")
    if not test_acts_dir.exists():
        raise FileNotFoundError(f"Test activations not found: {test_acts_dir}")

    dataset = ActivationShardDataset(test_acts_dir, batch_size=config.batch_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.float16)

    total_loss = 0.0
    total_recon = 0.0
    total_l1 = 0.0
    total_decorr = 0.0
    total_sparsity = 0.0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(f"cuda:{gpu_id}")

            if config.use_amp:
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

    results = {
        "expansion": config.expansion_factor,
        "latent_dim": config.input_dim * config.expansion_factor,
        "avg_total_loss": total_loss / batches,
        "avg_recon_loss": total_recon / batches,
        "avg_l1_loss": total_l1 / batches,
        "avg_decorr_loss": total_decorr / batches,
        "avg_sparsity": total_sparsity / batches,
        "test_batches": batches,
    }

    print(f"  Loss: total={results['avg_total_loss']:.4f} recon={results['avg_recon_loss']:.4f} "
          f"l1={results['avg_l1_loss']:.4f} decorr={results['avg_decorr_loss']:.6f}")
    print(f"  Sparsity: {results['avg_sparsity']:.1%}")

    return results


def main():
    output_dir = Path("/scratch2/f004ndc/RL-Decoder with SAE Features/checkpoints/gpt2-small/sae")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify test activations exist
    test_acts_dir = Path("/tmp/gpt2_gsm8k_acts_test/gsm8k/test")
    if not test_acts_dir.exists():
        print(f"Error: Test activations not found at {test_acts_dir}")
        sys.exit(1)

    # Expansion levels to train
    expansions = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    checkpoints = {}
    gpu_id = 0

    print(f"Training SAEs: {expansions}")
    print(f"Output directory: {output_dir}")

    # Train all SAEs (sequential to avoid memory issues)
    for expansion in expansions:
        try:
            checkpoint_path, train_info = train_sae(expansion, gpu_id % 6, output_dir)
            checkpoints[expansion] = checkpoint_path
        except Exception as e:
            print(f"✗ Error training {expansion}x: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Evaluate all checkpoints
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)

    eval_results = []
    for expansion in expansions:
        if expansion in checkpoints:
            try:
                result = evaluate_sae(checkpoints[expansion], gpu_id % 6)
                eval_results.append(result)
            except Exception as e:
                print(f"✗ Error evaluating {expansion}x: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save results to JSON
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print results table
    print("\n" + "=" * 140)
    print("EVALUATION TABLE - SAE Test Performance on GSM8K")
    print("=" * 140)
    print(
        f"{'Expansion':<12} {'Latent Dim':<14} {'Total Loss':<16} {'Recon Loss':<16} "
        f"{'L1 Loss':<16} {'Decorr Loss':<16} {'Sparsity':<12}"
    )
    print("-" * 140)
    for result in sorted(eval_results, key=lambda r: r["expansion"]):
        print(
            f"{result['expansion']}x{'':<9} {result['latent_dim']:<14} "
            f"{result['avg_total_loss']:<16.6f} {result['avg_recon_loss']:<16.6f} "
            f"{result['avg_l1_loss']:<16.6f} {result['avg_decorr_loss']:<16.8f} "
            f"{result['avg_sparsity']:<12.1%}"
        )
    print("=" * 140)

    return eval_results


if __name__ == "__main__":
    results = main()

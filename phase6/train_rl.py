#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
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


def reward_from_logits(logits: torch.Tensor, targets: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
    # REINFORCE-consistent default: reward depends on the sampled action.
    # (A top-5/argmax metric can still be tracked separately during evaluation.)
    exact = (sampled == targets)
    reward = torch.full_like(targets, -0.1, dtype=torch.float32)
    reward = torch.where(exact, torch.full_like(reward, 1.0), reward)
    return reward


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Supervised checkpoint to initialize from")
    p.add_argument("--dataset-train", default="phase6_results/dataset/gsm8k_expanded_train.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--kl-coeff", type=float, default=0.01)
    p.add_argument("--baseline-momentum", type=float, default=0.95)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--results-dir", default="phase6_results/results")
    p.add_argument("--checkpoints-dir", default="phase6_results/checkpoints")
    p.add_argument("--min-supervised-val-top1", type=float, default=0.0,
                   help="Warn (but continue) if supervised checkpoint val top1 is below this")
    p.add_argument("--reward-scheme", choices=["sampled_exact_only"], default="sampled_exact_only",
                   help="Reward shaping used for RL (recorded in outputs for reproducibility)")
    p.add_argument("--kl-direction", choices=["current_to_ref"], default="current_to_ref",
                   help="KL regularization direction (recorded in outputs for reproducibility)")
    return p.parse_args()


def load_supervised_checkpoint(path: str | Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("stage") != "supervised":
        print(f"Warning: checkpoint stage is {ckpt.get('stage')}, expected 'supervised'")
    exp_cfg = DecoderExperimentConfig.from_dict(ckpt["experiment_config"])
    model_cfg = ArithmeticDecoderConfig.from_dict(ckpt["model_config"])
    model = ArithmeticDecoder(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    return ckpt, exp_cfg, model_cfg, model


def main():
    args = parse_args()
    ckpt, exp_cfg, model_cfg, model = load_supervised_checkpoint(args.checkpoint, args.device)
    set_seed(exp_cfg.seed)

    # Soft gating only: warn if supervised checkpoint looks weak.
    history = ckpt.get("history", [])
    if history:
        best_val_top1 = max((row.get("val", {}).get("top1_accuracy", 0.0) for row in history), default=0.0)
        if best_val_top1 < args.min_supervised_val_top1:
            print(
                f"Warning: supervised val top1 ({best_val_top1:.4f}) < requested threshold "
                f"({args.min_supervised_val_top1:.4f}); continuing anyway."
            )

    ref_model = ArithmeticDecoder(model_cfg).to(args.device).eval()
    ref_model.load_state_dict(deepcopy(model.state_dict()))

    records = load_records(args.dataset_train, max_records=args.max_records)
    verify_schema(records)
    records = [r for r in records if r.get("gsm8k_split") == "train"] or records
    train_records, val_records = split_records_by_example(records, exp_cfg.val_fraction, exp_cfg.seed)

    train_loader = DataLoader(
        Phase6RecordDataset(train_records, exp_cfg),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_record_items,
    )
    val_loader = DataLoader(
        Phase6RecordDataset(val_records, exp_cfg),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_record_items,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    running_baseline = 0.0

    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_metric = None
    history_out: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_policy_loss = 0.0
        epoch_kl = 0.0
        epoch_reward = 0.0
        epoch_n = 0

        for batch in train_loader:
            x = batch["x"].to(args.device)
            y = batch["y"].to(args.device)

            logits = model(x)
            with torch.no_grad():
                ref_logits = ref_model(x)

            dist = Categorical(logits=logits)
            sampled = dist.sample()
            logp_sample = dist.log_prob(sampled)
            reward = reward_from_logits(logits.detach(), y, sampled).to(args.device)

            reward_mean = float(reward.mean().item())
            running_baseline = args.baseline_momentum * running_baseline + (1.0 - args.baseline_momentum) * reward_mean
            advantage = reward - running_baseline

            rl_loss = -(logp_sample * advantage.detach()).mean()
            # Explicit KL(current || ref) to match the Phase 6 specification.
            log_p = F.log_softmax(logits, dim=-1)
            log_q = F.log_softmax(ref_logits, dim=-1)
            p_probs = log_p.exp()
            kl = (p_probs * (log_p - log_q)).sum(dim=-1).mean()
            loss = rl_loss + args.kl_coeff * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bsz = y.numel()
            epoch_policy_loss += float(rl_loss.item()) * bsz
            epoch_kl += float(kl.item()) * bsz
            epoch_reward += float(reward.sum().item())
            epoch_n += int(bsz)

        val_metrics = evaluate_batches(model.eval(), val_loader, args.device)
        row = {
            "epoch": epoch,
            "train_mean_reward": epoch_reward / max(1, epoch_n),
            "train_policy_loss": epoch_policy_loss / max(1, epoch_n),
            "train_kl": epoch_kl / max(1, epoch_n),
            "running_reward_baseline": running_baseline,
            "reward_scheme": args.reward_scheme,
            "kl_direction": args.kl_direction,
            "val": val_metrics,
        }
        history_out.append(row)
        print(
            f"[RL {exp_cfg.name}] epoch {epoch}/{args.epochs} "
            f"reward={row['train_mean_reward']:.4f} kl={row['train_kl']:.4f} "
            f"val_top1={val_metrics['top1_accuracy']:.4f} val_top5={val_metrics['top5_accuracy']:.4f}"
        )

        delta = val_metrics.get("delta_logprob_vs_gpt2")
        delta_key = float(delta) if delta is not None else -1e9
        metric_key = (-(val_metrics["top1_accuracy"]), -delta_key)
        if best_metric is None or metric_key < best_metric:
            best_metric = metric_key
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    best_val = evaluate_batches(model.eval(), val_loader, args.device)

    ckpt_out = Path(args.checkpoints_dir) / f"{exp_cfg.name}_rl.pt"
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "rl",
            "initialized_from": str(args.checkpoint),
            "experiment_config": exp_cfg.to_dict(),
            "model_config": model_cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "history": history_out,
            "reward_scheme": args.reward_scheme,
            "kl_direction": args.kl_direction,
        },
        ckpt_out,
    )

    result = {
        "config_name": exp_cfg.name,
        "stage": "rl",
        "initialized_from": str(args.checkpoint),
        "dataset_train_path": str(args.dataset_train),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "kl_coeff": args.kl_coeff,
        "learning_rate": args.lr,
        "reward_scheme": args.reward_scheme,
        "kl_direction": args.kl_direction,
        "history": history_out,
        "checkpoint_path": str(ckpt_out),
    }
    out_path = Path(args.results_dir) / f"rl_{exp_cfg.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved RL checkpoint -> {ckpt_out}")
    print(f"Saved RL results    -> {out_path}")


if __name__ == "__main__":
    main()

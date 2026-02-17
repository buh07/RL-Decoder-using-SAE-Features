#!/usr/bin/env python3
"""
Phase 3 Causal Ablation: Feature importance via ablation on downstream tasks.
Supports zero-ablation, mean-ablation, and noise-based perturbations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sae_architecture import SparseAutoencoder


@dataclass
class FeatureImportanceResult:
    """Result of a single feature ablation."""

    feature_id: int
    """Index of ablated feature in SAE latent space."""

    baseline_loss: float
    """Model loss without ablation."""

    ablated_loss: float
    """Model loss with feature ablated."""

    loss_diff: float
    """Change in loss (ablated - baseline)."""

    relative_importance: float
    """Relative importance: loss_diff / baseline_loss."""

    task_accuracy_baseline: Optional[float] = None
    """Task accuracy without ablation."""

    task_accuracy_ablated: Optional[float] = None
    """Task accuracy with feature ablated."""

    task_accuracy_drop: Optional[float] = None
    """Accuracy drop (DropNone means not important for task)."""


class CausalAblationEvaluator:
    """Evaluate feature importance via causal ablation."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        model,
        device: str = "cuda:0",
        ablation_method: str = "zero",
        num_samplings: int = 1,
    ):
        """
        Args:
            sae: Trained SAE model
            model: Model to run causal tests on (e.g., language_model for likelihood scoring)
            device: Device for compute
            ablation_method: 'zero', 'mean' (replace with mean activation), or 'noise' (gaussian)
            num_samplings: Number of random seeds per ablation (for MC sampling)
        """
        self.sae = sae
        self.model = model
        self.device = device
        self.ablation_method = ablation_method
        self.num_samplings = num_samplings

        self.sae.to(device).eval()
        if model is not None:
            self.model.to(device).eval()

    def ablate_feature(
        self, latents: torch.Tensor, feature_id: int, seed: int = 0
    ) -> torch.Tensor:
        """
        Ablate a single feature in latent space.

        Args:
            latents: [batch, seq_len, latent_dim] or [seq_len, latent_dim]
            feature_id: Index to ablate
            seed: Random seed for noise-based ablation

        Returns:
            Modified latents with feature ablated
        """
        latents_new = latents.clone()

        if self.ablation_method == "zero":
            latents_new[..., feature_id] = 0.0
        elif self.ablation_method == "mean":
            # Replace with mean activation of that feature across the batch
            mean_val = latents[..., feature_id].mean()
            latents_new[..., feature_id] = mean_val
        elif self.ablation_method == "noise":
            # Replace with gaussian noise scaled to feature variance
            var = latents[..., feature_id].var()
            if var > 0:
                std = var.sqrt()
                rng = np.random.RandomState(seed)
                noise = torch.from_numpy(rng.normal(0, float(std), latents.shape[:-1])).float()
                if latents.dim() == 3:
                    noise = noise.to(latents.device)
                    latents_new[..., feature_id] = noise
                else:
                    latents_new[..., feature_id] = noise.to(latents.device)
        else:
            raise ValueError(f"Unknown ablation method: {self.ablation_method}")

        return latents_new

    def compute_reconstruction_loss(
        self,
        activations: torch.Tensor,
        latents: torch.Tensor,
        ablated_latents: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Compute reconstruction loss before and after ablation.

        Args:
            activations: Original activations [batch, seq_len, dim] or [seq_len, dim]
            latents: Original SAE latents [batch, seq_len, latent_dim] or [seq_len, latent_dim]
            ablated_latents: Modified SAE latents after ablation

        Returns:
            (baseline_loss, ablated_loss)
        """
        with torch.no_grad():
            recon_baseline = self.sae.decoder(latents)
            loss_baseline = F.mse_loss(recon_baseline, activations)

            recon_ablated = self.sae.decoder(ablated_latents)
            loss_ablated = F.mse_loss(recon_ablated, activations)

        return float(loss_baseline), float(loss_ablated)

    def compute_task_accuracy(
        self,
        activations: torch.Tensor,
        latents: torch.Tensor,
        ablated_latents: torch.Tensor,
        task_fn: Callable,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute task accuracy before and after ablation using a provided task function.

        Args:
            activations: Original activations
            latents: Original SAE latents
            ablated_latents: Modified SAE latents
            task_fn: Callable(latents) -> accuracy or loss

        Returns:
            (baseline_accuracy, ablated_accuracy)
        """
        if task_fn is None:
            return None, None

        with torch.no_grad():
            baseline_acc = task_fn(latents)
            ablated_acc = task_fn(ablated_latents)

        return baseline_acc, ablated_acc

    def evaluate_feature_importance(
        self,
        activations: torch.Tensor,
        latents: torch.Tensor,
        feature_ids: Optional[list[int]] = None,
        task_fn: Optional[Callable] = None,
        verbose: bool = True,
    ) -> list[FeatureImportanceResult]:
        """
        Evaluate importance of features by ablation.

        Args:
            activations: [batch, seq_len, dim] or [seq_len, dim]
            latents: [batch, seq_len, latent_dim] or [seq_len, latent_dim]
            feature_ids: List of feature IDs to ablate. If None, evaluate all.
            task_fn: Optional function(latents) -> accuracy for task-specific evaluation
            verbose: Print progress

        Returns:
            List of FeatureImportanceResult
        """
        if feature_ids is None:
            feature_ids = list(range(latents.shape[-1]))

        results = []
        iterator = tqdm(feature_ids, disable=not verbose)

        for feature_id in iterator:
            feature_results = []

            for sampling_idx in range(self.num_samplings):
                ablated_latents = self.ablate_feature(latents, feature_id, seed=sampling_idx)

                # Reconstruction loss
                loss_baseline, loss_ablated = self.compute_reconstruction_loss(
                    activations, latents, ablated_latents
                )

                # Task accuracy (if provided)
                task_acc_baseline, task_acc_ablated = self.compute_task_accuracy(
                    activations, latents, ablated_latents, task_fn
                )

                result = FeatureImportanceResult(
                    feature_id=feature_id,
                    baseline_loss=loss_baseline,
                    ablated_loss=loss_ablated,
                    loss_diff=loss_ablated - loss_baseline,
                    relative_importance=(loss_ablated - loss_baseline) / (loss_baseline + 1e-8),
                    task_accuracy_baseline=task_acc_baseline,
                    task_accuracy_ablated=task_acc_ablated,
                    task_accuracy_drop=(
                        task_acc_baseline - task_acc_ablated
                        if task_acc_baseline is not None
                        else None
                    ),
                )
                feature_results.append(result)

            # Average across samplings
            avg_result = FeatureImportanceResult(
                feature_id=feature_id,
                baseline_loss=np.mean([r.baseline_loss for r in feature_results]),
                ablated_loss=np.mean([r.ablated_loss for r in feature_results]),
                loss_diff=np.mean([r.loss_diff for r in feature_results]),
                relative_importance=np.mean([r.relative_importance for r in feature_results]),
                task_accuracy_baseline=(
                    np.mean([r.task_accuracy_baseline for r in feature_results if r.task_accuracy_baseline is not None])
                    if feature_results[0].task_accuracy_baseline is not None
                    else None
                ),
                task_accuracy_ablated=(
                    np.mean([r.task_accuracy_ablated for r in feature_results if r.task_accuracy_ablated is not None])
                    if feature_results[0].task_accuracy_ablated is not None
                    else None
                ),
                task_accuracy_drop=(
                    np.mean([r.task_accuracy_drop for r in feature_results if r.task_accuracy_drop is not None])
                    if feature_results[0].task_accuracy_drop is not None
                    else None
                ),
            )

            results.append(avg_result)

        return results

    def save_results(self, results: list[FeatureImportanceResult], path: Path):
        """Save feature importance results to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for r in results:
            records.append(
                {
                    "feature_id": r.feature_id,
                    "baseline_loss": r.baseline_loss,
                    "ablated_loss": r.ablated_loss,
                    "loss_diff": r.loss_diff,
                    "relative_importance": r.relative_importance,
                    "task_accuracy_baseline": r.task_accuracy_baseline,
                    "task_accuracy_ablated": r.task_accuracy_ablated,
                    "task_accuracy_drop": r.task_accuracy_drop,
                }
            )

        with open(path, "w") as f:
            json.dump(records, f, indent=2)

        print(f"[CausalAblation] Saved {len(results)} feature importance results to {path}")


class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance results."""

    @staticmethod
    def get_top_features(
        results: list[FeatureImportanceResult], metric: str = "loss_diff", top_k: int = 10
    ) -> list[FeatureImportanceResult]:
        """Get top-k features by metric."""
        sorted_results = sorted(results, key=lambda r: getattr(r, metric), reverse=True)
        return sorted_results[:top_k]

    @staticmethod
    def compute_statistics(results: list[FeatureImportanceResult]) -> dict:
        """Compute summary statistics."""
        loss_diffs = np.array([r.loss_diff for r in results])
        rel_imps = np.array([r.relative_importance for r in results])
        task_drops = np.array(
            [r.task_accuracy_drop for r in results if r.task_accuracy_drop is not None]
        )

        stats = {
            "num_features": len(results),
            "mean_loss_diff": float(np.mean(loss_diffs)),
            "median_loss_diff": float(np.median(loss_diffs)),
            "std_loss_diff": float(np.std(loss_diffs)),
            "max_loss_diff": float(np.max(loss_diffs)),
            "mean_relative_importance": float(np.mean(rel_imps)),
            "median_relative_importance": float(np.median(rel_imps)),
        }

        if len(task_drops) > 0:
            stats["mean_task_accuracy_drop"] = float(np.mean(task_drops))
            stats["median_task_accuracy_drop"] = float(np.median(task_drops))

        return stats

    @staticmethod
    def save_summary(results: list[FeatureImportanceResult], path: Path):
        """Save summary statistics to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "statistics": FeatureImportanceAnalyzer.compute_statistics(results),
            "top_10_by_loss_diff": [
                {
                    "feature_id": r.feature_id,
                    "loss_diff": r.loss_diff,
                    "relative_importance": r.relative_importance,
                }
                for r in FeatureImportanceAnalyzer.get_top_features(results, "loss_diff", 10)
            ],
        }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[FeatureAnalyzer] Saved summary to {path}")

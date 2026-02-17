#!/usr/bin/env python3
"""
Phase 3 Probes: Train probes to predict reasoning steps from SAE latents.
Supports linear and nonlinear probes with leakage detection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam

from phase3_alignment import ReasoningExample


class StepProbe(nn.Module):
    """Linear or nonlinear probe for predicting step types from SAE latents."""

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        if hidden_dim is None:
            # Linear probe
            self.net = nn.Linear(latent_dim, num_classes)
        else:
            # Nonlinear probe with hidden layer
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Input: latents [batch, latent_dim] or [seq_len, latent_dim]
        Output: logits [batch, num_classes]
        """
        return self.net(latents)


class ProbeDataset(Dataset):
    """Dataset for probe training: SAE latents -> step labels."""

    def __init__(
        self,
        reasoning_examples: list[ReasoningExample],
        sae_latents: dict[str, torch.Tensor],
        step_type_to_id: dict[str, int],
    ):
        """
        Args:
            reasoning_examples: List of ReasoningExample objects with token-level steps
            sae_latents: Dict mapping example_id -> latents [seq_len, latent_dim]
            step_type_to_id: Mapping from step type name to class ID
        """
        self.examples = reasoning_examples
        self.sae_latents = sae_latents
        self.step_type_to_id = step_type_to_id

        # Build token-level labels
        self.token_labels = []
        self.token_latents = []
        self.example_ids = []

        for ex in reasoning_examples:
            if ex.example_id not in sae_latents:
                continue

            latents = sae_latents[ex.example_id]  # [seq_len, latent_dim]
            seq_len = latents.shape[0]

            # Initialize all tokens as "other" (not in any step)
            labels = torch.full((seq_len,), step_type_to_id.get("other", 0), dtype=torch.long)

            # Mark tokens that belong to steps
            for step in ex.steps:
                step_class_id = step_type_to_id.get(step.step_type, step_type_to_id.get("other", 0))
                for tok_idx in range(step.start_token, min(step.end_token, seq_len)):
                    labels[tok_idx] = step_class_id

            # Append all tokens
            for tok_idx in range(seq_len):
                self.token_latents.append(latents[tok_idx])
                self.token_labels.append(labels[tok_idx])
                self.example_ids.append(ex.example_id)

        self.token_latents = torch.stack(self.token_latents)
        self.token_labels = torch.stack(self.token_labels)

    def __len__(self):
        return len(self.token_labels)

    def __getitem__(self, idx):
        return self.token_latents[idx], self.token_labels[idx], self.example_ids[idx]


class ProbeTrainer:
    """Train and evaluate step type prediction probes."""

    def __init__(
        self,
        latent_dim: int,
        step_types: list[str],
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cuda:0",
    ):
        self.latent_dim = latent_dim
        self.step_types = step_types
        self.step_type_to_id = {st: i for i, st in enumerate(step_types)}
        self.id_to_step_type = {i: st for st, i in self.step_type_to_id.items()}
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

        # Create probe
        self.probe = StepProbe(
            latent_dim=latent_dim,
            num_classes=len(step_types),
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)

        self.optimizer = Adam(self.probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        self.training_losses = []
        self.validation_metrics = {}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.probe.train()
        total_loss = 0.0
        num_batches = 0

        for latents, labels, _ in train_loader:
            latents = latents.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.probe(latents)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.training_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> dict:
        """
        Evaluate probe on a dataset.
        Returns: dict with accuracy, precision, recall, F1, per-class metrics
        """
        self.probe.eval()

        all_preds = []
        all_labels = []

        for latents, labels, _ in eval_loader:
            latents = latents.to(self.device)
            logits = self.probe(latents)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        }

        # Per-class metrics
        for class_id, class_name in self.id_to_step_type.items():
            class_mask = all_labels == class_id
            if class_mask.sum() > 0:
                class_acc = accuracy_score(
                    all_labels[class_mask], all_preds[class_mask]
                )
                metrics[f"acc_{class_name}"] = class_acc

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        verbose: bool = True,
    ):
        """Train probe for multiple epochs."""
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)

            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                self.validation_metrics = eval_metrics

                if verbose and (epoch + 1) % max(1, num_epochs // 5) == 0:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}: "
                        f"train_loss={train_loss:.4f} "
                        f"val_acc={eval_metrics['accuracy']:.4f} "
                        f"val_f1={eval_metrics['f1']:.4f}"
                    )
            elif verbose and (epoch + 1) % max(1, num_epochs // 5) == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f}")

    def save(self, path: Path):
        """Save probe state and metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "probe_state": self.probe.state_dict(),
                "latent_dim": self.latent_dim,
                "step_types": self.step_types,
                "step_type_to_id": self.step_type_to_id,
                "hidden_dim": self.hidden_dim,
                "training_losses": self.training_losses,
                "validation_metrics": self.validation_metrics,
            },
            path,
        )
        print(f"[ProbeTrainer] Saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = "cuda:0"):
        """Load probe from checkpoint."""
        state = torch.load(path, map_location=device, weights_only=False)

        trainer = cls(
            latent_dim=state["latent_dim"],
            step_types=state["step_types"],
            hidden_dim=state["hidden_dim"],
            device=device,
        )
        trainer.probe.load_state_dict(state["probe_state"])
        trainer.training_losses = state.get("training_losses", [])
        trainer.validation_metrics = state.get("validation_metrics", {})

        return trainer


def compute_leakage_metrics(
    baseline_accuracy: float,
    probe_accuracy: float,
    threshold: float = 0.05,
) -> dict:
    """
    Compute leakage metrics: gap between probe and baseline.

    Args:
        baseline_accuracy: Accuracy of random/majority-class baseline
        probe_accuracy: Accuracy of trained probe
        threshold: Flag leakage if gap > threshold

    Returns:
        dict with gap, flagged, etc.
    """
    gap = probe_accuracy - baseline_accuracy
    flagged = gap > threshold

    return {
        "baseline_accuracy": baseline_accuracy,
        "probe_accuracy": probe_accuracy,
        "gap": gap,
        "flagged": flagged,
        "threshold": threshold,
    }

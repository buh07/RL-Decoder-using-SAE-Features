#!/usr/bin/env python3
"""
Phase 3 Orchestration: Full pipeline for controlled CoT analysis.
Integrates alignment, probe training, and causal evaluation.

Usage:
    python phase3_pipeline.py --config phase3_config.yaml --output-dir results/phase3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model_registry import get_model
from phase3_alignment import GSM8KAligner, ReasoningExample
from phase3_causal import CausalAblationEvaluator, FeatureImportanceAnalyzer
from phase3_config import Phase3Config
from phase3_probes import ProbeDataset, ProbeTrainer
from sae_architecture import SparseAutoencoder
from sae_config import SAEConfig
from sae_training import ActivationShardDataset


class Phase3Pipeline:
    """End-to-end Phase 3 pipeline orchestrator."""

    def __init__(self, config: Phase3Config):
        self.config = config
        config.validate()

        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer and model
        self._setup_models()

        # Load SAE checkpoints
        self.saes = {}
        self._load_saes()

        print(f"[Phase3Pipeline] Ready. Output: {self.output_dir}")

    def _setup_models(self):
        """Load tokenizer and model."""
        print("[Phase3] Setting up tokenizer and model...")

        if self.config.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        else:
            # Infer from model registry
            model_info = get_model(self.config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_info.hf_id, local_files_only=True
            )

        # Load model (for future task evaluation)
        self.model = get_model(self.config.model_name)
        print(f"  Model: {self.config.model_name}")
        print(f"  Tokenizer: {self.tokenizer.__class__.__name__}")

    def _load_saes(self):
        """Load all SAE checkpoints."""
        print("[Phase3] Loading SAE checkpoints...")

        for ckpt_path in self.config.sae_checkpoints:
            print(f"  Loading {ckpt_path.name}...")

            torch.serialization.add_safe_globals([SAEConfig])
            state = torch.load(ckpt_path, map_location=self.config.device, weights_only=False)

            config = state["config"]
            sae = SparseAutoencoder(config)
            sae.load_state_dict(state["model_state"])
            sae.to(self.config.device)
            sae.eval()

            expansion_level = config.expansion_factor
            self.saes[expansion_level] = sae

        print(f"  Loaded {len(self.saes)} SAE(s)")

    def extract_and_align_steps(self) -> list[ReasoningExample]:
        """Extract reasoning steps from GSM8K and align to tokens."""
        print("\n[Phase3] Step 1: Extract and align reasoning steps...")

        aligned_path = self.output_dir / "gsm8k_aligned_train.jsonl"

        # If already exists, load
        if aligned_path.exists():
            print(f"  Loading cached alignments from {aligned_path}")
            examples = GSM8KAligner.load_aligned_dataset(aligned_path)
            return examples

        # Otherwise, extract and align
        aligner = GSM8KAligner(
            self.tokenizer,
            extraction_method=self.config.step_extraction_method,
            regex_patterns=self.config.step_regex_patterns,
            min_step_length=self.config.min_step_length,
            max_step_length=self.config.max_step_length,
        )

        examples = aligner.process_dataset(split="train", subsample=500)  # Subsample for efficiency
        aligner.save_aligned_dataset(examples, aligned_path)

        return examples

    def load_sae_activations(
        self, examples: list[ReasoningExample]
    ) -> dict[str, torch.Tensor]:
        """
        Load SAE latents for each example from sharded activation files.
        Returns dict mapping example_id -> latents [seq_len, latent_dim]
        """
        print("\n[Phase3] Step 2: Load SAE latents for aligned examples...")

        sae_latents = {}

        # This is a simplified version—in practice, you'd need to:
        # 1. Map examples to their corresponding activation shards
        # 2. Load those shards and extract the relevant token ranges
        # For now, we log what would need to happen:

        print(f"  Found {len(examples)} aligned examples")
        print(f"  Would load latents from {self.config.train_activation_dir}")
        print("  [TODO: Implement activation shard->example mapping]")

        # Placeholder: mock latents for demonstration
        for ex in examples[:10]:  # Mock first 10
            sae_latents[ex.example_id] = torch.randn(len(ex.tokens), 768, device="cpu")

        return sae_latents

    def train_step_probes(self, examples: list[ReasoningExample]) -> dict:
        """Train probes to predict step types from SAE latents."""
        print("\n[Phase3] Step 3: Train step type prediction probes...")

        probe_results = {}

        # Get unique step types
        all_step_types = set()
        for ex in examples:
            for step in ex.steps:
                all_step_types.add(step.step_type)
        all_step_types.add("other")
        step_types = sorted(list(all_step_types))

        print(f"  Step types: {step_types}")

        # Load SAE latents
        sae_latents_dict = self.load_sae_activations(examples)

        # For each SAE expansion level, train a probe
        for expansion, sae in self.saes.items():
            print(f"\n  Training probe for {expansion}x SAE...")

            latent_dim = sae.config.latent_dim

            # Create dataset
            dataset = ProbeDataset(examples, sae_latents_dict, {st: i for i, st in enumerate(step_types)})

            if len(dataset) == 0:
                print(f"    Skipping {expansion}x: no training data")
                continue

            # Split into train/val
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            train_loader = DataLoader(train_dataset, batch_size=self.config.probe_batch_size)
            val_loader = DataLoader(val_dataset, batch_size=self.config.probe_batch_size)

            # Train probe
            trainer = ProbeTrainer(
                latent_dim=latent_dim,
                step_types=step_types,
                hidden_dim=self.config.probe_hidden_dim,
                dropout=self.config.probe_dropout,
                learning_rate=self.config.probe_learning_rate,
                weight_decay=self.config.probe_weight_decay,
                device=self.config.device,
            )

            trainer.train(
                train_loader,
                eval_loader=val_loader,
                num_epochs=self.config.probe_epochs,
                verbose=self.config.verbose,
            )

            # Save probe
            probe_path = self.output_dir / f"probe_{expansion}x.pt"
            trainer.save(probe_path)

            probe_results[expansion] = {
                "accuracy": trainer.validation_metrics.get("accuracy", 0.0),
                "f1": trainer.validation_metrics.get("f1", 0.0),
                "latent_dim": latent_dim,
            }

        # Save summary
        summary_path = self.output_dir / "probe_results.json"
        with open(summary_path, "w") as f:
            json.dump(probe_results, f, indent=2)

        print(f"\n  Probe results saved to {summary_path}")

        return probe_results

    def evaluate_causal_importance(self) -> dict:
        """Run causal ablation to measure feature importance."""
        print("\n[Phase3] Step 4: Evaluate causal feature importance...")

        causal_results = {}

        for expansion, sae in self.saes.items():
            print(f"\n  Evaluating {expansion}x SAE...")

            # Load test activations (simplified)
            print(f"  Would load from {self.config.test_activation_dir}")
            print("  [TODO: Implement activation loading for test set]")

            # Mock evaluation
            evaluator = CausalAblationEvaluator(
                sae,
                model=None,
                device=self.config.device,
                ablation_method=self.config.ablation_method,
                num_samplings=self.config.num_perturbations_per_feature,
            )

            # In real pipeline, would evaluate top features
            print(f"    Loaded {expansion}x SAE with {sae.config.latent_dim}D latent space")

            causal_results[expansion] = {
                "status": "not-yet-implemented",
                "latent_dim": sae.config.latent_dim,
            }

        summary_path = self.output_dir / "causal_results.json"
        with open(summary_path, "w") as f:
            json.dump(causal_results, f, indent=2)

        return causal_results

    def run(self):
        """Run full Phase 3 pipeline."""
        print("=" * 80)
        print("PHASE 3: CONTROLLED CHAIN-OF-THOUGHT ANALYSIS")
        print("=" * 80)

        # Step 1: Extract and align
        examples = self.extract_and_align_steps()

        # Step 2: Train probes
        probe_results = self.train_step_probes(examples)

        # Step 3: Causal evaluation
        causal_results = self.evaluate_causal_importance()

        # Summary
        print("\n" + "=" * 80)
        print("PHASE 3 SUMMARY")
        print("=" * 80)
        print(f"Examples processed: {len(examples)}")
        print(f"SAEs evaluated: {len(self.saes)}")
        print(f"Probes trained: {len(probe_results)}")
        print(f"\nResults saved to: {self.output_dir}")

        return {
            "examples": len(examples),
            "probes": probe_results,
            "causal": causal_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Pipeline: Controlled CoT Analysis")
    parser.add_argument(
        "--config",
        type=Path,
        help="Phase3Config as JSON or YAML file",
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (gpt2, gpt2-medium, etc.)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="Layer to analyze",
    )
    parser.add_argument(
        "--sae-checkpoints",
        type=Path,
        nargs="+",
        help="Paths to SAE checkpoints to evaluate",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase3_results"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device (cuda:0, etc.)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample N examples for testing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Build config
    if args.config and args.config.exists():
        with open(args.config) as f:
            config_dict = json.load(f)
        config = Phase3Config(**config_dict)
    else:
        config = Phase3Config(
            model_name=args.model,
            layer=args.layer,
            sae_checkpoints=args.sae_checkpoints,
            output_dir=args.output_dir,
            device=args.device,
            verbose=args.verbose,
        )

    # Run pipeline
    pipeline = Phase3Pipeline(config)
    results = pipeline.run()

    print("\n✓ Phase 3 pipeline complete!")


if __name__ == "__main__":
    main()

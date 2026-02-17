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
from phase3_task_evaluation import GSM8KTaskEvaluator, CausalTaskEvaluator
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
                str(model_info.hf_local_path), local_files_only=True
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
    ) -> dict[int, dict[str, torch.Tensor]]:
        """
        Load activations for each example from sharded activation files.
        Returns dict mapping expansion -> dict(example_id -> latents [seq_len, latent_dim])
        """
        print("\n[Phase3] Step 2: Load activations for aligned examples...")

        sae_latents_all_expansions = {}

        for expansion, sae in self.saes.items():
            print(f"\n  Extracting latents for {expansion}x SAE ({sae.config.latent_dim}D)...")

            try:
                from pathlib import Path
                shard_dir = Path(self.config.train_activation_dir)
                shards = sorted(shard_dir.glob("*shard_*.pt"))
                
                # Filter to residual streams if both residual and MLP shards exist
                residual_shards = [s for s in shards if "residual" in s.name]
                if residual_shards:
                    shards = residual_shards
                
                print(f"    Loading from {len(shards)} shards...")
                
                latents_dict = {}
                example_idx = 0
                
                with torch.no_grad():
                    for shard_path in shards:
                        payload = torch.load(shard_path, map_location="cpu")
                        acts = payload["activations"]  # [num_sequences, seq_len, hidden_dim]
                        
                        # Process each sequence in the shard
                        for seq_idx in range(acts.shape[0]):
                            if example_idx >= len(examples):
                                break
                            
                            # Get this sequence's activations
                            seq_acts = acts[seq_idx]  # [seq_len, hidden_dim]
                            seq_acts = seq_acts.unsqueeze(0).to(self.config.device).float()  # [1, seq_len, hidden_dim]
                            
                            # Extract SAE latents
                            latents = sae.encoder(seq_acts)  # [1, seq_len, latent_dim]
                            
                            example_id = examples[example_idx].example_id
                            latents_dict[example_id] = latents.squeeze(0).cpu()  # [seq_len, latent_dim]
                            
                            example_idx += 1
                        
                        if example_idx >= len(examples):
                            break
                
                sae_latents_all_expansions[expansion] = latents_dict
                print(f"    ✓ Extracted latents for {len(latents_dict)} examples")

            except Exception as e:
                print(f"    ✗ Error loading activations: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n  Total: {len(sae_latents_all_expansions)} SAE expansion(s) loaded")

        return sae_latents_all_expansions

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

        # Load SAE latents for all expansions
        sae_latents_by_expansion = self.load_sae_activations(examples)

        # For each SAE expansion level, train a probe
        for expansion, sae in self.saes.items():
            print(f"\n  Training probe for {expansion}x SAE...")

            if expansion not in sae_latents_by_expansion:
                print(f"    Skipping {expansion}x: no latent data")
                continue

            sae_latents_dict = sae_latents_by_expansion[expansion]
            latent_dim = sae.config.latent_dim

            # Create dataset
            try:
                dataset = ProbeDataset(
                    examples, sae_latents_dict, {st: i for i, st in enumerate(step_types)}
                )
            except Exception as e:
                print(f"    Error creating dataset: {e}")
                continue

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

    def evaluate_causal_importance(self, examples: list[ReasoningExample]) -> dict:
        """Run causal ablation to measure feature importance."""
        print("\n[Phase3] Step 4: Evaluate causal feature importance...")

        causal_results = {}

        # Load test activations and latents
        print(f"  Loading test activations from {self.config.test_activation_dir}...")

        for expansion, sae in self.saes.items():
            print(f"\n  Evaluating {expansion}x SAE ({sae.config.latent_dim}D latent)...")

            try:
                # Load test activations
                test_dataset = ActivationShardDataset(
                    self.config.test_activation_dir, batch_size=self.config.causal_batch_size, shuffle=False
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=None, num_workers=self.config.num_workers
                )

                # Extract latents from test batch
                test_activations_list = []
                test_latents_list = []

                print(f"    Loading test batches...")
                batch_count = 0
                for batch in test_loader:
                    if batch_count >= 5:  # Limit to first 5 batches for quick validation
                        break

                    batch = batch.to(self.config.device).float()
                    with torch.no_grad():
                        latents = sae.encoder(batch)

                    test_activations_list.append(batch.cpu())
                    test_latents_list.append(latents.cpu())
                    batch_count += 1

                if not test_activations_list:
                    print(f"    No test data loaded")
                    continue

                # Concatenate batches
                test_activations = torch.cat(test_activations_list, dim=0)
                test_latents = torch.cat(test_latents_list, dim=0)

                print(f"    Loaded test activations: {test_activations.shape}")
                print(f"    Test latents: {test_latents.shape}")

                # Run causal ablation on top features
                evaluator = CausalAblationEvaluator(
                    sae=sae,
                    model=None,
                    device=self.config.device,
                    ablation_method=self.config.ablation_method,
                    num_samplings=self.config.num_perturbations_per_feature,
                )

                # Evaluate top-k features
                top_k = min(self.config.compute_feature_importance_top_k, test_latents.shape[-1])
                feature_ids = range(top_k)

                print(f"    Evaluating {top_k} features...")
                results = evaluator.evaluate_feature_importance(
                    activations=test_activations.to(self.config.device),
                    latents=test_latents.to(self.config.device),
                    feature_ids=list(feature_ids),
                    task_fn=None,
                    verbose=self.config.verbose,
                )

                # Save and analyze
                results_path = self.output_dir / f"causal_results_{expansion}x.json"
                evaluator.save_results(results, results_path)

                summary_path = self.output_dir / f"causal_summary_{expansion}x.json"
                FeatureImportanceAnalyzer.save_summary(results, summary_path)

                causal_results[expansion] = {
                    "num_features": len(results),
                    "top_feature_importance": results[0].loss_diff if results else None,
                    "results_saved": str(results_path),
                    "summary_saved": str(summary_path),
                }

            except Exception as e:
                print(f"    Error evaluating causal importance: {e}")
                import traceback
                traceback.print_exc()
                causal_results[expansion] = {"status": "error", "error": str(e)}

        # Save master summary
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
        causal_results = self.evaluate_causal_importance(examples)

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

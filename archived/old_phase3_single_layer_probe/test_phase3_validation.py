#!/usr/bin/env python3
"""
Quick validation test for Phase 3 pipeline using existing SAE data.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from phase3_config import Phase3Config
from phase3_pipeline import Phase3Pipeline

def main():
    """Run Phase 3 pipeline with existing GSM8K + SAE data."""

    print("=" * 80)
    print("PHASE 3 VALIDATION TEST")
    print("=" * 80)

    # Configuration for 12x SAE on GSM8K
    config = Phase3Config(
        dataset="gsm8k",
        model_name="gpt2",
        layer=6,
        activation_type="mlp_hidden",
        sae_checkpoints=[
            Path("/scratch2/f004ndc/RL-Decoder with SAE Features/checkpoints/gpt2-small/sae/sae_768d_12x_final.pt"),
        ],
        train_activation_dir=Path("/tmp/gpt2_gsm8k_acts/gsm8k/train"),
        test_activation_dir=Path("/tmp/gpt2_gsm8k_acts_test/gsm8k/test"),
        output_dir=Path("phase3_results/validation_test"),
        step_extraction_method="regex",
        probe_epochs=5,  # Quick test
        probe_batch_size=64,
        causal_batch_size=128,
        compute_feature_importance_top_k=20,  # Fast: only top 20 features
        device="cuda:0",
        verbose=True,
    )

    print("\n[Config]")
    print(f"  Model: {config.model_name}, Layer: {config.layer}")
    print(f"  SAEs: {len(config.sae_checkpoints)}")
    print(f"  Train acts: {config.train_activation_dir}")
    print(f"  Test acts: {config.test_activation_dir}")
    print(f"  Output: {config.output_dir}")

    # Validate
    try:
        config.validate()
        print("  ✓ Config valid")
    except Exception as e:
        print(f"  ✗ Config invalid: {e}")
        return 1

    # Initialize pipeline
    try:
        print("\n[Initialization]")
        pipeline = Phase3Pipeline(config)
        print("  ✓ Pipeline initialized")
    except Exception as e:
        print(f"  ✗ Pipeline init failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run pipeline
    try:
        print("\n[Running pipeline]")
        results = pipeline.run()
        print("  ✓ Pipeline completed")
    except Exception as e:
        print(f"  ✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Results summary
    print("\n[Results]")
    print(f"  Examples processed: {results['examples']}")
    print(f"  Probes trained: {len(results['probes'])}")
    print(f"  Causal evaluations: {len(results['causal'])}")

    print(f"\n✓ Validation complete. Results in: {config.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

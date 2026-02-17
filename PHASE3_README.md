# Phase 3: Controlled Chain-of-Thought Analysis Pipeline

## Overview

Phase 3 implements a complete, reusable pipeline for analyzing reasoning in LLMs through sparse autoencoders (SAEs). The pipeline:

1. **Extracts reasoning steps** from GSM8K dataset using configurable heuristics (regex, similarity, hybrid)
2. **Aligns steps to tokens** in the tokenized sequence
3. **Trains probes** on SAE latents to predict reasoning step types
4. **Performs causal ablation** to measure feature importance for task performance
5. **Reports leakage diagnostics** to flag potential identifiability issues

All components are **model-agnostic** and **SAE-configuration-agnostic**, supporting:
- Any LLM (GPT-2, GPT-2-medium, Pythia, Llama, etc.)
- Multiple SAE checkpoints simultaneously (any expansion levels: 4x, 12x, 20x, etc.)
- Custom datasets and reasoning domains beyond GSM8K

---

## Architecture

### Core Modules

#### `src/phase3_config.py`
**Configuration system** for the entire Phase 3 pipeline. Covers:
- Dataset and model selection
- SAE checkpoint paths (supports multiple expansions)
- Step extraction method (regex, similarity, hybrid)
- Probe architecture (linear vs nonlinear)
- Causal ablation parameters
- Output and logging settings

**Usage:**
```python
from phase3_config import Phase3Config

config = Phase3Config(
    dataset="gsm8k",
    model_name="gpt2",
    layer=6,
    sae_checkpoints=[
        Path("checkpoints/sae/sae_768d_4x_final.pt"),
        Path("checkpoints/sae/sae_768d_12x_final.pt"),
        Path("checkpoints/sae/sae_768d_20x_final.pt"),
    ],
    step_extraction_method="regex",
    output_dir=Path("phase3_results"),
)
config.validate()
```

#### `src/phase3_alignment.py`
**Reasoning step extraction and token-level alignment**. Components:

- **`ReasoningStep`**: Dataclass for a single reasoning step with token span
- **`ReasoningExample`**: Full example with extracted steps and token alignment
- **`ReasoningStepExtractor`**: Multi-method step extraction (regex, similarity, hybrid)
- **`TokenAligner`**: Maps extracted step text to token indices
- **`GSM8KAligner`**: End-to-end pipeline for GSM8K processing

**Features:**
- Character-to-token mapping for accurate alignment
- Configurable regex patterns for step type classification
- Similarity-based boundary detection (for models without explicit step markers)
- Saves aligned dataset as JSONL for reproducibility and inspection

**Usage:**
```python
from phase3_alignment import GSM8KAligner
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
aligner = GSM8KAligner(
    tokenizer,
    extraction_method="regex",
    regex_patterns={
        "step_header": r"(?:Step|step)\s+\d+[:.]?",
        "equation": r"(\d+[\s+\-*/]*)+\s*=",
    },
)

# Process dataset
examples = aligner.process_dataset(split="train", subsample=500)

# Save for reference
aligner.save_aligned_dataset(examples, Path("phase3_results/gsm8k_aligned.jsonl"))

# Load later
examples_loaded = GSM8KAligner.load_aligned_dataset(Path(...))
```

**Supported Extraction Methods:**
- `regex`: Pattern-based step detection (default, fastest)
- `similarity`: Semantic boundary detection via embeddings
- `hybrid`: Regex fallback to similarity if no regex matches found

#### `src/phase3_probes.py`
**Probe training for step prediction from SAE latents**. Components:

- **`StepProbe`**: Linear or nonlinear neural network probe
- **`ProbeDataset`**: PyTorch dataset mapping SAE latents → step type labels
- **`ProbeTrainer`**: Training loop with validation and metrics
- **`compute_leakage_metrics`**: Detect identifiability issues

**Features:**
- Supports linear probes (fastest) and nonlinear probes (more expressive)
- Token-level classification: each token labeled by its step type
- Tracks training losses and validation metrics (accuracy, F1, precision, recall)
- Saves probe checkpoints for inspection and reuse
- **Leakage detection**: Compares probe accuracy against random baseline

**Usage:**
```python
from phase3_probes import ProbeDataset, ProbeTrainer

# Create dataset
dataset = ProbeDataset(
    reasoning_examples=examples,
    sae_latents=sae_latents_dict,  # {example_id: latents[seq_len, latent_dim]}
    step_type_to_id={"equation": 0, "reasoning": 1, "other": 2},
)

# Train probe
trainer = ProbeTrainer(
    latent_dim=9216,  # e.g., 12x expansion of 768D
    step_types=["equation", "reasoning", "other"],
    hidden_dim=128,  # Nonlinear: hidden layer size; None for linear
    learning_rate=1e-3,
)

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

trainer.train(train_loader, val_loader, num_epochs=10)
trainer.save(Path("phase3_results/probe_12x.pt"))

# Check for leakage
metrics = trainer.evaluation_metrics
print(f"Probe accuracy: {metrics['accuracy']:.4f}")
```

#### `src/phase3_causal.py`
**Causal ablation and feature importance evaluation**. Components:

- **`FeatureImportanceResult`**: Dataclass for ablation result
- **`CausalAblationEvaluator`**: Ablate features and measure downstream impact
- **`FeatureImportanceAnalyzer`**: Summarize and analyze importance scores

**Features:**
- Multiple ablation methods: zero, mean-ablation, gaussian noise
- Reconstruction loss measurement (reconstruction vs ablated)
- Optional task-specific accuracy measurement
- Monte-Carlo sampling for statistical significance
- Selectable focus on top-k most important features (for efficiency)

**Ablation Methods:**
- `zero`: Replace feature with 0
- `mean`: Replace with mean activation of that feature
- `noise`: Replace with Gaussian noise (preserves statistics)

**Usage:**
```python
from phase3_causal import CausalAblationEvaluator, FeatureImportanceAnalyzer

evaluator = CausalAblationEvaluator(
    sae=sae_12x,
    model=None,  # Optional for downstream tasks
    device="cuda:0",
    ablation_method="zero",
    num_samplings=3,  # MC samples per feature
)

# Evaluate top 100 features
results = evaluator.evaluate_feature_importance(
    activations=test_activations,
    latents=test_latents,
    feature_ids=range(100),
    task_fn=None,  # No downstream task in this case
    verbose=True,
)

# Analyze
analyzer = FeatureImportanceAnalyzer()
top_features = analyzer.get_top_features(results, metric="loss_diff", top_k=10)
summary = analyzer.compute_statistics(results)
analyzer.save_summary(results, Path("phase3_results/causal_summary_12x.json"))
```

#### `src/phase3_pipeline.py`
**End-to-end orchestration script** that ties all components together.

**Workflow:**
1. Load tokenizer and SAE checkpoints (supports multiple)
2. Extract and align reasoning steps from dataset
3. Load SAE latents for aligned examples
4. Train step-prediction probes (one per SAE expansion)
5. Evaluate causal feature importance (one per SAE expansion)
6. Generate combined report

**CLI Usage:**
```bash
python src/phase3_pipeline.py \
  --model gpt2 \
  --layer 6 \
  --sae-checkpoints \
    checkpoints/sae/sae_768d_4x_final.pt \
    checkpoints/sae/sae_768d_12x_final.pt \
    checkpoints/sae/sae_768d_20x_final.pt \
  --output-dir phase3_results \
  --device cuda:0 \
  --verbose
```

**Python API:**
```python
from phase3_pipeline import Phase3Pipeline
from phase3_config import Phase3Config

config = Phase3Config(
    model_name="gpt2",
    layer=6,
    sae_checkpoints=[...],
    output_dir=Path("phase3_results"),
)

pipeline = Phase3Pipeline(config)
results = pipeline.run()
```

---

## Quick Start

### 1. Basic Usage (GSM8K + 12x SAE)

```bash
cd "RL-Decoder with SAE Features"
python src/phase3_pipeline.py \
  --model gpt2 \
  --layer 6 \
  --sae-checkpoints checkpoints/sae/sae_768d_12x_final.pt \
  --output-dir phase3_results \
  --verbose
```

### 2. Compare Multiple SAE Expansions

```bash
python src/phase3_pipeline.py \
  --model gpt2 \
  --layer 6 \
  --sae-checkpoints \
    checkpoints/sae/sae_768d_4x_final.pt \
    checkpoints/sae/sae_768d_8x_final.pt \
    checkpoints/sae/sae_768d_12x_final.pt \
  --output-dir phase3_results/multi_saes \
  --verbose
```

### 3. Custom Configuration

```python
from phase3_config import Phase3Config
from phase3_pipeline import Phase3Pipeline
from pathlib import Path

config = Phase3Config(
    model_name="gpt2-medium",
    layer=12,
    sae_checkpoints=[Path("my_custom_sae.pt")],
    step_extraction_method="hybrid",
    probe_hidden_dim=256,  # Nonlinear probe
    ablation_method="noise",
    output_dir=Path("custom_results"),
)

pipeline = Phase3Pipeline(config)
results = pipeline.run()
```

---

## Extending to New Models and Datasets

### Adding a New Model

1. Register model in `src/model_registry.py`:
   ```python
   MODEL_REGISTRY = {
       ...
       "llama-7b": ModelInfo(
           hf_id="meta-llama/Llama-2-7b-hf",
           hf_local_path="/path/to/cache",
           num_layers=32,
           hidden_dim=4096,
           default_probe_layer=16,
       ),
   }
   ```

2. Use in Phase 3:
   ```python
   config = Phase3Config(model_name="llama-7b", layer=16, ...)
   ```

### Adding a New Dataset

1. Create alignment class in `phase3_alignment.py`:
   ```python
   class CustomDatasetAligner:
       def process_example(self, ...):
           # Extract reasoning steps
           return ReasoningExample(...)
       
       def process_dataset(self, ...):
           # Load and process all examples
           return examples
   ```

2. Use in pipeline:
   ```python
   class Phase3Pipeline:
       def extract_and_align_steps(self):
           if self.config.dataset == "custom":
               aligner = CustomDatasetAligner(...)
               return aligner.process_dataset()
   ```

### Custom Probe Architecture

Extend `phase3_probes.py`:
```python
class CustomProbe(nn.Module):
    def __init__(self, latent_dim, num_classes):
        # Your custom architecture
        pass

# Then modify ProbeTrainer to use it
```

---

## Output Structure

```
phase3_results/
├── gsm8k_aligned_train.jsonl          # Token-aligned reasoning steps
├── probe_4x.pt                        # Probe checkpoint for 4x SAE
├── probe_12x.pt                       # Probe checkpoint for 12x SAE
├── probe_20x.pt                       # Probe checkpoint for 20x SAE
├── probe_results.json                 # Probe metrics per SAE
├── causal_results_4x.json             # Feature importance for 4x
├── causal_results_12x.json            # Feature importance for 12x
├── causal_summary_4x.json             # Causal summary statistics
├── causal_summary_12x.json
└── phase3_summary.json                # Overall results and metadata
```

---

## Key Concepts

### Reasoning Steps and Alignment

- **Extraction**: Identify reasoning "steps" in text (equations, reasoning blocks, etc.)
- **Alignment**: Map step text back to token positions in the tokenized sequence
- **Confidence**: Alignment confidence score (currently 1.0 for regex, <1 for heuristics)

### Step Types

Default types (customizable via config):
- `step_header`: Explicit step markers ("Step 1:", etc.)
- `equation`: Mathematical expressions
- `comparison`: Logical comparisons (>, <, etc.)
- `reasoning`: Explicit reasoning language ("therefore", "so", etc.)
- `other`: Remaining tokens

### Probe Training

- **Input**: SAE latents at each token position [seq_len, latent_dim]
- **Target**: Step type label for that token
- **Output**: Step type prediction
- **Metric**: Accuracy, F1, leakage gap

### Feature Importance (Causal)

- **Ablation**: Zero out a single SAE feature across the sequence
- **Measurement**: How much does reconstruction/task loss increase?
- **Interpretation**: High importance = feature likely involved in reasoning

---

## Limitations and Future Work

### Current Limitations

1. **Activation loading** not yet integrated (placeholder in pipeline)
   - Need to map aligned examples → activation shard files → token ranges
   - Implement `_load_sae_activations_for_examples()` with proper indexing

2. **Task accuracy metrics** not yet implemented
   - Need to: run model on ablated activations, measure task accuracy change
   - Implement task function for `CausalAblationEvaluator.compute_task_accuracy()`

3. **Simple alignment heuristics**
   - Could improve with learned alignment (attention-based)
   - Could use embedding similarity for step boundaries

### Future Enhancements

- [ ] Learned alignment via attention mechanisms
- [ ] Multi-layer analysis (track features across layers)
- [ ] Attention-based feature attribution
- [ ] Temporal coherence analysis (feature consistency across reasoning steps)
- [ ] Clustering features by reasoning type
- [ ] Interactive visualization dashboard

---

## Configuration Reference

See `src/phase3_config.py` for full parameter list and defaults. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | `"gsm8k"` | Dataset name |
| `model_name` | `"gpt2"` | Model identifier |
| `layer` | `6` | Layer to analyze |
| `step_extraction_method` | `"regex"` | How to find reasoning steps |
| `probe_hidden_dim` | `None` | Linear probe if None, else hidden layer size |
| `ablation_method` | `"zero"` | How to ablate features |
| `compute_feature_importance_top_k` | `100` | Top features to compute exactly |
| `leakage_threshold` | `0.05` | Flag if probe-baseline gap > 5% |

---

## References

- Alignment ideas: Token-level step mapping similar to Transformer Circuits interpretability work
- Probes: Inspired by diagnostic classifiers in mechanistic interpretability
- Causal ablation: Following standards from causal circuits research (Nanda et al., Conmy et al.)
- Leakage detection: From SAE identifiability literature (Anon 2025a)

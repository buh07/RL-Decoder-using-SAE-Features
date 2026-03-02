# Phase 6: RL Decoder with SAE Features — Implementation Plan

## Implementation Status Note (Historical Spec Context)

This document is the design/spec history for Phase 6. For live execution status and latest
empirical results, use `PROJECT_STATUS.md`.

- `phase6/collect_expanded_dataset.py` is implemented as a **fused collector** that performs both
  annotation extraction and feature computation in one pass.
- Phase 6 execution has been completed in this repo (dataset, supervised, RL follow-up, and full
  layer sweep). This file is retained as design context, not the canonical run-status tracker.
- The `compute_features.py` split described below remains a **planned refactor**, not the active
  implementation path.

## 1. Motivation (Summary)

Phases 1–5 established:

- **SAE features encode arithmetic reasoning.** Ridge probes on TopK SAE features at the `=` token predict log(|result|) with R²=0.977 at layer 7 (Phase 4r, Experiment A).
- **The encoding is causal.** Projecting the residual-stream difference Δh between two examples onto the SAE decoder column subspace and patching at the `=` token shifts log-prob by +0.107 toward the target result at layer 22 (Phase 4r, Experiment C, method="subspace"). 21/24 layers show positive causal effect.
- **Direct SAE-space steering does not work.** Three variants — full-space mean-diff (−2.854), probe-subspace mean-diff (−2.385), nearest-neighbour feature transfer (−2.589) — all yield negative Δlog_prob. Root causes: (1) the SAE encode→decode roundtrip introduces reconstruction error larger than the steering signal; (2) the high-C/low-C split is conceptually misaligned with self-improvement.
- **Reasoning is not localised to one layer.** Phase 3 showed single-layer probes cannot decode reasoning connectives (0% per-class accuracy). Arithmetic information flows across layers 7–22, with different layers serving different roles.

**Conclusion:** SAE features *read* the model's arithmetic reasoning reliably (probe) and *causally* (patching), but writing modified features back into the model is destructive. The natural next step is to build an external system that *reads* these features and produces correct answers — an RL-trained decoder that uses SAE features as a substitute for chain-of-thought tokens.

### Related Work

| Paper | Year | Key Idea | Relevance |
|-------|------|----------|-----------|
| Coconut (Hao et al., Meta) | 2024 | Feed hidden states back as continuous thought tokens; implicit BFS in latent space | Proves hidden states carry full reasoning signal without token decoding |
| CoLaR (Xiaomi) | 2025 | "Latent head" predicts next compressed embedding; dynamic compression ratio | Architecture template for our decoder head |
| LaTRO | 2025 | Variational RL: sample rationales, self-reward, update | Training framework for RL stage |
| HRPO | 2025 | Gated hybrid of token embeddings + hidden states; curriculum training | Gating mechanism and progressive curriculum |
| DLR | 2025 | Lightweight assistant generates hypotheses in latent space; frozen main model decodes | Closest architecture to ours (frozen LLM + external decoder) |
| DeepMind SAE Negative Results | 2025 | SAE features underperform raw activations on downstream tasks | Motivates benchmarking SAE features vs raw hidden states |

---

## 2. What We Could Learn

1. **Can an external decoder outperform GPT-2 medium's own next-token prediction for arithmetic?** If yes, SAE features carry exploitable information the model itself fails to use during generation.
2. **Do SAE features outperform raw hidden states as decoder input?** The DeepMind negative result (March 2025) suggests SAEs may lose task-relevant information. Our Phase 4 probe (R²=0.977 on SAE features) sets a high bar, but raw hidden states might be even better. This comparison is critical.
3. **Which layers matter for the decoder?** Phase 4r identified L22 as the causal peak, but the probe peaked at L7. The decoder's learned attention over layers will reveal whether early-layer encoding or late-layer computation is more useful for answer generation.
4. **Is multi-layer input necessary?** If a single layer suffices, SAE features are a practical CoT replacement. If the decoder needs many layers, the "compressed reasoning" claim is weaker.
5. **Does RL fine-tuning improve over supervised training?** If supervised training saturates quickly (small dataset), RL with shaped rewards may unlock better generalisation.
6. **Are the features interpretable in the decoder's decisions?** If the decoder's attention weights align with known arithmetic features (probe top-50), this validates SAE interpretability for downstream use.

---

## 3. Implementation

### 3.1 Data Pipeline

#### 3.1.1 Expand the Dataset

The current dataset has 674 GSM8K arithmetic annotations. This is too small for training a decoder. We need ~5,000–10,000 examples.

**Script:** `phase6/collect_expanded_dataset.py`

```
Source: Full GSM8K train split (7,473 problems, multiple annotations each)
        + GSM8K test split (1,319 problems) for held-out evaluation

Annotation extraction:
  - Parse all <<expr=result>> annotations (same regex as phase4/arithmetic_data_collector.py)
  - Filter: single-token results only (same constraint as Phase 4)
  - Expected yield: ~3,000–5,000 annotations from train, ~500–1,000 from test

Per annotation, store:
  - token_ids, tokens (full sequence)
  - expr_str, C (result value), eq_tok_idx, result_tok_idxs
  - gsm8k_split: "train" or "test" (for held-out eval)
```

#### 3.1.2 Compute Features for Expanded Dataset

**Script:** `phase6/compute_features.py` (planned refactor; current implementation is fused into `collect_expanded_dataset.py`)

For each annotation, run GPT-2 medium forward pass with capture hooks and compute:

```
For each record, for each layer L in 0..23:
  1. raw_hidden[L]     = model.transformer.h[L] output at eq_tok_idx     (1024,)
  2. sae_features[L]   = SAE[L].encode(normalize(raw_hidden[L]))         (12288,)
  3. baseline_logprob  = log_softmax(logits[eq_tok_idx])[result_tok_id]   scalar
  4. baseline_top5     = top-5 token predictions at eq_tok_idx            list

Output format per record:
  {
    "expr_str": "48/2",
    "C": 24.0,
    "result_token_id": 1187,        # GPT-2 token id for "24"
    "eq_tok_idx": 15,
    "token_ids": [...],
    "raw_hidden": Tensor (24, 1024),     # float16, ~49 KB
    "sae_features": Tensor (24, 12288),  # float16, ~590 KB (sparse)
    "baseline_logprob": -3.21,
    "gsm8k_split": "train"
  }

GPU parallelism: shard by record index across 3 GPUs, merge.
Estimated total size: ~5000 records × 640 KB ≈ 3.2 GB
```

**Important:** Store both `raw_hidden` and `sae_features` so we can benchmark both as decoder input (addressing the DeepMind concern).

#### 3.1.3 Train/Val/Test Split

```
Train:  GSM8K train-split annotations, randomly sample 80% → ~3,200 records
Val:    GSM8K train-split annotations, remaining 20%       → ~800 records
Test:   GSM8K test-split annotations (held out entirely)   → ~800 records
```

No overlap between GSM8K problems in train vs test.

---

### 3.2 Decoder Architecture

#### 3.2.1 Input Representation (Three Variants to Benchmark)

| Variant | Input per example | Dim | Rationale |
|---------|-------------------|-----|-----------|
| **A: Raw hidden states** | `raw_hidden[L]` at selected layers | 1024 × N_layers | Baseline: what can you get without SAEs? |
| **B: SAE features (sparse)** | `sae_features[L]` at selected layers | 12288 × N_layers (sparse) | Core hypothesis: SAE features as CoT substitute |
| **C: Hybrid** | Concat `raw_hidden[L]` + top-50 SAE feature activations per layer | (1024 + 50) × N_layers | Best of both: full signal + interpretable decomposition |

For each variant, test two layer selections:
- **Single-layer:** L22 only (causal peak)
- **Multi-layer:** L7, L12, L17, L22 (probe peak + causal layers, 4 layers)

This gives 3 × 2 = 6 configurations to benchmark.

#### 3.2.2 Architecture

**Script:** `phase6/decoder_model.py`

```python
class ArithmeticDecoder(nn.Module):
    """
    Reads SAE features (or raw hidden states) from a frozen GPT-2 medium
    and predicts the arithmetic result token.

    Architecture:
      1. Input projection: per-layer linear projections → d_model
      2. Layer aggregation: small transformer encoder (cross-layer attention)
         OR simple concatenation + MLP (ablation)
      3. Output head: project to vocabulary logits (50257 for GPT-2 tokenizer)
    """

    def __init__(self, config):
        # config fields:
        #   input_dim: 1024 (raw) or 12288 (SAE) or 1074 (hybrid)
        #   n_layers_input: how many LLM layers to read from (1 or 4)
        #   d_model: decoder hidden dim (256 or 512)
        #   n_heads: 4 or 8
        #   n_decoder_layers: 2 or 4
        #   vocab_size: 50257
        #   use_sparse_input: bool (True for SAE features)

        # Per-layer input projection
        self.layer_projections = nn.ModuleList([
            nn.Linear(input_dim, d_model) for _ in range(n_layers_input)
        ])

        # Cross-layer transformer encoder
        # Treats each layer's projected features as one token in a sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True
        )
        self.cross_layer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_decoder_layers
        )

        # Learned [CLS]-style aggregation token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Output head
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, layer_features):
        """
        Args:
            layer_features: list of N_layers tensors, each (batch, input_dim)
                            For SAE input: sparse tensors or dense with many zeros
        Returns:
            logits: (batch, vocab_size)
        """
        # Project each layer's features
        projected = []
        for i, feat in enumerate(layer_features):
            projected.append(self.layer_projections[i](feat))  # (batch, d_model)

        # Stack as sequence: (batch, n_layers + 1, d_model)
        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls] + [p.unsqueeze(1) for p in projected], dim=1)

        # Cross-layer attention
        encoded = self.cross_layer_encoder(seq)  # (batch, n_layers + 1, d_model)

        # Take CLS output
        cls_out = encoded[:, 0, :]  # (batch, d_model)

        return self.output_head(cls_out)  # (batch, vocab_size)
```

For sparse SAE input: use a sparse-to-dense projection that only reads non-zero features. Since TopK SAE has exactly K=3686 active features (out of 12288), the projection effectively sees a 3686-dim input per layer.

```python
class SparseProjection(nn.Module):
    """Efficient projection for sparse SAE features."""
    def __init__(self, latent_dim, d_model):
        self.weight = nn.Parameter(torch.randn(d_model, latent_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, sparse_features):
        # sparse_features: (batch, latent_dim) with ~70% zeros
        # Standard matmul works; PyTorch handles sparsity efficiently on GPU
        return F.linear(sparse_features, self.weight, self.bias)
```

#### 3.2.3 Model Sizes

| Config | d_model | n_heads | n_decoder_layers | Params (approx) |
|--------|---------|---------|------------------|-----------------|
| Small  | 256     | 4       | 2                | ~2M             |
| Medium | 512     | 8       | 4                | ~15M            |

Start with Small. Scale to Medium only if Small underfits.

---

### 3.3 Training

#### 3.3.1 Stage 1: Supervised Warmup

**Script:** `phase6/train_supervised.py`

```
Objective: Cross-entropy loss on the correct result token
  loss = CrossEntropyLoss(decoder_logits, result_token_id)

Optimizer: AdamW, lr=1e-4, weight_decay=0.01
Scheduler: cosine with 100-step warmup
Batch size: 64
Epochs: 50 (with early stopping on val loss, patience=10)
GPU: single RTX 6000 (decoder is small)

Input: pre-computed features from phase6 dataset (no GPT-2 forward pass needed)
  - Load raw_hidden and/or sae_features from .pt file
  - Select layers [7, 12, 17, 22] (or [22] for single-layer)
  - Feed through decoder
  - Compare output logits to result_token_id

Metrics (computed on val set every epoch):
  - Top-1 accuracy: does argmax(logits) == result_token_id?
  - Top-5 accuracy: is result_token_id in top-5 predictions?
  - Mean log-prob of correct token
  - Per-operator breakdown (+, -, *, /)

Baseline comparison:
  - GPT-2 medium's own baseline_logprob at eq_tok (stored in dataset)
  - Random baseline (1/50257 vocab)
  - Majority-class baseline (most common result token in train set)
```

#### 3.3.2 Stage 2: RL Fine-Tuning

**Script:** `phase6/train_rl.py`

Only proceed to Stage 2 if Stage 1 shows the decoder can learn (val accuracy > random baseline).

```
Algorithm: REINFORCE with baseline (or GRPO if batch is large enough)

Reward function:
  r(prediction, target) =
    +1.0  if argmax(logits) == result_token_id       (exact match)
    +0.5  if result_token_id in top-5(logits)         (near miss)
    -0.1  otherwise                                   (wrong)

  Shaped variant (for multi-digit results, future extension):
    +1.0  if decoded number == C
    +0.5  if |decoded_number - C| / |C| < 0.1        (within 10%)
    -0.1  otherwise

Policy: the decoder outputs a distribution over vocab; sample from it
Baseline: running mean of rewards (variance reduction)

Update rule (REINFORCE):
  loss = -log π(a|s) × (r - baseline)
  where a = sampled token, s = SAE features, r = reward

Training:
  - Initialise from Stage 1 checkpoint
  - lr=1e-5 (10x lower than supervised)
  - 20 epochs, evaluate every epoch
  - KL penalty against Stage 1 policy (prevent collapse):
    loss_total = loss_rl + 0.01 * KL(π_current || π_supervised)
```

#### 3.3.3 Training Schedule

```
Phase 6a: Supervised training on 6 configurations (3 input types × 2 layer selections)
          → identify best 1-2 configurations
Phase 6b: RL fine-tuning on the best configuration(s)
Phase 6c: Analysis and ablations
```

Estimated compute per configuration:
- Feature loading: ~10 sec (pre-computed, from disk)
- Training (50 epochs, ~4000 train examples, batch 64): ~15 min on single GPU
- Total for 6 configs: ~90 min

---

### 3.4 Evaluation

**Script:** `phase6/evaluate_decoder.py`

#### 3.4.1 Quantitative Metrics

On held-out GSM8K test split:

| Metric | Description |
|--------|-------------|
| Top-1 accuracy | argmax(decoder) == result_token_id |
| Top-5 accuracy | result_token_id in top-5 |
| Mean log-prob | mean log_softmax(decoder_logits)[result_token_id] |
| Δlog-prob vs GPT-2 | decoder log-prob minus GPT-2's baseline_logprob |
| Per-operator accuracy | Breakdown by +, −, ×, ÷ |
| Per-magnitude accuracy | Breakdown by |C| buckets: [0,10), [10,100), [100,1000), [1000+) |

**Success criterion:** Decoder's mean log-prob of correct token > GPT-2 medium's baseline log-prob at the `=` token position. This means the decoder extracts *more* useful information from the features than GPT-2 does during autoregressive generation.

#### 3.4.2 Ablation Studies

1. **Layer ablation:** Train with all 4 layers, then test with each layer zeroed out. Which layer's removal hurts most?
2. **Feature count ablation:** For SAE input, restrict to top-K probe features (K=10, 25, 50, 100, all). How few features suffice?
3. **SAE vs raw:** Direct comparison of matched-architecture decoders on SAE features vs raw hidden states. Does the SAE decomposition help or hurt?
4. **Attention analysis:** Visualise the cross-layer attention weights. Does the decoder attend most to L22 (consistent with causal patching)?

#### 3.4.3 Interpretability Analysis

**Script:** `phase6/interpret_decoder.py`

For the SAE-input decoder:
1. Compute input gradient: ∂ log_prob(correct) / ∂ sae_features[L][k] for all L, k
2. Rank features by mean |gradient| across test set
3. Compare top-gradient features to Phase 4's top-50 probe features
4. If overlap > 50%: SAE features the decoder relies on are the same ones the probe identified as arithmetic → strong interpretability validation
5. If overlap < 20%: the decoder found different features → SAE interpretability claims need revision

---

### 3.5 File Structure

```
phase6/
├── collect_expanded_dataset.py     # Current path: fused extraction + feature computation
├── compute_features.py             # Planned refactor (split pipeline, not yet implemented)
├── decoder_model.py                # ArithmeticDecoder architecture
├── decoder_config.py               # Hyperparameter configs for 6 variants
├── train_supervised.py             # Stage 1: cross-entropy training
├── train_rl.py                     # Stage 2: REINFORCE fine-tuning
├── evaluate_decoder.py             # Quantitative evaluation on test set
├── interpret_decoder.py            # Gradient-based feature importance analysis
└── merge_results.py                # Aggregate results across configs

phase6_results/
├── dataset/
│   ├── gsm8k_expanded_train.pt     # ~4000 records with features
│   └── gsm8k_expanded_test.pt      # ~800 records with features
├── checkpoints/
│   ├── raw_single_supervised.pt    # Best checkpoint per config
│   ├── raw_multi_supervised.pt
│   ├── sae_single_supervised.pt
│   ├── sae_multi_supervised.pt
│   ├── hybrid_single_supervised.pt
│   └── hybrid_multi_supervised.pt
├── results/
│   ├── supervised_comparison.json   # All 6 configs, val/test metrics
│   ├── rl_results.json             # RL fine-tuning on best config
│   ├── ablation_results.json       # Layer/feature ablations
│   └── interpretability.json       # Gradient feature rankings
└── plots/
    ├── accuracy_comparison.png      # Bar chart: 6 configs + GPT-2 baseline
    ├── layer_attention.png          # Cross-layer attention heatmap
    └── feature_importance.png       # Top features by gradient vs probe overlap
```

---

### 3.6 Execution Plan

| Step | Script | GPU | Est. Time | Depends On |
|------|--------|-----|-----------|------------|
| 1. Generate expanded dataset + features (current fused path, 3 shards) | `collect_expanded_dataset.py` | GPU 0,1,2 | ~2–3 hr | — |
| 2. Train 6 supervised configs | `train_supervised.py` | GPU 0 (sequential) | ~90 min | Step 1 |
| 3. Evaluate supervised | `evaluate_decoder.py` | GPU 0 | ~10 min | Step 2 |
| 4. RL fine-tune best 1–2 configs | `train_rl.py` | GPU 0 | ~30 min | Step 3 |
| 5. Full evaluation + ablations | `evaluate_decoder.py` | GPU 0 | ~20 min | Step 4 |
| 6. Interpretability analysis | `interpret_decoder.py` | GPU 0 | ~10 min | Step 5 |
| **Total** | | | **~5 hours** | |

All steps are automatable. Steps 1–2 can run in background (tmux). Steps 3–7 can be a single pipeline script.

---

### 3.7 Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Dataset too small (674→~4000 train records) | Medium | Data augmentation: swap operand order for commutative ops; use both `eq_features` and `result_features` as input variants |
| SAE features worse than raw hidden states | Medium (DeepMind result) | Benchmark both; if raw wins, report as finding — still validates "internal representations as CoT substitute" even without SAE |
| Decoder memorises train set | Medium | Strict GSM8K problem-level train/test split; report per-problem (not per-annotation) accuracy |
| RL training unstable | Medium | KL penalty against supervised policy; start from strong supervised checkpoint; use simple REINFORCE before trying PPO |
| Top-1 accuracy near zero (vocab too large) | Low | Result tokens are a small subset of vocab (~200 number tokens). Can restrict output head to number tokens only as a fallback |
| Single-token result constraint too limiting | Low for now | Current Phase 4 dataset enforced this. For multi-token results (e.g., "124"), extend decoder to autoregressive generation in Phase 6b |

---

### 3.8 Success Criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| **Minimum viability** | Top-5 accuracy > 10% on test set | Decoder extracts *some* arithmetic signal from features |
| **Positive result** | Decoder log-prob > GPT-2 baseline log-prob | External decoder reads features better than the model uses them in generation |
| **Strong result** | Top-1 accuracy > 30% AND SAE input ≥ raw input | SAE features are a viable CoT substitute with interpretability benefits |
| **Negative result** | Top-5 accuracy ≤ random baseline across all configs | Features do not carry sufficient arithmetic signal for external decoding; revise approach |

A negative result is still publishable: it constrains the conditions under which SAE features can substitute for explicit reasoning.

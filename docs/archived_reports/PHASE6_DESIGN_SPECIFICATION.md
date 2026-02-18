# Phase 6: Controllable Chain-of-Thought SAE Extension
## Technical Design Specification

**Date Created**: 2026-02-17  
**Status**: Design & Planning Phase (Not Yet Implemented)  
**Expected Implementation Time**: 7 hours over 4 days  
**Research Impact**: Publication-level (mechanism + control + universality)

---

## Executive Summary

Phase 5 revealed a critical insight: **Feature selectivity ≠ task importance** (global correlation r=0.335). This is the key limitation Phase 6 addresses by shifting from task-level to **step-level causal testing**.

**Hypothesis**: Step-level ablations will yield much stronger causal signals (r > 0.7) because we're testing features during the operations they actually support, not globally.

**Outcome**: Framework for understanding and controlling reasoning via latent manipulation.

---

## Problem Statement (Why Phase 6 Matters)

### Phase 5 Limitations

| Metric | Result | Limitation |
|--------|--------|-----------|
| Global feature importance | r = 0.335 avg | Weak signal; conflates multiple operations |
| Phi-2 correlation | r = 0.754 | Good for one model but why? |
| GPT-2 correlation | r = 0.161 | Very weak; distributed representations |
| Causal mechanism | Unknown | Can we steer reasoning? |

### Phase 6 Hypothesis

**By testing features at the step they're meant for**, we eliminate noise:

```
Phase 5 (Global):
  Feature X → Affects overall task performance? ← Weak signal
  
Phase 6 (Step-level):
  Feature X → Disrupts DECOMPOSE step? ← Strong signal
  Feature X → Disrupts FORMAT step? ← Orthogonal signal
  
Result: Feature can be unimportant globally but critical for specific steps
```

---

## Technical Architecture

### 1. CoT-Aligned Activation Capture

**Goal**: Map activations to reasoning steps, not just tokens

```python
class CoTActivationCapture:
    """
    Extends Phase 4 capture with step-boundary alignment.
    
    Input: Text with explicit reasoning steps
    Output: Activations tensor + step indices
    """
    
    def __init__(self, model, step_markers=None):
        self.model = model
        self.step_markers = step_markers or {
            'parse': ['problem', 'given', 'input'],
            'decompose': ['step', 'first', 'then', 'next'],
            'operate': ['calculate', 'compute', 'solve', 'apply'],
            'verify': ['check', 'verify', 'confirm', 'correct'],
            'format': ['answer', 'therefore', 'result']
        }
        
    def detect_step_boundaries(self, tokens, token_ids):
        """
        Identify where reasoning steps occur in token sequence.
        
        Returns: [(start_idx, end_idx, step_type), ...]
        """
        boundaries = []
        current_step = None
        
        for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
            # Check if token initiates a step transition
            for step_type, markers in self.step_markers.items():
                if token.lower() in markers:
                    if current_step != step_type:
                        boundaries.append((i, step_type))
                        current_step = step_type
                    break
        
        return boundaries
    
    def capture_with_alignment(self, text_with_steps, layer=16):
        """
        Main capture pipeline with step alignment.
        """
        # Tokenize
        tokens = self.model.tokenizer.encode(text_with_steps)
        
        # Forward pass with hooks
        activations = []  # Shape: (seq_len, hidden_dim)
        with torch.no_grad():
            outputs = self.model(tokens, output_hidden_states=True)
            layer_acts = outputs.hidden_states[layer]
        
        # Detect step boundaries
        step_boundaries = self.detect_step_boundaries(
            self.model.tokenizer.decode(tokens),
            tokens
        )
        
        # Align: activations → steps
        step_activations = self._align_activations_to_steps(
            layer_acts,
            step_boundaries
        )
        
        return layer_acts, step_activations, step_boundaries
```

**Data Format**:
```
Input:  "Problem: 2+3*4. Step 1: Apply PEMDAS. Multiply: 3*4=12. 
         Step 2: Add: 2+12=14. Answer: 14"
         
Output: {
    'activations': Tensor(seq_len, hidden_dim),
    'steps': {
        'parse': (0, 5),        # Token indices for "Problem: 2+3*4"
        'decompose': (5, 15),   # "Step 1: Apply PEMDAS"
        'operate': (15, 30),    # "Multiply: 3*4=12"
        'operate': (30, 40),    # "Add: 2+12=14"
        'format': (40, 45),     # "Answer: 14"
    }
}
```

---

### 2. CoT-Aware SAE Architecture

**Goal**: Train SAE to predict reasoning steps from latents (auxiliary task)

```python
class CoTAwareSAE(SparseAutoencoder):
    """
    Extends Phase 4 SAE with step-prediction auxiliary loss.
    
    Primary: Reconstruct activations
    Auxiliary: Predict next reasoning step
    
    Hypothesis: If latents encode step info, step prediction = supervision
    """
    
    def __init__(self, 
                 input_dim=2560,
                 hidden_dim=20480,
                 n_steps=5,  # parse, decompose, operate, verify, format
                 weight_auxiliary=0.3):
        super().__init__(input_dim, hidden_dim)
        
        self.step_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_steps)
        )
        self.weight_auxiliary = weight_auxiliary
        self.step_encoder = self._create_step_encoder()
    
    def _create_step_encoder(self):
        """One-hot encode step types."""
        return {
            'parse': torch.tensor([1, 0, 0, 0, 0]),
            'decompose': torch.tensor([0, 1, 0, 0, 0]),
            'operate': torch.tensor([0, 0, 1, 0, 0]),
            'verify': torch.tensor([0, 0, 0, 1, 0]),
            'format': torch.tensor([0, 0, 0, 0, 1])
        }
    
    def compute_loss(self, x, step_labels=None):
        """
        Total loss = Reconstruction + Auxiliary step prediction
        
        Args:
            x: Activation batch (batch_size, input_dim)
            step_labels: Step type for each sample (batch_size,)
        """
        # Reconstruction
        latents = self.encode(x)
        x_recon = self.decode(latents)
        recon_loss = torch.nn.functional.mse_loss(x_recon, x)
        
        # Sparsity
        sparsity_loss = 1e-4 * torch.norm(latents, p=1, dim=1).mean()
        
        # Auxiliary: step prediction
        auxiliary_loss = 0
        if step_labels is not None:
            step_preds = self.step_predictor(latents)
            step_labels_encoded = torch.stack([
                self.step_encoder[label] for label in step_labels
            ]).to(x.device)
            auxiliary_loss = torch.nn.functional.cross_entropy(
                step_preds, 
                step_labels_encoded
            )
        
        # Weighted total
        total_loss = (
            recon_loss + 
            sparsity_loss + 
            self.weight_auxiliary * auxiliary_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'auxiliary_loss': auxiliary_loss,
            'step_accuracy': self._compute_step_accuracy(step_preds, step_labels)
            if step_labels is not None else 0
        }
    
    def _compute_step_accuracy(self, preds, labels):
        """Compute step prediction accuracy."""
        pred_steps = preds.argmax(dim=1)
        label_ints = torch.tensor([
            ['parse', 'decompose', 'operate', 'verify', 'format'].index(l)
            for l in labels
        ]).to(preds.device)
        return (pred_steps == label_ints).float().mean()
```

**Training**:
```python
# Phase 6.2 training loop
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

for epoch in range(20):
    for batch in dataloader:
        x, step_labels = batch  # (batch_size, hidden_dim), (batch_size,)
        
        loss_dict = sae.compute_loss(x, step_labels)
        loss_dict['total_loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log: reconstruction and step prediction accuracy
        log(epoch, loss_dict)

# Success if: recon_loss ≈ Phase 4, step_accuracy > 75% (vs. 20% random)
```

---

### 3. Step-Level Causal Ablation Testing

**Goal**: Measure which features are critical for each reasoning step

```python
class StepLevelCausalTester:
    """
    Test causal importance of features at the step they're meant for.
    
    Logic: Feature X is important for DECOMPOSE if:
      - Zeroing X during DECOMPOSE steps → quality drop > threshold
      - Zeroing X during other steps → quality unchanged
    """
    
    def __init__(self, model, sae, step_activations):
        self.model = model
        self.sae = sae
        self.step_activations = step_activations  # {step: Tensor(n, hidden_dim)}
    
    def test_feature_for_step(self, feature_idx, step_type):
        """
        Measure impact of feature on a specific step.
        
        Returns: quality_drop_ratio (0-1)
        """
        # Get baseline quality for this step (reconstructed activations)
        step_acts = self.step_activations[step_type]
        baseline_recon = self.sae.decode(self.sae.encode(step_acts))
        baseline_quality = torch.nn.functional.mse_loss(
            baseline_recon, step_acts
        )
        
        # Ablate: zero out feature
        latents = self.sae.encode(step_acts)
        latents_ablated = latents.clone()
        latents_ablated[:, feature_idx] = 0
        
        recon_ablated = self.sae.decode(latents_ablated)
        ablated_quality = torch.nn.functional.mse_loss(
            recon_ablated, step_acts
        )
        
        # Quality drop
        quality_drop = (ablated_quality - baseline_quality) / (baseline_quality + 1e-6)
        
        return float(quality_drop)
    
    def get_step_importance_ranking(self, step_type, n_top=50):
        """
        Rank all features by importance for a specific step.
        
        Returns: [(feature_idx, importance_score), ...]
        """
        rankings = []
        
        for feat_idx in range(self.sae.hidden_dim):
            importance = self.test_feature_for_step(feat_idx, step_type)
            rankings.append((feat_idx, importance))
        
        rankings.sort(key=lambda x: -x[1])
        return rankings[:n_top]
    
    def get_step_importance_matrix(self):
        """
        Create matrix: steps × features → importance
        
        Returns: Dict[step_type, Dict[feature_idx, importance]]
        """
        steps = ['parse', 'decompose', 'operate', 'verify', 'format']
        matrix = {}
        
        for step in steps:
            rankings = self.get_step_importance_ranking(step, n_top=50)
            matrix[step] = {feat: imp for feat, imp in rankings}
        
        return matrix
```

**Expected Output**:
```json
{
    "parse": {
        "feature_123": 0.45,
        "feature_456": 0.38,
        "feature_789": 0.28
    },
    "decompose": {
        "feature_111": 0.62,
        "feature_222": 0.55,
        "feature_333": 0.42
    },
    "operate": {
        "feature_444": 0.71,
        "feature_555": 0.68,
        "feature_666": 0.51
    },
    "verify": {
        "feature_777": 0.44,
        "feature_888": 0.39,
        "feature_999": 0.35
    },
    "format": {
        "feature_abc": 0.29,
        "feature_def": 0.25,
        "feature_ghi": 0.22
    }
}
```

**Key Insight**: Same feature (e.g., 123) might be:
- Critical for PARSE (0.45 importance)
- Unnecessary for OPERATE (<0.01 importance)
- This explains Phase 5 global weak signal!

---

### 4. Controllable Reasoning via Latent Steering

**Goal**: Prove we can control reasoning by manipulating latents

```python
class LatentSteering:
    """
    Steer reasoning properties by amplifying/dampening important features.
    """
    
    def __init__(self, model, sae, step_importance_matrix):
        self.model = model
        self.sae = sae
        self.step_importance_matrix = step_importance_matrix
    
    def increase_reasoning_length(self, text, target_length_ratio=1.5):
        """
        Make reasoning longer by boosting DECOMPOSE features.
        
        Logic: More decomposition → more steps
        """
        # Get important features for DECOMPOSE
        decompose_features = self.step_importance_matrix['decompose']
        top_features = sorted(
            decompose_features.items(),
            key=lambda x: -x[1]
        )[:20]  # Top 20 features
        
        # Forward pass with latent capture
        with torch.no_grad():
            # Get original latents
            tokens = self.model.tokenizer.encode(text)
            outputs = self.model(tokens, output_hidden_states=True)
            activations = outputs.hidden_states[-2]  # Second-to-last layer
            
            # Encode to SAE latents
            latents = self.sae.encode(activations)
            
            # Amplify top decompose features
            for feat_idx, _ in top_features:
                latents[:, feat_idx] *= 1.5  # 50% amplification
            
            # Generate with boosted latents
            # (This requires modified generation loop - see implementation details)
            ...
        
        return controlled_output
    
    def improve_reasoning_quality(self, text, target_accuracy_ratio=1.3):
        """
        Improve accuracy by boosting VERIFY features.
        """
        verify_features = self.step_importance_matrix['verify']
        top_features = sorted(
            verify_features.items(),
            key=lambda x: -x[1]
        )[:15]
        
        # Similar amplification process
        # Expected: More checking → fewer arithmetic errors
        ...
    
    def force_failure_for_testing(self, text):
        """
        Zero out critical PARSE features to test failure mode.
        
        Expected: Model misses problem constraints
        """
        parse_features = self.step_importance_matrix['parse']
        critical_features = sorted(
            parse_features.items(),
            key=lambda x: -x[1]
        )[:10]
        
        # Zero them out
        latents[:, [f[0] for f in critical_features]] = 0
        
        # Expected failure mode
        ...
```

**Validation Metrics**:
```python
# Length control
original_steps = count_reasoning_steps(model(text))
steered_steps = count_reasoning_steps(model_with_steering(text))
length_ratio = steered_steps / original_steps
assert length_ratio > 1.3, "Should increase by 30%+"

# Quality control  
original_accuracy = measure_accuracy(model(text))
steered_accuracy = measure_accuracy(model_with_boost(text))
quality_improvement = steered_accuracy / original_accuracy
assert quality_improvement > 1.2, "Should improve by 20%+"

# Predictability
ablated_output = model_with_zeros(text)
assert is_predictably_broken(ablated_output), "Failure should be interpretable"
```

---

### 5. Cross-Model Reasoning Universality Analysis

**Goal**: Identify which features are universal vs. model-specific

```python
class CrossModelUniversalityAnalysis:
    """
    Compare step-level feature importance across models.
    
    Question: Do different models use the same features for same reasoning steps?
    """
    
    def __init__(self, step_importance_matrices):
        """
        step_importance_matrices: Dict[model_name, Dict[step, Dict[feat, importance]]]
        """
        self.matrices = step_importance_matrices
        self.models = list(step_importance_matrices.keys())
    
    def compute_feature_overlap(self, step_type, top_k=20):
        """
        For a given step, how much do models agree on which features matter?
        
        Returns: Overlap percentage (how many top-20 features are shared)
        """
        top_features_by_model = {}
        
        for model in self.models:
            features = self.matrices[model][step_type]
            top_features = sorted(
                features.items(),
                key=lambda x: -x[1]
            )[:top_k]
            top_features_by_model[model] = set([f[0] for f in top_features])
        
        # Pairwise intersection
        overlaps = {}
        for i, model1 in enumerate(self.models):
            for model2 in self.models[i+1:]:
                intersection = (
                    top_features_by_model[model1] &
                    top_features_by_model[model2]
                )
                overlap_ratio = len(intersection) / top_k
                overlaps[f"{model1}↔{model2}"] = overlap_ratio
        
        return overlaps
    
    def compute_correlation_by_step(self):
        """
        Spearman correlation of feature importance across models by step.
        
        Expected pattern:
        - PARSE, DECOMPOSE, OPERATE: Higher correlation (universal operations)
        - FORMAT: Lower correlation (model-specific preferences)
        """
        steps = ['parse', 'decompose', 'operate', 'verify', 'format']
        correlations = {}
        
        for step in steps:
            model_pairs = []
            for i, model1 in enumerate(self.models):
                for model2 in self.models[i+1:]:
                    # Extract importance scores for same features
                    feats1 = self.matrices[model1][step]
                    feats2 = self.matrices[model2][step]
                    
                    common_features = set(feats1.keys()) & set(feats2.keys())
                    
                    scores1 = [feats1[f] for f in common_features]
                    scores2 = [feats2[f] for f in common_features]
                    
                    corr = scipy.stats.spearmanr(scores1, scores2)[0]
                    model_pairs.append((f"{model1}↔{model2}", corr))
            
            correlations[step] = model_pairs
        
        return correlations
    
    def generate_universality_report(self):
        """
        Comprehensive analysis: Are reasoning features universal or model-specific?
        """
        overlap_by_step = {}
        correlation_by_step = {}
        
        for step in ['parse', 'decompose', 'operate', 'verify', 'format']:
            overlap_by_step[step] = self.compute_feature_overlap(step, top_k=20)
            
        correlation_by_step = self.compute_correlation_by_step()
        
        report = {
            'overlap_analysis': overlap_by_step,
            'correlation_analysis': correlation_by_step,
            'summary': {
                'hypothesis_universal_reasoning': (
                    correlation_by_step.get('operate', [{}])[0].get(
                        list(correlation_by_step.values())[0][0][0], 0
                    ) > 0.6
                )
            }
        }
        
        return report
```

**Expected Results**:
```
Universality Matrix:
═════════════════════════════════════════════════════════════
Step Type  │ Feature Overlap │ Correlation │ Interpretation
═════════════════════════════════════════════════════════════
PARSE      │ 0.75 (15/20)   │ 0.68        │ Mostly universal
DECOMPOSE  │ 0.80 (16/20)   │ 0.72        │ Highly universal
OPERATE    │ 0.85 (17/20)   │ 0.75        │ Highly universal ✅
VERIFY     │ 0.60 (12/20)   │ 0.45        │ Mixed
FORMAT     │ 0.35 (7/20)    │ 0.22        │ Model-specific ❌
═════════════════════════════════════════════════════════════

Interpretation:
- Core reasoning (parse → operate) uses shared features
- Output formatting is model-specific
- Reasoning is partially universal (can transfer features for decompose/operate)
```

---

## Implementation Roadmap

### Day 1: Capture & Architecture (2 hours)
```python
# Task 6.1: Step-aligned capture
python phase6/phase6_cot_capture.py \
  --model phi-2 \
  --data datasets/gsm8k_with_cot.jsonl \
  --output phase6_results/activations/phi2_cot_aligned.pt

# Task 6.2: CoT-aware SAE
python phase6/phase6_cot_sae.py \
  --activations phase6_results/activations/ \
  --epochs 20 \
  --weight_auxiliary 0.3 \
  --output phase6_results/checkpoints/cot_sae_phi2.pt
```

### Day 2: Step-Level Causal Testing (1.5 hours)
```python
# Task 6.3: Causal ablation by step
python phase6/phase6_step_causal_test.py \
  --activations phase6_results/activations/ \
  --sae phase6_results/checkpoints/cot_sae_phi2.pt \
  --output phase6_results/causal_by_step/
```

### Day 3: Steering Experiments (1 hour)
```python
# Task 6.4: Latent steering
python phase6/phase6_latent_steering.py \
  --model phi-2 \
  --sae phase6_results/checkpoints/cot_sae_phi2.pt \
  --importance_matrix phase6_results/causal_by_step/ \
  --experiments length_control quality_control failure_mode \
  --output phase6_results/steering_experiments/
```

### Day 4: Cross-Model Analysis (1.5 hours)
```python
# Task 6.5: Universality analysis
python phase6/phase6_cross_model_reasoning.py \
  --importance_matrices phase6_results/causal_by_step/ \
  --models phi-2 pythia-1.4b gpt2-medium gemma-2b \
  --output phase6_results/universality_analysis/
```

---

## Success Criteria & Validation

### Technical Thresholds
| Metric | Phase 5 Baseline | Phase 6 Target | Validation |
|--------|------------------|----------------|-----------|
| Feature importance correlation | r = 0.335 | r > 0.7 | Step-specific > global |
| Step prediction accuracy | N/A | >75% | Auxiliary task validation |
| Reasoning length control | N/A | >30% increase | Steered output analysis |
| Quality improvement via steering | N/A | >20% | Accuracy measurement |
| Feature overlap across models | N/A | >60% for core ops | Universality matrix |

### Publication Readiness
- [ ] Step-level correlations validated on all 4 models
- [ ] Steering experiments produce interpretable, controllable outputs
- [ ] Cross-model analysis answers universality question
- [ ] Results reproduce across dataset seeds
- [ ] Code documented and reproducible

---

## Risks & Mitigations

### Risk 1: Step-level correlations still weak
**Mitigation**: Implement finer-grained steps (substeps within OPERATE: "multiply" vs. "add")

### Risk 2: Steering doesn't work (latent modifications fail to control output)
**Mitigation**: Requires modified generation loop; fallback to frozen-latent inference

### Risk 3: No clear universality pattern
**Mitigation**: Still publishable as "model-specific reasoning"; explains diversity

### Risk 4: Computational cost exceeds budget
**Mitigation**: Focus on 2 models (Phi-2 + Pythia); skip Gemma/GPT-2 if needed

---

## Expected Outcomes & Publications

### Paper 1: "Interpretability vs. Causality in Neural Reasoning"
- Explain Phase 5 paradox: Why Phi-2 has causal features but GPT-2 doesn't
- Use Phase 6 step-level data as evidence
- Implications for interpretability research

### Paper 2: "Steering Reasoning Without Fine-Tuning"
- Real demonstrations of controllable reasoning via latent manipulation
- Practical applications for model alignment and debugging
- Experimental validation of steering efficacy

### Paper 3: "Are Reasoning Features Universal?" (Optional)
- Combines Phase 5.3 (feature transfer) + Phase 6.5 (cross-model step analysis)
- Answers fundamental question about compositional reasoning

---

## References to Existing Work

- **Phase 4**: SAE training infrastructure (reuse wholesale)
- **Phase 4B**: Feature statistics and sparsity validation
- **Phase 5**: Methodology for importance testing and semantic descriptions
- **Phase 5.3**: Feature transfer baseline (feed Phase 6 universality analysis)

---

## Appendix: Code Templates

Ready-to-implement stubs for all Phase 6 components are provided in pseudocode above. Full implementation requires:

1. Integration with Phase 4 model/dataset infrastructure
2. Step boundary detection heuristics (regex-based or transformer-based)
3. Modified generation loop for latent steering (requires model internals)
4. Statistical testing and visualization (matplotlib/seaborn)

**Estimated time to full code**: 5-7 hours for experienced practitioner.


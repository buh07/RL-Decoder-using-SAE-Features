# Interpretable LLM Architecture: Complete Implementation Template

**Project Title**: Building a Faithful Deep-Layer Decoder for Human-Interpretable LLM Reasoning via Sparse Features and Reinforcement Learning

**Status**: Implementation Template  
**Last Updated**: January 25, 2026  
**Primary Goal**: Create an interpretable LLM architecture that reveals intermediate reasoning through RL-trained sparse feature decoders

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Design](#architecture-design)
3. [Phase 1: Setup & Preparation](#phase-1-setup--preparation)
4. [Phase 2: Sparse Autoencoder (SAE) Training](#phase-2-sparse-autoencoder-training)
5. [Phase 3: Supervised Decoder Pre-training](#phase-3-supervised-decoder-pre-training)
6. [Phase 4: RL-Based Decoder Fine-tuning](#phase-4-rl-based-decoder-fine-tuning)
7. [Phase 5: Feature Interpretation & Labeling](#phase-5-feature-interpretation--labeling)
8. [Phase 6: Human-Interpretable Output Generation](#phase-6-human-interpretable-output-generation)
9. [Validation & Testing](#validation--testing)
10. [Implementation Checklist](#implementation-checklist)
11. [Key Challenges & Mitigations](#key-challenges--mitigations)
12. [Success Metrics](#success-metrics)
13. [References & Prior Work](#references--prior-work)

---

## Project Overview

### The Problem You're Solving

Large language models make predictions through opaque, distributed computations across dozens of layers. While chain-of-thought (CoT) explanations provide post-hoc reasoning steps, they are **fundamentally unfaithful** to how the model actually computes—transformers reason in parallel, not sequentially, and intermediate CoT steps often contain errors that the model implicitly corrects without updating the text.

This project directly observes the model's **true intermediate reasoning** by:
1. Extracting sparse, monosemantic features from mid-layer hidden states via Sparse Autoencoders
2. Training a lightweight decoder with RL to predict next tokens while maintaining faithfulness
3. Converting extracted features into human-readable explanations

### Core Hypothesis

**Mid-layer hidden states (layers 8-10 for 32-layer models, optimized per scale) encode sufficient semantic information to reconstruct the model's reasoning when decoded via sparse, interpretable features trained with multi-objective RL.**

### Success Criteria

- **Faithfulness**: KL divergence between RL-trained decoder and frozen model < 0.5 nats
- **Interpretability**: >85% of SAE features labeled with high-confidence semantic meaning
- **Accuracy**: <3% degradation in task performance vs. frozen model
- **Generalization**: >80% OOD accuracy (held-out task domains)
- **Sparsity**: 50-100 active SAE features per token (maintaining interpretability)

---

## Architecture Design

### High-Level System Diagram

```
Pretrained LLM (frozen, generative - e.g., Llama-7B to 70B)
    ↓ [Extract hidden states @ layer L]
    ↓ Layer Selection: L = optimal_semantic_layer(model_size)
    ↓   For 24-32L: L ≈ 8-10
    ↓   For 48-64L: L ≈ 16-20
    ↓   For 96-128L: L ≈ 24-32 (to be determined)
    
Residual Stream Activations (d_model = 4096 for 7B models)
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 1: Feature Extraction                     │
│ Sparse Autoencoder (SAE)                        │
│   Input: residual stream (4096)                 │
│   Encoder: Linear → 40k latents (10x expansion) │
│   Bottleneck: Sparse (L0 penalty during train)  │
│   Decoder: 40k → 4096 reconstruction            │
│ Output: ~50-100 active features per token       │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 2: RL-Trained Decoder                     │
│ Architecture:                                   │
│   Input: Sparse SAE features (40k dims)         │
│   Dense layer: 40k → 4096 (or linear → vocab)   │
│   Optional: Multi-head attention over features  │
│ Training: PPO with multi-objective rewards      │
│   - Faithfulness (KL to frozen model)           │
│   - Monosemanticity (sparsity + consistency)    │
│   - Task accuracy (verifiable correctness)      │
│ Output: Logit distribution for next token       │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 3: Feature Interpretation                 │
│ Processes:                                      │
│   - Activation patching (importance scoring)    │
│   - LLM-based feature labeling                  │
│   - Semantic clustering                         │
│ Output: feature_id → human_label mapping        │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Stage 4: Human-Readable Output Generation       │
│ Options:                                        │
│   A. Feature importance ranking (no training)   │
│   B. Trained explanation decoder (supervised)   │
│   C. LLM verbalization (dynamic, expensive)     │
│ Output: Natural language explanation            │
└─────────────────────────────────────────────────┘
    ↓
Final Output to User:
  - Next token prediction
  - Feature activation trace (which features fired)
  - Human-readable reasoning explanation
  - Confidence scores / uncertainty estimates
```

### Key Design Decisions

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Layer Selection** | Mid-layers (8-10 for 32L) | Peak semantic coherence; maximum info density |
| **Feature Space** | 10x SAE expansion | Balance between interpretability and expressiveness |
| **Training Algorithm** | PPO (not RLOO/DPO) | Stable on multi-objective; critic helps with sparse rewards; KL regularization prevents hacking |
| **Feature Extraction** | Unsupervised SAEs (not probes) | Monosemantic features > dense neurons; generalizes across domains |
| **Faithfulness Metric** | KL divergence (not output matching) | Asymmetric penalty for confident wrong predictions |
| **Output Generation** | Hybrid (ranking + trained decoder) | Start simple (interpretable), add sophistication (naturalness) |

---

## Phase 1: Setup & Preparation

### 1.1 Environment Configuration

**Hardware Requirements**:
- **SAE Training**: 1x A100 (40GB) minimum; 2x preferred for parallelization
- **RL Training**: 2x A100 (80GB total) for batch size 32, gradient accumulation
- **Feature Labeling**: 1x A100 or CPU-only (embarrassingly parallel)
- **Inference**: 1x V100 or A10 sufficient (small decoder)
- **Total Training Time**: 2-3 weeks for full pipeline on 7B model

**Software Stack**:
```
PyTorch 2.0+
Transformers (HuggingFace)
TRL (for RL training utilities)
Wandb (experiment tracking)
SAE training code (Anthropic/OpenAI open-source)
```

### 1.2 Model Selection

**Recommended Starting Model**: Llama-3.2-1B or Llama-3.1-7B
- Proven interpretability characteristics
- Manageable training overhead
- Published SAE results for validation

**Scaling**: Plan upgrades
- Week 1-2: Llama-3.2-1B (rapid prototyping)
- Week 3-4: Llama-3.1-7B (publication results)
- Week 5+: Llama-3.1-13B or 70B (frontier results, if feasible)

### 1.3 Data Preparation

**Training Data Sources**:
- **Pretraining Dataset**: Use model's original training distribution (if accessible)
- **Task-Specific Data**: Collect for evaluation benchmarks
  - Arithmetic: GSM8K subset, MATH
  - Logic: BoolQ, LogiQA
  - Reading comprehension: SQuAD-subset, RACE
  - General: Next-token prediction on diverse documents

**Data Split**:
- Supervised pretraining: 100M tokens
- RL training: 10B tokens (can reuse/cycle through batches)
- Evaluation (clean): 100k tokens from held-out domain
- OOD evaluation: 50k tokens from different domain

**Storage**:
- Raw data: ~50-100 GB (depending on tokenization)
- Preprocessed shards: ~200 GB
- SAE checkpoints: ~50 GB per layer
- RL checkpoints: ~30 GB per iteration

### 1.4 Baseline Measurements

Before starting training, record baselines for later comparison:

```yaml
Frozen LLM Baseline:
  - Next-token accuracy on held-out data: ____%
  - CoT reasoning quality (BLEU vs human): ____
  - Speed (tokens/sec): ____
  - Memory (GB): ____

Layer-wise Logit Lens:
  - For layers 4, 8, 12, 16, 20, 24, 28, 32:
    - Next-token accuracy at that layer: ____%
    - KL divergence from final layer: ____ nats

Task-Specific Baselines:
  - GSM8K accuracy: ____%
  - BoolQ accuracy: ____%
  - SQuAD F1: ____
```

Record these so you can measure improvement.

---

## Phase 2: Sparse Autoencoder (SAE) Training

### 2.1 SAE Architecture & Hyperparameters

**SAE Configuration** (for 7B model, layer 8):

```python
class SparseAutoencoder(nn.Module):
    def __init__(self):
        # Dimensions
        self.d_in = 4096  # residual stream width
        self.d_sae = 40960  # SAE dictionary (10x expansion)
        
        # Encoder: residual_stream → sparse latents
        self.encoder = nn.Linear(self.d_in, self.d_sae)
        
        # Decoder: sparse latents → residual_stream reconstruction
        self.decoder = nn.Linear(self.d_sae, self.d_in)
        
        # Learned bias for sparsity control
        self.bias = nn.Parameter(torch.zeros(self.d_sae))
        
        # Dead neuron detector
        self.dead_feature_count = torch.zeros(self.d_sae)

SAE_HYPERPARAMS = {
    'expansion_ratio': 10,          # 10x larger than input (proven optimal for LLMs)
    'sparsity_coefficient': 0.01,   # L0 penalty strength (λ in SAE loss)
    'learning_rate': 1e-4,          # Conservative (SAE training is sensitive)
    'batch_size': 128,              # Large batches for sparse loss stability
    'num_steps': 50000,             # ~100-200B tokens of training
    'activation_fn': 'relu',        # ReLU for sparsity
    'use_gradient_checkpointing': True,  # Memory efficiency
    'warm_up_steps': 1000,          # Gradual sparsity increase
    'resample_dead_neurons': True,  # Restart dead features
    'dead_feature_threshold': 1e-7, # Below this activation = dead
}

TRAINING_SCHEDULE = {
    'phase_1': {'sparsity_coeff': 0.001, 'steps': 10000},  # Warm up
    'phase_2': {'sparsity_coeff': 0.01, 'steps': 30000},   # Main training
    'phase_3': {'sparsity_coeff': 0.001, 'steps': 10000},  # Fine-tune sparsity
}
```

### 2.2 SAE Training Loop

**Pseudocode**:

```python
frozen_lm = load_model("llama-7b").eval()
sae = SparseAutoencoder()
optimizer = Adam(sae.parameters(), lr=SAE_HYPERPARAMS['learning_rate'])

for step in range(SAE_HYPERPARAMS['num_steps']):
    # Get batch of data
    batch = next(dataloader)
    tokens = frozen_lm.tokenize(batch)
    
    # Forward through frozen LLM (no grad)
    with torch.no_grad():
        hidden_states = frozen_lm.get_hidden_states(tokens, layer=8)  # (batch, seq_len, 4096)
    
    # SAE forward pass
    sae_latents = sae.encode(hidden_states)  # (batch, seq_len, 40960)
    sae_reconstruction = sae.decode(sae_latents)  # (batch, seq_len, 4096)
    
    # Compute SAE loss
    reconstruction_loss = mse_loss(sae_reconstruction, hidden_states)
    sparsity_penalty = SAE_HYPERPARAMS['sparsity_coefficient'] * l0_norm(sae_latents)
    loss_sae = reconstruction_loss + sparsity_penalty
    
    # Backward pass
    loss_sae.backward()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    # Monitoring
    if step % 100 == 0:
        log(f"Step {step}: recon_loss={reconstruction_loss:.4f}, "
            f"sparsity={l0_norm(sae_latents):.1f}, loss_total={loss_sae:.4f}")
    
    # Resample dead neurons every 1000 steps
    if step % 1000 == 0:
        sae.resample_dead_neurons(threshold=SAE_HYPERPARAMS['dead_feature_threshold'])
    
    # Adjust sparsity schedule
    if step == 10000:
        optimizer.param_groups[0]['lr'] = 1e-5  # Reduce LR mid-training

# Save checkpoint
save_checkpoint(sae, f"sae_layer8_step50k.pt")
```

### 2.3 SAE Quality Validation

**What to measure**:

```yaml
Reconstruction Quality:
  - MSE between input and reconstruction: < 0.01
  - KL divergence vs. original activations: < 0.5 nats
  - Explained variance ratio: > 0.95
  
Sparsity:
  - Average L0 per sample: 50-100 active features
  - Dead feature ratio: < 5%
  - Feature usage entropy: 1.0-1.5 (not all features used equally, but distributed)

Feature Quality (on sample of 100 features):
  - Average interpretability score (human judgment): > 0.8/1.0
  - Coverage: Top-10 features cover ~30% of mass
  - No obvious duplicates: < 5% semantic overlap
```

### 2.4 Multi-Layer SAE (Optional, for robustness)

Train SAEs on 3-5 layers (8, 10, 12, 14, 16) to compare:
- Which layers have cleanest interpretable features?
- Do features transfer across layers?
- Use best-performing layer for RL training

**Decision Rule**: Choose layer with highest explained variance + highest interpretability.

---

## Phase 3: Supervised Decoder Pre-training

### 3.1 Decoder Architecture

```python
class RLDecoderBase(nn.Module):
    def __init__(self, sae_dim=40960, vocab_size=128000, hidden_dim=512):
        # Simple linear decoder (start here)
        self.decoder_linear = nn.Linear(sae_dim, vocab_size)
        
        # Optional: lightweight transformer for more expressivity
        self.use_transformer = False  # Try this if linear underfits
        if self.use_transformer:
            self.transformer = nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=2048
            )
            self.input_proj = nn.Linear(sae_dim, hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, sae_features):
        # sae_features: (batch, seq_len, sae_dim)
        if not self.use_transformer:
            logits = self.decoder_linear(sae_features)  # (batch, seq_len, vocab_size)
        else:
            hidden = self.input_proj(sae_features)
            hidden = self.transformer(hidden)
            logits = self.output_proj(hidden)
        return logits

DECODER_HYPERPARAMS = {
    'architecture': 'linear',  # Start simple, upgrade if needed
    'learning_rate': 5e-5,     # Small (finetuning, not pretraining)
    'batch_size': 32,
    'num_epochs': 3,
    'max_grad_norm': 1.0,
    'use_mixed_precision': True,  # FP16 training
    'gradient_accumulation_steps': 1,
}
```

### 3.2 Supervised Pre-training

**Goal**: Initialize decoder to be faithful to frozen model (supervised baseline).

```python
decoder = RLDecoderBase()
optimizer = AdamW(decoder.parameters(), lr=DECODER_HYPERPARAMS['learning_rate'])
sae = load_checkpoint("sae_layer8_step50k.pt")
frozen_lm = load_model("llama-7b").eval()

for epoch in range(DECODER_HYPERPARAMS['num_epochs']):
    for batch in train_loader:
        tokens = frozen_lm.tokenize(batch)
        
        # Get ground truth from frozen model
        with torch.no_grad():
            hidden_states = frozen_lm.get_hidden_states(tokens, layer=8)
            sae_features = sae.encode(hidden_states)
            logits_frozen = frozen_lm.get_logits(hidden_states)
        
        # Decoder predictions
        logits_decoder = decoder(sae_features)
        
        # Supervised loss: match frozen model's predictions
        loss = cross_entropy_loss(logits_decoder, logits_frozen.argmax(dim=-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 
                                       DECODER_HYPERPARAMS['max_grad_norm'])
        optimizer.step()
        optimizer.zero_grad()
    
    # Evaluate
    val_loss = evaluate(decoder, sae, frozen_lm, val_loader)
    kl_divergence = compute_kl(logits_decoder, logits_frozen)
    accuracy = compute_accuracy(logits_decoder, logits_frozen)
    
    print(f"Epoch {epoch}: loss={val_loss:.4f}, kl={kl_divergence:.3f}, acc={accuracy:.3f}")

# Save initialized decoder
save_checkpoint(decoder, "decoder_supervised_init.pt")
```

**Success Criteria for Supervised Phase**:
- KL divergence < 1.5 nats (decoder roughly matches frozen model)
- Accuracy > 75% (next-token prediction accuracy)
- Training converges smoothly (no divergence)

---

## Phase 4: RL-Based Decoder Fine-tuning

### 4.1 Multi-Objective Reward Design

**Core Principle**: Optimize three objectives simultaneously with dynamic weighting.

```python
class MultiObjectiveReward:
    def __init__(self):
        self.alpha = 0.5    # Faithfulness weight
        self.beta = 0.3     # Monosemanticity weight
        self.gamma = 0.2    # Task accuracy weight
        
        # Schedule: adjust weights over time
        self.weight_schedule = {
            'phase_1': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2},  # Init phases
            'phase_2': {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3},  # Balanced
            'phase_3': {'alpha': 0.3, 'beta': 0.2, 'gamma': 0.5},  # Final
        }
    
    def compute_reward(self, 
                       logits_decoder, logits_frozen,  # faithfulness
                       sae_features,                    # monosemanticity
                       task_accuracy,                   # accuracy
                       phase='phase_1'):
        """
        Compute total reward for RL training.
        """
        # 1. FAITHFULNESS REWARD: KL divergence (frozen vs decoder)
        kl_div = torch.nn.functional.kl_div(
            torch.log_softmax(logits_decoder, dim=-1),
            torch.softmax(logits_frozen, dim=-1),
            reduction='batchmean'
        )
        r_faithfulness = -kl_div  # Negative because we want to minimize KL
        
        # 2. MONOSEMANTICITY REWARD: Sparsity + consistency
        l0_norm = torch.count_nonzero(sae_features, dim=-1).float().mean()
        entropy = -torch.sum(
            sae_features * torch.log(sae_features + 1e-8), dim=-1
        ).mean()
        r_monosemanticity = -(0.01 * l0_norm + 0.001 * entropy)
        
        # 3. TASK ACCURACY REWARD
        accuracy = task_accuracy.float().mean()
        r_task = accuracy - 0.5  # Centered around 0
        
        # 4. COMBINE WITH CURRENT PHASE WEIGHTS
        weights = self.weight_schedule[phase]
        r_total = (
            weights['alpha'] * r_faithfulness +
            weights['beta'] * r_monosemanticity +
            weights['gamma'] * r_task
        )
        
        # 5. ADD KL PENALTY: prevent diverging from frozen model
        kl_penalty = 0.1 * kl_div  # Subtract to penalize large divergence
        r_final = r_total - kl_penalty
        
        return {
            'r_faithfulness': r_faithfulness.item(),
            'r_monosemanticity': r_monosemanticity.item(),
            'r_task': r_task.item(),
            'r_total_weighted': r_total.item(),
            'r_final': r_final.item(),
            'kl_divergence': kl_div.item(),
            'l0_norm': l0_norm.item(),
        }

# Tracking during training
reward_tracker = {
    'r_faithfulness': [],
    'r_monosemanticity': [],
    'r_task': [],
    'kl_divergence': [],
    'l0_norm': [],
}
```

### 4.2 PPO Training Configuration

**Algorithm**: Proximal Policy Optimization (PPO) with KL regularization

```python
PPO_CONFIG = {
    # Learning
    'learning_rate': 5e-5,
    'num_ppo_epochs': 3,           # Inner loop iterations per batch
    'clip_epsilon': 0.2,            # Max policy change per step
    'gamma': 0.99,                  # Discount factor
    'gae_lambda': 0.95,             # GAE parameter
    
    # Optimization
    'batch_size': 32,
    'rollout_size': 64,             # Collect 64 trajectories before update
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    
    # Regularization
    'kl_penalty': 0.1,              # KL divergence penalty from frozen model
    'entropy_bonus': 0.01,          # Encourage exploration
    'value_loss_coef': 1.0,         # Weight of value function loss
    
    # Training
    'total_training_steps': 10000,
    'eval_frequency': 500,
    'checkpoint_frequency': 1000,
    'warmup_steps': 500,
    
    # Scheduling
    'lr_schedule': 'linear',        # linear, cosine, constant
    'weight_update_frequency': 2000, # Adjust alpha/beta/gamma every N steps
}

class PPOTrainer:
    def __init__(self, decoder, frozen_lm, sae, config):
        self.decoder = decoder
        self.frozen_lm = frozen_lm
        self.sae = sae
        self.config = config
        
        # PPO components
        self.optimizer = AdamW(decoder.parameters(), lr=config['learning_rate'])
        self.value_network = ValueNetwork()  # Separate critic
        self.value_optimizer = AdamW(self.value_network.parameters(), 
                                     lr=config['learning_rate'])
        
        # Tracking
        self.global_step = 0
        self.rewards_history = []
        
    def compute_advantages(self, rewards, values):
        """Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config['gamma'] * values[t+1] - values[t]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages)
    
    def train_step(self, batch):
        """Single PPO update step"""
        tokens = batch['tokens']
        
        # Get frozen model outputs and SAE features
        with torch.no_grad():
            hidden_frozen = self.frozen_lm.get_hidden_states(tokens, layer=8)
            logits_frozen = self.frozen_lm.get_logits(hidden_frozen)
            sae_features = self.sae.encode(hidden_frozen)
        
        # Decoder predictions
        logits_decoder = self.decoder(sae_features)
        
        # Compute rewards
        task_acc = (logits_decoder.argmax(dim=-1) == logits_frozen.argmax(dim=-1))
        reward_dict = self.compute_reward(
            logits_decoder, logits_frozen, sae_features, task_acc
        )
        
        # Compute advantages via value network
        values = self.value_network(sae_features)
        advantages = self.compute_advantages(reward_dict['r_final'], values)
        returns = advantages + values.detach()
        
        # PPO loss
        log_probs_old = torch.log_softmax(logits_decoder, dim=-1)
        log_probs_new = torch.log_softmax(self.decoder(sae_features), dim=-1)
        ratio = torch.exp(log_probs_new - log_probs_old.detach())
        
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 
                                 1 - self.config['clip_epsilon'],
                                 1 + self.config['clip_epsilon']) * advantages
        
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Value loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # Entropy bonus
        entropy = -(torch.softmax(logits_decoder, dim=-1) * 
                   torch.log_softmax(logits_decoder, dim=-1)).sum(dim=-1).mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.config['value_loss_coef'] * value_loss - 
                     self.config['entropy_bonus'] * entropy)
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 
                                       self.config['max_grad_norm'])
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            **reward_dict
        }
    
    def train_epoch(self, train_loader):
        """Full epoch of RL training"""
        losses = []
        for batch_idx, batch in enumerate(train_loader):
            loss_dict = self.train_step(batch)
            losses.append(loss_dict)
            
            if batch_idx % 100 == 0:
                print(f"Step {self.global_step}: "
                      f"loss={loss_dict['loss']:.4f}, "
                      f"kl={loss_dict['kl_divergence']:.3f}, "
                      f"l0={loss_dict['l0_norm']:.1f}")
            
            # Evaluate every N steps
            if self.global_step % self.config['eval_frequency'] == 0:
                self.evaluate(val_loader)
            
            # Checkpoint every N steps
            if self.global_step % self.config['checkpoint_frequency'] == 0:
                save_checkpoint(self.decoder, 
                               f"decoder_rl_step{self.global_step}.pt")
            
            # Update weights based on performance
            if self.global_step % self.config['weight_update_frequency'] == 0:
                self.adjust_weights()
        
        return losses
    
    def adjust_weights(self):
        """Dynamically adjust alpha/beta/gamma based on training progress"""
        # If faithfulness is already very high, reduce alpha
        if self.rewards_history[-1]['kl_divergence'] < 0.5:
            self.reward_computer.alpha *= 0.95
        
        # If sparsity is degrading, increase beta
        if self.rewards_history[-1]['l0_norm'] > 120:
            self.reward_computer.beta *= 1.1
```

### 4.3 Training Loop (Main RL Phase)

```python
# Initialize
trainer = PPOTrainer(decoder, frozen_lm, sae, PPO_CONFIG)
rl_reward = MultiObjectiveReward()

# Training phases
for phase_name, num_steps in [('phase_1', 2000), 
                               ('phase_2', 6000), 
                               ('phase_3', 2000)]:
    print(f"\n=== RL Training Phase: {phase_name} ===")
    
    for step in range(num_steps):
        batch = next(rl_train_loader)
        
        # Compute reward with current phase weights
        loss_dict = trainer.train_step(batch, phase=phase_name)
        
        # Log to wandb
        wandb.log({
            'phase': phase_name,
            **loss_dict,
            'step': trainer.global_step,
        })
        
        # Print progress
        if (step + 1) % 500 == 0:
            print(f"{phase_name} - Step {step+1}: "
                  f"KL={loss_dict['kl_divergence']:.3f}, "
                  f"L0={loss_dict['l0_norm']:.1f}, "
                  f"Total Reward={loss_dict['r_final']:.3f}")
    
    # Phase transition: save checkpoint
    save_checkpoint(trainer.decoder, f"decoder_rl_{phase_name}.pt")

print("\n✓ RL training complete!")
```

### 4.4 Critical Validation: Random Baseline Test

**DO NOT SKIP THIS**: Run this test to verify your reward function is working.

```python
def random_baseline_test():
    """
    Train decoder with RANDOM rewards (not real rewards).
    If it improves similarly to real training → your reward is broken!
    """
    print("\n=== RANDOM BASELINE TEST ===")
    
    # Clone decoder
    decoder_random = copy.deepcopy(decoder_trained)
    optimizer_random = AdamW(decoder_random.parameters(), lr=1e-5)
    
    # Train with random rewards
    initial_kl = evaluate_kl(decoder_random, frozen_lm, sae)
    
    for step in range(1000):
        batch = next(rl_train_loader)
        
        # Forward pass
        hidden_frozen = frozen_lm.get_hidden_states(batch['tokens'], layer=8)
        sae_features = sae.encode(hidden_frozen)
        logits_decoder = decoder_random(sae_features)
        
        # RANDOM REWARD (meaningless)
        r_random = torch.randn(1).item()
        
        # Gradient step
        loss = -r_random  # Maximize random reward (absurd!)
        loss.backward()
        optimizer_random.step()
        optimizer_random.zero_grad()
    
    final_kl = evaluate_kl(decoder_random, frozen_lm, sae)
    
    print(f"Initial KL: {initial_kl:.3f}")
    print(f"Final KL (random reward): {final_kl:.3f}")
    print(f"KL change: {final_kl - initial_kl:.3f}")
    
    # Verdict
    if abs(final_kl - initial_kl) < 0.1:
        print("✓ PASS: Random baseline didn't improve. Your reward is meaningful.")
    else:
        print("✗ FAIL: Random baseline improved! Something is wrong with your reward.")
        raise Exception("Reward function validation failed")

random_baseline_test()
```

---

## Phase 5: Feature Interpretation & Labeling

### 5.1 Feature Importance Scoring via Activation Patching

**Goal**: Determine which SAE features actually matter for the model's predictions.

```python
class ActivationPatcher:
    def __init__(self, decoder, frozen_lm, sae):
        self.decoder = decoder
        self.frozen_lm = frozen_lm
        self.sae = sae
    
    def compute_feature_importance(self, test_data, top_k=100):
        """
        Score each feature via ablation (gradient-based for efficiency).
        """
        importance_scores = {}
        
        for feature_idx in range(self.sae.d_sae):
            # Compute gradient of loss w.r.t. feature activations
            feature_importance = 0.0
            num_batches = 0
            
            for batch in test_data:
                tokens = batch['tokens']
                
                # Forward pass with gradient tracking
                hidden = self.frozen_lm.get_hidden_states(tokens, layer=8)
                sae_features = self.sae.encode(hidden)
                logits_decoder = self.decoder(sae_features)
                logits_frozen = self.frozen_lm.get_logits(hidden)
                
                # Loss: KL divergence
                loss = torch.nn.functional.kl_div(
                    torch.log_softmax(logits_decoder, dim=-1),
                    torch.softmax(logits_frozen, dim=-1)
                )
                
                # Gradient w.r.t. feature
                if sae_features[:, feature_idx].sum() > 0:  # Feature is active
                    grad = torch.autograd.grad(
                        loss, sae_features, 
                        create_graph=False, retain_graph=True
                    )[0][:, feature_idx].abs().mean()
                    feature_importance += grad.item()
                    num_batches += 1
            
            importance_scores[feature_idx] = feature_importance / max(num_batches, 1)
        
        # Rank and return top-k
        ranked = sorted(importance_scores.items(), 
                       key=lambda x: x[1], 
                       reverse=True)
        return ranked[:top_k], importance_scores

# Compute importances
patcher = ActivationPatcher(decoder, frozen_lm, sae)
top_features, all_scores = patcher.compute_feature_importance(val_loader, top_k=1000)

print(f"\nTop-10 Important Features:")
for rank, (feature_idx, score) in enumerate(top_features[:10]):
    print(f"  {rank+1}. Feature #{feature_idx}: importance={score:.4f}")
```

### 5.2 Automated Feature Labeling via LLM

**Goal**: Assign human-readable labels to each feature.

```python
class FeatureLabeler:
    def __init__(self, labeling_model="gpt-3.5-turbo", sae=None, frozen_lm=None):
        self.labeling_model = labeling_model  # Use OpenAI API or local LLM
        self.sae = sae
        self.frozen_lm = frozen_lm
    
    def get_max_activating_tokens(self, feature_idx, data, top_k=50):
        """
        Find tokens that maximally activate a given feature.
        """
        max_activations = []
        
        for batch in data:
            tokens = batch['tokens']
            hidden = self.frozen_lm.get_hidden_states(tokens, layer=8)
            sae_features = self.sae.encode(hidden)  # (batch, seq_len, 40960)
            
            # Get activations for this feature
            feature_activations = sae_features[:, :, feature_idx]  # (batch, seq_len)
            
            # Find top tokens
            for b in range(feature_activations.shape[0]):
                for t in range(feature_activations.shape[1]):
                    if feature_activations[b, t] > 0:
                        token_id = tokens[b, t].item()
                        activation = feature_activations[b, t].item()
                        max_activations.append({
                            'token': self.frozen_lm.tokenizer.decode([token_id]),
                            'activation': activation,
                        })
        
        # Sort and return top-k
        max_activations.sort(key=lambda x: x['activation'], reverse=True)
        return max_activations[:top_k]
    
    def label_features(self, feature_indices, data):
        """
        Use LLM to infer labels for features.
        """
        labels = {}
        
        # Batch query: ask LLM to label multiple features at once
        batch_size = 100
        for i in range(0, len(feature_indices), batch_size):
            batch_indices = feature_indices[i:i+batch_size]
            
            # Prepare batch
            batch_features = []
            for feature_idx in batch_indices:
                top_tokens = self.get_max_activating_tokens(feature_idx, data, top_k=20)
                token_strs = [t['token'].strip() for t in top_tokens]
                batch_features.append({
                    'feature_id': feature_idx,
                    'top_tokens': ', '.join(token_strs[:10]),
                })
            
            # Create prompt
            prompt = "For each feature, infer what concept it represents based on max-activating tokens:\n\n"
            for feat in batch_features:
                prompt += f"Feature {feat['feature_id']}: [{feat['top_tokens']}] → Label: "
            
            # Query LLM (batched = 100x cheaper)
            response = call_llm(self.labeling_model, prompt)
            
            # Parse response
            for line in response.split('\n'):
                if ' → Label:' in line:
                    parts = line.split(' → Label:')
                    feature_id = int(parts[0].split()[-1])
                    label = parts[1].strip()
                    labels[feature_id] = label
        
        return labels

# Run feature labeling
labeler = FeatureLabeler("gpt-3.5-turbo", sae=sae, frozen_lm=frozen_lm)
top_1000_features = [f[0] for f in top_features]
feature_labels = labeler.label_features(top_1000_features, val_loader)

# Save
save_json(feature_labels, "feature_labels.json")
print(f"✓ Labeled {len(feature_labels)} features")

# Show sample
for feature_id in list(feature_labels.keys())[:10]:
    print(f"  Feature {feature_id}: {feature_labels[feature_id]}")
```

### 5.3 Feature Quality Evaluation

```python
def evaluate_feature_labels(feature_labels, data, sample_size=100):
    """
    Spot-check feature labels for quality.
    """
    print("\n=== Feature Label Quality Evaluation ===\n")
    
    sample_features = random.sample(list(feature_labels.keys()), 
                                   min(sample_size, len(feature_labels)))
    
    human_scores = []
    
    for feature_id in sample_features:
        label = feature_labels[feature_id]
        top_tokens = get_max_activating_tokens(feature_id, data, top_k=20)
        
        print(f"Feature {feature_id}:")
        print(f"  Label: {label}")
        print(f"  Top tokens: {[t['token'] for t in top_tokens[:5]]}")
        
        # Get human judgment
        score = input("  Quality (0-1): ")
        human_scores.append(float(score))
    
    avg_quality = np.mean(human_scores)
    print(f"\nAverage quality: {avg_quality:.2f}/1.0")
    
    if avg_quality > 0.8:
        print("✓ PASS: Feature labels are of high quality")
    else:
        print("⚠ WARNING: Consider re-labeling features")
    
    return avg_quality

quality_score = evaluate_feature_labels(feature_labels, val_loader, sample_size=50)
```

---

## Phase 6: Human-Interpretable Output Generation

### 6.1 Feature Importance Ranking (Method 1—Simple, No Training)

**Output**: Ranked list of active features per prediction.

```python
def generate_feature_ranking_explanation(sae_features, feature_labels, top_k=5):
    """
    Convert sparse features to ranked explanation (no training required).
    """
    # Get active features
    active_idx = torch.nonzero(sae_features).squeeze()
    active_vals = sae_features[active_idx]
    
    # Sort by magnitude
    sorted_idx = torch.argsort(active_vals, descending=True)[:top_k]
    top_features = active_idx[sorted_idx]
    top_scores = active_vals[sorted_idx]
    
    # Generate explanation
    explanation = "**Key Reasoning Components (Ranked by Importance):**\n"
    for rank, (feat_id, score) in enumerate(zip(top_features, top_scores)):
        label = feature_labels.get(feat_id.item(), f"unknown_feature_{feat_id}")
        explanation += f"{rank+1}. `{label}` (activation: {score:.3f})\n"
    
    return explanation

# Test
test_sae_features = sae.encode(hidden_states)
explanation = generate_feature_ranking_explanation(test_sae_features[0], feature_labels)
print(explanation)
# Output:
# **Key Reasoning Components (Ranked by Importance):**
# 1. `arithmetic_operator` (activation: 0.984)
# 2. `numerical_operand` (activation: 0.871)
# 3. `addition_operation` (activation: 0.853)
# 4. `intermediate_result` (activation: 0.719)
# 5. `answer_generation` (activation: 0.645)
```

### 6.2 Trained Explanation Decoder (Method 2—Natural Language)

**Training data preparation**:

```python
def create_explanation_training_data(decoder, sae, frozen_lm, data_loader, num_samples=10000):
    """
    Generate pairs of (sparse_features → natural_language_explanation).
    """
    training_pairs = []
    
    for batch in data_loader:
        if len(training_pairs) >= num_samples:
            break
        
        tokens = batch['tokens']
        hidden = frozen_lm.get_hidden_states(tokens, layer=8)
        sae_features = sae.encode(hidden)
        
        for t in range(sae_features.shape[1]):
            # Get active features for this position
            active_features = torch.nonzero(sae_features[:, t]).squeeze()
            
            if len(active_features) > 0:
                # Create feature description
                feature_labels_active = [feature_labels[f.item()] for f in active_features[:10]]
                feature_text = ", ".join(feature_labels_active)
                
                # Template-based explanation (can be refined manually)
                template_explanation = f"The model used: {feature_text}"
                
                training_pairs.append({
                    'sae_features': sae_features[:, t].detach().cpu().numpy(),
                    'explanation': template_explanation,
                })
    
    return training_pairs

# Generate training data
expl_train_data = create_explanation_training_data(decoder, sae, frozen_lm, 
                                                  train_loader, num_samples=5000)
save_dataset(expl_train_data, "explanation_training_data.pkl")
```

**Train explanation decoder**:

```python
class ExplanationDecoder(nn.Module):
    def __init__(self, sae_dim=40960, vocab_size=50000, hidden_dim=512):
        super().__init__()
        self.feature_embed = nn.Embedding(sae_dim, hidden_dim)
        self.aggregation = nn.Linear(hidden_dim, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2048
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, sae_features):
        # sae_features: (batch, sae_dim)
        # Embed and aggregate active features
        active_idx = torch.nonzero(sae_features, as_tuple=True)[1]
        active_vals = sae_features[torch.arange(sae_features.shape[0]), active_idx]
        
        embs = self.feature_embed(active_idx)
        weighted = embs * self.aggregation(active_vals.unsqueeze(-1))
        aggregated = weighted.sum(dim=0, keepdim=True)
        
        # Transform
        trans_out = self.transformer(aggregated)
        
        # Generate
        logits = self.output_proj(trans_out)
        return logits

# Training
expl_decoder = ExplanationDecoder()
opt = AdamW(expl_decoder.parameters(), lr=1e-4)

for epoch in range(3):
    for batch_idx, pair in enumerate(expl_train_data):
        sae_feats = torch.from_numpy(pair['sae_features']).float().cuda()
        expl_text = pair['explanation']
        expl_ids = tokenizer.encode(expl_text)
        
        # Forward
        logits = expl_decoder(sae_feats)
        
        # Loss
        loss = cross_entropy_loss(logits, torch.tensor(expl_ids[:-1]))
        loss.backward()
        opt.step()
        opt.zero_grad()

save_checkpoint(expl_decoder, "explanation_decoder.pt")
```

### 6.3 Dynamic Visualization Dashboard

```python
def create_explanation_dashboard(input_text, decoder, sae, frozen_lm, feature_labels):
    """
    Generate comprehensive human-readable output.
    """
    tokens = tokenizer.encode(input_text)
    
    # Forward pass
    hidden = frozen_lm.get_hidden_states(tokens, layer=8)
    sae_features = sae.encode(hidden)
    logits = decoder(sae_features)
    
    # Next token prediction
    next_token_id = logits[-1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    
    # Method 1: Feature ranking
    ranking_explanation = generate_feature_ranking_explanation(sae_features[-1], feature_labels)
    
    # Method 2: Trained explanation
    expl_decoder = load_checkpoint("explanation_decoder.pt")
    narrative_explanation = expl_decoder(sae_features[-1]).argmax(dim=-1)
    narrative_text = tokenizer.decode(narrative_explanation)
    
    # Build dashboard
    dashboard = f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║                    INTERPRETABLE LLM REASONING                 ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Input: {input_text}
    
    ────────────────────────────────────────────────────────────────
    PREDICTION
    ────────────────────────────────────────────────────────────────
    Next Token: `{next_token}` (confidence: {logits[-1].max():.3f})
    
    ────────────────────────────────────────────────────────────────
    REASONING TRACE (Feature Activations)
    ────────────────────────────────────────────────────────────────
    {ranking_explanation}
    
    ────────────────────────────────────────────────────────────────
    INTERPRETATION (Natural Language)
    ────────────────────────────────────────────────────────────────
    {narrative_text}
    
    ────────────────────────────────────────────────────────────────
    """
    
    return dashboard

# Usage
dashboard = create_explanation_dashboard(
    "What is 17 + 3?",
    decoder, sae, frozen_lm, feature_labels
)
print(dashboard)
```

---

## Validation & Testing

### V1: Unit Tests

```python
def test_sae_reconstruction():
    """Verify SAE can reconstruct hidden states"""
    hidden = torch.randn(32, 4096)
    sae_features = sae.encode(hidden)
    reconstruction = sae.decode(sae_features)
    
    mse = torch.mean((hidden - reconstruction) ** 2)
    assert mse < 0.01, f"Reconstruction MSE {mse} > 0.01"
    print("✓ SAE reconstruction test passed")

def test_decoder_faithfulness():
    """Verify decoder matches frozen model"""
    hidden = frozen_lm.get_hidden_states(test_tokens, layer=8)
    sae_feats = sae.encode(hidden)
    
    logits_decoder = decoder(sae_feats)
    logits_frozen = frozen_lm.get_logits(hidden)
    
    kl = kl_divergence(logits_decoder, logits_frozen).mean()
    assert kl < 0.5, f"KL divergence {kl} > 0.5"
    print("✓ Decoder faithfulness test passed")

def test_feature_sparsity():
    """Verify features are sparse"""
    hidden = frozen_lm.get_hidden_states(test_tokens, layer=8)
    sae_feats = sae.encode(hidden)
    
    l0_per_sample = torch.count_nonzero(sae_feats, dim=-1).float().mean()
    assert 50 < l0_per_sample < 100, f"L0 {l0_per_sample} not in [50, 100]"
    print(f"✓ Sparsity test passed (L0={l0_per_sample:.1f})")
```

### V2: Task-Specific Evaluation

```python
def evaluate_on_tasks(decoder, sae, frozen_lm, tasks):
    """
    Evaluate decoder on diverse reasoning tasks.
    """
    results = {}
    
    for task_name, task_data in tasks.items():
        accuracies = []
        
        for example in task_data:
            # Frozen LLM baseline
            output_frozen = frozen_lm.generate(example['input'])
            acc_frozen = 1.0 if output_frozen == example['target'] else 0.0
            
            # Decoder output
            hidden = frozen_lm.get_hidden_states(example['input'], layer=8)
            sae_feats = sae.encode(hidden)
            logits_decoder = decoder(sae_feats)
            output_decoder = logits_decoder.argmax(dim=-1).item()
            acc_decoder = 1.0 if output_decoder == example['target'] else 0.0
            
            accuracies.append(acc_decoder)
        
        results[task_name] = {
            'accuracy': np.mean(accuracies),
            'num_examples': len(task_data),
        }
    
    return results

tasks = {
    'arithmetic': load_task_data('gsm8k', subset_size=1000),
    'logic': load_task_data('boolq', subset_size=1000),
    'reading': load_task_data('squad', subset_size=1000),
}

eval_results = evaluate_on_tasks(decoder, sae, frozen_lm, tasks)
for task, results in eval_results.items():
    print(f"{task}: {results['accuracy']:.1%} accuracy")
```

### V3: Interpretability Evaluation

```python
def evaluate_interpretability(decoder, sae, frozen_lm, feature_labels):
    """
    Measure quality of extracted features and explanations.
    """
    print("\n=== INTERPRETABILITY EVALUATION ===\n")
    
    # 1. Feature label coverage
    num_labeled = len([f for f in feature_labels.values() if f != 'noise'])
    total_features = sae.d_sae
    coverage = num_labeled / total_features
    print(f"Feature Label Coverage: {coverage:.1%} ({num_labeled}/{total_features})")
    
    # 2. Feature importance concentration
    importance_scores = compute_feature_importance(decoder, sae, frozen_lm)
    top_10_mass = sum(sorted(importance_scores.values(), reverse=True)[:10])
    top_100_mass = sum(sorted(importance_scores.values(), reverse=True)[:100])
    print(f"Top-10 features cover: {top_10_mass:.1%} of total importance")
    print(f"Top-100 features cover: {top_100_mass:.1%} of total importance")
    
    # 3. Explanation faithfulness
    explanations = []
    for batch in test_loader:
        sae_feats = sae.encode(frozen_lm.get_hidden_states(batch['tokens'], layer=8))
        expl = generate_explanation(sae_feats, feature_labels)
        explanations.append(expl)
    
    # Validate via activation patching
    faithfulness_scores = []
    for expl in explanations[:100]:  # Sample for speed
        fidelity = validate_explanation_faithfulness(expl, decoder, frozen_lm)
        faithfulness_scores.append(fidelity)
    
    avg_faithfulness = np.mean(faithfulness_scores)
    print(f"Average Explanation Faithfulness: {avg_faithfulness:.1%}")
    
    print("\n✓ Interpretability evaluation complete")

evaluate_interpretability(decoder, sae, frozen_lm, feature_labels)
```

---

## Implementation Checklist

### Phase 1: Setup ✓
- [ ] Environment configured (hardware, software)
- [ ] Model loaded and verified
- [ ] Data pipeline tested
- [ ] Baseline measurements recorded
- [ ] Experiment tracking (wandb) set up

### Phase 2: SAE Training ✓
- [ ] SAE architecture defined
- [ ] Training loop implemented
- [ ] SAE training completed (50k steps)
- [ ] SAE checkpoint saved
- [ ] Reconstruction quality verified (MSE < 0.01)
- [ ] Sparsity validated (50-100 L0)
- [ ] Sample features inspected manually

### Phase 3: Supervised Pre-training ✓
- [ ] Decoder architecture defined
- [ ] Supervised training loop implemented
- [ ] Pre-training completed (3 epochs)
- [ ] Decoder checkpoint saved
- [ ] KL divergence < 1.5 nats
- [ ] Accuracy > 75%

### Phase 4: RL Fine-tuning ✓
- [ ] Multi-objective reward function designed
- [ ] PPO trainer implemented
- [ ] Phase 1 (2k steps): faithfulness focus
- [ ] Phase 2 (6k steps): balanced training
- [ ] Phase 3 (2k steps): accuracy focus
- [ ] RL decoder checkpoint saved
- [ ] Random baseline test PASSED
- [ ] KL divergence < 0.5 nats
- [ ] Task accuracy degradation < 3%

### Phase 5: Feature Interpretation ✓
- [ ] Activation patching implemented
- [ ] Feature importance computed
- [ ] Top 1000 features identified
- [ ] Feature labeling via LLM started
- [ ] Batch labeling script working
- [ ] Feature labels saved to JSON
- [ ] Manual quality check (>0.8 average)
- [ ] Feature clustering performed

### Phase 6: Output Generation ✓
- [ ] Method 1 (feature ranking) implemented and tested
- [ ] Method 2 (explanation decoder) training data created
- [ ] Method 2 decoder trained and saved
- [ ] Dashboard visualization built
- [ ] Faithfulness validation integrated
- [ ] End-to-end pipeline tested

### Validation & Testing ✓
- [ ] All unit tests passing
- [ ] Task-specific evaluation completed
- [ ] Interpretability metrics computed
- [ ] OOD generalization tested
- [ ] Output quality spot-checked by humans
- [ ] Performance regression test passed

### Documentation & Reproducibility ✓
- [ ] Code committed to version control
- [ ] Hyperparameters documented
- [ ] Experiment logs saved
- [ ] Key figures generated
- [ ] Results summarized

---

## Key Challenges & Mitigations

| Challenge | Symptom | Mitigation |
|-----------|---------|-----------|
| **SAE dead neurons** | Sparsity collapses to few features | Resample dead neurons every 1000 steps; use gradient normalization |
| **Reward hacking** | Random baseline improves too | Implement KL penalty; run random baseline test |
| **Feature collapse** | All samples use same features | Increase entropy penalty β; check weight schedule |
| **Over-faithfulness** | KL → 0 but task accuracy drops | Reduce α weight; increase γ task weight in later phases |
| **Poor feature labels** | Labels don't match semantics | Use batch LLM labeling; manually verify top-100 features |
| **Unfaithful explanations** | Explanations don't match ablations | Add faithfulness loss to explanation decoder; validate with activation patching |
| **Slow inference** | Explanations take too long | Use simpler Method 1 (ranking); cache feature labels |
| **OOD generalization** | Good on train, poor on test | Evaluate on multiple domains; increase diversity in RL data |
| **High computational cost** | Training takes too long | Use gradient checkpointing; reduce batch size; skip full circuit analysis |

---

## Success Metrics

### Quantitative Metrics

| Metric | Target | Baseline | Measurement |
|--------|--------|----------|-------------|
| **Faithfulness** | KL < 0.5 nats | - | KL divergence between decoder & frozen model |
| **Sparsity** | 50-100 L0/token | - | Average active features per token |
| **Task Accuracy** | <3% degradation | 100% (frozen) | Next-token prediction accuracy |
| **Feature Quality** | >85% labeled | - | High-confidence semantic labels |
| **Generalization** | >80% OOD | - | Task accuracy on unseen domains |
| **SAE Reconstruction** | MSE < 0.01 | - | MSE between original & reconstructed hidden states |
| **Explanation Coverage** | Top-10 explain 30% | - | Fraction of output variance from top features |

### Qualitative Metrics

- [ ] Explanations are intuitive to domain experts
- [ ] Feature labels align with human intuition
- [ ] Reasoning traces are interpretable (not gibberish)
- [ ] Users can understand model decisions
- [ ] No hallucinated explanations

---

## References & Prior Work

### Foundational Papers

1. **Towards Monosemanticity** (Anthropic, 2023)
   - URL: https://transformer-circuits.pub/2023/monosemantic-features
   - Key contribution: Sparse autoencoders for interpretability

2. **Scaling and Evaluating Sparse Autoencoders** (OpenAI, 2024)
   - URL: https://cdn.openai.com/papers/sparse-autoencoders.pdf
   - Key finding: SAE scaling laws; 16M latent SAE on GPT-4

3. **A Mechanistic Study of RL-Induced Reasoning via Steering Vectors** (ICLR 2026)
   - Key contribution: RL training of layer-wise interventions
   - Parallel to your work on RL-trained decoders

### Faithful Reasoning

4. **FaithLM: Towards Faithful Explanations for LLMs** (2023)
   - Iterative refinement of explanations for faithfulness
   - Complements your mid-layer approach

5. **Drift: Dual-Reward Probabilistic Inference** (2025)
   - Dual-reward framework for faithful reasoning
   - Inspiration for your multi-objective rewards

6. **Chain-of-Thought Is Not Explainability** (Oxford AIGI, 2025)
   - Demonstrates CoT faithfulness gap
   - Motivation for your direct hidden-state approach

### Related Interpretability

7. **Circuit Tracing with Transcoders** (Anthropic)
   - Layer-wise decomposition with sparse features
   - Precedent for your decoder architecture

8. **RouteSAE: Multi-Layer Sparse Autoencoders** (2025)
   - Dynamic routing across layers
   - Could improve your decoder robustness

9. **Mechanistic Interpretability for AI Safety** (2024)
   - Review of interpretability techniques
   - Context for your work in AI alignment

---

## Appendix: Useful Commands

### Training Commands

```bash
# Phase 2: SAE training
python train_sae.py \
  --model_name llama-7b \
  --layer 8 \
  --expansion_ratio 10 \
  --sparsity_coeff 0.01 \
  --num_steps 50000 \
  --save_path checkpoints/sae_layer8.pt

# Phase 3: Supervised pre-training
python train_decoder_supervised.py \
  --sae_path checkpoints/sae_layer8.pt \
  --model_name llama-7b \
  --num_epochs 3 \
  --batch_size 32 \
  --save_path checkpoints/decoder_supervised.pt

# Phase 4: RL fine-tuning
python train_decoder_rl.py \
  --decoder_path checkpoints/decoder_supervised.pt \
  --sae_path checkpoints/sae_layer8.pt \
  --model_name llama-7b \
  --num_steps 10000 \
  --alpha 0.5 --beta 0.3 --gamma 0.2 \
  --save_path checkpoints/decoder_rl.pt

# Phase 5: Feature labeling
python label_features.py \
  --sae_path checkpoints/sae_layer8.pt \
  --decoder_path checkpoints/decoder_rl.pt \
  --model_name llama-7b \
  --num_features 1000 \
  --save_path results/feature_labels.json

# Phase 6: Generate explanations
python generate_explanations.py \
  --decoder_path checkpoints/decoder_rl.pt \
  --sae_path checkpoints/sae_layer8.pt \
  --feature_labels_path results/feature_labels.json \
  --input "What is 17 + 3?" \
  --output_format dashboard
```

### Evaluation Commands

```bash
# Validate SAE quality
python evaluate_sae.py \
  --sae_path checkpoints/sae_layer8.pt \
  --model_name llama-7b

# Evaluate decoder faithfulness
python evaluate_decoder.py \
  --decoder_path checkpoints/decoder_rl.pt \
  --sae_path checkpoints/sae_layer8.pt \
  --model_name llama-7b

# Measure interpretability
python evaluate_interpretability.py \
  --decoder_path checkpoints/decoder_rl.pt \
  --sae_path checkpoints/sae_layer8.pt \
  --feature_labels_path results/feature_labels.json

# Run random baseline test
python validate_rewards.py \
  --decoder_path checkpoints/decoder_rl.pt \
  --sae_path checkpoints/sae_layer8.pt
```

---

## Project Timeline

**Week 1-2**: Setup, SAE training on Llama-1B
**Week 3**: Supervised decoder pre-training
**Week 4-5**: RL fine-tuning, Phase 1-3 cycles
**Week 6**: Feature labeling (parallelizable)
**Week 7**: Explanation generation, visualization
**Week 8**: Validation, testing, documentation

**Publication Target**: ICLR 2026 Workshops or BlackboxNLP 2026

---

## Contact & Attribution

This project builds on research from:
- Anthropic (SAE methodology, monosemanticity)
- OpenAI (SAE scaling, mechanistic interpretability)
- DeepSeek (RL-based reasoning)
- Academic labs (University of Toronto, UC Berkeley, Stanford)

---

**Last Updated**: January 25, 2026  
**Status**: Ready for Implementation  
**Version**: 1.0


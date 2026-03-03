# RL-Decoder with SAE Features — Project Status

**Last Updated:** March 2, 2026 (Phase 6 complete; Phase 7 two-track protocol + portability work in progress)

---

## Phase Structure

| Phase | Name | Status | Key Result |
|-------|------|--------|------------|
| **1** | Ground-Truth Validation | ✅ Complete | R²=0.967–0.999 on BFS/Stack/Logic |
| **2** | Multi-Layer SAE Training | ✅ Complete | 24 GPT-2 SAEs (12×) + 98 SAEs across 4 models |
| **2r** | SAE Retrain — TopK (GPT-2) | ✅ Complete | 24 TopK SAEs, exactly 30% sparsity, loss 0.019–0.033 |
| **3** | Reasoning Flow Tracing | ✅ Complete | Feature-flow heatmaps; ~50% active (~50% sparse) |
| **4** | Arithmetic Feature Probing | ✅ Complete (v1) | R²=0.985 at layer 8; features predictive but not causally sufficient |
| **4r** | Arithmetic Probing — TopK + Subspace | ✅ Complete | R²=0.977 (L7); causal Δlog_prob=+0.107 at L22 (21/24 layers positive) |
| **5** | Feature Interpretation + Causal Steering | ✅ Complete | Feature contexts (L22 F11823 active 97%); mean-diff steering uniformly negative → confirms distributed encoding |
| **6** | Decoder Benchmark + Full Layer Sweep | ✅ Complete | Full sweep done (102/102, 0 failed); best test top1=0.5628 (`raw_block8_00_07`) |
| **7** | Causal CoT Verification Auditor | 🟡 Two-track in progress | GPT-2 protocol gates pending; Qwen2.5-7B portability pilot scaffolded |

---

## Phase 1 — Ground-Truth Validation ✅

**Scripts:** `phase1/`
**Results:** `phase1_results/`

Trained SAEs on three controlled environments where the ground-truth latent state is known:

| Environment | R² (best) | R² (worst) | Expansions tested |
|-------------|-----------|------------|-------------------|
| BFS graph traversal | 0.987 | 0.976 | 4×, 8×, 12× |
| Stack machine | 0.989 | 0.967 | 4×, 8×, 12× |
| Logic puzzle | 0.999 | 0.990 | 4×, 8×, 12× |

All pass R²>0.95, confirming the SAE can reconstruct known latent states.

**Canonical artifact paths (current references):**
- BFS: `phase1_results/v2/gpu1_bfs/phase1_results.json`
- Stack: `phase1_results/v2/gpu2_stack/phase1_results.json`
- Logic: `phase1_results/v3/gpu3_logic/phase1_results.json`

Note: `phase1_results/` also contains earlier exploratory and intermediate runs.

---

## Phase 2 — Multi-Layer SAE Training ✅

**Scripts:** `phase2/`
**Results:** `phase2_results/`

```
phase2_results/
  activations/        — 98 activation files (4 models × all layers)
  saes_gpt2_12x/      — 24 SAEs for GPT-2 medium, 12× expansion
  saes_all_models/    — 98 SAEs across GPT-2 medium, Phi-2, Gemma-2B, Pythia-1.4B
```

Key facts:
- GPT-2 medium SAEs: 12× expansion (12 288 features), trained per layer (0–23)
- All-model SAEs: 8× expansion, one per layer per model
- Training fixed: `use_relu=True`, `normalize_decoder()` after every step, correct decorrelation loss

---

## Phase 2r — SAE Retrain with TopK Activation ✅

**Goal:** Replace ReLU+L1 with hard TopK activation (K=3 686 ≈ 30% of 12 288) to hit
the ~30% sparsity target and improve feature monosemanticity before re-running Phase 4.

**Why TopK over stronger L1:**
- L1 requires ~50× increase to halve sparsity, entering feature-death territory
- TopK guarantees exact sparsity by construction; no L1 tuning needed
- Typically yields better reconstruction quality at the same sparsity level (Anthropic post-Scaling Monosemanticity practice)

**Scope:** GPT-2 medium only (debug/validate first).  Other models to follow after results confirmed.

**Checkpoints:** `phase2_results/saes_gpt2_12x_topk/saes/gpt2-medium_layer{N}_sae.pt`

**Run command:**
```bash
tmux new-session -d -s retrain
tmux send-keys -t retrain "cd '/scratch2/f004ndc/RL-Decoder with SAE Features' && \
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 phase2/train_multilayer_saes.py \
  --activations-dir phase2_results/activations \
  --output-dir phase2_results/saes_gpt2_12x_topk/saes \
  --model-filter gpt2-medium \
  --expansion-factor 12 \
  --use-topk --topk-k 3686 \
  --epochs 10 --device cuda:0" Enter
```

---

## Phase 4r — Arithmetic Probing (TopK SAEs + Subspace Steering) ✅

**Output:** `phase4_results/topk/`

| Experiment | Key Result |
|---|---|
| A — Ridge probe | R²=0.977 (result tok, layer 7); no instability vs v1 |
| B — Co-activation | computation-bridge 3.2%, result-encoder 2.3%, operand-tracker 1.9% |
| C — Subspace steering | **+0.107 Δlog_prob at layer 22**; 21/24 layers positive (mean +0.037) |

**Key findings:**
- TopK SAEs (30% sparsity) produce stable probing targets — the extreme instability at layers 5/7 in v1 is gone.
- Subspace steering flipped the causal test from uniformly negative (v1) to **positive across layers 12–23**, peaking at layer 22.
- Arithmetic computation in GPT-2 medium is causally concentrated in the **upper half of the network** (layers ~12–23); early layers carry no causal signal.
- The feature taxonomy (computation-bridge / result-encoder / operand-tracker) is robust across both SAE variants.

---

## Phase 3 — Reasoning Flow Tracing ✅

**Scripts:** `phase3/`
**Results:** `phase3_results/reasoning_flow/`

`reasoning_flow_tracer.py` registers forward hooks on every GPT-2 medium block, encodes through the layer's SAE, and records active features per token.  Interactive analysis is available in `interactive_reasoning_tracker.ipynb`.

Key finding: roughly half of the 12 288 features are active on average (~50% active, ~50% sparse), and computation vs. background tokens differ by only a small amount in raw active-feature count (~1%). This motivates Phase 4's targeted probing approach instead of raw feature-count comparisons.

---

## Phase 4 — Arithmetic Feature Probing ✅

**Scripts:** `phase4/`
**Results:** `phase4_results/`

Three progressively stronger experiments to locate arithmetic features.  674 annotations across 200 GSM8K examples.

| Script | Experiment | Key Result |
|--------|-----------|------------|
| `arithmetic_data_collector.py` | Data collection | 674 `<<expr=result>>` annotations, SAE features saved at 3 positions per annotation |
| `arithmetic_value_probe.py` | A — Ridge probe | R²=0.985 (result tok, layer 8); R²=0.856 (pre-`=` tok, layer 8) |
| `feature_coactivation.py` | B — Co-activation | 9 412 computation-bridge, 6 603 result-encoder, 5 678 operand-tracker features identified |
| `causal_patch_test.py` | C — Causal patching | All layers negative Δlog_prob (best: −0.34 at L0, worst: −7.97 at L9) |

**Key findings:**
- **Experiment A**: SAE features at *result* tokens are near-perfect linear predictors of the arithmetic result (R²=0.985, layer 8).  Features at operand tokens are also strong (R²=0.856).  Layers 5 and 7 show instability at the `=` token.
- **Experiment B**: ~23 500 features (~1.9% of 12 288) have selectivity >0.1 for at least one arithmetic role; the majority (271K total occurrences) are background.  Computation-bridge features (active at both `=` and result) peak at layer 11.
- **Experiment C**: Causal patching of top-128 probe features *decreases* target log-prob at every layer.  Features are highly predictive of values but swapping them in isolation disrupts rather than transfers arithmetic computation — evidence that the result encoding is distributed across many co-dependent features, not a localized causal bottleneck.

**Run order (for re-runs):**
```bash
# 1. Collect (GPU, ~20 min)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 phase4/arithmetic_data_collector.py --device cuda:0

# 2. Value probe (CPU)
.venv/bin/python3 phase4/arithmetic_value_probe.py

# 3. Co-activation (CPU)
.venv/bin/python3 phase4/feature_coactivation.py

# 4. Causal patch (GPU)
CUDA_VISIBLE_DEVICES=7 .venv/bin/python3 phase4/causal_patch_test.py --device cuda:0
```

---

## Phase 5 — Feature Interpretation + Causal Steering ✅

**Scripts:** `phase5/`
**Results:** `phase5_results/`
**Scope:** GPT-2 medium first; extend to other models once validated.

Two experiments using the Phase 4r TopK SAEs and dataset as input:

| Script | Experiment | Key Result |
|---|---|---|
| `feature_interpreter.py` | A — Feature contexts | Feature 11823 (computation_bridge, L22) active on 651/674 records; all top features labeled with role + max-activating contexts |
| `arithmetic_steerer.py` | B — Mean-diff steering | All layers negative (best: −2.85 at L17, α=0.5); confirms naive feature-space steering disrupts computation |

**Experiment A — Feature interpreter (✅):**
1. Load 674 records from `phase4_results/topk/collection/`
2. For each of top-50 probe features at layers 14–23: rank records by activation value at the result token, extract ±6-token context window, record expression and result value
3. Outputs per-feature JSON cards + a layer-22 summary table

Key finding: At layer 22, the most active arithmetic feature (11823, `computation_bridge`) is
active on 651/674 records (97%). The top features are predominantly labeled `background` or
`eq_sign` by the co-activation threshold (0.1), with a handful of `operand` and
`computation_bridge` features. All top-10 features at L22 are active on ≥571/674 records,
suggesting the arithmetic-predictive features at this layer are very broadly active.

**Experiment B — Mean-diff steering (✅):**
1. Split dataset (472 train: 237 high-|C|, 235 low-|C|; 202 test)
2. Compute `Δf[L] = mean(result_features[L] | high-C) − mean(result_features[L] | low-C)` in SAE feature space
3. For test examples: encode residual at layer L → add α×Δf[L] → decode → inject → measure log_prob shift
4. All 96 (layer, α) combinations produce negative Δlog_prob (range: −2.85 to −12.1)

Key finding: Dense mean-diff feature-space injection disrupts arithmetic computation regardless
of layer or steering strength — consistent with v1 causal patching failure. The only method that
produced positive causal effects is the *subspace projection* approach from Phase 4r (which
preserves non-arithmetic co-dependent features). This confirms that arithmetic representation is
highly distributed and non-linear: the mean-diff direction in feature space is not an
independently-editable causal knob.

---

## Phase 6 — Decoder Benchmark + Layer Sweep ✅

**Scripts:** `phase6/`
**Primary results:** `phase6_results/`

Phase 6 execution summary:
- Expanded dataset generated (`phase6_results/dataset/gsm8k_expanded_{train,test}.pt`)
- Supervised benchmark completed for 6 base configs
- RL top-2 follow-up completed (`phase6_results/rl_runs/20260226_104147_p6rl/`)
- Full Phase 6 layer sweep completed (`phase6_results/sweeps/20260226_164900_phase6_full_sweep/phase6_full/`)

### Base supervised benchmark (6 configs)

| Config | Test top-1 | Test top-5 | Test Δlogprob vs GPT-2 |
|---|---:|---:|---:|
| `raw_multi_l7_l12_l17_l22` | 0.5075 | 0.7605 | +2.2919 |
| `hybrid_multi_l7_l12_l17_l22` | 0.5086 | 0.7584 | +2.2798 |
| `sae_multi_l7_l12_l17_l22` | 0.3356 | 0.6362 | +1.6297 |

### RL top-2 follow-up (vs supervised baselines)

| Config | Δtop-1 | Δtop-5 | Δ(Δlogprob vs GPT-2) |
|---|---:|---:|---:|
| `raw_multi_l7_l12_l17_l22` | +0.0100 | -0.0024 | -0.0801 |
| `hybrid_multi_l7_l12_l17_l22` | +0.0003 | -0.0008 | -0.0484 |

Interpretation: RL was mixed and mostly not beneficial on top-5 and `delta_logprob_vs_gpt2`.

### Full Phase 6 sweep (run: `20260226_164900_phase6_full_sweep`)

- Scope: `43 raw + 43 hybrid + 16 sae = 102` configs
- Completion: `102/102` done, `0` failed
- Artifacts present: `102` eval JSONs, `102` supervised JSONs, `102` checkpoints

Best test metrics across the sweep:
- Best top-1: `raw_block8_00_07` = `0.5628`
- Best top-5: `hybrid_block4_04_07` = `0.7860`
- Best `delta_logprob_vs_gpt2`: `raw_block4_04_07` = `+2.5717`

Practical finding: strongest readout appears in early-mid blocks (`block4_04_07`, `block8_00_07`), not only in the prior `spread4_current` layer set.

**Known bookkeeping gap:** `pipeline.done` and canonical copied summary/report are missing for this sweep run because the coordinator was not resumed after completion. The run artifacts themselves are complete and valid.

---

## Phase 7 — Causal CoT Verification Auditor 🟡

**Scripts:** `phase7/`
**Current state:** implemented and smoke/calibration validated; now running a two-track plan:
- Track A: GPT-2 protocol-gate validation
- Track B: Qwen2.5-7B portability pilot

Completed prerequisites:
- Trace dataset built (`phase7_results/dataset/gsm8k_step_traces_{train,test,all}.pt`)
- Calibration mini-sweep summary (`phase7_results/results/20260226_121004_layer_sweep_calib_calibration_summary.json`)
- Causal dry-run throughput artifact (`phase7_results/interventions/20260226_121004_layer_sweep_calib_calibration_causal_throughput.json`)
- Phase 7 CLIs hardened for both script and module invocation (`python phase7/<script>.py` and `python -m phase7.<script>`)
- Phase 7 smoke unittest suite added under `phase7/tests/` for invocation, registry contract, and metadata contract
- Model portability scaffolding added:
  - `phase7/model_adapters/base.py`
  - `phase7/model_registry.py`
  - `phase7/model_adapters/qwen25_7b_adapter.py`
  - `phase7/causal_intervention_engine.py` now supports `--dry-run` with structured unsupported status for non-GPT causal subspace runs
  - Note: non-GPT causal patching still requires model-specific SAE assets/loaders; currently implemented full patching path is GPT-2.

### Phase 7 pivot criteria (gates before CoT-improvement use)

Phase 7 is now the primary contribution track for CoT faithfulness verification.
Required gates before using auditor scores for reranking/filtering/reward shaping:
- `causal_auditor` AUROC on controls: `>= 0.85`
- Unfaithful-control false positive rate: `<= 0.05`
- At least one explicit `high latent_only / low causal_auditor` case
- Claim-boundary disclaimer preserved in benchmark output JSON

Latest gate matrix snapshot (`phase7_results/results/trackA_gate_matrix_fixv3_v2.json`):
- Checkpoints evaluated: `raw_every2_even`, `hybrid_every2_even`, `raw_middle12_06_17`
- AUROC gate: **pass** (`0.9676` on all three)
- FPR gate: pass (`0.018` on all three)
- Separation gate: pass (`readout_high_causal_fail_cases_n` in `1648–1726`)
- Reporting gate: pass (A/B/C tracks + claim-boundary disclaimer present)

Interpretation: Track-A gate criteria are now satisfied for the fixv3 matrix run; next work should focus on broader robustness/portability and Phase 8 reranking design validation.

Minimal validation sequence:
1. Generate controls: `phase7/generate_cot_controls.py`
2. Parse/align controls: `phase7/parse_cot_to_states.py`
3. Run shortlist causal interventions: `phase7/causal_intervention_engine.py` workflow
4. Audit traces: `phase7/causal_audit.py`
5. Calibrate thresholds: `phase7/calibrate_audit_thresholds.py`
6. Benchmark tracks A/B/C: `phase7/benchmark_faithfulness.py`

### Protocol validity vs model capability (locked interpretation)

- GPT-2 is the **protocol-validation model** for fast white-box iteration.
- Qwen2.5-7B is the **external-relevance model** for CoT-capable behavior checks.
- A failed GPT-2 natural-language CoT trace is not by itself a protocol failure.
- A protocol pass requires intervention-backed separation (not decoder accuracy alone).

Reference scope and literature baseline:
- `docs/PHASE7_AUDITOR_PROTOCOL_SCOPE.md`

### Phase 8 launch policy (reranking-first)

Phase 8 starts only after Phase 7 gates pass. The first optimization method is
offline **CoT reranking** (not reward shaping first):

1. Sample multiple CoT candidates per prompt.
2. Score with the Phase 7 causal-faithfulness auditor.
3. Select highest gated score candidate.
4. Evaluate answer quality + faithfulness against baseline decoding.

Phase 8 execution details:
- `docs/PHASE8_RERANKING_PLAN.md`

---

## Directory Map

```
RL-Decoder with SAE Features/
├── src/                     — SAE library (architecture, training, config)
├── datasets/                — GSM8K and other raw data
│
├── phase1/                  — Ground-truth environment scripts
├── phase1_results/          — BFS/Stack/Logic SAE checkpoints + R² metrics
│
├── phase2/                  — Multi-layer activation capture + SAE training
├── phase2_results/
│   ├── activations/         — 98 activation files (gpt2-medium_layer{N}_activations.pt …)
│   ├── saes_gpt2_12x/       — 24 × 12× GPT-2 SAEs
│   └── saes_all_models/     — 98 SAEs across 4 frontier models
│
├── phase3/                  — Reasoning flow tracer + interactive notebook
├── phase3_results/
│   └── reasoning_flow/      — Feature-flow PNGs, JSON
│
├── phase4/                  — Arithmetic feature probing (3 experiments)
├── phase4_results/
│   ├── collection/          — gsm8k_arithmetic_dataset.pt (shared input)
│   ├── probe/               — R² by layer, top features per layer
│   ├── coactivation/        — Selectivity heatmaps, category distribution
│   ├── patching/            — Δlog_prob by layer, causal effect heatmap (v1, ReLU)
│   └── topk/                — Phase 4r results (TopK SAEs + subspace steering)
│
├── phase5/                  — Feature interpretation + causal steering
├── phase5_results/
│   ├── feature_interpretations/ — Per-feature JSON cards (layers 14–23) + L22 summary
│   └── steering/            — Steering heatmap + line plot + steering_results.json
│
├── phase6/                  — Decoder benchmark pipeline scripts
├── phase6_results/
│   ├── dataset/             — Expanded GSM8K arithmetic dataset (train/test)
│   ├── results/             — Base supervised benchmark outputs
│   ├── rl_runs/             — RL follow-up run-scoped artifacts
│   └── sweeps/              — Full layer-sweep run outputs
│
├── phase7/                  — Causal CoT verification auditor scripts
├── phase7_results/
│   ├── dataset/             — Step-trace datasets
│   ├── sweeps/              — Calibration and future broad-sweep outputs
│   └── interventions/       — Causal check artifacts
│
├── archived/                — Superseded / broken work (kept for reference)
│   ├── old_phase3_single_layer_probe/   (GPT-2 small single-layer probes)
│   ├── old_phase3_results/
│   ├── old_phase4_broken_sae_training/  (use_relu=False era)
│   ├── old_phase4_results/
│   ├── old_phase5_broken_analysis/
│   ├── old_phase5_broken_results/
│   └── old_phase6_docs/
│
└── docs/                    — Design documents and reports
```

---

## Key Environment Details

- Python 3.12.3, CUDA 12.8, PyTorch 2.6
- 4 GPUs (0–3) on primary node; additional GPUs (5, 7) available
- Activate with `source setup_env.sh` or `.venv/bin/python3`
- `weights_only=False` required for `torch.load` on SAE checkpoints (contain PosixPath)
- Hook target for GPT-2 activations: `model.transformer.h[i]` (full block, not `.mlp`)

# Active Tasks

**Last Updated:** March 7, 2026

## Project State
- GPT-2 Phase 7 is closed.
- Track C disposition for GPT-2 is negative under current protocol.
- Final GPT-2 Phase 7 configuration is two-track (`text=0.50`, `latent=0.50`) with structural penalties.

Primary closure references:
- `docs/PHASE7V3_TRACKC_FINDINGS.md`
- `docs/TRACKC_NEGATIVE_FINDING.md`
- `phase7_results/results/optionbc_final_phase7_optionbc_20260306_092554_phase7_optionbc.json`

## Completed (GPT-2 Closure)
- [x] R0: Negative finding lock-in and archive manifest.
- [x] R1: Confidence-margin implementation and benchmark integration.
- [x] R2: Trajectory-coherence implementation and benchmark path.
- [x] R3: Contrastive probe implementation and corrected split path.
- [x] R4: Representation geometry diagnostic path.
- [x] R5.1: Decision matrix executed.
- [x] R5.2: Closure branch selected for GPT-2 (`all-fail` for Track C deployment relevance).

## GPT-2 Policy
- [x] No further Track C optimization on GPT-2 under the current protocol.
- [x] Keep Track C as a documented negative result for GPT-2.
- [x] Treat two-track auditor as final GPT-2 Phase 7 output.

## Next Priority: Qwen Diagnostic-First Inquiry

### Q0 (Go/No-Go, required before full Qwen Stage 3)
Goal: determine if Qwen hidden states actually separate faithful vs unfaithful control variants enough to justify full pipeline cost.

#### Q0 protocol
- Model: `qwen2.5-7b`
- Sample: 50 traces
- Controls: faithful + unfaithful variants from existing generator
- Forward mode: control-conditioned
- Anchor: `=` token (or eq-like anchor policy)
- Metric: within-trace hidden-state L2 separation between faithful and unfaithful variants

#### Q0 required outputs
- `phase7_results/results/qwen_q0_hidden_state_separation_<run_tag>.json`
- `phase7_results/results/qwen_q0_go_nogo_<run_tag>.json`

#### Q0 go/no-go criteria
- `mean_within_trace_l2_delta > 0` with stable distribution (not near-zero collapse)
- non-trivial variant separation across a majority of sampled traces
- no split/parse contract violations

### Stage 3 (Qwen full pipeline) dependency
- [ ] Start full Qwen Stage 3 **only if Q0 passes**.
- [ ] If Q0 fails, stop and document "blocked_no_hidden_state_separation".

## Qwen Preparation Track (In Progress)
- [ ] Capture Qwen activation tensors for layer-wise SAE training (`phase2_results/activations`).
- [ ] Launch Qwen SAE training split across GPUs 5/6/7 with deterministic layer partitioning.
- [ ] Emit Qwen SAE training summary under `phase2_results/saes_qwen25_7b_12x_topk/`.
- [ ] Rebuild Qwen training summary index after all layers complete.

Note:
- This preparation track does not change GPT-2 closure status.
- Qwen claims remain gated by Q0/Qwen-specific evidence once assets are ready.

## Deferred / Out Of Scope For Current Pass
- Cross-model publication claims beyond Qwen Q0.
- Phase 8 reranking based on GPT-2 Track C.

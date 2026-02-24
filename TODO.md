# Active Tasks

**Last Updated:** February 23, 2026

For full status and key results, see [PROJECT_STATUS.md](PROJECT_STATUS.md).

---

## Phase Status Summary

| Phase | Name | Status |
|-------|------|--------|
| 1 | Ground-Truth Validation | ✅ Complete |
| 2 | Multi-Layer SAE Training | ✅ Complete |
| 2r | SAE Retrain — TopK | ✅ Complete |
| 3 | Reasoning Flow Tracing | ✅ Complete |
| 4 | Arithmetic Feature Probing (v1, ReLU) | ✅ Complete |
| 4r | Arithmetic Probing — TopK + Subspace | ✅ Complete |
| 5 | Feature Interpretation + Causal Steering | ✅ Complete |

---

## Outstanding / Next Steps

### Infrastructure
- [ ] Clean up `sae_logs/` — many logs are from superseded runs; keep only
      the phase4b and phase5 task4 logs that correspond to current checkpoints.
- [ ] Verify `checkpoints/gpt2-small/sae/` contents — these standalone
      single-layer GPT-2 small checkpoints predate the multi-layer work;
      document or prune.

### Potential Phase 6 Work (not started)
If extending the project, the natural next step is **step-level causal analysis**:
- Align activations to reasoning step boundaries (parse / decompose / operate / verify)
- Train CoT-aware SAEs with an auxiliary step-prediction loss
- Perform per-step causal ablations (stronger signal than task-level ablations)
- Latent steering experiments to control reasoning length/quality
See [docs/archived_reports/PHASE6_DESIGN_SPECIFICATION.md](docs/archived_reports/PHASE6_DESIGN_SPECIFICATION.md) for a full design spec.

### Paper / Documentation
- [ ] Write up Phase 4r + Phase 5 findings as a coherent narrative
      (subspace steering positive result is the publishable claim).
- [ ] Add description of TopK vs ReLU tradeoffs to `overview.tex`.

---

## Environment & Infrastructure (completed)

- [x] Python 3.12.3, CUDA 12.8, PyTorch 2.6 environment documented
- [x] `setup_env.sh` installs minimal deps and pins tokenizer artifacts
- [x] Secrets in `.env`; `weights_only=False` noted for SAE checkpoint loading
- [x] GSM8K downloaded to `datasets/raw/gsm8k/`
- [x] Activation hooks validated: <50% overhead, >1000 tokens/s

# Documentation Index

**Last Updated:** March 1, 2026

## Essential Documents

| File | Purpose |
|------|---------|
| [README.md](../README.md) | Project overview, structure, run commands |
| [PROJECT_STATUS.md](../PROJECT_STATUS.md) | Phase-by-phase results, key findings, environment notes |
| [TODO.md](../TODO.md) | Outstanding tasks and next steps |
| [overview.tex](../overview.tex) | Formal research design document |

## Source Documentation

| File | Purpose |
|------|---------|
| [src/SECTION3_README.md](../src/SECTION3_README.md) | Activation capture infrastructure (`activation_capture.py`, `model_registry.py`) |
| [src/SECTION4_README.md](../src/SECTION4_README.md) | SAE architecture and training (`sae_architecture.py`, `sae_config.py`, `sae_training.py`) |

## Future Work Planning Docs

| File | Purpose |
|------|---------|
| [TODO.md](../TODO.md) | Execution tracker and source of truth for pending work |
| [experiments/PHASE7_LAYER_SWEEP_REVIEW.md](../experiments/PHASE7_LAYER_SWEEP_REVIEW.md) | Objective alignment, layer-sweep rationale, and runtime framing |
| [phase6_implementation.md](../phase6_implementation.md) | Historical Phase 6 design baseline (not the live execution-status source) |

## Key Recent Artifacts

| Path | What it contains |
|------|------------------|
| `phase6_results/sweeps/20260226_164900_phase6_full_sweep/phase6_full/` | Completed Phase 6 full sweep run (102 train+eval configs) |
| `phase6_results/rl_runs/20260226_104147_p6rl/` | Phase 6 RL top-2 follow-up run outputs |
| `phase7_results/results/20260226_121004_layer_sweep_calib_calibration_summary.json` | Phase 7 calibration mini-sweep summary |
| `phase7_results/interventions/20260226_121004_layer_sweep_calib_calibration_causal_throughput.json` | Phase 7 causal dry-run throughput summary |
| `phase7_results/dataset/build_summary.json` | Phase 7 step-trace dataset build summary |

## Archived Reports

Execution reports and design documents from earlier runs are in
[docs/archived_reports/](archived_reports/):

| File | What it covers |
|------|---------------|
| `PHASE3_FULLSCALE_EXECUTION_GUIDE.md` | GPT-2 small single-layer probe execution (superseded) |
| `PHASE4_EXECUTION_REPORT.md` | Phase 4 frontier SAE training run log (broken SAEs, superseded) |
| `PHASE4_FINAL_SUMMARY.md` | Phase 4 summary (broken SAEs, superseded) |
| `PHASE4_IMPLEMENTATION_SUMMARY.md` | Phase 4 implementation notes (superseded) |
| `PHASE5_COMPLETION_REPORT.md` | Phase 5 Tasks 1–3 completion report (broken SAEs, superseded) |
| `PHASE5_TASK4_REFACTOR_STATUS.md` | Multi-layer SAE refactor status (old numbering) |
| `PHASE5_TASK4_TRAINING_STATUS.md` | Multi-layer SAE training status (old numbering) |
| `PHASE5_TASK4_ANALYSIS_REPORT.md` | Multi-layer SAE transfer analysis (old numbering, now Phase 3) |
| `PHASE5_TASK4_SPECIFICATION.md` | Multi-layer SAE specification (old numbering, now Phase 2/3) |
| `PHASE6_DESIGN_SPECIFICATION.md` | Step-level CoT extension design (proposed, not started) |

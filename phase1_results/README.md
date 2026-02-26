# Phase 1 Results Directory Notes

`phase1_results/` contains multiple generations of Phase 1 runs (early exploratory runs plus later improved runs).

## Canonical Phase 1 artifacts (used in current project summaries)

- BFS: `phase1_results/v2/gpu1_bfs/phase1_results.json`
- Stack: `phase1_results/v2/gpu2_stack/phase1_results.json`
- Logic: `phase1_results/v3/gpu3_logic/phase1_results.json`

## Why multiple versions exist

- Early runs were used to validate the causality pipeline quickly and had weaker reconstruction.
- Later runs (`v2`, `v3`) improved training stability and are the basis for the reported Phase 1 metrics in `PROJECT_STATUS.md`.

Use `PROJECT_STATUS.md` as the source of truth for the current Phase 1 summary.

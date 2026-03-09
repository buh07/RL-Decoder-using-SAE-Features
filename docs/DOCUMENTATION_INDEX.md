# Documentation Index

**Last Updated:** March 9, 2026

## Core Project Docs
| File | Purpose |
|---|---|
| `README.md` | High-level project summary and current status |
| `PROJECT_STATUS.md` | Phase-by-phase status and canonical results |
| `TODO.md` | Active work queue and next-step execution plan |
| `overview.tex` | Research design and framing |

## Phase 7 GPT-2 Closure Docs
| File | Purpose |
|---|---|
| `docs/PHASE7V3_TRACKC_FINDINGS.md` | Final GPT-2 Track C closure and decision matrix outcome |
| `docs/TRACKC_NEGATIVE_FINDING.md` | Final negative-result claim boundary for GPT-2 Track C |
| `docs/TRACKC_NEGATIVE_TO_POSITIVE_STORY.md` | Canonical narrative from coherence-confounded negatives to Option C arithmetic positive baseline |
| `experiments/PHASE7_LAYER_SWEEP_REVIEW.md` | Closure-oriented review summary and open inquiry boundary |

## Phase 7 Key Result Artifacts
| Path | Purpose |
|---|---|
| `phase7_results/results/trackc_phase7v3_closure_note_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json` | Closure-grade non-smoke summary |
| `phase7_results/results/trackc_phase7v3_decision_phase7v3_20260305_234011_phase7v3_closure_non_smoke.json` | v3 decision artifact |
| `phase7_results/results/optionb_canary_decision_phase7_optionbc_20260306_092554_phase7_optionbc.json` | Option B fair profile comparison decision |
| `phase7_results/results/optionc_probe_phase7_optionbc_20260306_092554_phase7_optionbc.json` | Option C contrastive probe outcome |
| `phase7_results/results/optionbc_final_phase7_optionbc_20260306_092554_phase7_optionbc.json` | Combined Option B/C closure summary |
| `phase7_results/results/phase7_sae_trajectory_pathb_20260306_234123_phase7_sae_trajectory_pathb.json` | Path B feature-set swap summary |
| `phase7_results/results/phase7_sae_trajectory_pathc_robust_20260307_001237_phase7_sae_trajectory_pathc_robust.json` | Path C robust variant-exclusion + bootstrap/CV result |
| `phase7_results/results/phase7_mixed_trajectory_validation_phase7_mixed_trajectory_20260307_012248_phase7_mixed_trajectory_validation.json` | Mixed hidden+SAE ladder decision artifact |
| `phase7_results/results/mixed_gain_sweep_summary.json` | Multi-seed mixed-model stability sweep (baseline feature family) |
| `phase7_results/results/mixed_result_top50_seed_sweep_summary.json` | Multi-seed mixed-model stability sweep (`result_top50`) |
| `phase7_results/results/trackc_intervention_archive_manifest.json` | Archive index for Track C intervention-era artifacts |

## Option C Arithmetic Baseline Artifacts
| Path | Purpose |
|---|---|
| `phase7_results/results/optionc_summary_20260309_084825_phase7_optionc_generated_qwen_arith.json` | Canonical arithmetic Option C summary |
| `phase7_results/results/optionc_eval_20260309_084825_phase7_optionc_generated_qwen_arith_full.json` | Full arithmetic Option C evaluation metrics |
| `phase7_results/results/optionc_claim_boundary_20260309_084825_phase7_optionc_generated_qwen_arith_full.json` | Arithmetic claim-boundary decision artifact |
| `phase7_results/results/optionc_stress_20260309_130420_optionc_stress_full_rigor.json` | Full arithmetic stress summary (permutation + ablation/reg + multiseed, lexical-excluded training) |
| `phase7_results/results/trackc_arithmetic_significance_reframe_20260309_130420_optionc_stress_full_rigor.json` | Arithmetic significance reframe with p-value-primary policy |

## Option C Stress Artifacts
| Path | Purpose |
|---|---|
| `experiments/run_optionc_stress.sh` | Full Option C stress runner (permutation + ablation/reg + multiseed) |
| `phase7/stress_test_optionc_probe.py` | Option C-native stress evaluator with p-value-primary decision fields |
| `phase7/merge_g2_stress_decision.py` | Cross-domain merger for eval + stress decisions (`*_stress_validated`) |
| `phase7_results/results/optionc_stress_20260309_130420_optionc_stress_full_rigor.json` | Arithmetic stress result (primary: pass, p=0.000999) |
| `phase7_results/results/optionc_stress_20260309_130420_optionc_stress_full_rigor.md` | Arithmetic stress human-readable summary |
| `phase7_results/results/trackc_arithmetic_significance_reframe_20260309_130420_optionc_stress_full_rigor.json` | Arithmetic significance reframe with p-value-primary policy |

## Phase G2 Cross-Task Validation Artifacts
| Path | Purpose |
|---|---|
| `experiments/run_phase7_g2_cross_task.sh` | PrOntoQA → EntailmentBank cross-task gate orchestrator |
| `phase7_results/results/optionc_summary_20260309_141106_phase7_g2_domain_decoder_fix_prontoqa.json` | PrOntoQA Option C full summary (domain-decoder, CV AUROC 0.964, strict eval PASS) |
| `phase7_results/results/optionc_claim_boundary_20260309_141106_phase7_g2_domain_decoder_fix_prontoqa_full.json` | PrOntoQA claim boundary artifact for canonical G2 lineage |
| `phase7_results/results/optionc_eval_20260309_141106_phase7_g2_domain_decoder_fix_prontoqa_full.json` | PrOntoQA detailed eval (zero overlap, strict gate pass) |
| `phase7_results/results/optionc_summary_20260309_141106_phase7_g2_domain_decoder_fix_entailmentbank.json` | EntailmentBank Option C full summary (domain-decoder, CV AUROC 0.999, strict eval PASS) |
| `phase7_results/results/optionc_claim_boundary_20260309_141106_phase7_g2_domain_decoder_fix_entailmentbank_full.json` | EntailmentBank claim boundary artifact for canonical G2 lineage |
| `phase7_results/results/optionc_eval_20260309_141106_phase7_g2_domain_decoder_fix_entailmentbank_full.json` | EntailmentBank detailed eval (zero overlap, strict gate pass) |
| `phase7_results/results/trackc_g2_cross_task_decision_20260309_141106_phase7_g2_domain_decoder_fix.json` | Cross-domain decision on strict eval gates (PASS) |
| `phase7_results/results/optionc_stress_20260309_145050_optionc_stress_g2fix_prontoqa.json` | PrOntoQA full stress result (primary fail due regularization stability) |
| `phase7_results/results/optionc_stress_20260309_145052_optionc_stress_g2fix_entailmentbank.json` | EntailmentBank full stress result (primary pass) |
| `phase7_results/results/trackc_g2_cross_task_decision_20260309_141106_phase7_g2_domain_decoder_fix_stress_validated.json` | Stress-validated cross-domain decision (not publishable_cross_domain yet) |

## Option C Pipeline Scripts
| Path | Purpose |
|---|---|
| `phase7/contradictory_pair_prepare.py` | Contradictory pair generation (arithmetic, prontoqa, entailmentbank domains) |
| `phase7/internal_consistency_optionc.py` | SAE transition feature extraction (per-layer sharded) |
| `phase7/evaluate_optionc.py` | Option C evaluation: merge features + decoder + CV + claim gate |
| `phase7/stress_test_optionc_probe.py` | Option C-native stress evaluator |
| `experiments/run_phase7_optionc_generated.sh` | Option C full pipeline runner |
| `phase7/tests/test_optionc_pair_logic.py` | Unit tests for Option C pair labeling logic |

## Active Inquiry Docs
| File | Purpose |
|---|---|
| `TODO.md` | Active work queue: PrOntoQA stress-stability remediation after canonical domain-decoder eval pass |

## Historical Ablation Lineages
| Path | Purpose |
|---|---|
| `phase7_results/results/trackc_g2_cross_task_decision_20260309_124650_phase7_g2_cross_task_gpu135.json` | Pre domain-decoder-fix cross-task decision (historical ablation: PrOntoQA fail due decoder mismatch) |

## Archived Material
Historical execution reports and superseded plans are under `docs/archived_reports/`.

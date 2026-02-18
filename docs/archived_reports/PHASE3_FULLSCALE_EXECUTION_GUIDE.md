# Phase 3 Full-Scale Execution Guide

## Scope

- Full GSM8K evaluation with SAE expansions (4x-20x available now).
- Optional: train missing expansions (2x, 22x-32x) later.

## Pipeline Summary

Phase 3 runs four stages per SAE expansion:

1. Alignment: extract reasoning steps and align to tokens.
2. Latent extraction: encode activations through the SAE.
3. Probe training: predict step types from SAE latents.
4. Causal ablation: rank features by impact.

## Validation Baseline

- 500 GSM8K examples validated end-to-end.
- 12x SAE probe reached 100% accuracy and F1 in validation.
- Causal ablation evaluated 20 features with summaries saved.

## Run Phase 3 (GPUs 0-3)

```bash
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features
bash phase3/phase3_orchestrate.sh
```

Attach:
```bash
tmux attach -t phase3_fullscale
```

Monitor:
```bash
bash phase3_results/full_scale/monitor.sh
```

## Merge Results

```bash
python phase3/phase3_merge_results.py --input-dir phase3_results/full_scale
```

## Outputs

phase3_results/full_scale/ contains:
- results_gpu0.json
- results_gpu1.json
- results_gpu2.json
- results_gpu3.json
- log_gpu0.txt
- log_gpu1.txt
- log_gpu2.txt
- log_gpu3.txt
- REPORT.txt

Checkpoint path (model-specific):
- checkpoints/gpt2-small/sae/sae_768d_*x_final.pt

## Troubleshooting

- If a GPU window stalls, restart inside tmux:
  ```bash
  tmux send-keys -t phase3_fullscale:gpu0 C-c
  tmux send-keys -t phase3_fullscale:gpu0 "python phase3/phase3_full_scale.py --gpu-id 0 --sae-expansions 4 6 8 --output-file phase3_results/full_scale/results_gpu0.json" Enter
  ```

- If HF metadata requests warn about unauthenticated access, it is safe to ignore for offline cached data.

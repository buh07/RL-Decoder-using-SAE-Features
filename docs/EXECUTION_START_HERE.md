# Execution Start Guide

Scope: full GSM8K evaluation and SAE training orchestration.

## Step 0: Activate Environment

```bash
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features
source .venv/bin/activate
```

## Step 1: Train SAEs (GPUs 0-1)

```bash
python sae/sae_train_all_gsm8k_orchestrator.py --gpu-ids 0 1
```

Monitor:
```bash
bash sae/monitor_sae_training.sh
```

## Step 2: Run Phase 3 Full-Scale (GPUs 0-3)

```bash
bash phase3/phase3_orchestrate.sh
```

Attach to tmux:
```bash
tmux attach -t phase3_fullscale
```

## Step 3: Monitor Phase 3

```bash
bash phase3_results/full_scale/monitor.sh
```

## Step 4: Merge Results

```bash
python phase3/phase3_merge_results.py --input-dir phase3_results/full_scale
cat phase3_results/full_scale/REPORT.txt
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

## Optional: Train Missing Expansions (GPUs 4-7)

```bash
python sae/sae_train_missing_expansions.py --expansions 2 22 24 26 28 30 32 --num-gpus 4 --base-gpu 4
```

Checkpoint path (model-specific):
- checkpoints/gpt2-small/sae/sae_768d_*x_final.pt

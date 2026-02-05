# Experiment Tracking Setup

We rely on [Weights & Biases](https://wandb.ai) for long-running SAE/decoder
jobs. Follow the checklist below before launching any training:

1. Copy `configs/tracking/wandb_template.yaml` to a safe location (or update it
   in-place) with the correct `project` and `entity`.
2. Export your API key: `export WANDB_API_KEY=...`.
3. Run `python scripts/init_wandb_project.py --notes "phase1 smoke test"` to
   materialize a config file under `results/runs/`. The script appends tags and
   warns if the API key is missing.
4. Pass the generated YAML path into training jobs (e.g.,
   `python train_decoder.py --wandb-config results/runs/wandb_run_*.yaml`).
5. Record each run in `docs/logbook.md` with a link to the wandb dashboard.

Future improvements:
- add CI check to ensure wandb configs point to non-personal projects
- capture WANDB_MODE=offline option for restricted environments

# SETUP

## 1. Create & Activate the Virtual Environment
1. Run `./setup_env.sh` (or `PYTHON=python3.12 ./setup_env.sh` if you need to pin the interpreter). The script will create `.venv/` when missing, activate it, and attempt to install all required packages.
2. Manual activation commands (after the script runs once):
   - bash/zsh: `source .venv/bin/activate`
   - fish: `source .venv/bin/activate.fish`
   - PowerShell: `.venv\\Scripts\\Activate.ps1`
3. If the cluster blocks outbound internet (current default), pip will fail to download wheels. Work around this by pointing `PIP_INDEX_URL` to an internal mirror or manually copying wheels into the machine before re-running `setup_env.sh`. Run `SKIP_PIP=1 ./setup_env.sh` only after you have preinstalled packages another way so tokenizer syncing still happens.
4. Populate a local `.env` file (ignored by git) with secrets such as `WANDB_API_KEY=...`; the setup script auto-sources it before running.
5. `.venv/` stays ignored via `.gitignore`; never commit its contents.

## 2. Environment & Infrastructure Checklist (TODO Step 1)
- **Runtime Specs**: Verified `python --version` → 3.12.3 and `nvcc --version` → CUDA 12.8.93 on this machine. Record any future deviations (driver downgrades, cuDNN revisions) here so activation capture runs are reproducible.
- **Dependency Bootstrap (`setup_env.sh`)**: The script installs the minimal stack—`torch>=2.0`, `transformers>=4.37`, `trl>=0.8`, `accelerate>=0.25`, and `datasets`. Extend the `REQUIRED_PACKAGES` array in the script whenever new tooling (e.g., wandb, rich logging) is needed.
- **Tracking Defaults (Weights & Biases)**: Use the `rl-decoder-phase1` project under your W&B entity. Export `WANDB_API_KEY` in your shell or store it in the `.env`; the script now sources that file and logs whether the key is present.
- **Tokenizer Asset Sync**: Tokenizers are sourced from `/scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2` and copied into `assets/tokenizers/gpt2`. Override `TOKENIZER_SOURCE` or `TOKENIZER_DEST` when necessary (e.g., different base model) before running the script so every workstation gets identical `tokenizer.json`, `merges.txt`, and `vocab.json`.

Update this document as you solidify each decision so SETUP.md remains the single source of truth for onboarding.

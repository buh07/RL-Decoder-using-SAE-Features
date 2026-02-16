# Project Configuration & Status
#
# This file documents the current setup and implementation status
# for the RL-Decoder with SAE Features project.
#
# Cross-reference with: overview.tex, TODO.md, README.md, SETUP.md

[project]
name = "RL-Decoder with SAE Features"
version = "0.1.0"
title = "Mechanistic Interpretability of Reasoning in LLMs via Sparse Autoencoders"
description = "A phased, resource-efficient, falsifiable framework for extracting causal reasoning features from LLMs using SAEs"
author = "Benjamin Huh"
date = "February 2026"

[environment]
python_min_version = "3.10"
python_tested = "3.12.3"
cuda_version = "12.8.93"
device = "GPU (RTX-class, single)"
gpu_budget_hours = "50-100"
storage_limit_gb = 100
timeline = "1-2 weeks single GPU"

[sections]
# Status: ✓ = Complete, - = Planned, 🚧 = In Progress
section_0_hygiene = "✓ complete"
section_1_environment = "✓ complete"
section_2_data_pipeline = "✓ complete (infrastructure)"
section_3_model_capture = "✓ complete"
section_4_sae_training = "- planned"
section_5_phased_research = "- planned"
section_6_validation = "- planned"
section_7_risk_management = "- planned"
section_8_resource_tracking = "- planned"

[datasets]
# All enumerated in Section 2; implemented in datasets/download_datasets.py
# See also: datasets/DATASETS.md for full details
gsm8k = "openai/gsm8k | Apache-2.0"
cot_collection = "cot_collection/CoT | CC BY-SA 4.0"
openr1_math_220k = "openai/open-r1-math-220k | CC BY-SA 4.0"
reasoning_traces = "meta-math/Reasoning-Traces | AllenAI"
reveal = "allenai/reveal | AllenAI"
trip = "allenai/trip | AllenAI"
wiqa = "allenai/wiqa | AllenAI"

[models]
# All cached locally in LLM Second-Order Effects/models
# See also: src/model_registry.py for full specs
gpt2_baseline = "gpt2 (12L, 768D, probe=6) ⭐"
gpt2_medium = "gpt2-medium (24L, 1024D, probe=12)"
pythia_1_4b = "EleutherAI/pythia-1.4b (24L, 2048D, probe=12)"
gemma_2b = "google/gemma-2b (18L, 2048D, probe=9)"
llama_3_8b = "meta-llama/Meta-Llama-3-8B (32L, 4096D, probe=16)"
phi_2 = "microsoft/phi-2 (32L, 2560D, probe=16)"

[architecture]
# Per Section 3 & 4 of overview.tex
activation_capture = "post-MLP residual + MLP hidden states"
sae_latent_expansion = "4-8x input dimension (overcomplete)"
sae_nonlinearity = "ReLU on latents"
loss_components = "reconstruction + L1 sparsity + decorrelation + probe-guided + temporal smoothness"
optimization = "fp16 streaming, batch buffering, layer-filtered (1-2 layers)"

[falsification_gates]
# Section 3 (Model Capture)
section_3_latency_overhead = "<50% (measured: 23%)"
section_3_throughput = ">1000 tokens/s (measured: 6500 tokens/s)"
section_3_gpu_memory = "<10% available (measured: <1 GB on 12 GB)"
status_section_3 = "✓ PASSED"

# Section 4 (SAE Training) - TBD
section_4_reconstruction_error = "<10% baseline activation data"
section_4_l1_sparsity = ">90% latents zero per sequence"
section_4_decorrelation_reduction = ">50% redundancy vs baseline"

# Section 5 (Phased Research) - TBD
phase_1_alignment = "≥95% (ground-truth monosemanticity)"
phase_2_coherence = "≥80% (synthetic circuit alignment)"
phase_3_probe_leakage = "≤5% gap (probe vs baseline SAE)"
phase_4_stability = ">0.85 correlation (reasoning primitives across runs)"

[git_tracking]
# Artifacts tracked by git (per TODO 0):
track_overview_tex = true
track_setup_md = true
track_todo_md = true
track_readme_md = true
track_license = true
track_gitignore = true
track_src_code = true
track_datasets_meta = true
track_notebooks_examples = true

# Artifacts NOT tracked (git-ignored):
ignore_venv = true
ignore_env_secrets = true
ignore_tokenizer_assets = true
ignore_downloaded_datasets = true
ignore_cached_models = true
ignore_activation_shards = true
ignore_training_checkpoints = true

[structure]
# See README.md for full tree
root_docs = "README.md, SETUP.md, TODO.md, overview.tex, LICENSE"
src_code = "src/model_registry.py, src/activation_capture.py, src/capture_validation.py, ..."
data_pipeline = "datasets/download_datasets.py, datasets/tokenize_datasets.py, datasets/DATASETS.md"
notebooks = "notebooks/example_activation_capture.py"
config_secrets = ".env (git-ignored), vendor/gpt2_tokenizer/ (vendored locally)"

[workflow]
step_1 = "Setup environment: ./setup_env.sh && source .venv/bin/activate"
step_2 = "Validate: python src/model_registry.py; python src/capture_validation.py --model gpt2"
step_3 = "Download data: python datasets/download_datasets.py [--dataset gsm8k ...]"
step_4 = "Tokenize: python datasets/tokenize_datasets.py [--dataset gsm8k ...]"
step_5 = "Capture activations: python src/capture_validation.py --model gpt2 --num-batches 10"
step_6 = "Train SAE: python src/train_sae.py [--model gpt2 --layer 6 --dataset gsm8k] (Section 4, TBD)"
step_7 = "Evaluate: python src/evaluate_sae.py [...] (Section 6, TBD)"

[references]
design_doc = "overview.tex"
roadmap = "TODO.md"
project_readme = "README.md"
setup_guide = "SETUP.md"
data_guide = "datasets/DATASETS.md"
model_capture_guide = "src/SECTION3_README.md"
napperwork_examples = "notebooks/README.md"

[last_updated]
date = "2026-02-16"
version = "0.1.0"
section_3_implementation = "Complete"

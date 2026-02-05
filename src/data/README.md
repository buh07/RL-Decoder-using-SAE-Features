# Data Pipeline (Phase 1 scaffold)

Goals:
- Mirror tokenizer + preprocessing for gpt2 baseline.
- Stream sharded datasets for supervised (100M tokens) + RL buffer (10B tokens) + eval splits.

Phase 1 Action Items:
1. Implement `src/data/dataloader.py` that wraps HF `datasets` with deterministic tokenization.
2. Add CLI (`scripts/preprocess_dataset.py`) for caching token shards into `data/shards/{split}`.
3. Benchmark throughput on available RTX 6000 Ada GPUs once environment is active.

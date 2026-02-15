# Reasoning Datasets Overview

This document tracks the corpora referenced in TODO Step 2 and how they are normalized by `datasets/download_datasets.py`.

## Usage
1. Activate the project venv (`source .venv/bin/activate`).
2. Run `python datasets/download_datasets.py` to pull every dataset into `datasets/raw/<name>/<split>.jsonl`.
3. Add `--dataset gsm8k trip` to restrict the download list, or `--force` to overwrite existing files.
4. Inspect the generated `datasets/raw/manifest.json` for the exact spec (Hugging Face path, splits, columns).

> **Licensing**: Every dataset below carries its own license (Apache-2.0, CC BY-SA 4.0, AllenAI terms, etc.). Confirm your use is compliant before distributing derived data.

## Dataset Details

| Dataset | HF Path | Splits | Question Column | Answer Column | CoT Column | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | `openai/gsm8k` (subset `main`) | train/test | `question` | `answer` | _none_ (answer embeds reasoning) | Grade-school math; treat the provided answer as final response after stripping the boxed value. |
| `cot_collection` | `cot_collection/CoT` | train | `question` | `answer` | `rationale` | Aggregated chain-of-thought exemplars with dataset/source metadata for provenance checks. |
| `openr1_math_220k` | `openai/open-r1-math-220k` | train | `prompt` | `response` | `chain_of_thought` | Distilled math reasoning traces; optional `difficulty` tag for curriculum scheduling. |
| `reasoning_traces` | `meta-math/Reasoning-Traces` | train/validation | `problem` | `answer` | `chain_of_thought` | MetaMath-style proofs with labeled subject field (algebra, geometry, etc.). |
| `reveal` | `allenai/reveal` | train/validation/test | `question` | `answer` | `rationale` | Requires citation/context fields to verify factual support for rationales. |
| `trip` | `allenai/trip` | train/validation/test | `question` | `answer` | `program` | Each rationale is a program trace over supporting facts (column `context`). |
| `wiqa` | `allenai/wiqa` | train/validation/test | `question_stem` | `answer_label` | `para_steps` | Uses multiple-choice answers (labels `more`, `less`, `no_effect`). Additional columns capture perturbations/effects. |

### Column Handling Notes
- Every record is normalized to `{question, answer, cot, source_dataset, ...extras}` in the JSONL output. Missing CoT fields are returned as `null`.
- Extra fields preserved per dataset:
  - `cot_collection`: `dataset`, `source`
  - `openr1_math_220k`: `difficulty`
  - `reasoning_traces`: `subject`
  - `reveal`: `context`, `citation`
  - `trip`: `context`
  - `wiqa`: `perturbation`, `effect1`, `effect2`
- `gsm8k` is intentionally left without an explicit CoT column; downstream parsing should split the answer string if a boxed solution format is required.

### Sharded Tokenization Plan
1. **Output Layout**: Serialized shards live under `datasets/tokenized/gpt2/<dataset>/<split>/shard_XXXXX.pt` with a shared `datasets/tokenized/gpt2/manifest.json`. Each shard contains `seq_len`-length sequences plus metadata (`dataset`, `split`, `seq_len`, `num_sequences`, `token_count`, `source_files`, `shard_index`).
2. **Tokenizer + Formatting**: Always load the pinned GPT-2 tokenizer from `assets/tokenizers/gpt2`. For every JSONL record, concatenate:
   ```
   {question}

   {cot if present}

   Answer:
   {answer}
   ```
   strip redundant whitespace, append a single EOS token, and keep CoT optional.
3. **Chunking Parameters**: Default `seq_len=2048`, `tokens_per_shard=2_097_152` (→ 1024 sequences/shard). Set `pad_token_id = eos_token_id` so leftover buffers can be padded deterministically. Emit shards as `torch.int32` tensors to keep file sizes modest.
4. **CLI Controls**: `datasets/tokenize_datasets.py` (see below) accepts:
   - `--tokenizer-dir`, `--raw-dir`, `--tokenized-dir`
   - `--dataset` / `--split` filters (multiple allowed)
   - `--seq-len`, `--tokens-per-shard`
   - `--world-size` and `--rank` to deterministically divide work across array jobs (record index modulo world size)
   - `--resume`/`--force` flags to skip or overwrite shards
5. **Manifest + Provenance**: After each run, regenerate `datasets/tokenized/gpt2/manifest.json` summarizing every shard plus a `sha256` of the tensor payload. Training jobs can round-robin shards via `manifest[rank::world_size]`, and checkpoints should log the last shard processed.
6. **Staging for Multi-GPU Training**: Before SAE training, copy only the shards assigned to that node onto local NVMe (`/tmp/<job>/shards`). Each process memory-maps two shards at a time and advances `current_shard = (current_shard + world_size) % len(manifest)` each epoch, enabling elastic scaling from 1–8 RTX 6000 GPUs.
7. **Future Enhancements**: Add dataset-specific cleaning hooks (e.g., remove GSM8K boxed answers), shard-level checksum validation, and an optional temporal alignment file once CoT-to-token span logic is finalized.

### Temporal Alignment Plan
We need token-accurate spans tying rationales to SAE features. Alignment artifacts will live under `datasets/alignment/<dataset>/<split>.align.jsonl` with the schema:

```json
{
  "example_id": "cot_collection_train_000123",
  "sequence_index": 42,
  "shard_path": "datasets/tokenized/gpt2/cot_collection/train/shard_00007.pt",
  "text_offset_ranges": [[0, 128], [256, 344]],
  "token_offset_ranges": [[0, 64], [160, 186]],
  "step_labels": ["Hypothesis", "Constraint"],
  "confidence": 0.87,
  "notes": "regex-aligned; pending manual audit"
}
```

Alignment workflow:
1. **Stable IDs**: `datasets/tokenize_datasets.py` will emit `datasets/tokenized/gpt2/index.jsonl` mapping each `example_id` to `{shard_path, sequence_index}` so alignment consumers can jump straight to the right tensor slice.
2. **Span Extraction**: Dataset-specific heuristics (regex for “Step N”, program parsing for TRIP, multi-choice mapping for WIQA) create character spans inside the rendered prompt. When no explicit structure exists (e.g., GSM8K), fall back to heuristics like splitting on `\n\n`.
3. **Token Projection**: Re-tokenize the rendered text with the pinned GPT-2 tokenizer, convert character spans to token ranges, and store both. This keeps the alignment stable even if downstream tokenization logic changes.
4. **Confidence + Audits**: Each alignment row includes a confidence score (1.0 for explicit spans, <0.9 for fuzzy matches). Weekly manual audits flip `notes` to “verified” for a sampled subset; automated scripts can prioritize low-confidence rows.
5. **Consumption**: Probe-guided SAE training loads the alignment JSONL to attach supervision targets per reasoning step; analysis notebooks use it to aggregate causal metrics by span.

### Data QC & Validation
1. **Schema Checks**: `datasets/qc_validate.py` (to be authored) will confirm required columns exist, strings are non-empty, and alignment files reference valid `example_id`s & token ranges.
2. **Coverage Metrics**: Post-tokenization, record per-dataset stats (`total_tokens`, `% with CoT`, `avg_seq_len`, `alignment_coverage`) into `datasets/reports/YYYY-MM-DD.json`. Fail the pipeline if values drift beyond configured tolerances.
3. **Spot-Checks**: Lightweight notebooks in `notebooks/data_checks/` will render random samples with highlighted alignment spans to ensure formatting survived normalization.
4. **Invariance Tests**: Run whitespace/punctuation perturbations on random examples, re-tokenize, and ensure token IDs change by <1%. Flag datasets where formatting instability exceeds the threshold.
5. **Checksum Enforcement**: Re-hash raw JSONL files and shard tensors before training. `datasets/qc_verify.py` will read each `.meta.json` entry, recompute SHA256, and compare against the manifest.

### Open Tasks / Future Work
- Validate that each Hugging Face path is accessible from the target compute cluster; substitute mirrors if necessary.
- Augment the script with dataset-specific cleaning (e.g., removing the boxed `\boxed{}` wrappers, normalizing multiple-choice options).
- Add licensing metadata to `datasets/raw/manifest.json` once confirmed so compliance checks are automated.

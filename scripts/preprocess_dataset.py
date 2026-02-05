"""CLI entry for building token shards for Phase 1."""
import argparse
from pathlib import Path

from src.data.dataloader import DataConfig, ShardConfig, preprocess


def parse_args():
    p = argparse.ArgumentParser(description="Tokenize datasets into GPT-2 format shards")
    p.add_argument("--dataset", default="wikitext", help="HF dataset identifier")
    p.add_argument("--dataset-config", default="wikitext-103-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument(
        "--tokenizer-path",
        default="/scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2",
        help="Directory containing tokenizer.json/merges/vocab",
    )
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--tokens-per-shard", type=int, default=2_097_152)
    p.add_argument("--output-dir", default="data/shards/train")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DataConfig(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
    )
    shard_cfg = ShardConfig(output_dir=Path(args.output_dir), tokens_per_shard=args.tokens_per_shard)
    preprocess(cfg, shard_cfg)


if __name__ == "__main__":
    main()

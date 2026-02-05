# Tokenizer Artifact Hashes

Once `scripts/sync_tokenizer_assets.py` successfully copies the GPT-2 tokenizer
files into `assets/tokenizers/gpt2`, record their SHA256 hashes here so we can
prove reproducibility and detect drift.

| file | sha256 |
| --- | --- |
| tokenizer.json | _pending_ |
| merges.txt | _pending_ |
| vocab.json | _pending_ |

**How to update:**

1. Run `python scripts/sync_tokenizer_assets.py` from the project root. The
   script will copy the files and print their hashes.
2. Paste the hashes above and commit both the assets and this document.

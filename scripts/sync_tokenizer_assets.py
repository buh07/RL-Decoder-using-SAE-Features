"""Copy tokenizer artifacts from cached HuggingFace model snapshots.

The gpt2 model cache we have locally does not expose tokenizer files at the
root level, so this helper script copies the assets we need (tokenizer.json,
merges.txt, vocab.json) into the repository where other tooling can pick them
up easily. It also prints SHA256 hashes so we can document provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, Tuple

REQUIRED_FILES = ("tokenizer.json", "merges.txt", "vocab.json")


def read_snapshot_dir(model_root: Path) -> Path:
    """Return the path containing the actual files inside an HF cache."""
    refs_main = model_root / "refs" / "main"
    snapshots_dir = model_root / "snapshots"

    if refs_main.exists() and snapshots_dir.exists():
        snapshot_hash = refs_main.read_text().strip()
        snapshot_path = snapshots_dir / snapshot_hash
        if snapshot_path.exists():
            return snapshot_path
        raise FileNotFoundError(f"Snapshot directory {snapshot_path} missing")
    return model_root


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_files(source_dir: Path, dest_dir: Path) -> Iterable[Tuple[str, str]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    recorded = []
    for filename in REQUIRED_FILES:
        src = source_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")
        dst = dest_dir / filename
        shutil.copy2(src, dst)
        recorded.append((filename, compute_sha256(dst)))
    return recorded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync tokenizer assets into repo")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2"),
        help="Root of the cached HuggingFace model (contains refs/ + snapshots/)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("assets/tokenizers/gpt2"),
        help="Destination directory inside the repo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = read_snapshot_dir(args.model_root)
    print(f"[INFO] Using snapshot directory: {snapshot_dir}")
    records = sync_files(snapshot_dir, args.dest)
    print("[INFO] Copied tokenizer files:\n")
    for filename, sha in records:
        print(f"  - {filename}\tsha256={sha}")


if __name__ == "__main__":
    main()

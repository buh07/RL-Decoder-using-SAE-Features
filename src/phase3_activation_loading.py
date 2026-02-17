#!/usr/bin/env python3
"""
Phase 3 Activation Loading: Map aligned examples to activation shards and extract latents.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from phase3_alignment import ReasoningExample
from sae_architecture import SparseAutoencoder
from sae_training import ActivationShardDataset


class ActivationShardIndex:
    """Build and query an index of activation shards to find which shard contains each sequence."""

    def __init__(self, shard_dir: Path):
        """
        Initialize index by reading shard manifests.

        Args:
            shard_dir: Directory containing activation shards (*.pt) and manifest.json
        """
        self.shard_dir = shard_dir
        self.shard_index = []  # List of (shard_idx, start_seq, num_seqs)
        self.manifest = None

        self._build_index()

    def _build_index(self):
        """Read manifest.json to build sequence offset index."""
        manifest_path = self.shard_dir / "manifest.json"

        if not manifest_path.exists():
            print(f"[ActivationShardIndex] No manifest found at {manifest_path}")
            print("  Building index from shard files...")
            self._build_index_from_shards()
            return

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Expected manifest format:
        # {
        #   "total_chunks": N,
        #   "shards": [
        #     {"shard_idx": 0, "filename": "..._shard_000000.pt", "num_chunks": 208},
        #     ...
        #   ]
        # }

        cumulative_seq = 0
        if "shards" in self.manifest:
            for shard_info in self.manifest["shards"]:
                shard_idx = shard_info.get("shard_idx", 0)
                num_chunks = shard_info.get("num_chunks", 0)
                self.shard_index.append((shard_idx, cumulative_seq, num_chunks))
                cumulative_seq += num_chunks

        print(f"[ActivationShardIndex] Indexed {len(self.shard_index)} shards")

    def _build_index_from_shards(self):
        """Fallback: scan shard files directly."""
        shard_files = sorted(self.shard_dir.glob("*_shard_*.pt"))

        cumulative_seq = 0
        for shard_idx, shard_file in enumerate(shard_files):
            try:
                # Load and check shape
                data = torch.load(shard_file, map_location="cpu", weights_only=True)
                num_chunks = data.shape[0] if isinstance(data, torch.Tensor) else 1
                self.shard_index.append((shard_idx, cumulative_seq, num_chunks))
                cumulative_seq += num_chunks
            except Exception as e:
                print(f"  Warning: Could not load {shard_file}: {e}")

        print(f"[ActivationShardIndex] Found {len(self.shard_index)} shards by scanning")

    def find_shard(self, sequence_id: int) -> Optional[tuple[int, int, int]]:
        """
        Find which shard contains a sequence.

        Args:
            sequence_id: Global sequence index

        Returns:
            (shard_idx, start_seq, end_seq) or None if not found
        """
        for shard_idx, start_seq, num_seqs in self.shard_index:
            end_seq = start_seq + num_seqs
            if start_seq <= sequence_id < end_seq:
                return shard_idx, start_seq, end_seq

        return None


class ActivationExtractor:
    """Extract SAE latents from raw activations for aligned reasoning examples."""

    def __init__(
        self,
        activation_dir: Path,
        sae: SparseAutoencoder,
        device: str = "cuda:0",
    ):
        """
        Args:
            activation_dir: Directory with activation shards
            sae: Trained SAE model (used to encode activations → latents)
            device: Device for compute
        """
        self.activation_dir = activation_dir
        self.sae = sae
        self.device = device

        self.sae.to(device).eval()

        # Build index of shards
        self.shard_index = ActivationShardIndex(activation_dir)

        # Cache loaded shards to avoid repeated disk I/O
        self._shard_cache = {}

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        """Load a shard from disk (with caching)."""
        if shard_idx in self._shard_cache:
            return self._shard_cache[shard_idx]

        # Try to find shard file
        shard_files = sorted(self.activation_dir.glob("*_shard_*.pt"))
        if shard_idx >= len(shard_files):
            raise IndexError(f"Shard {shard_idx} not found (only {len(shard_files)} shards)")

        shard_file = shard_files[shard_idx]

        print(f"[ActivationExtractor] Loading shard {shard_idx}: {shard_file.name}")
        data = torch.load(shard_file, map_location="cpu", weights_only=True)

        self._shard_cache[shard_idx] = data
        return data

    @torch.no_grad()
    def extract_latents_for_example(
        self,
        example: ReasoningExample,
        sequence_id: int,
        start_token: int = 0,
        end_token: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract SAE latents for a subset of tokens in an example.

        Args:
            example: ReasoningExample with token-level information
            sequence_id: Global sequence ID in the dataset
            start_token: Start token index in sequence
            end_token: End token index (exclusive); if None, use all

        Returns:
            latents: [seq_len, latent_dim] from SAE encoder
        """
        if end_token is None:
            end_token = len(example.tokens)

        # Find shard containing this sequence
        shard_info = self.shard_index.find_shard(sequence_id)
        if shard_info is None:
            raise ValueError(f"Sequence {sequence_id} not found in activation shards")

        shard_idx, shard_start_seq, shard_end_seq = shard_info
        seq_in_shard = sequence_id - shard_start_seq

        # Load shard
        shard_data = self._load_shard(shard_idx)

        # Extract this sequence's activations
        activations = shard_data[seq_in_shard, start_token:end_token, :].to(self.device).float()

        # Encode with SAE
        with torch.no_grad():
            latents = self.sae.encoder(activations)  # [seq_len, latent_dim]

        return latents.cpu()

    def extract_latents_for_examples(
        self,
        examples: list[ReasoningExample],
        sequence_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Extract latents for multiple examples.

        Args:
            examples: List of ReasoningExample
            sequence_ids: Corresponding global sequence IDs

        Returns:
            dict: {example_id: latents[seq_len, latent_dim]}
        """
        result = {}

        for example, seq_id in zip(examples, sequence_ids):
            try:
                latents = self.extract_latents_for_example(example, seq_id)
                result[example.example_id] = latents
            except Exception as e:
                print(f"  Warning: Could not extract latents for {example.example_id}: {e}")

        return result


class ActivationBatchExtractor:
    """Extract latents in batches for efficiency (avoids repeated SAE passes)."""

    def __init__(
        self,
        activation_dir: Path,
        sae: SparseAutoencoder,
        device: str = "cuda:0",
    ):
        """
        Args:
            activation_dir: Directory with activation shards
            sae: Trained SAE model
            device: Device for compute
        """
        self.activation_dir = activation_dir
        self.sae = sae
        self.device = device

        self.sae.to(device).eval()
        self.shard_index = ActivationShardIndex(activation_dir)
        self._shard_cache = {}

    def extract_latents_batch(
        self,
        examples: list[ReasoningExample],
        sequence_ids: list[int],
        batch_size: int = 256,
    ) -> dict[str, torch.Tensor]:
        """
        Extract latents for multiple examples using batch processing for efficiency.

        Args:
            examples: List of ReasoningExample
            sequence_ids: Corresponding global sequence IDs
            batch_size: Token batch size for encoding

        Returns:
            dict: {example_id: latents[seq_len, latent_dim]}
        """
        result = {}
        activation_batch = []
        batch_metadata = []

        for example, seq_id in zip(examples, sequence_ids):
            try:
                shard_info = self.shard_index.find_shard(seq_id)
                if shard_info is None:
                    continue

                shard_idx, shard_start_seq, _ = shard_info
                seq_in_shard = seq_id - shard_start_seq

                if shard_idx not in self._shard_cache:
                    shard_files = sorted(self.activation_dir.glob("*_shard_*.pt"))
                    if shard_idx < len(shard_files):
                        self._shard_cache[shard_idx] = torch.load(
                            shard_files[shard_idx], map_location="cpu", weights_only=True
                        )

                if shard_idx in self._shard_cache:
                    shard_data = self._shard_cache[shard_idx]
                    activations = shard_data[seq_in_shard, :, :].float()

                    activation_batch.append(activations)
                    batch_metadata.append((example.example_id, len(activations)))

                    # Process batch when full
                    if len(activation_batch) >= batch_size:
                        result.update(
                            self._encode_batch(activation_batch, batch_metadata)
                        )
                        activation_batch = []
                        batch_metadata = []

            except Exception as e:
                print(f"  Warning: Could not process {example.example_id}: {e}")

        # Process remaining
        if activation_batch:
            result.update(self._encode_batch(activation_batch, batch_metadata))

        return result

    @torch.no_grad()
    def _encode_batch(
        self,
        activation_batch: list[torch.Tensor],
        metadata: list[tuple[str, int]],
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of activations to latents."""
        result = {}

        for activations, (example_id, seq_len) in zip(activation_batch, metadata):
            activations = activations.to(self.device)
            latents = self.sae.encoder(activations)  # [seq_len, latent_dim]
            result[example_id] = latents.cpu()

        return result

#!/usr/bin/env python3
"""
Activation capture hooks for extracting post-MLP residual streams and MLP hidden states.
Streams activations to disk in fp16 format, layer-filtered for single-GPU memory efficiency.
"""
from __future__ import annotations

import io
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class CaptureConfig:
    """Configuration for activation capture."""
    
    output_dir: Path
    """Root directory for activation shards."""
    
    layer_indices: list[int]
    """Which layers to capture (e.g., [6] for single mid-layer, [6, 9, 12] for multiple)."""
    
    capture_residual: bool = True
    """Capture post-MLP residual streams (X_out from attention+MLP block)."""
    
    capture_mlp_hidden: bool = True
    """Capture MLP hidden states (intermediate after MLP nonlinearity, before output proj)."""
    
    dtype: torch.dtype = torch.float16
    """Data type for saving (fp16 to conserve disk space)."""
    
    max_activations_per_file: int = 1000
    """Flush activations to disk after collecting this many token batches."""
    
    enable_gradient: bool = False
    """If False, wrap hooks in torch.no_grad() context."""


class ActivationCapture:
    """
    Manages PyTorch hooks to capture and stream activations from specified layers.
    
    Hooks attach to transformer layers and record post-MLP residuals and/or MLP internals.
    Activations are streamed to disk in fp16 shards with metadata for later alignment.
    """
    
    def __init__(self, model: nn.Module, config: CaptureConfig):
        self.model = model
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.activations: dict[str, list[Tensor]] = defaultdict(list)
        self.metadata: list[dict] = []
        self.shard_count = 0
        self.token_count = 0
        self.hooks: list[Any] = []
        
    def _get_hook_fn(self, layer_idx: int, hook_type: str) -> Callable:
        """Return a hook function for a given layer and activation type."""
        
        def hook_fn(module: nn.Module, input: tuple, output: Tensor) -> None:
            if not self.config.enable_gradient:
                output = output.detach()
            else:
                output = output.clone()
            
            # Convert to fp16 and move to CPU to save GPU memory
            output_fp16 = output.to(dtype=self.config.dtype, device="cpu")
            
            key = f"layer_{layer_idx}_{hook_type}"
            self.activations[key].append(output_fp16)
            
            # Track shape for metadata
            batch_size, seq_len, hidden_dim = output_fp16.shape
            self.token_count += batch_size * seq_len
            
            # Auto-flush if buffer is large
            if len(self.activations[key]) >= self.config.max_activations_per_file:
                self._flush_activations()
        
        return hook_fn
    
    def attach_hooks(self, get_layer_module: Callable[[int], nn.Module]) -> None:
        """
        Attach hooks to specified layers.
        
        Args:
            get_layer_module: Callable that takes layer_idx and returns the transformer block module.
                              E.g., model.transformer.h[idx] for GPT-2.
        """
        for layer_idx in self.config.layer_indices:
            module = get_layer_module(layer_idx)
            
            if self.config.capture_residual:
                hook = module.register_forward_hook(
                    self._get_hook_fn(layer_idx, "residual")
                )
                self.hooks.append(hook)
            
            if self.config.capture_mlp_hidden:
                # For GPT-2 style: module.mlp.c_proj takes the hidden state input
                # We hook the MLP's output (before residual addition) by attaching to mlp module
                try:
                    mlp_module = module.mlp
                    hook = mlp_module.register_forward_hook(
                        self._get_hook_fn(layer_idx, "mlp_hidden")
                    )
                    self.hooks.append(hook)
                except AttributeError:
                    print(f"[warn] Layer {layer_idx} has no .mlp attribute; skipping MLP hook.")
    
    def _flush_activations(self) -> None:
        """Write accumulated activations to disk as a single shard."""
        if not self.activations:
            return
        
        for key, acts in self.activations.items():
            if not acts:
                continue
            
            # Stack into a single tensor (batch_size, seq_len, hidden_dim)
            # Shapes vary if sequences have different lengths; pad if needed.
            max_seq_len = max(a.shape[1] for a in acts)
            batch_size_total = sum(a.shape[0] for a in acts)
            hidden_dim = acts[0].shape[2]
            
            stacked = torch.zeros(
                batch_size_total, max_seq_len, hidden_dim,
                dtype=self.config.dtype,
                device="cpu"
            )
            
            offset = 0
            seq_lens = []
            for act in acts:
                bs, seq_len, hd = act.shape
                stacked[offset : offset + bs, :seq_len, :] = act
                seq_lens.extend([seq_len] * bs)
                offset += bs
            
            # Save shard
            shard_path = self.config.output_dir / f"{key}_shard_{self.shard_count:06d}.pt"
            payload = {
                "activations": stacked,
                "seq_lens": torch.tensor(seq_lens, dtype=torch.int32),
            }
            torch.save(payload, shard_path)
            
            # Save metadata
            meta_path = shard_path.with_suffix(".meta.json")
            meta = {
                "shard_index": self.shard_count,
                "key": key,
                "shape": [int(x) for x in stacked.shape],
                "dtype": str(self.config.dtype),
                "num_sequences": batch_size_total,
                "max_seq_len": max_seq_len,
                "hidden_dim": hidden_dim,
                "token_count": int(batch_size_total * max_seq_len),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            
            print(
                f"[shard] {key} -> {shard_path.name} "
                f"(seqs={batch_size_total}, max_seq_len={max_seq_len}, tokens~{meta['token_count']})"
            )
        
        self.shard_count += 1
        self.activations.clear()
    
    def remove_hooks(self) -> None:
        """Unregister all attached hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def close(self) -> None:
        """Finalize capture: flush remaining activations and save manifest."""
        self._flush_activations()
        self.remove_hooks()
        
        # Write global manifest
        manifest = {
            "model_config": asdict(self.config),
            "layer_indices": self.config.layer_indices,
            "total_shards": self.shard_count,
            "total_tokens_captured": self.token_count,
            "dtype": str(self.config.dtype),
        }
        manifest_path = self.config.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[manifest] wrote {manifest_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_gpt2_capture(
    model: nn.Module,
    output_dir: Path,
    layer_indices: list[int],
    capture_residual: bool = True,
    capture_mlp_hidden: bool = True,
) -> ActivationCapture:
    """
    Factory for GPT-2 style models (e.g., gpt2, gpt2-medium).
    
    Args:
        model: Loaded GPT-2 model.
        output_dir: Where to save activation shards.
        layer_indices: Which transformer layers to capture.
        capture_residual: Whether to capture post-layer residuals.
        capture_mlp_hidden: Whether to capture MLP hidden states.
    
    Returns:
        ActivationCapture instance configured for GPT-2.
    """
    config = CaptureConfig(
        output_dir=output_dir,
        layer_indices=layer_indices,
        capture_residual=capture_residual,
        capture_mlp_hidden=capture_mlp_hidden,
    )
    capture = ActivationCapture(model, config)
    
    def get_gpt2_layer(idx: int) -> nn.Module:
        return model.transformer.h[idx]
    
    capture.attach_hooks(get_gpt2_layer)
    return capture


if __name__ == "__main__":
    print("Activation capture module. Import and use create_gpt2_capture() or ActivationCapture directly.")

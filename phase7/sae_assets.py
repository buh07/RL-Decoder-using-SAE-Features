#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_architecture import SparseAutoencoder  # type: ignore
from sae_config import SAEConfig  # type: ignore


def model_key_safe(model_key: str) -> str:
    return str(model_key).replace("/", "-")


def sae_checkpoint_path(*, saes_dir: Path, model_key: str, layer: int) -> Path:
    return Path(saes_dir) / f"{model_key_safe(model_key)}_layer{int(layer)}_sae.pt"


def activation_stats_path(*, activations_dir: Path, model_key: str, layer: int) -> Path:
    return Path(activations_dir) / f"{model_key_safe(model_key)}_layer{int(layer)}_activations.pt"


def load_sae_for_layer(
    *,
    saes_dir: Path,
    model_key: str,
    layer: int,
    device: str,
    use_half_on_cuda: bool = True,
) -> SparseAutoencoder:
    ckpt_path = sae_checkpoint_path(saes_dir=saes_dir, model_key=model_key, layer=layer)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = dict(ckpt["config"])
    cfg = SAEConfig(
        input_dim=int(c["input_dim"]),
        expansion_factor=int(c["expansion_factor"]),
        use_relu=bool(c.get("use_relu", True)),
        use_topk=bool(c.get("use_topk", False)),
        topk_k=int(c.get("topk_k", 0)),
        use_amp=False,
    )
    sae = SparseAutoencoder(cfg)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device).eval()
    if bool(use_half_on_cuda) and str(device).startswith("cuda"):
        sae = sae.half()
    return sae


def load_saes(
    *,
    saes_dir: Path,
    model_key: str,
    num_layers: int,
    device: str,
    use_half_on_cuda: bool = True,
) -> Dict[int, SparseAutoencoder]:
    out: Dict[int, SparseAutoencoder] = {}
    for layer in range(int(num_layers)):
        out[int(layer)] = load_sae_for_layer(
            saes_dir=saes_dir,
            model_key=model_key,
            layer=int(layer),
            device=device,
            use_half_on_cuda=bool(use_half_on_cuda),
        )
    return out


def load_norm_stats_for_layer(
    *,
    activations_dir: Path,
    model_key: str,
    layer: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    path = activation_stats_path(activations_dir=activations_dir, model_key=model_key, layer=layer)
    if not path.exists():
        raise FileNotFoundError(f"Missing activation stats: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    acts = payload["activations"] if isinstance(payload, dict) else payload
    if not isinstance(acts, torch.Tensor):
        raise TypeError(f"Unexpected activation payload type for layer={layer}: {type(acts).__name__}")
    if acts.ndim == 3:
        acts = acts.reshape(-1, acts.shape[-1])
    acts = acts.float()
    return acts.mean(dim=0).to(device), acts.std(dim=0).clamp_min(1e-6).to(device)


def load_norm_stats(
    *,
    activations_dir: Path,
    model_key: str,
    num_layers: int,
    device: str,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in range(int(num_layers)):
        try:
            out[int(layer)] = load_norm_stats_for_layer(
                activations_dir=activations_dir,
                model_key=model_key,
                layer=int(layer),
                device=device,
            )
        except FileNotFoundError:
            continue
    return out


def infer_sae_dim(saes: Dict[int, SparseAutoencoder]) -> int:
    if not saes:
        return 0
    first = next(iter(saes.values()))
    return int(first.decoder.weight.shape[1])


def can_load_sae_assets(*, saes_dir: Optional[str], activations_dir: Optional[str], model_key: str, num_layers: int) -> Tuple[bool, Optional[str]]:
    if not saes_dir:
        return False, f"missing_model_sae_dir:{model_key}"
    sd = Path(saes_dir)
    if not sd.exists():
        return False, f"missing_sae_dir_path:{sd}"
    missing_layers = []
    for layer in range(int(num_layers)):
        p = sae_checkpoint_path(saes_dir=sd, model_key=model_key, layer=layer)
        if not p.exists():
            missing_layers.append(layer)
            if len(missing_layers) >= 3:
                break
    if missing_layers:
        return False, f"missing_sae_checkpoints:{model_key}:layers={missing_layers}"
    if activations_dir:
        ad = Path(activations_dir)
        if not ad.exists():
            return False, f"missing_activations_dir_path:{ad}"
    return True, None

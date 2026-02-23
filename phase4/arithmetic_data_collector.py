#!/usr/bin/env python3
"""
Phase 7: Arithmetic Data Collector
====================================
Collects SAE feature vectors at key token positions for GSM8K arithmetic
annotations.  This dataset is shared by all three probing experiments
(Experiments A, B, and C).

Algorithm
---------
For each of the first --num-examples GSM8K examples:

  1. Build the full text:  question + "\\n" + answer.
  2. Find all <<expression=result>> annotations with a regex.
  3. Tokenise with return_offsets_mapping=True to map char positions → token
     indices.  Identify four key positions per annotation:
       • eq_tok     — the ``=`` sign inside the annotation (just before the
                      result); the last ``=`` in the annotation string.
       • pre_eq_toks — all tokens of the expression before ``=`` (the
                      operand/operator region).
       • result_toks — tokens of the result C (between ``=`` and ``>>``).
       • outer_eq_tok — the ``=`` that precedes ``<<`` in the surrounding
                       text, if one exists within 12 characters of ``<<``.
  4. Run a single GPT-2 medium forward pass with output hooks registered on
     every transformer block.
  5. For each (annotation, layer) pair, encode the relevant activations
     through the layer's SAE and record the sparse latent vector h.
  6. Store features at eq_tok and the mean of pre_eq_toks / result_toks as
     float16 tensors of shape [NUM_LAYERS, LATENT_DIM].

Output
------
  phase7/results/collection/gsm8k_arithmetic_dataset.pt
    A list of record dicts, one per annotation, containing:
      - 'example_idx'    : int
      - 'ann_text'       : str (full <<expression=result>>)
      - 'expr_str'       : str (part before =)
      - 'C'              : float (result value)
      - 'log_abs_C'      : float (log(|C|+1), probe target)
      - 'eq_tok_idx'     : int (token index of = in annotation)
      - 'pre_eq_tok_idxs': List[int]
      - 'result_tok_idxs': List[int]
      - 'outer_eq_tok_idx': int or None
      - 'eq_features'    : Tensor[NUM_LAYERS, LATENT_DIM] float16
      - 'pre_eq_features' : Tensor[NUM_LAYERS, LATENT_DIM] float16 (mean)
      - 'result_features' : Tensor[NUM_LAYERS, LATENT_DIM] float16 (mean)
      - 'token_ids'      : List[int] (full sequence)
      - 'tokens'         : List[str] (decoded strings)

Usage
-----
  cd "/scratch2/f004ndc/RL-Decoder with SAE Features"
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python3 phase7/arithmetic_data_collector.py \\
      --saes-dir      phase5_results/multilayer_gpt2_12x/saes \\
      --activations-dir phase4_results/activations_multilayer \\
      --gsm8k-path    datasets/raw/gsm8k/train.jsonl \\
      --output-dir    phase7/results/collection \\
      --num-examples  200 \\
      --device        cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_architecture import SparseAutoencoder
from sae_config import SAEConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LAYERS = 24
HIDDEN_DIM = 1024
LATENT_DIM = 12288  # 12 × 1024
MODEL_ID = "gpt2-medium"

# Matches <<expression=result>> where result is a decimal number.
# The expression may contain multiple operators (e.g. 100-50-30=20).
ANN_RE = re.compile(r"<<([^>]*?)=\s*(-?\d+(?:\.\d+)?)\s*>>")

# ---------------------------------------------------------------------------
# SAE and normalisation utilities (mirrors reasoning_flow_tracer.py)
# ---------------------------------------------------------------------------

def load_saes(saes_dir: Path, device: str) -> Dict[int, SparseAutoencoder]:
    """Load all 24 layer SAEs from ``gpt2-medium_layer{N}_sae.pt`` checkpoints.

    Parameters
    ----------
    saes_dir : Path
        Directory containing the checkpoint files.
    device : str
        Torch device string (e.g. ``"cuda:0"``).

    Returns
    -------
    Dict[int, SparseAutoencoder]
        Mapping from layer index to a loaded, eval-mode SAE.
    """
    saes: Dict[int, SparseAutoencoder] = {}
    for layer_idx in range(NUM_LAYERS):
        ckpt_path = saes_dir / f"gpt2-medium_layer{layer_idx}_sae.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAE checkpoint missing: {ckpt_path}\n"
                f"Run phase5_task4_train_multilayer_saes.py first."
            )
        # weights_only=False: checkpoints store PosixPath objects (PyTorch 2.6+)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cd = ckpt["config"]
        cfg = SAEConfig(
            input_dim=cd["input_dim"],
            expansion_factor=cd["expansion_factor"],
            use_relu=cd.get("use_relu", True),
            use_amp=False,
        )
        sae = SparseAutoencoder(cfg)
        sae.load_state_dict(ckpt["model_state_dict"])
        sae = sae.to(device).eval()
        saes[layer_idx] = sae

    print(f"  Loaded {len(saes)} SAEs  "
          f"(expansion={saes[0].config.expansion_factor}x, "
          f"latent_dim={saes[0].config.latent_dim})")
    return saes


def load_norm_stats(
    activations_dir: Path,
    device: str,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute per-layer normalisation statistics from the training activation files.

    The SAEs were trained on activations that had been normalised to zero mean
    and unit variance computed across the training corpus.  The same transform
    must be applied at inference time so that the learned feature detectors
    respond at the correct activation magnitudes.

    Parameters
    ----------
    activations_dir : Path
        Directory containing ``gpt2-medium_layer{N}_activations.pt`` files.
    device : str
        Target device for the returned tensors.

    Returns
    -------
    Dict[int, Tuple[Tensor, Tensor]]
        layer_idx → (mean, std), each of shape (HIDDEN_DIM,).
    """
    stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in range(NUM_LAYERS):
        path = activations_dir / f"gpt2-medium_layer{layer_idx}_activations.pt"
        if not path.exists():
            print(f"  WARNING: no activation file for layer {layer_idx}")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        acts = payload["activations"] if isinstance(payload, dict) else payload
        if acts.dim() == 3:
            acts = acts.reshape(-1, acts.shape[-1])
        acts = acts.float()
        stats[layer_idx] = (
            acts.mean(dim=0).to(device),
            acts.std(dim=0).clamp_min(1e-6).to(device),
        )
    print(f"  Loaded norm stats for {len(stats)}/{NUM_LAYERS} layers.")
    return stats


def normalize(
    x: torch.Tensor,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Apply pre-computed (mean, std) normalisation to a 1-D or 2-D tensor."""
    if stats is None:
        return x
    mean, std = stats
    return (x - mean) / std


def unnormalize(
    x: torch.Tensor,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Invert normalisation: useful in Experiment C causal patching."""
    if stats is None:
        return x
    mean, std = stats
    return x * std + mean

# ---------------------------------------------------------------------------
# Forward-hook infrastructure
# ---------------------------------------------------------------------------

def register_hooks(model) -> Tuple[Dict[int, torch.Tensor], List]:
    """Register output hooks on all 24 GPT-2 medium transformer blocks.

    The hook captures the post-block residual-stream state (the first element
    of the ``GPT2Block`` output tuple) immediately after each block's MLP.
    All tensors are moved to CPU right away to avoid filling GPU memory.

    Returns
    -------
    activation_store : Dict[int, Tensor]
        Populated on each forward pass; shape (1, seq_len, HIDDEN_DIM).
    handles : List
        Removable hook handles.
    """
    activation_store: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activation_store[layer_idx] = hidden.detach().cpu()
        return hook_fn

    for i in range(NUM_LAYERS):
        handle = model.transformer.h[i].register_forward_hook(make_hook(i))
        handles.append(handle)

    return activation_store, handles

# ---------------------------------------------------------------------------
# Annotation parsing and token alignment
# ---------------------------------------------------------------------------

def find_tokens_for_span(
    offset_mapping: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    """Return all token indices whose character span overlaps [char_start, char_end).

    Parameters
    ----------
    offset_mapping : List[Tuple[int, int]]
        Output of ``tokenizer(..., return_offsets_mapping=True)``.
    char_start, char_end : int
        Half-open character range.

    Returns
    -------
    List[int]
        Token indices that overlap with the given range, in order.
    """
    result = []
    for i, (s, e) in enumerate(offset_mapping):
        if e > char_start and s < char_end and e > s:
            result.append(i)
    return result


def find_token_for_char(
    offset_mapping: List[Tuple[int, int]],
    char_pos: int,
) -> Optional[int]:
    """Find the single token index that contains char_pos."""
    for i, (s, e) in enumerate(offset_mapping):
        if s <= char_pos < e:
            return i
    # Fall back: find the token that starts at or just before char_pos
    best = None
    for i, (s, e) in enumerate(offset_mapping):
        if s <= char_pos:
            best = i
    return best


def parse_annotations(text: str) -> List[dict]:
    """Extract all <<expression=result>> annotations from ``text``.

    For each annotation, computes the character positions of:
      • the ``=`` sign inside the annotation
      • the expression string (before ``=``)
      • the result value C

    Parameters
    ----------
    text : str
        Full example text (question + answer).

    Returns
    -------
    List[dict]
        One dict per annotation with keys:
        ``ann_text``, ``expr_str``, ``C``,
        ``inner_eq_char``, ``expr_chars``, ``result_chars``, ``ann_span``,
        ``outer_eq_char`` (char of preceding ``=`` or None).
    """
    results = []
    for m in ANN_RE.finditer(text):
        expr_str = m.group(1)
        C = float(m.group(2))

        ann_text = m.group(0)
        ann_start = m.start()
        ann_end = m.end()

        # Position of the '=' inside the annotation (last '=' in '<<expr=result>>')
        # m.start(2) - 1 points to the '=' between expr_str and result (after any spaces)
        # We scan backward from m.start(2) to find the exact '=' char
        inner_eq_char = ann_start + ann_text.rfind("=")

        # Expression characters: from after '<<' to just before the '='
        expr_start_char = ann_start + 2         # skip '<<'
        expr_end_char = inner_eq_char            # up to (not including) '='

        # Result characters: from after '=' to before '>>'
        result_start_char = inner_eq_char + 1
        result_end_char = ann_end - 2            # exclude '>>'

        # Look for an outer '=' that immediately precedes '<<' in the text
        # (e.g. "48/2 = <<48/2=24>>" — the '=' before '<<')
        outer_eq_char: Optional[int] = None
        search_start = max(0, ann_start - 12)
        substr = text[search_start:ann_start]
        idx = substr.rfind("=")
        if idx != -1:
            outer_eq_char = search_start + idx

        results.append({
            "ann_text":       ann_text,
            "expr_str":       expr_str.strip(),
            "C":              C,
            "inner_eq_char":  inner_eq_char,
            "expr_chars":     (expr_start_char, expr_end_char),
            "result_chars":   (result_start_char, result_end_char),
            "ann_span":       (ann_start, ann_end),
            "outer_eq_char":  outer_eq_char,
        })
    return results

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features_at_positions(
    tok_indices: List[int],
    layer_activations: Dict[int, torch.Tensor],
    saes: Dict[int, SparseAutoencoder],
    norm_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: str,
    aggregate: str = "mean",
) -> torch.Tensor:
    """Encode residual-stream activations at given token positions through SAEs.

    For each of the 24 layers, takes the residual-stream vector(s) at the
    requested token positions, normalises them, encodes through the layer's
    SAE, and (optionally) averages over positions.

    Parameters
    ----------
    tok_indices : List[int]
        Token positions to extract.  If multiple, the latent vectors are
        averaged together before returning (when aggregate="mean").
    layer_activations : Dict[int, Tensor]
        Output of the forward hooks; layer_idx → (1, seq_len, HIDDEN_DIM).
    saes : Dict[int, SparseAutoencoder]
        Loaded SAEs for all layers.
    norm_stats : Dict[int, Tuple[Tensor, Tensor]]
        Per-layer normalisation statistics.
    device : str
        Computation device.
    aggregate : str
        "mean" to average over token positions; "first" to use the first only.

    Returns
    -------
    Tensor
        Shape (NUM_LAYERS, LATENT_DIM), dtype float16.
    """
    all_layers: List[torch.Tensor] = []
    for layer_idx in range(NUM_LAYERS):
        acts = layer_activations[layer_idx]  # (1, seq_len, HIDDEN_DIM)
        seq_len = acts.shape[1]
        valid_idx = [i for i in tok_indices if 0 <= i < seq_len]
        if not valid_idx:
            all_layers.append(torch.zeros(LATENT_DIM, dtype=torch.float16))
            continue

        # Gather activations at requested positions  → (n_pos, HIDDEN_DIM)
        vecs = acts[0, valid_idx, :].to(device)  # float32

        # Apply per-layer normalisation
        stats = norm_stats.get(layer_idx)
        vecs_norm = normalize(vecs, stats)

        # Encode through SAE → (n_pos, LATENT_DIM)
        with torch.no_grad():
            h = saes[layer_idx].encode(vecs_norm)  # ReLU applied inside encode()

        if aggregate == "mean":
            h = h.mean(dim=0)   # (LATENT_DIM,)
        else:
            h = h[0]            # (LATENT_DIM,)

        all_layers.append(h.half().cpu())

    return torch.stack(all_layers, dim=0)  # (NUM_LAYERS, LATENT_DIM)

# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(args) -> None:
    device = args.device
    saes_dir = Path(args.saes_dir)
    activations_dir = Path(args.activations_dir)
    gsm8k_path = Path(args.gsm8k_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gsm8k_arithmetic_dataset.pt"

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"\n[1/4] Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(device).eval()
    print(f"  Model on {device}.")

    # -----------------------------------------------------------------------
    # Load SAEs and norm stats
    # -----------------------------------------------------------------------
    print("\n[2/4] Loading SAEs …")
    saes = load_saes(saes_dir, device)

    print("\n[3/4] Loading normalisation statistics …")
    norm_stats = load_norm_stats(activations_dir, device)

    # Register forward hooks
    activation_store, handles = register_hooks(model)

    # -----------------------------------------------------------------------
    # Process GSM8K examples
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Processing {args.num_examples} examples from {gsm8k_path} …")
    records: List[dict] = []
    n_skipped = 0

    with open(gsm8k_path) as f:
        raw_examples = [json.loads(line) for line in f][:args.num_examples]

    for ex_idx, ex in enumerate(raw_examples):
        text = ex["question"] + "\n" + ex["answer"]
        annotations = parse_annotations(text)
        if not annotations:
            n_skipped += 1
            continue

        # Tokenise with character-level offset mapping
        enc = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = enc["input_ids"].to(device)
        offset_mapping: List[Tuple[int, int]] = enc["offset_mapping"][0].tolist()
        tokens_str: List[str] = [
            tokenizer.decode([tid]) for tid in input_ids[0].tolist()
        ]

        # Forward pass — activations captured in activation_store
        with torch.no_grad():
            model(input_ids)

        for ann_idx, ann in enumerate(annotations):
            # --- Map character positions → token indices -------------------
            eq_toks = find_tokens_for_span(
                offset_mapping,
                ann["inner_eq_char"],
                ann["inner_eq_char"] + 1,
            )
            pre_eq_toks = find_tokens_for_span(
                offset_mapping,
                ann["expr_chars"][0],
                ann["expr_chars"][1],
            )
            result_toks = find_tokens_for_span(
                offset_mapping,
                ann["result_chars"][0],
                ann["result_chars"][1],
            )
            outer_eq_tok: Optional[int] = None
            if ann["outer_eq_char"] is not None:
                outer_eq_tok = find_token_for_char(
                    offset_mapping, ann["outer_eq_char"]
                )

            # Require at least one token per key position
            if not eq_toks or not pre_eq_toks or not result_toks:
                n_skipped += 1
                continue

            # --- Extract SAE features at each position -------------------
            eq_features = extract_features_at_positions(
                eq_toks, activation_store, saes, norm_stats, device, aggregate="first"
            )
            pre_eq_features = extract_features_at_positions(
                pre_eq_toks, activation_store, saes, norm_stats, device, aggregate="mean"
            )
            result_features = extract_features_at_positions(
                result_toks, activation_store, saes, norm_stats, device, aggregate="mean"
            )

            C = ann["C"]
            log_abs_C = math.log(abs(C) + 1.0)  # +1 avoids log(0)

            records.append({
                "example_idx":       ex_idx,
                "ann_idx":           ann_idx,
                "ann_text":          ann["ann_text"],
                "expr_str":          ann["expr_str"],
                "C":                 C,
                "log_abs_C":         log_abs_C,
                "eq_tok_idx":        eq_toks[0],
                "pre_eq_tok_idxs":   pre_eq_toks,
                "result_tok_idxs":   result_toks,
                "outer_eq_tok_idx":  outer_eq_tok,
                "eq_features":       eq_features,       # (24, 12288) float16
                "pre_eq_features":   pre_eq_features,   # (24, 12288) float16
                "result_features":   result_features,   # (24, 12288) float16
                "token_ids":         input_ids[0].tolist(),
                "tokens":            tokens_str,
            })

        if (ex_idx + 1) % 25 == 0 or (ex_idx + 1) == len(raw_examples):
            print(f"  {ex_idx+1}/{len(raw_examples)} examples, "
                  f"{len(records)} annotations collected …")

    # Remove hooks
    for h in handles:
        h.remove()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"\nSaving {len(records)} annotation records to {output_path} …")
    torch.save(records, output_path)

    # Summary statistics
    C_vals = [r["C"] for r in records]
    ops = {}
    for r in records:
        expr = r["expr_str"]
        for op in ["*", "/", "+", "-"]:
            if op in expr:
                ops[op] = ops.get(op, 0) + 1
                break

    print(f"\n=== Collection Summary ===")
    print(f"  Total annotations : {len(records)}")
    print(f"  Skipped           : {n_skipped}")
    print(f"  C range           : [{min(C_vals):.2f}, {max(C_vals):.2f}]")
    print(f"  Op distribution   : {ops}")
    feature_bytes = sum(
        r["eq_features"].numel() * 2
        + r["pre_eq_features"].numel() * 2
        + r["result_features"].numel() * 2
        for r in records
    )
    print(f"  Dataset size      : {feature_bytes / 1e9:.2f} GB (float16 tensors)")
    print(f"  Saved to          : {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect SAE features for GSM8K arithmetic annotations."
    )
    parser.add_argument(
        "--saes-dir",
        default="phase2_results/saes_gpt2_12x/saes",
        help="Directory with gpt2-medium_layer{N}_sae.pt files.",
    )
    parser.add_argument(
        "--activations-dir",
        default="phase2_results/activations",
        help="Directory with gpt2-medium_layer{N}_activations.pt files.",
    )
    parser.add_argument(
        "--gsm8k-path",
        default="datasets/raw/gsm8k/train.jsonl",
        help="Path to GSM8K train.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="phase4_results/collection",
        help="Directory to save the dataset.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of GSM8K examples to process.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device (e.g. cuda:0, cpu).",
    )

    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()

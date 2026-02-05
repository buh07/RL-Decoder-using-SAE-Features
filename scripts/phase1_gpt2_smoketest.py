"""Quick sanity check for gpt2 hidden-state extraction."""
import argparse
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(args):
    model_dir = Path(args.model_dir).expanduser()
    print(f"[INFO] Loading model from {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval().to(args.device)

    sample = "The quick brown fox jumps over the lazy dog. " * 20
    batch = tok(sample, return_tensors="pt").to(args.device)

    start = time.time()
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
    duration = time.time() - start

    hidden = out.hidden_states[args.layer_idx]
    tokens = batch["input_ids"].shape[-1]
    print(f"[INFO] Tokens processed: {tokens}")
    print(f"[INFO] Layer {args.layer_idx} hidden shape: {tuple(hidden.shape)}")
    print(f"[INFO] Runtime: {duration:.3f}s ({tokens / duration:.1f} tok/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/scratch2/f004ndc/LLM Second-Order Effects/models/models--gpt2")
    parser.add_argument("--layer-idx", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(parser.parse_args())

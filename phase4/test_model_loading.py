#!/usr/bin/env python3
"""Test model loading for Phase 4."""

import argparse
import logging
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


def load_model(model_name: str, device: torch.device):
    """Load model and tokenizer from HuggingFace."""
    logger.info(f"Loading {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=str(device),
            torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument("--models", type=str, nargs="+", default=["gpt2-medium"],
                       help="Models to test")
    parser.add_argument("--gpu-id", type=int, default=0)
    
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 60)
    logger.info("MODEL LOADING TEST")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info("")
    
    for model_name in args.models:
        logger.info(f"Testing {model_name}...")
        try:
            model, tokenizer = load_model(model_name, device)
            logger.info(f"  ✓ Loaded successfully")
            logger.info(f"    Hidden dim: {model.config.hidden_size}")
            logger.info(f"    Layers: {model.config.num_hidden_layers}")
            logger.info(f"    Vocab size: {len(tokenizer)}")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
    
    logger.info("")
    logger.info("Model loading test complete!")

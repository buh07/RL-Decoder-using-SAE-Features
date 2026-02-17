#!/usr/bin/env python3
"""
Phase 3 Alignment: Extract reasoning steps and map to token positions.
Supports regex, similarity-based, and hybrid step extraction.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class ReasoningStep:
    """A single reasoning step with token-level span information."""

    step_id: int
    """Step number (0-indexed)."""

    text: str
    """Text of the step."""

    start_token: int
    """Start token index in the full tokenized sequence."""

    end_token: int
    """End token index (exclusive)."""

    step_type: str
    """Category: 'equation', 'comparison', 'reasoning', 'other'."""

    confidence: float = 1.0
    """Alignment confidence [0, 1]."""

    @property
    def num_tokens(self) -> int:
        return self.end_token - self.start_token


@dataclass
class ReasoningExample:
    """Full reasoning example with aligned steps."""

    example_id: str
    """Unique example identifier."""

    full_text: str
    """Full input + reasoning + answer."""

    tokens: list[int]
    """Tokenized full text."""

    steps: list[ReasoningStep]
    """Extracted and aligned reasoning steps."""

    answer: str
    """Ground-truth answer."""

    question: str
    """Initial question."""


class ReasoningStepExtractor:
    """Extract reasoning steps from text using configurable methods."""

    def __init__(
        self,
        method: str = "regex",
        regex_patterns: Optional[dict[str, str]] = None,
        similarity_threshold: float = 0.5,
        min_step_length: int = 5,
        max_step_length: int = 256,
    ):
        self.method = method
        self.regex_patterns = regex_patterns or {}
        self.similarity_threshold = similarity_threshold
        self.min_step_length = min_step_length
        self.max_step_length = max_step_length

    def extract_regex_steps(self, text: str) -> list[tuple[str, str]]:
        """Extract steps using regex patterns. Returns (step_text, step_type)."""
        lines = text.split("\n")
        steps = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Classify by pattern
            step_type = "other"
            if self.regex_patterns.get("step_header") and re.search(
                self.regex_patterns["step_header"], line
            ):
                step_type = "step_header"
            elif self.regex_patterns.get("equation") and re.search(
                self.regex_patterns["equation"], line
            ):
                step_type = "equation"
            elif self.regex_patterns.get("comparison") and re.search(
                self.regex_patterns["comparison"], line
            ):
                step_type = "comparison"
            elif self.regex_patterns.get("reasoning") and re.search(
                self.regex_patterns["reasoning"], line
            ):
                step_type = "reasoning"

            if len(line) >= self.min_step_length and len(line) <= self.max_step_length:
                steps.append((line, step_type))

        return steps

    def extract_similarity_steps(self, text: str, tokenizer) -> list[tuple[str, str]]:
        """
        Extract steps by detecting semantic boundaries via embeddings.
        Simplified version: split by high-similarity boundaries.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        steps = []

        for sent in sentences:
            sent = sent.strip()
            if self.min_step_length <= len(sent) <= self.max_step_length:
                steps.append((sent, "sentence"))

        return steps

    def extract_steps(self, text: str, tokenizer=None) -> list[tuple[str, str]]:
        """Extract reasoning steps using selected method."""
        if self.method == "regex":
            return self.extract_regex_steps(text)
        elif self.method == "similarity":
            return self.extract_similarity_steps(text, tokenizer)
        elif self.method == "hybrid":
            regex_steps = self.extract_regex_steps(text)
            if not regex_steps:
                return self.extract_similarity_steps(text, tokenizer)
            return regex_steps
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")


class TokenAligner:
    """Align extracted steps to token indices in tokenized sequences."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def align_steps_to_tokens(
        self,
        full_text: str,
        tokens: list[int],
        steps: list[tuple[str, str]],
    ) -> list[ReasoningStep]:
        """
        Align step texts to token indices via character-to-token mapping.
        Returns aligned steps with token spans.
        """
        # Build character offset to token index mapping
        char_to_token = {}
        current_char = 0
        for token_idx, token_id in enumerate(tokens):
            token_text = self.tokenizer.decode([token_id])
            for offset in range(len(token_text)):
                char_to_token[current_char + offset] = token_idx
            current_char += len(token_text)

        aligned_steps = []
        step_id = 0

        for step_text, step_type in steps:
            # Find character position of step in full text
            char_pos = full_text.find(step_text)
            if char_pos == -1:
                continue

            # Map to token range
            start_token = char_to_token.get(char_pos)
            end_token = char_to_token.get(char_pos + len(step_text) - 1)

            if start_token is not None and end_token is not None:
                aligned_steps.append(
                    ReasoningStep(
                        step_id=step_id,
                        text=step_text,
                        start_token=start_token,
                        end_token=end_token + 1,  # Exclusive
                        step_type=step_type,
                        confidence=1.0,
                    )
                )
                step_id += 1

        return aligned_steps


class GSM8KAligner:
    """Full pipeline for extracting and aligning GSM8K reasoning steps."""

    def __init__(
        self,
        tokenizer,
        extraction_method: str = "regex",
        regex_patterns: Optional[dict[str, str]] = None,
        min_step_length: int = 5,
        max_step_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.extractor = ReasoningStepExtractor(
            method=extraction_method,
            regex_patterns=regex_patterns,
            min_step_length=min_step_length,
            max_step_length=max_step_length,
        )
        self.aligner = TokenAligner(tokenizer)

    def process_example(
        self, question: str, answer: str, reasoning_text: Optional[str] = None
    ) -> ReasoningExample:
        """
        Process a single GSM8K example.
        reasoning_text should contain the chain-of-thought explanation.
        """
        if reasoning_text is None:
            reasoning_text = ""

        # Combine question + reasoning for reasoning step extraction
        full_text = f"{question}\n{reasoning_text}\n{answer}"
        tokens = self.tokenizer.encode(full_text)

        # Extract reasoning steps
        steps_text = self.extractor.extract_steps(reasoning_text)

        # Align to tokens
        aligned_steps = self.aligner.align_steps_to_tokens(full_text, tokens, steps_text)

        # Create example
        example_id = f"gsm8k_{hash(full_text) % 1000000:06d}"

        return ReasoningExample(
            example_id=example_id,
            full_text=full_text,
            tokens=tokens,
            steps=aligned_steps,
            answer=answer,
            question=question,
        )

    def process_dataset(
        self, split: str = "train", subsample: Optional[int] = None
    ) -> list[ReasoningExample]:
        """
        Load and process GSM8K dataset.

        Args:
            split: 'train' or 'test'
            subsample: If set, only process first N examples

        Returns:
            List of ReasoningExample objects
        """
        print(f"[GSM8KAligner] Loading GSM8K {split} split...")
        dataset = load_dataset("gsm8k", "main", split=split)

        if subsample:
            dataset = dataset.select(range(min(subsample, len(dataset))))

        examples = []
        for idx, item in enumerate(dataset):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)}")

            # GSM8K format: question + answer (answer contains full working)
            question = item["question"]
            answer = item["answer"]

            # Try to extract reasoning from answer (usually formatted as steps)
            example = self.process_example(question, answer, reasoning_text=answer)
            examples.append(example)

        print(f"[GSM8KAligner] Processed {len(examples)} examples")
        return examples

    def save_aligned_dataset(self, examples: list[ReasoningExample], output_path: Path):
        """Save aligned examples to JSONL for reference and further processing."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for ex in examples:
                record = {
                    "example_id": ex.example_id,
                    "question": ex.question,
                    "answer": ex.answer,
                    "num_tokens": len(ex.tokens),
                    "num_steps": len(ex.steps),
                    "steps": [
                        {
                            "step_id": s.step_id,
                            "text": s.text,
                            "start_token": s.start_token,
                            "end_token": s.end_token,
                            "step_type": s.step_type,
                            "confidence": s.confidence,
                        }
                        for s in ex.steps
                    ],
                }
                f.write(json.dumps(record) + "\n")

        print(f"[GSM8KAligner] Saved {len(examples)} aligned examples to {output_path}")

    @staticmethod
    def load_aligned_dataset(path: Path) -> list[ReasoningExample]:
        """Load aligned examples from JSONL."""
        examples = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                steps = [
                    ReasoningStep(
                        step_id=s["step_id"],
                        text=s["text"],
                        start_token=s["start_token"],
                        end_token=s["end_token"],
                        step_type=s["step_type"],
                        confidence=s.get("confidence", 1.0),
                    )
                    for s in data["steps"]
                ]
                ex = ReasoningExample(
                    example_id=data["example_id"],
                    full_text=data.get("full_text", ""),
                    tokens=[],
                    steps=steps,
                    answer=data["answer"],
                    question=data["question"],
                )
                examples.append(ex)
        return examples


if __name__ == "__main__":
    # Quick test
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    aligner = GSM8KAligner(
        tokenizer,
        extraction_method="regex",
        regex_patterns={
            "step_header": r"(?:Step|step)\s+\d+[:.]?",
            "equation": r"(\d+[\s+\-*/]*)+\s*=",
        },
    )

    # Process first 10 examples
    examples = aligner.process_dataset(split="train", subsample=10)
    print(f"\nProcessed {len(examples)} examples")
    if examples:
        ex = examples[0]
        print(f"Example ID: {ex.example_id}")
        print(f"Question: {ex.question[:100]}")
        print(f"Tokens: {len(ex.tokens)}")
        print(f"Steps: {len(ex.steps)}")
        for step in ex.steps[:3]:
            print(f"  [{step.start_token}:{step.end_token}] {step.text[:50]}")

    # Save
    output_path = Path("/tmp/gsm8k_aligned_train.jsonl")
    aligner.save_aligned_dataset(examples, output_path)

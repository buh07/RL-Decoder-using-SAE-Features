#!/usr/bin/env python3
"""
Phase 3 Task Evaluation: Measure task accuracy for GSM8K with feature ablations.
"""
from __future__ import annotations

import re
from typing import Optional

import torch

from phase3_alignment import ReasoningExample


class GSM8KTaskEvaluator:
    """Evaluate task performance on GSM8K examples with feature ablations."""

    @staticmethod
    def extract_final_answer(text: str) -> Optional[str]:
        """
        Extract the final numeric answer from GSM8K format.

        GSM8K answers are typically formatted as:
        "... calculation ... = 42" with final answer on last line after "####"

        Args:
            text: Answer text

        Returns:
            Extracted number as string, or None
        """
        # Try to extract from "####" marker
        if "####" in text:
            answer_part = text.split("####")[-1].strip()
            # Extract first number
            match = re.search(r"-?\d+\.?\d*", answer_part)
            if match:
                return match.group(0)

        # Fallback: try to find last number in text
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return numbers[-1]

        return None

    @staticmethod
    def is_correct_answer(
        predicted_answer: str,
        ground_truth_answer: str,
        metric: str = "exact_match",
        numeric_tolerance: float = 1e-3,
    ) -> bool:
        """
        Check if predicted answer matches ground truth.

        Args:
            predicted_answer: Extracted predicted answer
            ground_truth_answer: Extracted ground truth answer
            metric: 'exact_match', 'contains_answer', or 'numeric_close'
            numeric_tolerance: Tolerance for numeric comparison

        Returns:
            True if answer is correct
        """
        if metric == "exact_match":
            return predicted_answer.strip() == ground_truth_answer.strip()

        elif metric == "contains_answer":
            return ground_truth_answer.strip() in predicted_answer or predicted_answer in ground_truth_answer

        elif metric == "numeric_close":
            try:
                pred_num = float(predicted_answer)
                truth_num = float(ground_truth_answer)
                return abs(pred_num - truth_num) < numeric_tolerance
            except (ValueError, TypeError):
                return False

        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def evaluate_example(
        example: ReasoningExample,
        metric: str = "exact_match",
        numeric_tolerance: float = 1e-3,
    ) -> dict:
        """
        Evaluate a single example's answer (without ablation).

        Args:
            example: ReasoningExample with question, answer
            metric: Evaluation metric
            numeric_tolerance: Tolerance for numeric comparison

        Returns:
            dict with predicted_answer, ground_truth_answer, is_correct
        """
        # Extract ground truth
        ground_truth = GSM8KTaskEvaluator.extract_final_answer(example.answer)

        return {
            "example_id": example.example_id,
            "ground_truth_answer": ground_truth,
            "is_correct": ground_truth is not None,  # Placeholder
            "metric": metric,
        }

    @staticmethod
    def compute_task_loss(
        example: ReasoningExample,
        predicted_answer: str,
        metric: str = "exact_match",
        numeric_tolerance: float = 1e-3,
    ) -> float:
        """
        Compute loss for task evaluation (higher = worse).

        For use in causal evaluation: task_fn for ablations.

        Args:
            example: ReasoningExample
            predicted_answer: Model's predicted answer
            metric: Evaluation metric
            numeric_tolerance: Tolerance for numeric comparison

        Returns:
            Loss value (0 = correct, 1 = incorrect)
        """
        ground_truth = GSM8KTaskEvaluator.extract_final_answer(example.answer)

        if ground_truth is None:
            return 1.0  # Unknown answer = loss 1

        is_correct = GSM8KTaskEvaluator.is_correct_answer(
            predicted_answer, ground_truth, metric=metric, numeric_tolerance=numeric_tolerance
        )

        return 0.0 if is_correct else 1.0


class ReasoningExplanationEvaluator:
    """Evaluate reasoning quality (step coverage, coherence, etc.)."""

    @staticmethod
    def count_reasoning_steps(example: ReasoningExample) -> int:
        """Count number of extracted reasoning steps."""
        return len(example.steps)

    @staticmethod
    def compute_step_coverage(example: ReasoningExample) -> float:
        """
        Compute what fraction of tokens are covered by reasoning steps.

        Returns:
            Fraction [0, 1]
        """
        if len(example.tokens) == 0:
            return 0.0

        covered_tokens = set()
        for step in example.steps:
            for tok_idx in range(step.start_token, step.end_token):
                covered_tokens.add(tok_idx)

        return len(covered_tokens) / len(example.tokens)

    @staticmethod
    def compute_step_overlap(example: ReasoningExample) -> float:
        """
        Compute average overlap between consecutive steps.

        High overlap = poorly separated steps. Low overlap = clean boundaries.

        Returns:
            Fraction [0, 1]
        """
        if len(example.steps) < 2:
            return 0.0

        overlaps = []
        for i in range(len(example.steps) - 1):
            step_i = example.steps[i]
            step_j = example.steps[i + 1]

            # Overlap = gap between steps (0 if adjacent, >0 if overlap)
            gap = step_j.start_token - step_i.end_token
            overlap = max(0, -gap)  # Convert negative gap to overlap

            overlaps.append(overlap)

        if overlaps:
            return sum(overlaps) / sum(
                len(s) for s in example.steps
            )  # Normalize by total step tokens

        return 0.0

    @staticmethod
    def evaluate_reasoning_quality(example: ReasoningExample) -> dict:
        """
        Compute reasoning quality metrics for an example.

        Returns:
            dict with step_count, step_coverage, step_overlap, etc.
        """
        return {
            "example_id": example.example_id,
            "step_count": ReasoningExplanationEvaluator.count_reasoning_steps(example),
            "step_coverage": ReasoningExplanationEvaluator.compute_step_coverage(example),
            "step_overlap": ReasoningExplanationEvaluator.compute_step_overlap(example),
        }

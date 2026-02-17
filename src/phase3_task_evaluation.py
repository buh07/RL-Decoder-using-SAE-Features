#!/usr/bin/env python3
"""
Phase 3 Task Evaluation: Measure task performance (GSM8K math accuracy) with and without ablations.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class TaskEvaluationResult:
    """Result of task evaluation on a single example."""

    example_id: str
    """Example identifier."""

    baseline_correct: bool
    """Whether model solved task without ablation."""

    expected_answer: str
    """Expected answer."""

    baseline_answer: Optional[str] = None
    """Model's answer without ablation."""

    ablation_results: dict[int, bool] = None
    """Mapping feature_id -> whether model solved task with that feature ablated."""

    causality_score: float = 0.0
    """Fraction of features whose ablation caused failure."""


class GSM8KTaskEvaluator:
    """Evaluate task performance on GSM8K math problems."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @staticmethod
    def extract_final_answer(text: str) -> Optional[str]:
        """
        Extract the final numerical answer from solution text.
        
        Looks for patterns like:
        - "#### 42" (standard GSM8K format)
        - "Answer: 42"
        - "The answer is 42"
        """
        # Try standard GSM8K format (#### answer)
        match = re.search(r"####\s*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return match.group(1).strip()

        # Try "Answer:" format
        match = re.search(r"(?:Answer|answer)[\s:]+([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return match.group(1).strip()

        # Try "The answer is" format
        match = re.search(r"(?:The answer is|the answer is)[\s:]*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return match.group(1).strip()

        return None

    @staticmethod
    def answers_match(answer1: str, answer2: str, numeric_tolerance: float = 1e-3) -> bool:
        """
        Check if two answers match (numerically or exactly).
        
        Args:
            answer1: First answer
            answer2: Second answer
            numeric_tolerance: Tolerance for numeric matching
            
        Returns:
            True if answers match
        """
        # Clean whitespace
        answer1 = answer1.strip()
        answer2 = answer2.strip()

        # Exact match
        if answer1 == answer2:
            return True

        # Numeric match
        try:
            val1 = float(answer1)
            val2 = float(answer2)
            return abs(val1 - val2) < numeric_tolerance
        except ValueError:
            pass

        # Contains match
        if answer1 in answer2 or answer2 in answer1:
            return True

        return False

    def evaluate_example(
        self,
        example_id: str,
        expected_answer: str,
        baseline_output: str,
        numeric_tolerance: float = 1e-3,
    ) -> TaskEvaluationResult:
        """
        Evaluate task performance on a single example.
        
        Args:
            example_id: Example identifier
            expected_answer: Expected answer (from GSM8K)
            baseline_output: Model's output without ablation
            numeric_tolerance: Tolerance for numeric matching
            
        Returns:
            TaskEvaluationResult
        """
        # Extract model's answer
        model_answer = self.extract_final_answer(baseline_output)

        # Check if correct
        baseline_correct = False
        if model_answer:
            baseline_correct = self.answers_match(model_answer, expected_answer, numeric_tolerance)

        result = TaskEvaluationResult(
            example_id=example_id,
            expected_answer=expected_answer,
            baseline_answer=model_answer,
            baseline_correct=baseline_correct,
        )

        return result

    def compute_causality_from_ablations(
        self,
        baseline_correct: bool,
        ablation_correctness: dict[int, bool],
    ) -> float:
        """
        Compute causality score: fraction of features whose ablation caused failure.
        
        Args:
            baseline_correct: Whether task was solved without ablation
            ablation_correctness: Dict mapping feature_id -> correctness with ablation
            
        Returns:
            Causality score [0, 1]
        """
        if not baseline_correct or not ablation_correctness:
            return 0.0

        # Count features whose ablation broke the solution
        causes_failure = sum(1 for correct in ablation_correctness.values() if not correct)

        causality_score = causes_failure / len(ablation_correctness)

        return causality_score

    def evaluate_batch(
        self,
        example_ids: list[str],
        expected_answers: list[str],
        baseline_outputs: list[str],
        numeric_tolerance: float = 1e-3,
    ) -> list[TaskEvaluationResult]:
        """Evaluate a batch of examples."""
        results = []

        for ex_id, exp_ans, baseline_out in zip(example_ids, expected_answers, baseline_outputs):
            result = self.evaluate_example(ex_id, exp_ans, baseline_out, numeric_tolerance)
            results.append(result)

        return results

    @staticmethod
    def compute_accuracy_statistics(results: list[TaskEvaluationResult]) -> dict:
        """Compute summary statistics over batch of results."""
        correct = sum(1 for r in results if r.baseline_correct)
        total = len(results)

        stats = {
            "num_examples": total,
            "num_correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "num_incorrect": total - correct,
        }

        return stats


class CausalTaskEvaluator:
    """Evaluate how feature ablations impact task performance (causal link)."""

    def __init__(self, task_evaluator: GSM8KTaskEvaluator, device: str = "cuda:0"):
        self.task_evaluator = task_evaluator
        self.device = device

    def evaluate_feature_causal_importance(
        self,
        example_output_baseline: str,
        expected_answer: str,
        ablation_outputs: dict[int, str],
        numeric_tolerance: float = 1e-3,
    ) -> dict[int, float]:
        """
        Evaluate causal importance of features based on task performance.
        
        Args:
            example_output_baseline: Model output without ablation
            expected_answer: Expected answer
            ablation_outputs: Dict mapping feature_id -> model output with ablation
            numeric_tolerance: Tolerance for numeric answers
            
        Returns:
            Dict mapping feature_id -> causal_importance [0, 1]
            (1 = ablating feature breaks the solution)
        """
        # Evaluate baseline
        baseline_ans = self.task_evaluator.extract_final_answer(example_output_baseline)
        baseline_correct = self.task_evaluator.answers_match(
            baseline_ans or "", expected_answer, numeric_tolerance
        )

        if not baseline_correct:
            # If baseline is already wrong, no causal analysis possible
            return {}

        causality_dict = {}

        for feature_id, ablation_output in ablation_outputs.items():
            ablation_ans = self.task_evaluator.extract_final_answer(ablation_output)
            ablation_correct = self.task_evaluator.answers_match(
                ablation_ans or "", expected_answer, numeric_tolerance
            )

            # Importance: 1 if ablation breaks solution, 0 otherwise
            causality_dict[feature_id] = 0.0 if ablation_correct else 1.0

        return causality_dict

    def evaluate_batch_causal(
        self,
        baseline_outputs: dict[str, str],
        expected_answers: dict[str, str],
        ablation_outputs_per_feature: dict[int, dict[str, str]],
        numeric_tolerance: float = 1e-3,
    ) -> dict[int, dict[str, float]]:
        """
        Compute causal importance for all features and examples.
        
        Args:
            baseline_outputs: Dict[example_id -> output]
            expected_answers: Dict[example_id -> answer]
            ablation_outputs_per_feature: Dict[feature_id -> Dict[example_id -> output]]
            
        Returns:
            Dict[feature_id -> Dict[example_id -> causality_score]]
        """
        results = {}

        for feature_id, feature_ablation_outputs in ablation_outputs_per_feature.items():
            results[feature_id] = {}

            for example_id, ablation_output in feature_ablation_outputs.items():
                baseline_out = baseline_outputs.get(example_id, "")
                expected_ans = expected_answers.get(example_id, "")

                if not baseline_out or not expected_ans:
                    continue

                causality_scores = self.evaluate_feature_causal_importance(
                    baseline_out, expected_ans, {feature_id: ablation_output}, numeric_tolerance
                )

                results[feature_id][example_id] = causality_scores.get(feature_id, 0.0)

        return results

    @staticmethod
    def compute_feature_importance_from_causal(
        causal_scores: dict[int, dict[str, float]]
    ) -> dict[int, float]:
        """
        Aggregate causal scores to feature-level importance.
        
        Args:
            causal_scores: Dict[feature_id -> Dict[example_id -> score]]
            
        Returns:
            Dict[feature_id -> mean_importance]
        """
        feature_importance = {}

        for feature_id, example_scores in causal_scores.items():
            if example_scores:
                feature_importance[feature_id] = np.mean(list(example_scores.values()))
            else:
                feature_importance[feature_id] = 0.0

        return feature_importance

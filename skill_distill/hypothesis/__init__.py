"""Minimal Hypothesis DSL for teacher knowledge extraction."""

from skill_distill.hypothesis.dsl import (
    evaluate_behavior,
    evaluate_condition,
    validate_hypothesis,
)
from skill_distill.hypothesis.evaluate import evaluate_hypothesis_on_frame

__all__ = [
    "evaluate_behavior",
    "evaluate_condition",
    "evaluate_hypothesis_on_frame",
    "validate_hypothesis",
]

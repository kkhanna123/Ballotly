"""Aggregation, scoring, and recommendation generation."""

from .aggregator import Analyzer, OverallAssessment
from .scoring import (
    Recommendation,
    RecommendationEngine,
    score_to_grade,
    score_to_label,
)

__all__ = [
    "Analyzer",
    "OverallAssessment",
    "Recommendation",
    "RecommendationEngine",
    "score_to_grade",
    "score_to_label",
]

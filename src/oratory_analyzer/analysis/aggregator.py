"""Aggregation: turn per-metric results into one overall assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..domain.frame import VideoMetadata
from ..metrics.base import MetricResult
from .scoring import (
    Recommendation,
    RecommendationEngine,
    score_to_grade,
    score_to_label,
)


@dataclass
class OverallAssessment:
    """The complete analytical result for one analyzed clip."""

    overall_score: float
    grade: str
    label: str
    metrics: Dict[str, MetricResult]
    recommendations: List[Recommendation]
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    unavailable: Dict[str, MetricResult] = field(default_factory=dict)
    frames_analyzed: int = 0
    frames_with_speaker: int = 0
    duration_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def tracking_coverage(self) -> float:
        if self.frames_analyzed == 0:
            return 0.0
        return self.frames_with_speaker / self.frames_analyzed

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall_score": self.overall_score,
            "grade": self.grade,
            "label": self.label,
            "frames_analyzed": self.frames_analyzed,
            "frames_with_speaker": self.frames_with_speaker,
            "tracking_coverage": round(self.tracking_coverage, 3),
            "duration_seconds": round(self.duration_seconds, 2),
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "unavailable": sorted(self.unavailable),
            "notes": self.notes,
            "metrics": {
                name: {
                    "title": r.title or r.name,
                    "score": r.score,
                    "summary": r.summary,
                    "category": r.category,
                    "coverage": round(r.coverage, 3),
                    "weight": r.weight,
                    "stats": r.stats,
                }
                for name, r in self.metrics.items()
            },
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


class Analyzer:
    """Combines metric results into a weighted overall score and assessment.

    Each metric contributes proportionally to ``weight * coverage`` so that
    metrics evaluated on few usable frames don't dominate (or distort) the
    overall grade.
    """

    def __init__(
        self,
        *,
        strength_threshold: float = 80.0,
        weakness_threshold: float = 60.0,
        low_coverage_threshold: float = 0.5,
        min_assess_coverage: float = 0.1,
        recommender: Optional[RecommendationEngine] = None,
    ) -> None:
        self.strength_threshold = strength_threshold
        self.weakness_threshold = weakness_threshold
        self.low_coverage_threshold = low_coverage_threshold
        # Below this coverage a metric is reported as "not assessed" rather than
        # scored — we won't grade or coach on data we couldn't reliably measure.
        self.min_assess_coverage = min_assess_coverage
        self.recommender = recommender or RecommendationEngine()

    def analyze(
        self,
        results: Dict[str, MetricResult],
        *,
        frames_analyzed: int = 0,
        frames_with_speaker: int = 0,
        video_meta: Optional[VideoMetadata] = None,
    ) -> OverallAssessment:
        if not results:
            raise ValueError("Cannot analyze: no metric results provided")

        # Split into metrics we can assess vs. those with too little visibility.
        assessed: Dict[str, MetricResult] = {}
        unavailable: Dict[str, MetricResult] = {}
        for name, r in results.items():
            if r.coverage >= self.min_assess_coverage:
                assessed[name] = r
            else:
                unavailable[name] = r

        if not assessed:
            raise ValueError(
                "No metric had enough visible frames to assess "
                "(was the speaker detected?)."
            )

        weighted_sum = 0.0
        weight_total = 0.0
        strengths: List[str] = []
        weaknesses: List[str] = []
        notes: List[str] = []

        for name, r in assessed.items():
            effective_weight = r.weight * r.coverage
            weighted_sum += r.score * effective_weight
            weight_total += effective_weight

            if r.score >= self.strength_threshold:
                strengths.append(name)
            elif r.score < self.weakness_threshold:
                weaknesses.append(name)

            if r.coverage < self.low_coverage_threshold:
                notes.append(
                    f"'{name}' assessed on only "
                    f"{round(r.coverage * 100)}% of frames — treat as low confidence."
                )

        for name, r in unavailable.items():
            notes.append(
                f"'{r.title or name}' could not be assessed — the required "
                "landmarks were rarely visible (e.g. hands/torso out of frame)."
            )

        overall = round(weighted_sum / weight_total, 1) if weight_total > 0 else 0.0
        recommendations = self.recommender.generate(assessed)

        return OverallAssessment(
            overall_score=overall,
            grade=score_to_grade(overall),
            label=score_to_label(overall),
            metrics=assessed,
            recommendations=recommendations,
            strengths=strengths,
            weaknesses=weaknesses,
            unavailable=unavailable,
            frames_analyzed=frames_analyzed,
            frames_with_speaker=frames_with_speaker,
            duration_seconds=video_meta.duration_seconds if video_meta else 0.0,
            notes=notes,
        )

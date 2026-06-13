"""Score → grade mapping and recommendation generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..metrics.base import MetricResult

# (min_score_inclusive, letter, label) ordered high → low.
_GRADE_BANDS = [
    (90.0, "A", "Excellent"),
    (80.0, "B", "Strong"),
    (70.0, "C", "Competent"),
    (60.0, "D", "Developing"),
    (0.0, "F", "Needs work"),
]


def score_to_grade(score: float) -> str:
    for threshold, letter, _ in _GRADE_BANDS:
        if score >= threshold:
            return letter
    return "F"


def score_to_label(score: float) -> str:
    for threshold, _, label in _GRADE_BANDS:
        if score >= threshold:
            return label
    return "Needs work"


@dataclass
class Recommendation:
    """A prioritized, actionable coaching note tied to a metric."""

    metric_name: str
    title: str
    detail: str
    drill: str
    priority: float  # higher == more important to address

    def to_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric_name,
            "title": self.title,
            "detail": self.detail,
            "drill": self.drill,
            "priority": round(self.priority, 1),
        }


# Concrete coaching advice per metric, surfaced when the metric scores low.
_TIPS: Dict[str, Dict[str, str]] = {
    "eye_contact": {
        "title": "Hold the audience with your eyes",
        "detail": (
            "You looked away or down at notes too often. Judges weight perceived "
            "conviction heavily, and steady eye contact is the strongest signal of it."
        ),
        "drill": (
            "Reduce notes to a few keyword cue cards. Practise delivering each "
            "argument while looking at three fixed points in the room, holding each "
            "for a full sentence before moving on."
        ),
    },
    "head_stability": {
        "title": "Steady your head",
        "detail": (
            "Frequent head swaying or bobbing reads as nervous energy and pulls "
            "focus from your words."
        ),
        "drill": (
            "Record a 60s rebuttal while consciously keeping your head still, "
            "moving it only to deliberately address a different part of the room."
        ),
    },
    "facial_expressivity": {
        "title": "Let your face carry the argument",
        "detail": (
            "Your expression stayed flat. A varied brow and mouth make points land "
            "and signal that you believe what you are saying."
        ),
        "drill": (
            "Mark your case for emphasis beats. Practise raising your brows on key "
            "claims and varying your mouth shape so emotion matches content."
        ),
    },
    "posture": {
        "title": "Anchor an upright, balanced stance",
        "detail": (
            "Leaning, uneven shoulders, or a dropped head undercut your authority "
            "before you speak."
        ),
        "drill": (
            "Stand with feet shoulder-width, weight even, shoulders level and back. "
            "Practise your intro from this 'home base' and return to it between points."
        ),
    },
    "gestures": {
        "title": "Make your hands work for you",
        "detail": (
            "Your hand movement fell outside the effective range — either too still "
            "or too busy. Purposeful gestures emphasise structure and contrast."
        ),
        "drill": (
            "Assign one clear gesture to each signpost (e.g. counting points on "
            "fingers, a flat hand to dismiss an opponent's claim). Rest hands "
            "between gestures rather than fidgeting."
        ),
    },
}


class RecommendationEngine:
    """Turns low-scoring metrics into prioritized coaching recommendations."""

    def __init__(self, focus_threshold: float = 70.0) -> None:
        self.focus_threshold = focus_threshold

    def generate(self, results: Dict[str, MetricResult]) -> List[Recommendation]:
        recs: List[Recommendation] = []
        for name, result in results.items():
            if result.score >= self.focus_threshold:
                continue
            tip = _TIPS.get(name)
            if tip is None:
                continue
            # Priority scales with how far below target and the metric's weight.
            priority = (self.focus_threshold - result.score) * result.weight
            recs.append(
                Recommendation(
                    metric_name=name,
                    title=tip["title"],
                    detail=tip["detail"],
                    drill=tip["drill"],
                    priority=priority,
                )
            )
        recs.sort(key=lambda r: r.priority, reverse=True)
        return recs

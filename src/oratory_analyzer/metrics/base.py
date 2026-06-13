"""Metric framework: result type, abstract base, and scoring helpers.

Every metric consumes the full sequence of :class:`FrameLandmarks` for the
identified speaker and returns a :class:`MetricResult` whose ``score`` is
normalized to ``0..100`` where **higher is always better** for oratory. This
uniform convention lets the analysis layer aggregate heterogeneous metrics into
a single overall grade without special-casing each one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..domain.geometry import clamp
from ..domain.landmarks import FrameLandmarks

# Which landmark group a metric needs.
FACE = "face"
POSE = "pose"


@dataclass
class MetricResult:
    """Outcome of evaluating one metric over a clip."""

    name: str
    score: float  # 0..100, higher == better
    summary: str
    category: str  # FACE or POSE
    title: str = ""  # human-facing title (set by the registry; falls back to name)
    stats: Dict[str, float] = field(default_factory=dict)
    series: List[float] = field(default_factory=list)  # per-frame signal for plots
    coverage: float = 1.0  # fraction of frames with usable data for this metric
    weight: float = 1.0  # relative weight in the overall score (set by the registry)

    def __post_init__(self) -> None:
        self.score = round(clamp(self.score, 0.0, 100.0), 1)
        self.coverage = clamp(self.coverage, 0.0, 1.0)
        if not self.title:
            self.title = self.name.replace("_", " ").title()


class Metric(ABC):
    """Abstract oratory metric.

    Subclasses declare which landmark group they require and implement
    :meth:`compute`. They should degrade gracefully when frames are missing the
    required landmarks, reporting ``coverage`` accordingly.
    """

    #: Stable identifier (snake_case) used as a dict key and in reports.
    name: str = "metric"
    #: Human-facing title.
    title: str = "Metric"
    #: FACE or POSE.
    requires: str = FACE
    #: Weight in the overall score (relative).
    weight: float = 1.0

    @abstractmethod
    def compute(self, frames: Sequence[FrameLandmarks]) -> MetricResult:
        """Evaluate the metric over the clip."""

    # --- shared helpers -------------------------------------------------

    @property
    def category(self) -> str:
        return self.requires

    def _empty_result(self, summary: str = "No usable frames for this metric.") -> MetricResult:
        return MetricResult(
            name=self.name,
            score=0.0,
            summary=summary,
            category=self.requires,
            coverage=0.0,
        )


# --- scoring helpers (pure functions) -----------------------------------


def fraction_score(fraction: float) -> float:
    """Map a fraction in ``[0, 1]`` to a ``0..100`` score."""
    return clamp(fraction, 0.0, 1.0) * 100.0


def band_score(
    value: float,
    ideal_low: float,
    ideal_high: float,
    hard_low: float,
    hard_high: float,
) -> float:
    """Score that is 100 inside ``[ideal_low, ideal_high]`` and falls to 0 at
    the hard bounds, linearly in between.

    Useful for "healthy range" metrics (gestures, expressivity) where both too
    little and too much are undesirable.
    """
    if hard_low > ideal_low or hard_high < ideal_high:
        raise ValueError("hard bounds must enclose the ideal band")
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        if value <= hard_low:
            return 0.0
        return (value - hard_low) / (ideal_low - hard_low) * 100.0
    # value > ideal_high
    if value >= hard_high:
        return 0.0
    return (hard_high - value) / (hard_high - ideal_high) * 100.0


def tolerance_score(value: float, ideal: float, tolerance: float) -> float:
    """Score 100 when ``value == ideal``, decaying linearly to 0 at
    ``ideal ± tolerance`` (one-sided distance)."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    deviation = abs(value - ideal)
    return clamp(1.0 - deviation / tolerance, 0.0, 1.0) * 100.0

"""Metric registry.

Collects the available metrics and runs the subset whose required landmark group
is present in the data. New metrics can be registered without touching the
pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..domain.landmarks import FrameLandmarks
from .base import FACE, HANDS, POSE, Metric, MetricResult
from .eye_contact import EyeContactMetric
from .facial_expressivity import FacialExpressivityMetric
from .gestures import GestureMetric
from .hand_gestures import HandGestureMetric
from .head_stability import HeadStabilityMetric
from .posture import PostureMetric


class MetricRegistry:
    """Holds metric instances and evaluates the applicable ones over a clip."""

    def __init__(self, metrics: Sequence[Metric]) -> None:
        self._metrics: List[Metric] = list(metrics)
        names = [m.name for m in self._metrics]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate metric names: {names}")

    @classmethod
    def default(cls) -> "MetricRegistry":
        return cls(
            [
                EyeContactMetric(),
                HeadStabilityMetric(),
                FacialExpressivityMetric(),
                PostureMetric(),
                GestureMetric(),
                HandGestureMetric(),
            ]
        )

    @property
    def metrics(self) -> List[Metric]:
        return list(self._metrics)

    def names(self) -> List[str]:
        return [m.name for m in self._metrics]

    def evaluate(self, frames: Sequence[FrameLandmarks]) -> Dict[str, MetricResult]:
        """Run every metric whose required landmark group appears in any frame.

        Metrics requiring data that is entirely absent (e.g. pose metrics when
        only face tracking ran) are skipped rather than reported as zeros.
        """
        has_face = any(f.has_face for f in frames)
        has_pose = any(f.has_pose for f in frames)
        has_hands = any(f.has_hands for f in frames)
        present = {FACE: has_face, POSE: has_pose, HANDS: has_hands}
        results: Dict[str, MetricResult] = {}
        for metric in self._metrics:
            if not present.get(metric.requires, True):
                continue
            result = metric.compute(frames)
            result.weight = metric.weight  # carry weight forward for aggregation
            result.title = metric.title
            results[metric.name] = result
        return results

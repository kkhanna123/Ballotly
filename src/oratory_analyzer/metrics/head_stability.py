"""Head-stability metric.

Measures how much the speaker's head drifts and jitters. Excessive swaying or
bobbing reads as nervousness and distracts from the argument; a small amount of
natural motion is fine. We score primarily on frame-to-frame jitter, normalized
by face size so the result is independent of how close the speaker is to the
camera.
"""

from __future__ import annotations

from typing import List, Sequence

from ..domain import FaceMesh
from ..domain.geometry import euclidean, mean, stddev
from ..domain.landmarks import FrameLandmarks
from .base import FACE, Metric, MetricResult, tolerance_score


class HeadStabilityMetric(Metric):
    name = "head_stability"
    title = "Head Stability"
    requires = FACE
    weight = 1.0

    def __init__(self, jitter_tolerance: float = 0.04) -> None:
        # jitter (mean per-frame nose displacement / face width) at which the
        # score reaches 0. ~0 jitter => 100.
        self.jitter_tolerance = jitter_tolerance

    def compute(self, frames: Sequence[FrameLandmarks]) -> MetricResult:
        total = len(frames)
        positions: List[tuple] = []  # (x, y) normalized by face width
        usable = 0
        for f in frames:
            if not f.has_face:
                positions.append(None)  # type: ignore[arg-type]
                continue
            face = f.face
            width = euclidean(face[FaceMesh.LEFT_CHEEK], face[FaceMesh.RIGHT_CHEEK])
            if width == 0:
                positions.append(None)  # type: ignore[arg-type]
                continue
            nose = face[FaceMesh.NOSE_TIP]
            positions.append((nose.x / width, nose.y / width))
            usable += 1

        if usable < 2:
            return self._empty_result("Not enough face frames to assess stability.")

        # Frame-to-frame displacement between consecutive *usable* frames.
        displacements: List[float] = []
        prev = None
        for p in positions:
            if p is None:
                prev = None
                continue
            if prev is not None:
                displacements.append(
                    ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5
                )
            prev = p

        jitter = mean(displacements) if displacements else 0.0
        score = tolerance_score(jitter, ideal=0.0, tolerance=self.jitter_tolerance)
        coverage = usable / total if total else 0.0

        if score >= 80:
            summary = "Composed, steady head carriage — projects confidence."
        elif score >= 50:
            summary = "Some head movement; tighten it up to look more assured."
        else:
            summary = "Noticeable head swaying/bobbing — a common sign of nerves."

        return MetricResult(
            name=self.name,
            score=score,
            summary=summary,
            category=FACE,
            stats={
                "mean_jitter": round(jitter, 4),
                "position_spread": round(
                    stddev([p[0] for p in positions if p is not None]), 4
                ),
            },
            # series of per-frame displacement, aligned by emitting 0 at starts.
            series=displacements,
            coverage=coverage,
        )

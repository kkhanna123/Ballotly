"""Eye-contact metric.

Estimates how consistently the speaker faces the audience/camera, combining head
yaw (nose offset relative to the face mid-line) with iris position inside the
eye (when iris landmarks are available). Strong, steady eye contact is one of
the highest-impact oratory behaviours for debate.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..domain import FaceMesh
from ..domain.geometry import euclidean, mean
from ..domain.landmarks import FaceLandmarks, FrameLandmarks
from .base import FACE, Metric, MetricResult, fraction_score


def _iris_horizontal_offset(face: FaceLandmarks) -> Optional[float]:
    """Average absolute horizontal iris offset within the eyes (0 == centered)."""
    if not face.has_irises:
        return None
    offsets = []
    for outer_i, inner_i, iris_i in (
        (FaceMesh.LEFT_EYE_OUTER, FaceMesh.LEFT_EYE_INNER, FaceMesh.LEFT_IRIS_CENTER),
        (FaceMesh.RIGHT_EYE_OUTER, FaceMesh.RIGHT_EYE_INNER, FaceMesh.RIGHT_IRIS_CENTER),
    ):
        outer, inner, iris = face[outer_i], face[inner_i], face[iris_i]
        width = euclidean(outer, inner)
        if width == 0:
            continue
        center_x = (outer.x + inner.x) / 2.0
        offsets.append(abs(iris.x - center_x) / width)
    if not offsets:
        return None
    return mean(offsets)


def _gaze_deviation(face: FaceLandmarks) -> Optional[float]:
    """Combined horizontal gaze deviation (0 == looking straight ahead)."""
    left = face[FaceMesh.LEFT_CHEEK]
    right = face[FaceMesh.RIGHT_CHEEK]
    width = euclidean(left, right)
    if width == 0:
        return None
    center_x = (left.x + right.x) / 2.0
    nose = face[FaceMesh.NOSE_TIP]
    head_yaw = abs(nose.x - center_x) / width

    iris = _iris_horizontal_offset(face)
    if iris is None:
        return head_yaw
    # Weight head orientation more than eye darting; both matter.
    return 0.6 * head_yaw + 0.4 * iris


class EyeContactMetric(Metric):
    name = "eye_contact"
    title = "Eye Contact"
    requires = FACE
    weight = 1.5  # high-impact behaviour

    def __init__(self, forward_threshold: float = 0.18) -> None:
        self.forward_threshold = forward_threshold

    def compute(self, frames: Sequence[FrameLandmarks]) -> MetricResult:
        total = len(frames)
        series = []
        deviations = []
        for f in frames:
            if not f.has_face:
                series.append(float("nan"))
                continue
            dev = _gaze_deviation(f.face)  # type: ignore[arg-type]
            if dev is None:
                series.append(float("nan"))
                continue
            series.append(dev)
            deviations.append(dev)

        if not deviations:
            return self._empty_result("No face detected to assess eye contact.")

        forward = [d for d in deviations if d <= self.forward_threshold]
        forward_fraction = len(forward) / len(deviations)
        score = fraction_score(forward_fraction)
        coverage = len(deviations) / total if total else 0.0

        pct = round(forward_fraction * 100)
        if forward_fraction >= 0.85:
            summary = f"Excellent — facing the audience {pct}% of the time."
        elif forward_fraction >= 0.65:
            summary = f"Solid eye contact ({pct}%), with room to reduce looking away."
        else:
            summary = (
                f"Eye contact only {pct}% of the time — likely glancing at notes "
                "or turning away too often."
            )

        return MetricResult(
            name=self.name,
            score=score,
            summary=summary,
            category=FACE,
            stats={
                "forward_fraction": round(forward_fraction, 3),
                "mean_gaze_deviation": round(mean(deviations), 3),
            },
            series=series,
            coverage=coverage,
        )

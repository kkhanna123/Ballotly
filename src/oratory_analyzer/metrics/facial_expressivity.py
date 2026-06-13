"""Facial-expressivity metric.

A flat, frozen face disengages an audience; an animated face that varies brow
and mouth signals conviction. We quantify expressivity as the temporal
variability (standard deviation) of normalized brow-raise and mouth-opening,
then score it against a healthy band (too flat is penalized; extreme
over-animation is mildly penalized).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..domain import FaceMesh
from ..domain.geometry import euclidean, mean, stddev
from ..domain.landmarks import FaceLandmarks, FrameLandmarks
from .base import FACE, Metric, MetricResult, band_score


def _brow_and_mouth(face: FaceLandmarks) -> Optional[Tuple[float, float]]:
    """Return (brow_raise, mouth_open), both normalized by face height."""
    face_height = euclidean(face[FaceMesh.FOREHEAD_TOP], face[FaceMesh.CHIN])
    if face_height == 0:
        return None
    brow_y = (face[FaceMesh.LEFT_BROW].y + face[FaceMesh.RIGHT_BROW].y) / 2.0
    eye_y = (face[FaceMesh.LEFT_EYE_TOP].y + face[FaceMesh.RIGHT_EYE_TOP].y) / 2.0
    brow_raise = (eye_y - brow_y) / face_height  # larger == brows lifted
    mouth_open = euclidean(face[FaceMesh.UPPER_LIP], face[FaceMesh.LOWER_LIP]) / face_height
    return brow_raise, mouth_open


class FacialExpressivityMetric(Metric):
    name = "facial_expressivity"
    title = "Facial Expressivity"
    requires = FACE
    weight = 0.8

    def __init__(
        self,
        *,
        ideal_low: float = 0.012,
        ideal_high: float = 0.12,
        hard_low: float = 0.0,
        hard_high: float = 0.30,
    ) -> None:
        self.ideal_low = ideal_low
        self.ideal_high = ideal_high
        self.hard_low = hard_low
        self.hard_high = hard_high

    def compute(self, frames: Sequence[FrameLandmarks]) -> MetricResult:
        total = len(frames)
        brows: List[float] = []
        mouths: List[float] = []
        mouth_series: List[float] = []
        for f in frames:
            if not f.has_face:
                mouth_series.append(float("nan"))
                continue
            bm = _brow_and_mouth(f.face)  # type: ignore[arg-type]
            if bm is None:
                mouth_series.append(float("nan"))
                continue
            brows.append(bm[0])
            mouths.append(bm[1])
            mouth_series.append(bm[1])

        if len(brows) < 2:
            return self._empty_result("Not enough face frames to assess expressivity.")

        expressivity = stddev(brows) + stddev(mouths)
        score = band_score(
            expressivity, self.ideal_low, self.ideal_high, self.hard_low, self.hard_high
        )
        coverage = len(brows) / total if total else 0.0

        if expressivity < self.ideal_low:
            summary = "Facial expression is flat — vary brow and mouth to convey conviction."
        elif expressivity > self.ideal_high:
            summary = "Highly animated face — ensure expressions stay purposeful."
        else:
            summary = "Engaging, varied facial expression."

        return MetricResult(
            name=self.name,
            score=score,
            summary=summary,
            category=FACE,
            stats={
                "brow_variability": round(stddev(brows), 4),
                "mouth_variability": round(stddev(mouths), 4),
                "expressivity": round(expressivity, 4),
            },
            series=mouth_series,
            coverage=coverage,
        )

"""Posture metric.

Combines three pose cues into one posture score:

* **Shoulder levelness** — uneven shoulders read as a lopsided stance.
* **Torso uprightness** — leaning the torso left/right looks unsteady.
* **Head-over-shoulders** — head dropping forward (slouch) undercuts authority.

Each sub-cue is scored 0..100 and averaged. Low-visibility landmarks are
ignored so the metric degrades gracefully on partial detections.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..domain import Pose
from ..domain.geometry import (
    horizontal_tilt_deg,
    mean,
    vertical_lean_deg,
)
from ..domain.landmarks import Point3D, PoseLandmarks
from ..domain.landmarks import FrameLandmarks
from .base import POSE, Metric, MetricResult, tolerance_score

_MIN_VIS = 0.5


def _visible(p: Point3D) -> bool:
    return p.visibility >= _MIN_VIS


def _midpoint(a: Point3D, b: Point3D) -> Point3D:
    return a.midpoint(b)


def _shoulder_tilt(pose: PoseLandmarks) -> Optional[float]:
    ls, rs = pose[Pose.LEFT_SHOULDER], pose[Pose.RIGHT_SHOULDER]
    if not (_visible(ls) and _visible(rs)):
        return None
    return abs(horizontal_tilt_deg(ls, rs))


def _torso_lean(pose: PoseLandmarks) -> Optional[float]:
    ls, rs = pose[Pose.LEFT_SHOULDER], pose[Pose.RIGHT_SHOULDER]
    lh, rh = pose[Pose.LEFT_HIP], pose[Pose.RIGHT_HIP]
    if not all(_visible(p) for p in (ls, rs, lh, rh)):
        return None
    return abs(vertical_lean_deg(_midpoint(ls, rs), _midpoint(lh, rh)))


def _head_drop(pose: PoseLandmarks) -> Optional[float]:
    """Forward head droop: how far the nose sits below the ear line, relative to
    shoulder width. ~0 when the head is upright."""
    le, re = pose[Pose.LEFT_EAR], pose[Pose.RIGHT_EAR]
    ls, rs = pose[Pose.LEFT_SHOULDER], pose[Pose.RIGHT_SHOULDER]
    nose = pose[Pose.NOSE]
    if not all(_visible(p) for p in (le, re, ls, rs, nose)):
        return None
    shoulder_width = abs(rs.x - ls.x)
    if shoulder_width == 0:
        return None
    ear_y = (le.y + re.y) / 2.0
    # Positive when nose drops below ears (head down).
    return max(0.0, (nose.y - ear_y)) / shoulder_width


class PostureMetric(Metric):
    name = "posture"
    title = "Posture & Stance"
    requires = POSE
    weight = 1.2

    def __init__(
        self,
        *,
        shoulder_tol_deg: float = 12.0,
        lean_tol_deg: float = 12.0,
        head_drop_tol: float = 0.6,
    ) -> None:
        self.shoulder_tol_deg = shoulder_tol_deg
        self.lean_tol_deg = lean_tol_deg
        self.head_drop_tol = head_drop_tol

    def compute(self, frames: Sequence[FrameLandmarks]) -> MetricResult:
        total = len(frames)
        tilts: List[float] = []
        leans: List[float] = []
        drops: List[float] = []
        per_frame_score: List[float] = []
        usable = 0

        for f in frames:
            if not f.has_pose:
                per_frame_score.append(float("nan"))
                continue
            pose = f.pose
            tilt = _shoulder_tilt(pose)
            lean = _torso_lean(pose)
            drop = _head_drop(pose)
            sub = []
            if tilt is not None:
                tilts.append(tilt)
                sub.append(tolerance_score(tilt, 0.0, self.shoulder_tol_deg))
            if lean is not None:
                leans.append(lean)
                sub.append(tolerance_score(lean, 0.0, self.lean_tol_deg))
            if drop is not None:
                drops.append(drop)
                sub.append(tolerance_score(drop, 0.0, self.head_drop_tol))
            if sub:
                per_frame_score.append(mean(sub))
                usable += 1
            else:
                per_frame_score.append(float("nan"))

        if usable == 0:
            return self._empty_result("No usable pose frames to assess posture.")

        valid_scores = [s for s in per_frame_score if s == s]  # filter NaN
        score = mean(valid_scores)
        coverage = usable / total if total else 0.0

        if score >= 80:
            summary = "Upright, balanced posture — commands the room."
        elif score >= 55:
            summary = "Generally sound posture with intermittent lean or slouch."
        else:
            summary = "Posture undermines presence — leaning, uneven, or slouched."

        return MetricResult(
            name=self.name,
            score=score,
            summary=summary,
            category=POSE,
            stats={
                "mean_shoulder_tilt_deg": round(mean(tilts), 2),
                "mean_torso_lean_deg": round(mean(leans), 2),
                "mean_head_drop": round(mean(drops), 3),
            },
            series=per_frame_score,
            coverage=coverage,
        )

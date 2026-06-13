"""Gesture / hand-movement metric.

Purposeful gestures reinforce points; frozen hands look stiff while constant
flailing looks frantic. We measure hand *speed* in shoulder-widths per second
(making it independent of camera distance and frame rate) and score it against a
healthy band. We also report how often the hands are active vs. resting.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..domain import Pose
from ..domain.geometry import mean
from ..domain.landmarks import Point3D, PoseLandmarks
from ..domain.landmarks import FrameLandmarks
from .base import POSE, Metric, MetricResult, band_score

_MIN_VIS = 0.4


def _shoulder_width(pose: PoseLandmarks) -> Optional[float]:
    ls, rs = pose[Pose.LEFT_SHOULDER], pose[Pose.RIGHT_SHOULDER]
    if ls.visibility < _MIN_VIS or rs.visibility < _MIN_VIS:
        return None
    w = ((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) ** 0.5
    return w or None


def _wrist(pose: PoseLandmarks, idx: int) -> Optional[Point3D]:
    p = pose[idx]
    return p if p.visibility >= _MIN_VIS else None


class GestureMetric(Metric):
    name = "gestures"
    title = "Hand Gestures"
    requires = POSE
    weight = 1.0

    def __init__(
        self,
        *,
        ideal_low: float = 0.05,
        ideal_high: float = 0.6,
        hard_low: float = 0.0,
        hard_high: float = 2.0,
    ) -> None:
        # bounds in shoulder-widths per second
        self.ideal_low = ideal_low
        self.ideal_high = ideal_high
        self.hard_low = hard_low
        self.hard_high = hard_high

    def compute(self, frames: Sequence[FrameLandmarks]) -> MetricResult:
        total = len(frames)
        speeds: List[float] = []
        active_count = 0
        considered = 0
        prev = None  # (lw, rw, width, timestamp)

        for f in frames:
            if not f.has_pose:
                prev = None
                continue
            pose = f.pose
            width = _shoulder_width(pose)
            lw = _wrist(pose, Pose.LEFT_WRIST)
            rw = _wrist(pose, Pose.RIGHT_WRIST)
            if width is None or (lw is None and rw is None):
                prev = None
                continue
            considered += 1

            # Active = at least one wrist raised above the hip line.
            hip_y = (pose[Pose.LEFT_HIP].y + pose[Pose.RIGHT_HIP].y) / 2.0
            raised = any(w is not None and w.y < hip_y for w in (lw, rw))
            if raised:
                active_count += 1

            if prev is not None:
                p_lw, p_rw, _, p_ts = prev
                dt = f.timestamp - p_ts
                if dt > 0:
                    disp = []
                    if lw is not None and p_lw is not None:
                        disp.append(lw.distance_to(p_lw))
                    if rw is not None and p_rw is not None:
                        disp.append(rw.distance_to(p_rw))
                    if disp:
                        speeds.append((mean(disp) / width) / dt)
            prev = (lw, rw, width, f.timestamp)

        if considered == 0:
            return self._empty_result("No usable pose frames to assess gestures.")

        gesture_speed = mean(speeds) if speeds else 0.0
        score = band_score(
            gesture_speed,
            self.ideal_low,
            self.ideal_high,
            self.hard_low,
            self.hard_high,
        )
        active_fraction = active_count / considered
        coverage = considered / total if total else 0.0

        if gesture_speed < self.ideal_low:
            summary = "Hands are very still — add deliberate gestures for emphasis."
        elif gesture_speed > self.ideal_high:
            summary = "Lots of hand motion — channel it into fewer, purposeful gestures."
        else:
            summary = "Well-paced, purposeful gesturing."

        return MetricResult(
            name=self.name,
            score=score,
            summary=summary,
            category=POSE,
            stats={
                "gesture_speed_sw_per_s": round(gesture_speed, 3),
                "active_fraction": round(active_fraction, 3),
            },
            series=speeds,
            coverage=coverage,
        )

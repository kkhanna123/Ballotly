"""Hand-gesture metric (detailed, from MediaPipe Hands).

Where the pose-based ``gestures`` metric only sees wrists, this metric uses the
full 21-point hand model to assess gesturing much more precisely:

* **Presence** — how often the hands are actually visible/up (coverage).
* **Motion** — hand movement speed in hand-widths per second (purposeful
  gesturing vs. frozen or frantic), scored against a healthy band.
* **Articulation** — variability of hand *openness* (fingers spreading/closing,
  pointing, counting) as a sign of expressive, varied gestures.

Camera-distance-independent: everything is normalized by hand size.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..domain import Hands
from ..domain.geometry import euclidean, mean, stddev
from ..domain.landmarks import FrameLandmarks, HandLandmarks, Point3D
from .base import HANDS, Metric, MetricResult, band_score


def _hand_scale(hand: HandLandmarks) -> Optional[float]:
    """Reference size = wrist→middle-finger-base distance (>0)."""
    scale = euclidean(hand[Hands.WRIST], hand[Hands.MIDDLE_MCP])
    return scale or None


def _hand_center(hand: HandLandmarks) -> Point3D:
    return hand[Hands.WRIST]


def _openness(hand: HandLandmarks, scale: float) -> float:
    """Mean fingertip distance from the wrist, normalized by hand scale.

    Larger when the hand is open/spread, smaller for a closed fist.
    """
    wrist = hand[Hands.WRIST]
    dists = [euclidean(wrist, hand[tip]) for tip in Hands.FINGERTIPS]
    return mean(dists) / scale


class HandGestureMetric(Metric):
    name = "hand_gestures"
    title = "Hand Gestures (detailed)"
    requires = HANDS
    weight = 1.0

    def __init__(
        self,
        *,
        ideal_low: float = 0.15,
        ideal_high: float = 2.5,
        hard_low: float = 0.0,
        hard_high: float = 8.0,
    ) -> None:
        # bounds in hand-widths per second
        self.ideal_low = ideal_low
        self.ideal_high = ideal_high
        self.hard_low = hard_low
        self.hard_high = hard_high

    def compute(self, frames: Sequence[FrameLandmarks]) -> MetricResult:
        total = len(frames)
        speeds: List[float] = []
        openness_series: List[float] = []
        considered = 0
        prev = None  # (center, scale, timestamp)

        for f in frames:
            if not f.has_hands:
                prev = None
                continue
            # Use the most prominent hand (largest scale) as the gesture proxy.
            scored = [(s, h) for h in f.hands if (s := _hand_scale(h)) is not None]
            if not scored:
                prev = None
                continue
            scale, hand = max(scored, key=lambda sh: sh[0])
            considered += 1
            openness_series.append(_openness(hand, scale))

            center = _hand_center(hand)
            if prev is not None:
                p_center, _p_scale, p_ts = prev
                dt = f.timestamp - p_ts
                if dt > 0:
                    speeds.append((euclidean(center, p_center) / scale) / dt)
            prev = (center, scale, f.timestamp)

        if considered == 0:
            return self._empty_result("No hands detected to assess hand gestures.")

        gesture_speed = mean(speeds) if speeds else 0.0
        score = band_score(
            gesture_speed, self.ideal_low, self.ideal_high, self.hard_low, self.hard_high
        )
        coverage = considered / total if total else 0.0
        articulation = stddev(openness_series)

        if gesture_speed < self.ideal_low:
            summary = "Hands are visible but barely moving — add deliberate gestures."
        elif gesture_speed > self.ideal_high:
            summary = "Very busy hands — consolidate into fewer, purposeful gestures."
        else:
            summary = "Purposeful, well-paced hand gestures."

        return MetricResult(
            name=self.name,
            score=score,
            summary=summary,
            category=HANDS,
            stats={
                "gesture_speed_hw_per_s": round(gesture_speed, 3),
                "mean_openness": round(mean(openness_series), 3),
                "openness_variability": round(articulation, 3),
                "hands_present_fraction": round(coverage, 3),
            },
            series=openness_series,
            coverage=coverage,
        )

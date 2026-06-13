"""Pure geometric helpers operating on :class:`Point3D` values.

Everything here is deterministic and dependency-free so it can be exhaustively
unit-tested. Angles are returned in degrees unless noted otherwise.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

from .landmarks import Point3D


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def euclidean(a: Point3D, b: Point3D, *, use_z: bool = False) -> float:
    """Distance between two points (thin wrapper for symmetry of the API)."""
    return a.distance_to(b, use_z=use_z)


def angle_between(a: Point3D, vertex: Point3D, c: Point3D) -> float:
    """Interior angle (degrees) at ``vertex`` formed by points ``a`` and ``c``.

    Returns 0.0 if either arm has zero length (degenerate).
    """
    v1x, v1y = a.x - vertex.x, a.y - vertex.y
    v2x, v2y = c.x - vertex.x, c.y - vertex.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    cos_theta = (v1x * v2x + v1y * v2y) / (n1 * n2)
    cos_theta = clamp(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def line_angle_deg(a: Point3D, b: Point3D) -> float:
    """Angle of the directed segment ``a -> b`` relative to the horizontal.

    Returns a value in ``(-180, 180]``. Because image ``y`` grows downward, a
    positive angle means ``b`` is *below* ``a`` on screen. For posture checks we
    usually take ``abs(...)`` of the tilt from horizontal.
    """
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def horizontal_tilt_deg(left: Point3D, right: Point3D) -> float:
    """Signed tilt (degrees) of the ``left``→``right`` segment from horizontal.

    Used for shoulder / eye level: 0 == perfectly level. Sign indicates which
    side is lower on screen (positive => right point lower).
    """
    angle = line_angle_deg(left, right)
    # Fold into [-90, 90] so a near-horizontal line reads as a small tilt.
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def vertical_lean_deg(top: Point3D, bottom: Point3D) -> float:
    """Signed lean (degrees) of a nominally-vertical segment from vertical.

    ``top`` is the upper point (e.g. shoulder midpoint), ``bottom`` the lower
    (e.g. hip midpoint). 0 == perfectly upright. Positive => leaning so the top
    is to the right of the bottom on screen.
    """
    dx = top.x - bottom.x
    dy = top.y - bottom.y  # negative because top is higher on screen
    # Angle from the vertical axis.
    return math.degrees(math.atan2(dx, -dy))


def centroid(points: Sequence[Point3D]) -> Point3D:
    """Arithmetic mean point of a non-empty sequence."""
    if not points:
        raise ValueError("centroid requires at least one point")
    n = float(len(points))
    return Point3D(
        x=sum(p.x for p in points) / n,
        y=sum(p.y for p in points) / n,
        z=sum(p.z for p in points) / n,
        visibility=min(p.visibility for p in points),
    )


def eye_aspect_ratio(
    outer: Point3D, inner: Point3D, top: Point3D, bottom: Point3D
) -> float:
    """Eye Aspect Ratio (EAR): vertical opening / horizontal width.

    ~0.3 for an open eye, dropping toward ~0.1 during a blink. Returns 0.0 if
    the eye width is degenerate.
    """
    width = euclidean(outer, inner)
    if width == 0.0:
        return 0.0
    height = euclidean(top, bottom)
    return height / width


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Centered simple moving average; output length equals input length.

    Edges shrink the window rather than padding. ``window`` must be >= 1.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(values)
    out: List[float] = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def mean(values: Iterable[float]) -> float:
    """Mean of an iterable; 0.0 for an empty input (safe default for metrics)."""
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def stddev(values: Sequence[float]) -> float:
    """Population standard deviation; 0.0 for fewer than two samples."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / n
    return math.sqrt(var)


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Linear-interpolated percentile (``pct`` in ``[0, 100]``).

    Returns ``None`` for an empty input.
    """
    if not values:
        return None
    if not 0.0 <= pct <= 100.0:
        raise ValueError("pct must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac

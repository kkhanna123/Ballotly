"""Core landmark value objects.

These dataclasses are the contract between the landmark-extraction layer (which
may be backed by MediaPipe, DeepLabCut, etc.) and every analytical consumer
(metrics, analysis, reporting). They are deliberately free of any third-party
dependency so the bulk of the system can be unit-tested with synthetic data.

Coordinate convention
---------------------
All coordinates are **normalized** to the frame: ``x`` and ``y`` are in
``[0, 1]`` with the origin at the top-left of the image (so ``y`` grows
downward, matching image conventions). ``z`` is a relative depth where smaller
(more negative) values are closer to the camera; it is roughly normalized to
the same scale as ``x``. ``visibility`` is a ``[0, 1]`` confidence that the
point is present and not occluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Point3D:
    """A single normalized landmark point.

    Frozen/immutable so landmarks can be safely shared and hashed.
    """

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "visibility"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or math.isnan(float(value)):
                raise ValueError(f"Point3D.{name} must be a real number, got {value!r}")

    def distance_to(self, other: "Point3D", *, use_z: bool = False) -> float:
        """Euclidean distance to ``other`` in normalized units."""
        dx = self.x - other.x
        dy = self.y - other.y
        if use_z:
            dz = self.z - other.z
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        return math.sqrt(dx * dx + dy * dy)

    def midpoint(self, other: "Point3D") -> "Point3D":
        """Midpoint between two points (visibility = min of the two)."""
        return Point3D(
            x=(self.x + other.x) / 2.0,
            y=(self.y + other.y) / 2.0,
            z=(self.z + other.z) / 2.0,
            visibility=min(self.visibility, other.visibility),
        )

    def as_tuple(self, *, include_z: bool = False) -> Tuple[float, ...]:
        return (self.x, self.y, self.z) if include_z else (self.x, self.y)


@dataclass(frozen=True)
class FaceLandmarks:
    """MediaPipe Face Mesh output: 468 (or 478 with irises) normalized points."""

    points: Tuple[Point3D, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("FaceLandmarks requires at least one point")

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> Point3D:
        return self.points[index]

    @property
    def has_irises(self) -> bool:
        """True when iris landmarks (indices 468-477) are present."""
        return len(self.points) >= 478


@dataclass(frozen=True)
class PoseLandmarks:
    """MediaPipe Pose output: 33 normalized body landmarks."""

    points: Tuple[Point3D, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("PoseLandmarks requires at least one point")

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> Point3D:
        return self.points[index]


@dataclass(frozen=True)
class HandLandmarks:
    """MediaPipe Hands output: 21 normalized landmarks for a single hand.

    ``handedness`` is ``"Left"``, ``"Right"``, or ``"Unknown"`` as reported by
    the detector (from the camera's point of view).
    """

    points: Tuple[Point3D, ...]
    handedness: str = "Unknown"

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("HandLandmarks requires at least one point")
        if self.handedness not in ("Left", "Right", "Unknown"):
            raise ValueError(f"invalid handedness: {self.handedness!r}")

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> Point3D:
        return self.points[index]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in normalized coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("BoundingBox max must be >= min on each axis")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-union with another box (0.0 when disjoint)."""
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)
        iw = max(0.0, ix_max - ix_min)
        ih = max(0.0, iy_max - iy_min)
        intersection = iw * ih
        union = self.area + other.area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    @classmethod
    def from_points(cls, points: Sequence[Point3D], *, padding: float = 0.0) -> "BoundingBox":
        """Tightest box containing ``points``, optionally padded (clamped to [0,1])."""
        if not points:
            raise ValueError("Cannot build a BoundingBox from zero points")
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return cls(
            x_min=max(0.0, min(xs) - padding),
            y_min=max(0.0, min(ys) - padding),
            x_max=min(1.0, max(xs) + padding),
            y_max=min(1.0, max(ys) + padding),
        )


@dataclass(frozen=True)
class FrameLandmarks:
    """All landmark data extracted from a single video frame for one speaker.

    Either ``face`` or ``pose`` (or both) may be ``None`` when a detector fails
    on that frame. ``index`` is the 0-based frame number and ``timestamp`` is
    seconds from the start of the video.
    """

    index: int
    timestamp: float
    face: Optional[FaceLandmarks] = None
    pose: Optional[PoseLandmarks] = None
    hands: Tuple[HandLandmarks, ...] = ()
    face_box: Optional[BoundingBox] = None
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def has_face(self) -> bool:
        return self.face is not None

    @property
    def has_pose(self) -> bool:
        return self.pose is not None

    @property
    def has_hands(self) -> bool:
        return len(self.hands) > 0

    @property
    def num_hands(self) -> int:
        return len(self.hands)

    @property
    def is_empty(self) -> bool:
        """True when no landmarks of any kind were detected (a dropped frame)."""
        return self.face is None and self.pose is None and not self.hands

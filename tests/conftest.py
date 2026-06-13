"""Shared synthetic-data builders for the test suite.

These factories let us construct realistic ``FrameLandmarks`` without any video
file or ML model, so the analytical core can be tested deterministically.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from oratory_analyzer.domain import FaceMesh, Hands, Pose
from oratory_analyzer.domain.landmarks import (
    BoundingBox,
    FaceLandmarks,
    FrameLandmarks,
    HandLandmarks,
    Point3D,
    PoseLandmarks,
)


def make_face(
    *,
    n: int = 478,
    yaw: float = 0.0,
    eye_open: float = 0.30,
    brow_raise: float = 0.0,
    mouth_open: float = 0.0,
    center_x: float = 0.5,
    center_y: float = 0.45,
) -> FaceLandmarks:
    """Build a synthetic face whose key landmarks reflect the requested pose.

    ``yaw`` shifts the nose horizontally to simulate looking off-camera.
    ``eye_open`` sets the vertical eye opening (EAR numerator). ``brow_raise``
    lifts the eyebrows. ``mouth_open`` sets lip separation.
    """
    pts: List[Point3D] = [Point3D(center_x, center_y, 0.0, 1.0) for _ in range(n)]

    def put(idx: int, x: float, y: float, z: float = 0.0) -> None:
        pts[idx] = Point3D(x, y, z, 1.0)

    # Nose tip carries yaw (horizontal offset from face center).
    put(FaceMesh.NOSE_TIP, center_x + yaw, center_y, -0.02)
    put(FaceMesh.CHIN, center_x, center_y + 0.12)
    put(FaceMesh.FOREHEAD_TOP, center_x, center_y - 0.12)
    put(FaceMesh.LEFT_CHEEK, center_x - 0.10, center_y)
    put(FaceMesh.RIGHT_CHEEK, center_x + 0.10, center_y)

    # Eyes: width fixed at 0.06, vertical opening = eye_open * width.
    half_h = (eye_open * 0.06) / 2.0
    # Left eye around (center_x - 0.05)
    lx = center_x - 0.05
    put(FaceMesh.LEFT_EYE_OUTER, lx - 0.03, center_y - 0.03)
    put(FaceMesh.LEFT_EYE_INNER, lx + 0.03, center_y - 0.03)
    put(FaceMesh.LEFT_EYE_TOP, lx, center_y - 0.03 - half_h)
    put(FaceMesh.LEFT_EYE_BOTTOM, lx, center_y - 0.03 + half_h)
    put(FaceMesh.LEFT_IRIS_CENTER, lx, center_y - 0.03)
    # Right eye around (center_x + 0.05)
    rx = center_x + 0.05
    put(FaceMesh.RIGHT_EYE_OUTER, rx + 0.03, center_y - 0.03)
    put(FaceMesh.RIGHT_EYE_INNER, rx - 0.03, center_y - 0.03)
    put(FaceMesh.RIGHT_EYE_TOP, rx, center_y - 0.03 - half_h)
    put(FaceMesh.RIGHT_EYE_BOTTOM, rx, center_y - 0.03 + half_h)
    put(FaceMesh.RIGHT_IRIS_CENTER, rx, center_y - 0.03)

    # Eyebrows sit above eyes; brow_raise lifts them further.
    put(FaceMesh.LEFT_BROW, lx, center_y - 0.06 - brow_raise)
    put(FaceMesh.RIGHT_BROW, rx, center_y - 0.06 - brow_raise)

    # Mouth: width fixed, vertical opening = mouth_open.
    put(FaceMesh.MOUTH_LEFT, center_x - 0.04, center_y + 0.07)
    put(FaceMesh.MOUTH_RIGHT, center_x + 0.04, center_y + 0.07)
    put(FaceMesh.UPPER_LIP, center_x, center_y + 0.07 - mouth_open / 2.0)
    put(FaceMesh.LOWER_LIP, center_x, center_y + 0.07 + mouth_open / 2.0)

    return FaceLandmarks(points=tuple(pts))


def make_pose(
    *,
    shoulder_tilt: float = 0.0,
    lean: float = 0.0,
    left_wrist: Optional[tuple] = None,
    right_wrist: Optional[tuple] = None,
    visibility: float = 1.0,
) -> PoseLandmarks:
    """Build a synthetic upright pose.

    ``shoulder_tilt`` raises the right shoulder by that many normalized units
    (creating an uneven-shoulder posture). ``lean`` shifts the shoulders
    horizontally relative to the hips (torso lean). Wrists default to a neutral
    rest position near the hips.
    """
    pts: List[Point3D] = [Point3D(0.5, 0.5, 0.0, visibility) for _ in range(Pose.NUM_LANDMARKS)]

    def put(idx: int, x: float, y: float, z: float = 0.0, vis: float = visibility) -> None:
        pts[idx] = Point3D(x, y, z, vis)

    shoulder_y = 0.35
    hip_y = 0.70
    put(Pose.LEFT_SHOULDER, 0.40 + lean, shoulder_y)
    put(Pose.RIGHT_SHOULDER, 0.60 + lean, shoulder_y - shoulder_tilt)
    put(Pose.LEFT_HIP, 0.43, hip_y)
    put(Pose.RIGHT_HIP, 0.57, hip_y)
    put(Pose.NOSE, 0.50 + lean, 0.20)
    put(Pose.LEFT_EAR, 0.46 + lean, 0.22)
    put(Pose.RIGHT_EAR, 0.54 + lean, 0.22)

    lw = left_wrist if left_wrist is not None else (0.40, 0.68)
    rw = right_wrist if right_wrist is not None else (0.60, 0.68)
    put(Pose.LEFT_WRIST, lw[0], lw[1])
    put(Pose.RIGHT_WRIST, rw[0], rw[1])
    put(Pose.LEFT_ELBOW, 0.38, 0.52)
    put(Pose.RIGHT_ELBOW, 0.62, 0.52)

    return PoseLandmarks(points=tuple(pts))


def make_hand(
    *,
    center: tuple = (0.5, 0.6),
    openness: float = 1.0,
    handedness: str = "Right",
) -> HandLandmarks:
    """Build a synthetic hand. ``openness`` scales fingertip distance from wrist.

    Hand scale (wrist→middle-MCP) is fixed at 0.05, so the openness metric
    (mean fingertip distance / scale) is monotonic in ``openness``.
    """
    cx, cy = center
    pts = [Point3D(cx, cy, 0.0, 1.0) for _ in range(Hands.NUM_LANDMARKS)]
    pts[Hands.WRIST] = Point3D(cx, cy, 0.0, 1.0)
    pts[Hands.MIDDLE_MCP] = Point3D(cx, cy - 0.05, 0.0, 1.0)
    tip_dist = openness * 0.12
    spread = (-0.04, -0.02, 0.0, 0.02, 0.04)
    for tip, dx in zip(Hands.FINGERTIPS, spread):
        pts[tip] = Point3D(cx + dx, cy - tip_dist, 0.0, 1.0)
    return HandLandmarks(points=tuple(pts), handedness=handedness)


def make_frame(
    index: int = 0,
    timestamp: float = 0.0,
    *,
    face: Optional[FaceLandmarks] = None,
    pose: Optional[PoseLandmarks] = None,
    hands: Optional[tuple] = None,
    with_face: bool = True,
    with_pose: bool = True,
    with_hands: bool = False,
) -> FrameLandmarks:
    """Assemble a FrameLandmarks, defaulting to a neutral well-posed speaker.

    Hands are opt-in (``with_hands=True`` or an explicit ``hands`` tuple) since
    most face/pose tests don't need them.
    """
    if face is None and with_face:
        face = make_face()
    if pose is None and with_pose:
        pose = make_pose()
    if hands is None:
        hands = (make_hand(),) if with_hands else ()
    box = None
    if face is not None:
        box = BoundingBox.from_points(face.points, padding=0.02)
    return FrameLandmarks(
        index=index, timestamp=timestamp, face=face, pose=pose, hands=tuple(hands), face_box=box
    )


@pytest.fixture
def neutral_frame() -> FrameLandmarks:
    """A single well-posed, forward-facing speaker frame."""
    return make_frame()

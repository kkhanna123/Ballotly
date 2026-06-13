"""Pluggable landmark extraction.

The abstract interfaces and the fakes have no third-party dependencies and are
always importable. The MediaPipe-backed extractors are imported lazily via
:func:`get_mediapipe_extractors` so that environments without MediaPipe (e.g.
the unit-test runner) can still import this package.
"""

from .base import FaceExtractor, HandExtractor, LandmarkPipeline, PoseExtractor
from .fakes import (
    ScriptedFaceExtractor,
    ScriptedHandExtractor,
    ScriptedPoseExtractor,
)

__all__ = [
    "FaceExtractor",
    "PoseExtractor",
    "HandExtractor",
    "LandmarkPipeline",
    "ScriptedFaceExtractor",
    "ScriptedPoseExtractor",
    "ScriptedHandExtractor",
    "MediaPipeExtractors",
    "get_mediapipe_extractors",
]


class MediaPipeExtractors:
    """Bundle of the (optional) face/pose/hand extractors for a run."""

    def __init__(self, face=None, pose=None, hands=None) -> None:
        self.face = face
        self.pose = pose
        self.hands = hands


def get_mediapipe_extractors(
    *,
    extract_face: bool = True,
    extract_pose: bool = True,
    extract_hands: bool = True,
    max_num_faces: int = 1,
    max_num_hands: int = 2,
) -> MediaPipeExtractors:
    """Construct the requested MediaPipe extractors (any may be ``None``).

    Imported lazily to avoid a hard MediaPipe dependency at package import time.
    """
    face = pose = hands = None
    if extract_face:
        from .face import MediaPipeFaceExtractor

        face = MediaPipeFaceExtractor(max_num_faces=max_num_faces)
    if extract_pose:
        from .pose import MediaPipePoseExtractor

        pose = MediaPipePoseExtractor()
    if extract_hands:
        from .hands import MediaPipeHandExtractor

        hands = MediaPipeHandExtractor(max_num_hands=max_num_hands)
    return MediaPipeExtractors(face=face, pose=pose, hands=hands)

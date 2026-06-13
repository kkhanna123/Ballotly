"""Pluggable landmark extraction.

The abstract interfaces and the fakes have no third-party dependencies and are
always importable. The MediaPipe-backed extractors are imported lazily via
:func:`get_mediapipe_extractors` so that environments without MediaPipe (e.g.
the unit-test runner) can still import this package.
"""

from .base import FaceExtractor, LandmarkPipeline, PoseExtractor
from .fakes import ScriptedFaceExtractor, ScriptedPoseExtractor

__all__ = [
    "FaceExtractor",
    "PoseExtractor",
    "LandmarkPipeline",
    "ScriptedFaceExtractor",
    "ScriptedPoseExtractor",
    "get_mediapipe_extractors",
]


def get_mediapipe_extractors(
    *,
    extract_face: bool = True,
    extract_pose: bool = True,
    max_num_faces: int = 1,
):
    """Construct (face_extractor, pose_extractor), either may be ``None``.

    Imported lazily to avoid a hard MediaPipe dependency at package import time.
    """
    face = None
    pose = None
    if extract_face:
        from .face import MediaPipeFaceExtractor

        face = MediaPipeFaceExtractor(max_num_faces=max_num_faces)
    if extract_pose:
        from .pose import MediaPipePoseExtractor

        pose = MediaPipePoseExtractor()
    return face, pose

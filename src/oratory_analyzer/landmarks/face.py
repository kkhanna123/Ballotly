"""MediaPipe Face Mesh adapter.

Wraps ``mediapipe.solutions.face_mesh`` behind :class:`FaceExtractor`. With
``refine_landmarks=True`` MediaPipe returns 478 points (468 mesh + 10 iris),
which our metrics use for gaze/eye-contact estimation.

This module is intentionally thin and is excluded from unit-test coverage — it
is exercised by the ``requires_mediapipe`` integration test and the end-to-end
run, not by the pure unit suite.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..domain.landmarks import BoundingBox, FaceLandmarks, Point3D
from .base import FaceExtractor


class MediaPipeFaceExtractor(FaceExtractor):
    """Detects up to ``max_num_faces`` faces; ``extract`` returns the largest."""

    def __init__(
        self,
        *,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        import mediapipe as mp  # imported lazily so the core has no hard dep

        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _to_landmarks(mp_landmarks) -> FaceLandmarks:
        pts = tuple(
            Point3D(x=lm.x, y=lm.y, z=lm.z, visibility=1.0)
            for lm in mp_landmarks.landmark
        )
        return FaceLandmarks(points=pts)

    def extract_all(self, frame_bgr: np.ndarray) -> List[FaceLandmarks]:
        """Return every detected face mesh in the frame (possibly empty)."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return []
        return [self._to_landmarks(f) for f in results.multi_face_landmarks]

    def extract(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        faces = self.extract_all(frame_bgr)
        if not faces:
            return None
        # Return the largest face by bounding-box area (most prominent person).
        return max(faces, key=lambda f: BoundingBox.from_points(f.points).area)

    def close(self) -> None:
        self._mesh.close()

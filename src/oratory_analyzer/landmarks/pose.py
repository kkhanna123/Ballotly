"""MediaPipe Pose adapter.

Wraps ``mediapipe.solutions.pose`` behind :class:`PoseExtractor`. MediaPipe Pose
tracks the single most prominent person in frame, which for a speaker clip is
the speaker. Each landmark carries a ``visibility`` confidence we propagate.

Thin I/O edge — excluded from unit-test coverage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain.landmarks import Point3D, PoseLandmarks
from .base import PoseExtractor


class MediaPipePoseExtractor(PoseExtractor):
    def __init__(
        self,
        *,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        smooth_landmarks: bool = True,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: np.ndarray) -> Optional[PoseLandmarks]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return None
        pts = tuple(
            Point3D(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            for lm in results.pose_landmarks.landmark
        )
        return PoseLandmarks(points=pts)

    def close(self) -> None:
        self._pose.close()
